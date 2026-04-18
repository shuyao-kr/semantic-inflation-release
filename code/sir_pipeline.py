"""
Semantic Inflation Rate (SIR) pipeline.

For each row of the input table:
  1. Concatenate work_name + content + notes into a raw log string.
  2. Call a local, OpenAI-compatible LLM endpoint and ask for
       (a) a standardized ("de-embellished") restatement, and
       (b) an estimated work-hour count.
  3. Compute SDI (Semantic Deviation Index) as 1 - cosine similarity
     between Sentence-BERT embeddings of the raw and standardized texts.
  4. Compute SIR = SDI * ln(length(raw_text) + 1) / (leader_hours + epsilon).
  5. Write the augmented rows to an output CSV.

No cloud API is called. Point LOCAL_API_URL at your on-premise inference
server (e.g., a llama.cpp HTTP server loading a GGUF-quantized checkpoint).
"""

import argparse
import json
import math
import re
import sys
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Configuration (edit these or override with CLI flags)
# -----------------------------------------------------------------------------

LOCAL_API_URL = "http://127.0.0.1:5000/v1"
LOCAL_API_KEY = "dummy"                 # llama.cpp ignores the key value
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EPSILON = 1e-5                          # smoothing term in SIR denominator

SYSTEM_PROMPT = (
    "You are a strict corporate performance auditor. "
    "Ignore emotional coloring and rhetorical embellishment in the log. "
    "Produce a factual restatement and an estimated number of work hours."
)

USER_PROMPT_TEMPLATE = """\
Please analyze the following employee work log:
"{raw_text}"

Complete two tasks:
1. [Objective Restatement] Remove all non-factual adjectives and adverbs,
   and rewrite the core work facts in the most concise language possible.
2. [Work-Hour Estimation] Based on task difficulty and industry norms,
   estimate the reasonable time required (in hours) as a single number.

Return STRICTLY the following JSON format:
{{
    "standardized_text": "<rewritten content here>",
    "estimated_hours": <number>
}}
"""

DATA_COLUMNS = [
    "source_year", "source_month", "employee_id",
    "work_name", "content", "notes", "leader_hours",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_table(path: str) -> pd.DataFrame:
    """Load CSV (or XLSX) and validate schema."""
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}. "
            f"Expected schema: {DATA_COLUMNS}"
        )
    for c in ("work_name", "content", "notes"):
        df[c] = df[c].fillna("").astype(str)
    df["leader_hours"] = pd.to_numeric(df["leader_hours"], errors="coerce").fillna(0.0)
    df["raw_text"] = (
        "task: " + df["work_name"]
        + ". details: " + df["content"]
        + ". notes: " + df["notes"]
    )
    return df


def parse_llm_json(content: str) -> Tuple[str, float]:
    """Extract standardized_text and estimated_hours from model output."""
    if not content:
        return "", 0.0
    m = re.search(r"\{.*\}", content, flags=re.DOTALL)
    payload = m.group() if m else content
    try:
        data = json.loads(payload)
        std_text = str(data.get("standardized_text", "")).strip()
        hours = data.get("estimated_hours", 0)
        try:
            hours = float(hours)
        except (TypeError, ValueError):
            nums = re.findall(r"\d+\.?\d*", str(hours))
            hours = float(nums[0]) if nums else 0.0
        return std_text, hours
    except json.JSONDecodeError:
        # Fallback: return the whole content as the restatement, no hours
        return content.strip(), 0.0


def call_llm(client: OpenAI, raw_text: str, max_tokens: int = 500) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model="local-model",  # llama.cpp ignores this field
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(raw_text=raw_text)},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content
    except Exception as exc:                                # pragma: no cover
        print(f"[warn] LLM call failed: {exc}", file=sys.stderr)
        return None


def compute_sdi(embedder: SentenceTransformer,
                raw_text: str, std_text: str) -> float:
    if not std_text or len(std_text) < 2:
        return 0.0
    v_raw = embedder.encode([raw_text])
    v_std = embedder.encode([std_text])
    sim = float(cosine_similarity(v_raw, v_std)[0][0])
    return 1.0 - sim


def compute_sir(sdi: float, raw_len: int, leader_hours: float) -> float:
    return sdi * math.log(raw_len + 1) / (leader_hours + EPSILON)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  required=True,
                   help="Your input table (relative path, CSV or XLSX) "
                        "following the data_template/data_template.csv schema.")
    p.add_argument("--output", required=True,
                   help="Your output CSV path (relative). Suggested: your_results.csv.")
    p.add_argument("--model-name", default="unnamed-local-model",
                   help="Identifier recorded into the output CSV.")
    p.add_argument("--api-url",  default=LOCAL_API_URL)
    p.add_argument("--api-key",  default=LOCAL_API_KEY)
    p.add_argument("--max-tokens", type=int, default=500)
    args = p.parse_args()

    print(f"[info] Loading input from {args.input}")
    df = load_table(args.input)
    print(f"[info] {len(df)} rows loaded.")

    print(f"[info] Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"[info] Connecting to local LLM at {args.api_url}")
    client = OpenAI(base_url=args.api_url, api_key=args.api_key)

    outputs = []
    t0 = time.time()
    for idx, row in df.iterrows():
        raw_text = row["raw_text"]
        llm_output = call_llm(client, raw_text, max_tokens=args.max_tokens)
        std_text, ai_hours = parse_llm_json(llm_output or "")
        sdi = compute_sdi(embedder, raw_text, std_text)
        sir = compute_sir(sdi, len(raw_text), float(row["leader_hours"]))
        outputs.append({
            "model_name":          args.model_name,
            "source_year":         row["source_year"],
            "source_month":        row["source_month"],
            "employee_id":         row["employee_id"],
            "raw_text":            raw_text,
            "std_text":            std_text,
            "leader_hours":        float(row["leader_hours"]),
            "ai_estimated_hours":  ai_hours,
            "sdi":                 sdi,
            "inflation_rate":      sir,
        })
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[info] {idx + 1}/{len(df)}   elapsed={elapsed:.1f}s")

    out_df = pd.DataFrame(outputs)
    out_df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"[done] Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
