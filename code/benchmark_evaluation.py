"""
500-entry benchmark evaluation.

Samples N rows from the input table with a fixed random seed, queries the
local LLM for a single work-hour estimate per row (no restatement step,
minimal tokens so latency is dominated by the model), and reports
Pearson R and MAE against leader_hours. Each run appends one summary
line to the benchmark log CSV so that multiple model / quantization
configurations can be compared side-by-side.

No cloud API is used.
"""

import argparse
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import mean_absolute_error


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LOCAL_API_URL = "http://127.0.0.1:5000/v1"
LOCAL_API_KEY = "dummy"

DEFAULT_N    = 500
DEFAULT_SEED = 42

SYSTEM_PROMPT = "You are an objective work-hour evaluator."
USER_PROMPT_TEMPLATE = (
    "Estimate the reasonable duration (in hours) required for the following "
    "task. Output only a single number.\n{text}"
)

DATA_COLUMNS = [
    "source_year", "source_month", "employee_id",
    "work_name", "content", "notes", "leader_hours",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_and_filter(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing columns: {missing}")
    for c in ("work_name", "content", "notes"):
        df[c] = df[c].fillna("").astype(str)
    df["leader_hours"] = pd.to_numeric(df["leader_hours"], errors="coerce").fillna(0.0)
    df = df[df["leader_hours"] > 0].reset_index(drop=True)
    df["full_text"] = (
        df["work_name"] + " " + df["content"] + " " + df["notes"]
    ).str.strip()
    return df


def get_sample(df: pd.DataFrame, n: int, seed: int,
               cache_path: str) -> pd.DataFrame:
    """Draw a fixed-seed sample, caching it so all models see identical rows."""
    if os.path.exists(cache_path):
        print(f"[info] Loading cached sample from {cache_path}")
        return pd.read_csv(cache_path)
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)
    sample.to_csv(cache_path, index=False, encoding="utf-8-sig")
    print(f"[info] Cached sample of {len(sample)} rows to {cache_path}")
    return sample


def run_single(client: OpenAI, text: str,
               max_tokens: int = 10) -> tuple[float, float]:
    """Return (ai_hours, latency_ms) for one row."""
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model="local-model",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        nums = re.findall(r"\d+\.?\d*", raw)
        score = float(nums[0]) if nums else 0.0
    except Exception as exc:                                # pragma: no cover
        print(f"[warn] LLM call failed: {exc}", file=sys.stderr)
        score = 0.0
    latency_ms = (time.time() - t0) * 1000.0
    return score, latency_ms


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  required=True,
                   help="Your input table (relative path), following the "
                        "data_template/data_template.csv schema.")
    p.add_argument("--output", default="your_benchmark_log.csv",
                   help="Summary log (appended to across runs). Relative path.")
    p.add_argument("--sample-cache", default="your_benchmark_sample.csv",
                   help="Where to cache the drawn sample so subsequent model "
                        "runs see identical rows. Relative path.")
    p.add_argument("--model-name", required=True,
                   help="Label written into the summary log.")
    p.add_argument("--n",    type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-tokens", type=int, default=10)
    p.add_argument("--api-url",  default=LOCAL_API_URL)
    p.add_argument("--api-key",  default=LOCAL_API_KEY)
    p.add_argument("--save-rows", default=None,
                   help="Optional: path to save per-row AI estimates for later "
                        "robustness checks.")
    args = p.parse_args()

    df = load_and_filter(args.input)
    print(f"[info] {len(df)} valid rows after filtering leader_hours > 0.")

    sample = get_sample(df, n=args.n, seed=args.seed, cache_path=args.sample_cache)
    print(f"[info] Running {args.model_name} over {len(sample)} rows...")

    client = OpenAI(base_url=args.api_url, api_key=args.api_key)
    scores, latencies = [], []
    for i, row in enumerate(sample.itertuples(index=False)):
        s, ms = run_single(client, getattr(row, "full_text"),
                           max_tokens=args.max_tokens)
        scores.append(s)
        latencies.append(ms)
        if i % 50 == 0:
            print(f"[info] {i}/{len(sample)}")

    sample = sample.copy()
    sample["ai_score"] = scores
    sample["latency_ms"] = latencies

    correlation = sample["leader_hours"].corr(sample["ai_score"])
    mae         = mean_absolute_error(sample["leader_hours"], sample["ai_score"])
    avg_latency = float(np.mean(latencies))

    print("\n" + "=" * 40)
    print(f"Model [{args.model_name}]")
    print("=" * 40)
    print(f"Pearson R:   {correlation:.4f}")
    print(f"MAE:         {mae:.4f}")
    print(f"Avg latency: {avg_latency:.2f} ms/req")
    print("=" * 40)

    new_row = pd.DataFrame([{
        "model":       args.model_name,
        "pearson_r":   correlation,
        "mae":         mae,
        "latency_ms":  avg_latency,
        "n":           len(sample),
        "seed":        args.seed,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }])
    if os.path.exists(args.output):
        new_row.to_csv(args.output, mode="a", header=False, index=False)
    else:
        new_row.to_csv(args.output, index=False)
    print(f"[done] Appended summary to {args.output}")

    if args.save_rows:
        sample.to_csv(args.save_rows, index=False, encoding="utf-8-sig")
        print(f"[done] Saved per-row estimates to {args.save_rows}")


if __name__ == "__main__":
    main()
