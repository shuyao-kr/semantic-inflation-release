"""
analysis_tools.py

Post-hoc analysis utilities for SIR result CSVs produced by sir_pipeline.py.

This script implements two analyses referenced in the paper:

  1. Vocabulary analysis (Section IV.B, Table III):
     Split rows into a High-SIR group (top quartile by inflation_rate) and
     a Low-SIR group (bottom quartile), tokenize the raw text, and report
     the top-K most frequent terms per group.

  2. Bad-case extraction (Section V.B, Table IV):
     Rank rows by absolute error |ai_estimated_hours - leader_hours|
     and emit the top-K worst cases, separated into
     "over-estimation" (AI > leader) and "under-estimation" (AI < leader)
     buckets.

This script is intentionally simple. It does NOT call the LLM, does NOT
need a local inference server, and does NOT ship with any data; it
operates on per-row result CSVs you produce yourself. Stopword handling
is lightweight: for English a short built-in list is used, and for
multilingual inputs users may pass their own stopword list via
--stopwords-file.

Examples (replace your_*.csv with your own paths, all relative to the repo root)
--------
# Vocabulary comparison
python analysis_tools.py vocab \\
    --results your_results.csv \\
    --top-k 20 \\
    --output your_vocab.csv

# Top-20 bad cases
python analysis_tools.py bad-cases \\
    --results your_results.csv \\
    --top-k 20 \\
    --output your_bad_cases.csv
"""

import argparse
import re
import sys
from collections import Counter
from typing import Iterable, List, Set

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

REQUIRED_COLUMNS = ("raw_text", "leader_hours", "ai_estimated_hours", "inflation_rate")

DEFAULT_EN_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "but",
    "with", "at", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "this", "that", "these", "those", "it", "its", "i",
    "we", "you", "he", "she", "they", "them", "my", "our", "your", "their",
    "do", "did", "done", "have", "has", "had", "not", "no", "so", "than",
    "then", "there", "here", "if", "when", "while", "also", "just", "about",
    "into", "over", "under", "after", "before", "between", "because", "any",
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="ignore")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    df = df.drop_duplicates().copy()
    for c in ("leader_hours", "ai_estimated_hours", "inflation_rate"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df[df["leader_hours"] > 0]
    return df


def load_stopwords(path: str) -> Set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


# -----------------------------------------------------------------------------
# 1. Vocabulary analysis
# -----------------------------------------------------------------------------

# Token pattern: any run of word characters (Unicode-aware). For CJK inputs
# the caller should pre-tokenize `raw_text` (e.g., via jieba) before passing
# the CSV in, or supply a --token-regex that matches single characters.
DEFAULT_TOKEN_RE = r"\w+"


def tokenize(texts: Iterable[str],
             token_re: str,
             stopwords: Set[str]) -> Counter:
    pattern = re.compile(token_re, flags=re.UNICODE)
    counter: Counter = Counter()
    for t in texts:
        if not isinstance(t, str):
            continue
        for tok in pattern.findall(t.lower()):
            if tok in stopwords or len(tok) < 2:
                continue
            counter[tok] += 1
    return counter


def split_high_low(df: pd.DataFrame,
                   low_quantile: float = 0.25,
                   high_quantile: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_low  = df["inflation_rate"].quantile(low_quantile)
    q_high = df["inflation_rate"].quantile(high_quantile)
    low_df  = df[df["inflation_rate"] <= q_low]
    high_df = df[df["inflation_rate"] >= q_high]
    return high_df, low_df


def vocab_command(args: argparse.Namespace) -> None:
    df = load_results(args.results)

    stopwords = set(DEFAULT_EN_STOPWORDS)
    if args.stopwords_file:
        stopwords |= load_stopwords(args.stopwords_file)

    high_df, low_df = split_high_low(df,
                                     low_quantile=args.low_quantile,
                                     high_quantile=args.high_quantile)
    print(f"[info] N={len(df)}  High-SIR group N={len(high_df)}  "
          f"Low-SIR group N={len(low_df)}")
    print(f"[info] SIR quartiles: low<= {df['inflation_rate'].quantile(args.low_quantile):.4f}   "
          f"high>= {df['inflation_rate'].quantile(args.high_quantile):.4f}")

    high_counts = tokenize(high_df["raw_text"], args.token_regex, stopwords)
    low_counts  = tokenize(low_df["raw_text"],  args.token_regex, stopwords)

    top_high = high_counts.most_common(args.top_k)
    top_low  = low_counts.most_common(args.top_k)

    # Output as a side-by-side comparison table
    rows = []
    for i in range(args.top_k):
        h_tok, h_freq = top_high[i] if i < len(top_high) else ("", 0)
        l_tok, l_freq = top_low[i]  if i < len(top_low)  else ("", 0)
        rows.append({
            "rank":               i + 1,
            "high_sir_token":     h_tok,
            "high_sir_freq":      h_freq,
            "low_sir_token":      l_tok,
            "low_sir_freq":       l_freq,
        })
    out = pd.DataFrame(rows)

    print("\n" + out.to_string(index=False))
    if args.output:
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n[done] Wrote {args.output}")


# -----------------------------------------------------------------------------
# 2. Bad-case extraction
# -----------------------------------------------------------------------------

def bad_cases_command(args: argparse.Namespace) -> None:
    df = load_results(args.results)
    df = df.copy()
    df["abs_error"]    = (df["ai_estimated_hours"] - df["leader_hours"]).abs()
    df["signed_error"] = df["ai_estimated_hours"] - df["leader_hours"]

    over  = df[df["signed_error"] > 0].nlargest(args.top_k,  "abs_error")
    under = df[df["signed_error"] < 0].nlargest(args.top_k,  "abs_error")

    over = over.assign(error_type="over-estimation")
    under = under.assign(error_type="under-estimation")
    out = pd.concat([over, under], ignore_index=True)

    cols = ["error_type", "raw_text", "leader_hours",
            "ai_estimated_hours", "signed_error", "abs_error",
            "inflation_rate"]
    out = out[[c for c in cols if c in out.columns]]

    print(f"[info] Top {args.top_k} over-estimation cases and "
          f"top {args.top_k} under-estimation cases.")
    print(f"[info] Max |error|: over={over['abs_error'].max():.1f}h   "
          f"under={under['abs_error'].max():.1f}h")

    preview = out.copy()
    preview["raw_text"] = preview["raw_text"].str.slice(0, 80) + "..."
    print("\n" + preview.to_string(index=False))

    if args.output:
        out.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"\n[done] Wrote {args.output}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("vocab",
                        help="Top-K vocabulary in High-SIR vs Low-SIR groups.")
    pv.add_argument("--results",        required=True)
    pv.add_argument("--output",         default=None,
                    help="Optional output CSV path.")
    pv.add_argument("--top-k",          type=int,   default=20)
    pv.add_argument("--low-quantile",   type=float, default=0.25)
    pv.add_argument("--high-quantile",  type=float, default=0.75)
    pv.add_argument("--token-regex",    default=DEFAULT_TOKEN_RE,
                    help=r"Regex used for tokenization. Default: \w+. "
                         r"For raw CJK, pre-tokenize your raw_text column "
                         r"(e.g., with jieba) before running this tool.")
    pv.add_argument("--stopwords-file", default=None,
                    help="Optional file with one stopword per line. "
                         "Merged with the built-in small English list.")
    pv.set_defaults(func=vocab_command)

    pb = sub.add_parser("bad-cases",
                        help="Top-K AI-vs-leader errors, split into "
                             "over- and under-estimation.")
    pb.add_argument("--results", required=True)
    pb.add_argument("--output",  default=None)
    pb.add_argument("--top-k",   type=int, default=10)
    pb.set_defaults(func=bad_cases_command)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
