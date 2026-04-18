# Semantic Inflation Rate (SIR) — Reference Code

Reference implementation accompanying the paper

> *Can Quantized On-Premise LLMs Flag Semantic Inflation in Employee Work Logs? A Privacy-Preserving Decision-Support Workflow.*

This repository contains the **code** used to (i) run on-premise quantized-LLM inference over self-reported work logs, (ii) compute the Semantic Inflation Rate (SIR), (iii) benchmark multiple quantization levels and model families against leader-approved work hours, and (iv) perform the outlier-robustness checks discussed in Section IV.B of the paper.

## What is NOT in this repository

**No enterprise data is included.** The dataset used in the paper (8,603 monthly-report entries from a single Chinese EdTech R&D division) is restricted by an internal data-use agreement and cannot be released. For the same reason, we do not ship example records, a paraphrased sample, or translated fixtures — even a small sample would risk re-identification of specific employees or internal projects.

What we provide instead is:

- The code, as run in the paper.
- A **schema-only** data template (`data_template/data_template.csv`) containing just the column headers. Users must populate this file with their own records for the code to run.

Researchers wishing to reproduce our results on our specific dataset are welcome to contact the corresponding author; access will be considered case-by-case under the original data-use agreement.

## Repository layout

```
semantic-inflation-release/
├── README.md                       (this file)
├── requirements.txt                (Python dependencies)
├── code/
│   ├── sir_pipeline.py             (main pipeline: restatement + SDI + SIR per log)
│   ├── benchmark_evaluation.py     (500-entry benchmark for R / MAE across models)
│   ├── robustness_check.py         (trimming- and winsorization-based outlier checks)
│   └── analysis_tools.py           (vocabulary and bad-case analyses, Section IV.B / V.B)
└── data_template/
    └── data_template.csv           (headers only; populate before running)
```

## Dependencies

```
pip install -r requirements.txt
```

Required packages: `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`, `openai` (used only as a client against a **local** llama.cpp HTTP server).

## Serving the LLM on-premise

The code expects an OpenAI-compatible HTTP endpoint pointing at a **local** inference backend (the paper uses [llama.cpp](https://github.com/ggerganov/llama.cpp) loading a GGUF-quantized checkpoint of Qwen-2.5-7B-Instruct; any OpenAI-compatible local server will work). The default endpoint is `http://127.0.0.1:5000/v1` with a dummy API key. Edit the `LOCAL_API_URL` / `LOCAL_API_KEY` constants at the top of each script to match your setup.

**No cloud API is used.** All inference stays on the workstation that runs the code.

## Data schema (`data_template/data_template.csv`)

The pipeline expects a single table with the following columns. **Column names and types must match**; leave cells blank for records where a field does not apply.

| column | type | description |
|---|---|---|
| `source_year` | int | Calendar year of the monthly report. |
| `source_month` | int | Calendar month (1–12). |
| `employee_id` | string | An anonymized, stable per-employee identifier. Do **not** use real names. |
| `work_name` | string | Short task name or title recorded by the employee. |
| `content` | string | Free-text description of what was done. |
| `notes` | string | Optional supplementary notes. |
| `leader_hours` | float | Leader-approved work hours for this entry (S_leader in the paper). |

The pipeline concatenates `work_name`, `content`, and `notes` into the raw log used for SIR, and uses `leader_hours` as the reference signal in the regression against AI-estimated hours.

## Usage

In the example commands below, `your_*.csv` are placeholder filenames you should replace with your own input and output paths. All paths are relative to the repository root.

### 1. Run SIR over the full table

```
python code/sir_pipeline.py \
  --input  your_data.csv \
  --output your_results.csv \
  --model-name Qwen-2.5-7B-Q4_K_M
```

`your_data.csv` should follow the schema in `data_template/data_template.csv` (column headers only). `your_results.csv` will contain the original rows plus `std_text`, `ai_estimated_hours`, `sdi`, and `inflation_rate` columns.

### 2. Build and run the 500-entry benchmark

```
python code/benchmark_evaluation.py \
  --input  your_data.csv \
  --output your_benchmark_log.csv \
  --model-name Qwen-2.5-7B-Q4_K_M \
  --n 500 --seed 42
```

This (re-)samples `N` rows with a fixed seed, calls the local LLM for each, and appends a new row of `(model, R, MAE, latency_ms, timestamp)` to `your_benchmark_log.csv`. Repeat with different `--model-name` values to compare quantization levels.

### 3. Run the outlier-robustness check

```
python code/robustness_check.py --results your_results.csv
```

The script reports Pearson $R$ and MAE on (a) the full sample, (b) after trimming the top 1% of entries by SIR, and (c) after winsorizing AI-estimated hours and leader-approved hours at the 99th percentile. It also checks whether the Q2 > Q4 > Q8 ordering across Qwen-2.5-7B quantization variants is preserved (if results for multiple variants are supplied).

### 4. Vocabulary and bad-case analyses

`analysis_tools.py` reproduces the two post-hoc analyses described in the paper: the High-SIR vs. Low-SIR vocabulary comparison (Section IV.B, Table III) and the top-K AI-vs-leader error extraction (Section V.B, Table IV).

```
# High-SIR vs Low-SIR word-frequency comparison (top 20 per group)
python code/analysis_tools.py vocab \
    --results your_results.csv \
    --top-k 20 \
    --output your_vocab.csv

# Top-10 over-estimation + top-10 under-estimation cases
python code/analysis_tools.py bad-cases \
    --results your_results.csv \
    --top-k 10 \
    --output your_bad_cases.csv
```

The vocabulary tool uses a lightweight English stopword list by default. For CJK inputs, pre-tokenize the `raw_text` column (e.g., with `jieba`) before running, or pass a custom token regex and stopword file.

## Reproducibility notes

- All sampling uses fixed random seeds (`RANDOM_SEED = 42` in `benchmark_evaluation.py`).
- LLM decoding is deterministic-ish: `temperature = 0.1`, fixed max tokens, identical system prompt across models. Minor run-to-run drift on quantized models is expected.
- SIR values and correlations depend on the data the user provides; the absolute scale is dataset-dependent. The metric should be re-calibrated per department / institution before operational use.

## Privacy and intended use

This code is released for academic reproducibility. It is **not** a ready-to-deploy personnel-evaluation system. The paper positions SIR as a **decision-support indicator** for human reviewers, not as a definitive performance score. Any deployment on real employee data should:

1. Use anonymized identifiers, not real names, in inputs and outputs.
2. Keep raw logs, model weights, and derived embeddings on an access-controlled on-premise workstation.
3. Treat flagged logs as candidates for human review, not as automated judgments.
4. Periodically re-calibrate against fresh leader-approved ratings; do not reuse thresholds across departments without re-calibration.

## License

The code is released under the MIT License (see `LICENSE` if added). The dataset used in the paper is **not** covered by this license and is **not** redistributed.

## Citation

If you use this code, please cite the accompanying paper (full citation to be added upon publication).
