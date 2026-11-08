# Multi-Agent Calibration

A Python testbed asking whether three same-model agents with different personas become more accurate and better calibrated after reading each other's answers.

## What the saved pilot shows

30 hand-curated MCQs, three agents, two rounds (180 predictions). The original findings identify the model as llama-3.1-8b-instant via Groq; the historical run has no manifest, so its provider provenance cannot be independently verified from the logs.

| Measure | Independent round | Post-debate round |
|---|---:|---:|
| Individual accuracy | 75.56% | 72.22% |
| Mean stated confidence | 90.44% | 87.17% |
| ECE (10 bins) | 0.1489 | 0.1672 |
| Correctness Brier score | 0.1981 | 0.2192 |
| Majority vote accuracy | 76.67% | 80.00% |
| Confidence-weighted accuracy | 80.00% | 83.33% |

Votes in this table abstain on ties and count abstentions as incorrect. Original summaries used first-agent tie breaking. These policies can differ. The individual accuracy change is -3.33 percentage points; a paired question bootstrap gives a descriptive 95% interval of [-8.89, +2.22] points. This small convenience sample supports exploratory observations, not a general claim that debate harms reliability.

## Architecture

Question -> three independent persona responses -> synchronous debate rounds using only peers' previous-round responses -> final majority and confidence-weighted votes -> CSV logs, metrics and plots.

Agents use the same model, but persona instructions directly influence confidence. This controls model identity, not every source of bias. Each round is a fresh request: agents receive peer history but not their own previous response. A self-history control is an important next experiment.

## Reproduce without an API key

Requires Python 3.11+.

```sh
git clone https://github.com/shriya5752/multi-agent-calibration.git
cd multi-agent-calibration
python -m venv .venv
# Activate the virtual environment for your shell.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.audit_results
python -m src.run_experiment --mock --n-questions 8 --rounds 2 --seed 42
```

The audit reads committed logs without API calls. Mock runs only test the pipeline; they are not experimental evidence. New runs go into unique `runs/` directories, preserving the committed results. `--output-dir` must name a directory that does not exist. Each new run saves a manifest containing model, mode, seed, dataset hash, temperature and time. The seed makes mock sampling repeatable; it does not guarantee provider reproducibility.

For live calls, copy `.env.example` to `.env`, configure `LLM_API_KEY` and `LLM_BASE_URL`, then omit `--mock`. Live calls may incur provider costs. No live experiment was performed as part of this refinement.

## Reliability choices

Malformed answers or missing/out-of-range confidence raise an explicit error rather than becoming random predictions. Raw response text is retained for successful future runs. Voting ties abstain and count as incorrect; tied flags are saved. ECE includes confidence endpoints. Tests cover parsing, calibration boundaries, tie handling and synchronous round isolation.

## Files

- `src/agent.py`: prompts, response parsing and model calls
- `src/orchestrator.py`: synchronous rounds, transition logs and final voting
- `src/metrics.py`: accuracy, ECE, confidence drift and belief flips
- `src/audit_results.py`: offline baseline comparison, Brier score and paired bootstrap
- `src/run_experiment.py`: CLI, run manifest and plots
- `docs/findings.md`: qualified pilot interpretation
- `docs/interview-readiness.md`: prioritized extensions and interview narrative

## Limits

30 manually selected questions, one historical run, correlated same-model agents, confidence-shaped personas, no compute-matched no-peer control, no inference-cost accounting, no resume after failure, and no historical raw provider responses. Confidence is a stated probability of the chosen answer being correct; the Brier score here measures that binary event, not a full multiclass probability distribution. ECE depends on binning. Question bootstrap intervals do not capture repeated-run/model uncertainty or justify population generalization.
