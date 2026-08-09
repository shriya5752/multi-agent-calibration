# Multi-Agent Calibration: Does Debate Make LLM Agents Better Calibrated?

A small multi-agent system where three LLM agents (same base model, different personas)
independently answer a question, debate over multiple rounds by reading each other's
answers and reasoning, and revise their confidence. The project measures whether
**debate improves accuracy**, whether **stated confidence tracks actual correctness**
(calibration), and whether agents get swayed by confident-but-wrong peers.

This started as a scaled-down testbed for a larger research question I'm exploring
(metacognitive sensitivity and belief drift in multi-agent LLM deliberation) — this repo
is the self-contained, reproducible version of that idea.

## Why this, and not just another "multi-agent chatbot" repo

Most multi-agent demos show that agents *can* talk to each other. This project instead
asks a measurable question — **does talking to each other make them more accurate and
better calibrated, or does it just make them agree with whoever sounds confident?** —
and answers it with real metrics (calibration curves, expected calibration error,
belief-flip analysis), not just example transcripts.

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │           Question (MCQ)             │
                     └───────────────────┬───────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
     ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
     │  Agent A         │        │  Agent B         │        │  Agent C         │
     │  Cautious        │        │  Confident       │        │  Devil's         │
     │  Skeptic         │        │  Solver          │        │  Advocate        │
     └────────┬─────────┘        └────────┬─────────┘        └────────┬─────────┘
              │  answer + confidence(0-100) + reasoning  [ROUND 1]      │
              └───────────────────────────┼───────────────────────────┘
                                          ▼
                     each agent now sees the OTHER two agents'
                     round-1 answer, confidence, and reasoning
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
     ┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
     │  Agent A revises │        │  Agent B revises │        │  Agent C revises │
     └────────┬─────────┘        └────────┬─────────┘        └────────┬─────────┘
              │           [ROUND 2 ... N, same pattern]                │
              └───────────────────────────┼───────────────────────────┘
                                          ▼
                     ┌─────────────────────────────────────┐
                     │  Aggregate: majority vote  vs.       │
                     │  confidence-weighted vote            │
                     └─────────────────────────────────────┘
                                          │
                                          ▼
                     Every agent/round logged → metrics + plots
```

Every agent uses the **same underlying model** but a different system-prompt persona.
This is a deliberate control: it isolates belief dynamics (how confidence and answers
shift from social pressure within the debate) from raw model-capability differences,
which would otherwise confound the analysis.

## What gets measured

| Metric | What it tells you |
|---|---|
| **Accuracy by round** | Does debate actually help, or does groupthink cancel out the benefit of multiple perspectives? |
| **Calibration curve / Expected Calibration Error (ECE)** | When an agent says "90% confident," is it right ~90% of the time? Is it over- or under-confident? |
| **Confidence drift by round** | Do agents converge toward higher/lower confidence after seeing peers? |
| **Belief-flip rate** | How often does an agent change its answer after debate, and does that flip more often move it toward or away from the correct answer? |
| **Majority vote vs. confidence-weighted vote** | Which aggregation strategy actually performs better? |

## Project structure

```
multi-agent-calibration/
├── README.md
├── requirements.txt
├── .env.example
├── src/
│   ├── agent.py          # single agent call: prompt building + response parsing
│   ├── personas.py        # system prompts for the 3 agent personas
│   ├── orchestrator.py    # runs the multi-round debate loop, per question
│   ├── metrics.py         # calibration, ECE, flip-rate, accuracy calculations
│   ├── llm_client.py       # OpenAI-compatible client factory (Groq / OpenAI / etc)
│   └── run_experiment.py  # main entry point — run this
├── data/
│   └── questions.json     # 30 MCQ questions (mix of reasoning + CS/ML knowledge)
├── results/
│   ├── raw_logs.csv       # generated: one row per agent per round per question
│   ├── summary.csv        # generated: one row per question, final aggregated answer
│   ├── metrics.json       # generated: headline numbers
│   └── plots/              # generated: accuracy, calibration, drift plots
└── docs/
    └── findings.md         # write your analysis here after running a real experiment
```

## Setup

```bash
git clone <your-repo-url>
cd multi-agent-calibration
pip install -r requirements.txt
cp .env.example .env
# edit .env and add a free API key from https://console.groq.com
```

## Usage

**Test the pipeline first with zero API cost** (randomized fake responses — verifies
the plumbing works end-to-end before you spend real API calls):

```bash
python -m src.run_experiment --mock --n-questions 8 --rounds 2
```

**Run the real experiment:**

```bash
python -m src.run_experiment --rounds 2
```

Optional flags:

```bash
python -m src.run_experiment --n-questions 15 --rounds 3 --model llama-3.1-8b-instant
```

Output lands in `results/`: `raw_logs.csv`, `summary.csv`, `metrics.json`, and
three plots in `results/plots/`.

## Findings

_Run the real experiment and fill in `docs/findings.md`, then summarize the headline
result here. Things worth specifically checking once you have real data:_

- _Did round-2/3 accuracy actually beat round-1 accuracy, or did agents converge on a
  confidently-wrong answer?_
- _What was the ECE at round 1 vs. the final round — did debate make calibration better
  or worse?_
- _Look at `flip_hurt_rate` vs `flip_helped_rate` — are agents more often talked out of
  correct answers, or into them?_

## Limitations / what I'd extend with more time

- All three agents currently share one base model; a natural follow-up is giving each
  agent a *different* model family to separate "persona effect" from "model effect."
- The question set (30 MCQs) is small and hand-curated for interesting debate dynamics,
  not a standard benchmark — swapping in a larger sampled subset of MMLU or GSM8K would
  make results comparable to published work.
- Confidence is self-reported by the model in-context, which is a weaker signal than
  probing internal logits/log-probs directly — a stronger version of this project would
  extract token-level log-probabilities for the answer choice instead of asking the
  model to state a number.

## Why I built this

This is a scoped-down version of a research question I'm exploring in my engineering
capstone: metacognitive sensitivity (meta-d′) and belief drift in multi-agent LLM
deliberation. Building this smaller, fully-reproducible version let me test the core
mechanics — debate loops, confidence logging, calibration measurement — before applying
similar ideas at larger scale.
