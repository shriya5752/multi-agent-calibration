"""
Main entry point.

Usage:
    python -m src.run_experiment --mock                # no API key needed, tests pipeline
    python -m src.run_experiment                        # real run, needs LLM_API_KEY in .env
    python -m src.run_experiment --n-questions 10 --rounds 3

Outputs:
    results/raw_logs.csv       one row per agent per round per question
    results/summary.csv        one row per question with aggregated (final) answers
    results/plots/*.png        accuracy-by-round, calibration curve, confidence drift
    results/metrics.json       headline numbers (ECE, flip rate, aggregation accuracy)
"""

import argparse
import json
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import Agent
from src.personas import PERSONAS, PERSONA_ORDER
from src.orchestrator import run_experiment
from src import metrics as M

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_agents(client, mock, model):
    agents = []
    for key in PERSONA_ORDER:
        agents.append(Agent(
            name=key,
            persona_key=key,
            persona_prompt=PERSONAS[key],
            client=client,
            model=model,
            mock=mock,
        ))
    return agents


def make_plots(raw_df, summary_df, outdir):
    os.makedirs(outdir, exist_ok=True)

    # 1. Accuracy by round
    acc = M.accuracy_by_round(raw_df)
    plt.figure(figsize=(6, 4))
    plt.plot(acc["round"], acc["accuracy"], marker="o")
    plt.xlabel("Debate round")
    plt.ylabel("Accuracy")
    plt.title("Agent accuracy across debate rounds")
    plt.ylim(0, 1)
    plt.xticks(acc["round"])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "accuracy_by_round.png"), dpi=150)
    plt.close()

    # 2. Calibration curve (final round)
    last_round = raw_df["round"].max()
    curve = M.calibration_curve(raw_df, round_num=last_round)
    plt.figure(figsize=(5, 5))
    plt.plot([0, 100], [0, 1], "--", color="gray", label="Perfect calibration")
    if not curve.empty:
        plt.plot(curve["mean_confidence"], curve["accuracy"], marker="o", label="Observed")
    plt.xlabel("Mean stated confidence (%)")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Calibration curve (round {last_round})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "calibration_curve.png"), dpi=150)
    plt.close()

    # 3. Confidence drift by round, per agent
    drift = M.confidence_drift_by_round(raw_df)
    plt.figure(figsize=(6, 4))
    for agent_name, sub in drift.groupby("agent"):
        plt.plot(sub["round"], sub["confidence"], marker="o", label=agent_name)
    plt.xlabel("Debate round")
    plt.ylabel("Mean confidence (%)")
    plt.title("Confidence drift across rounds, by persona")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "confidence_drift.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true",
                         help="Run with fake randomized responses, no API key needed")
    parser.add_argument("--n-questions", type=int, default=None,
                         help="Limit number of questions (default: all)")
    parser.add_argument("--rounds", type=int, default=2, help="Number of debate rounds")
    parser.add_argument("--model", type=str, default="llama-3.1-8b-instant",
                         help="Model name to pass to the API")
    args = parser.parse_args()

    load_dotenv()

    with open(os.path.join(ROOT, "data", "questions.json")) as f:
        questions = json.load(f)
    if args.n_questions:
        questions = questions[: args.n_questions]

    client = None
    if not args.mock:
        from src.llm_client import get_client
        client = get_client()

    agents = build_agents(client, args.mock, args.model)

    print(f"Running {len(questions)} questions x {len(agents)} agents x {args.rounds} rounds "
          f"({'MOCK' if args.mock else args.model})...")

    pbar = tqdm(total=len(questions))

    def cb(done, total):
        pbar.n = done
        pbar.refresh()

    raw_logs, summaries = run_experiment(agents, questions, n_rounds=args.rounds, progress_cb=cb)
    pbar.close()

    raw_df = pd.DataFrame(raw_logs)
    summary_df = pd.DataFrame(summaries)

    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    raw_df.to_csv(os.path.join(results_dir, "raw_logs.csv"), index=False)
    summary_df.to_csv(os.path.join(results_dir, "summary.csv"), index=False)

    n_unparsed = (~raw_df["parsed_ok"]).sum()
    if n_unparsed:
        print(f"WARNING: {n_unparsed} responses could not be parsed and were "
              f"assigned a random fallback answer. Check raw_logs.csv (parsed_ok=False).")

    headline = {
        "n_questions": len(questions),
        "n_rounds": args.rounds,
        "accuracy_by_round": M.accuracy_by_round(raw_df).to_dict(orient="records"),
        "ece_round_1": M.expected_calibration_error(raw_df, round_num=1),
        "ece_final_round": M.expected_calibration_error(raw_df, round_num=args.rounds),
        "flip_stats": M.flip_rate(raw_df),
        "aggregation_accuracy": M.aggregation_accuracy(summary_df),
    }

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(headline, f, indent=2)

    make_plots(raw_df, summary_df, os.path.join(results_dir, "plots"))

    print("\n=== HEADLINE RESULTS ===")
    print(json.dumps(headline, indent=2))
    print(f"\nSaved: results/raw_logs.csv, results/summary.csv, results/metrics.json, "
          f"results/plots/*.png")


if __name__ == "__main__":
    main()
