"""
Orchestrator: runs the full debate pipeline over a set of questions.

Pipeline per question:
  Round 1: each agent answers independently (no cross-talk)
  Round 2..N: each agent sees every OTHER agent's previous-round
              answer + confidence + reasoning, then revises
  Final: aggregate via (a) majority vote and (b) confidence-weighted vote

Every agent's output at every round is logged as one row, so the full
history is reconstructible from results/raw_logs.csv alone.
"""

from collections import Counter


def run_debate_on_question(agents, question_obj, n_rounds=2):
    """Run the full multi-round debate for a single question.

    Returns a list of log rows (dicts) — one per agent per round —
    plus the final aggregated answer.
    """
    qid = question_obj["id"]
    question = question_obj["question"]
    options = question_obj["options"]
    correct = question_obj["answer"]

    logs = []
    round_responses = {}  # name -> response dict, most recent round

    for round_num in range(1, n_rounds + 1):
        this_round = {}
        for agent in agents:
            other_resp = None
            if round_num > 1:
                other_resp = {
                    name: r for name, r in round_responses.items()
                    if name != agent.name
                }
            resp = agent.respond(question, options, other_resp, round_num)
            this_round[agent.name] = resp

            prev_answer = round_responses.get(agent.name, {}).get("answer")
            flipped = (round_num > 1 and prev_answer is not None
                       and prev_answer != resp["answer"])

            logs.append({
                "question_id": qid,
                "round": round_num,
                "agent": agent.name,
                "persona": agent.persona_key,
                "answer": resp["answer"],
                "confidence": resp["confidence"],
                "correct_answer": correct,
                "is_correct": resp["answer"] == correct,
                "flipped_this_round": flipped,
                "flip_helped": (flipped and resp["answer"] == correct
                                 and prev_answer != correct),
                "flip_hurt": (flipped and prev_answer == correct
                              and resp["answer"] != correct),
                "parsed_ok": resp["parsed_ok"],
                "reasoning": resp["reasoning"],
            })

        round_responses = this_round

    # --- aggregation over final round ---
    final_answers = [r["answer"] for r in round_responses.values()]
    majority_answer = Counter(final_answers).most_common(1)[0][0]

    conf_weighted = {}
    for r in round_responses.values():
        conf_weighted[r["answer"]] = conf_weighted.get(r["answer"], 0) + r["confidence"]
    weighted_answer = max(conf_weighted, key=conf_weighted.get)

    return logs, {
        "question_id": qid,
        "correct_answer": correct,
        "majority_answer": majority_answer,
        "majority_correct": majority_answer == correct,
        "weighted_answer": weighted_answer,
        "weighted_correct": weighted_answer == correct,
    }


def run_experiment(agents, questions, n_rounds=2, progress_cb=None):
    """Run the debate pipeline across every question.

    Returns (raw_logs: list[dict], summary: list[dict])
    """
    all_logs = []
    all_summaries = []

    for i, q in enumerate(questions):
        logs, summary = run_debate_on_question(agents, q, n_rounds=n_rounds)
        all_logs.extend(logs)
        all_summaries.append(summary)
        if progress_cb:
            progress_cb(i + 1, len(questions))

    return all_logs, all_summaries
