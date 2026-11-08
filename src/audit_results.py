"""Offline, paired question-level audit of committed predictions."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from src import metrics as M


def vote(group, weighted=False):
    scores = Counter()
    for row in group.itertuples():
        scores[row.answer] += float(row.confidence) if weighted else 1
    winners = sorted(k for k, v in scores.items() if v == max(scores.values()))
    # Abstain on ties, rather than silently favoring the first persona.
    return winners[0] if len(winners) == 1 else None


def audit(df, seed=42, samples=5000):
    if df.duplicated(['question_id', 'agent', 'round']).any():
        raise ValueError('Duplicate predictions')
    if not df.parsed_ok.all():
        raise ValueError('Invalid predictions must be resolved before analysis')
    rounds = sorted(df['round'].unique())
    report = {'note': 'Descriptive reanalysis of saved logs; provenance is not independently verified.', 'rounds': {}}
    for r in rounds:
        data = df[df['round'] == r]
        row = {'individual_accuracy': float(data.is_correct.mean()), 'mean_confidence': float(data.confidence.mean()), 'ece': M.expected_calibration_error(data), 'correctness_brier': float(((data.confidence / 100 - data.is_correct.astype(float)) ** 2).mean())}
        for weighted, label in [(False, 'majority'), (True, 'weighted')]:
            outcomes = []
            ties = 0
            for _, group in data.groupby('question_id'):
                answer = vote(group, weighted)
                ties += answer is None
                outcomes.append(answer == group.correct_answer.iloc[0])
            row[label + '_accuracy_ties_incorrect'] = float(np.mean(outcomes))
            row[label + '_ties'] = ties
        report['rounds'][str(r)] = row
    # Keep all correlated personas together when resampling questions.
    paired = df.groupby(['question_id', 'round']).is_correct.mean().unstack()
    if paired.isna().any().any():
        raise ValueError('Unpaired question rounds')
    delta = (paired[rounds[-1]] - paired[rounds[0]]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = rng.choice(delta, size=(samples, len(delta)), replace=True).mean(axis=1)
    report['individual_accuracy_delta'] = float(delta.mean())
    report['question_bootstrap_95_percentile_interval'] = np.quantile(boot, [.025, .975]).tolist()
    report['bootstrap_seed'] = seed
    report['bootstrap_samples'] = samples
    report['flip_stats'] = M.flip_rate(df)
    report['confidence_by_persona'] = df.groupby(['round', 'persona']).confidence.mean().reset_index().to_dict('records')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', default='results/raw_logs.csv')
    parser.add_argument('--output', default='results/audit.json')
    args = parser.parse_args()
    result = audit(pd.read_csv(args.logs))
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
