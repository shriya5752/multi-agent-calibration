"""
Metrics for analyzing the debate logs.

- accuracy_by_round: does debate actually improve accuracy?
- calibration_curve / expected_calibration_error: does stated confidence
  match actual correctness?
- flip_rate: how often agents change their answer after seeing peers,
  split into "flip helped" vs "flip hurt"
"""

import pandas as pd
import numpy as np


def accuracy_by_round(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("round")["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
    )


def accuracy_by_round_and_persona(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["round", "persona"])["is_correct"]
        .mean()
        .reset_index()
        .rename(columns={"is_correct": "accuracy"})
    )


def calibration_curve(df: pd.DataFrame, n_bins=10, round_num=None):
    """Bin predictions by confidence decile, compute mean confidence vs
    actual accuracy in each bin. Returns a DataFrame ready to plot."""
    data = df if round_num is None else df[df["round"] == round_num]
    data = data.copy()
    data["bin"] = pd.cut(data["confidence"], bins=np.linspace(0, 100, n_bins + 1),
                          include_lowest=True)

    grouped = data.groupby("bin", observed=True).agg(
        mean_confidence=("confidence", "mean"),
        accuracy=("is_correct", "mean"),
        n=("is_correct", "size"),
    ).reset_index()

    return grouped.dropna(subset=["mean_confidence"])


def expected_calibration_error(df: pd.DataFrame, n_bins=10, round_num=None) -> float:
    """Weighted average |confidence - accuracy| across bins (0 = perfectly
    calibrated). Standard ECE definition."""
    curve = calibration_curve(df, n_bins=n_bins, round_num=round_num)
    total = curve["n"].sum()
    if total == 0:
        return float("nan")
    ece = np.sum(curve["n"] / total * np.abs(curve["mean_confidence"] / 100 - curve["accuracy"]))
    return float(ece)


def flip_rate(df: pd.DataFrame) -> dict:
    flips = df[df["round"] > 1]
    total = len(flips)
    if total == 0:
        return {"flip_rate": 0.0, "flip_helped_rate": 0.0, "flip_hurt_rate": 0.0, "n": 0}

    n_flipped = flips["flipped_this_round"].sum()
    n_helped = flips["flip_helped"].sum()
    n_hurt = flips["flip_hurt"].sum()

    return {
        "flip_rate": n_flipped / total,
        "flip_helped_rate": n_helped / max(n_flipped, 1),
        "flip_hurt_rate": n_hurt / max(n_flipped, 1),
        "n": int(total),
        "n_flipped": int(n_flipped),
    }


def confidence_drift_by_round(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["round", "agent"])["confidence"]
        .mean()
        .reset_index()
    )


def aggregation_accuracy(summary_df: pd.DataFrame) -> dict:
    return {
        "majority_vote_accuracy": summary_df["majority_correct"].mean(),
        "confidence_weighted_accuracy": summary_df["weighted_correct"].mean(),
    }
