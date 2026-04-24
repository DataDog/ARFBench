# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Shared logging and output utilities for evaluation scripts.
"""

import warnings
from typing import List, Dict, Tuple

import pandas as pd

from utils.compute_statistics import (
    compute_statistics,
    compute_multiclass_f1_metrics,
    compute_tier_weighted_f1,
)


def save_results_and_statistics(
    results_df: pd.DataFrame, output_prefix: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Save results and compute/save all statistics.

    Writes four CSV files:
        {output_prefix}_results.csv
        {output_prefix}_statistics.csv
        {output_prefix}_f1.csv
        {output_prefix}_tier_f1.csv

    Returns:
        Tuple of (stats_df, tier_f1_df)
    """
    results_filename = f"{output_prefix}_results.csv"
    results_df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")

    stats_df = compute_statistics(results_df)
    stats_filename = f"{output_prefix}_statistics.csv"
    stats_df.to_csv(stats_filename, index=False)
    print(f"Statistics saved to {stats_filename}")

    f1_df = compute_multiclass_f1_metrics(results_df)
    f1_filename = f"{output_prefix}_f1.csv"
    f1_df.to_csv(f1_filename, index=False)
    print(f"F1 metrics saved to {f1_filename}")

    tier_f1_df = compute_tier_weighted_f1(results_df, category_metrics_df=f1_df)
    tier_f1_filename = f"{output_prefix}_tier_f1.csv"
    tier_f1_df.to_csv(tier_f1_filename, index=False)
    print(f"Tier F1 metrics saved to {tier_f1_filename}")

    return stats_df, tier_f1_df


def print_evaluation_summary(
    stats_df: pd.DataFrame,
    tier_f1_df: pd.DataFrame,
    results_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
):
    """Print detailed evaluation summary including tier F1 scores."""
    if not stats_df.empty:
        print("\nEvaluation Summary:")
        print(stats_df.to_string(index=False))

        try:
            overall_accuracy = stats_df["overall_accuracy"].iloc[0]
            print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
        except (KeyError, IndexError):
            pass

    print("\nTier-Level Weighted F1 Scores:")
    if not tier_f1_df.empty:
        for _, row in tier_f1_df.iterrows():
            print(f"\n{row['tier']}:")
            print(f"  Weighted F1:        {row['weighted_f1']:.4f}")
            print(f"  Weighted Precision: {row['weighted_precision']:.4f}")
            print(f"  Weighted Recall:    {row['weighted_recall']:.4f}")
            print(f"  Total Questions:    {row['total_questions']}")
            if row.get("category_f1_scores"):
                print(f"  Categories:         {row['categories']}")
                print("  Per-Category F1:")
                for cat, f1_score in row["category_f1_scores"].items():
                    count = row["category_question_counts"][cat]
                    print(f"    - {cat}: {f1_score:.4f} (n={count})")
    else:
        print("  No tier F1 data available.")

    successful_questions = set(results_df["query_group"].unique())
    all_questions = set(benchmark_df["query_group"].unique())
    skipped_questions = all_questions - successful_questions

    if skipped_questions:
        print(f"\nSkipped Questions ({len(skipped_questions)}):")
        for q in sorted(skipped_questions):
            print(f"- {q}")


def log_progress(
    run_results: List[Dict],
    tier_counts: Dict,
    correct_count: int,
    processed: int,
    total: int,
):
    """Log progress with tier accuracy and F1 scores."""
    tier_acc_str = ""
    for tier_name in ["Tier 1", "Tier 2", "Tier 3"]:
        if tier_counts[tier_name]["total"] > 0:
            tier_acc = (
                100
                * tier_counts[tier_name]["correct"]
                / tier_counts[tier_name]["total"]
            )
            tier_acc_str += f", {tier_name}: {tier_acc:.2f}%"

    overall_acc = 100 * correct_count / processed if processed > 0 else 0
    print(
        f"Processed {processed}/{total} questions: "
        f"accuracy={overall_acc:.2f}%{tier_acc_str}"
    )

    if run_results:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                current_results_df = pd.DataFrame(run_results)
                tier_f1_df = compute_tier_weighted_f1(current_results_df)
                if not tier_f1_df.empty:
                    tier_f1_str = ""
                    for _, tier_row in tier_f1_df.iterrows():
                        tier_f1_str += (
                            f", {tier_row['tier']} F1:"
                            f" {tier_row['weighted_f1']:.4f}"
                        )
                    print(f"  Tier F1 scores{tier_f1_str}")
            except Exception:
                pass
