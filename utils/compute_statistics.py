# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Statistics computation and multiclass F1 score functions for ARFBench evaluation.
"""

import json
import random
import pandas as pd
import re
from typing import Optional, List
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for each model and task category using only the first
    run.

    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results with
            columns: run, model, task_category, is_correct, query_group,
            question, correct_answer

    Returns:
        pd.DataFrame: Statistics DataFrame with columns:
            model, task_category, accuracy, overall_accuracy,
            total_questions_with_images, tier_1_accuracy, tier_2_accuracy,
            tier_3_accuracy
    """
    if results_df.empty:
        return pd.DataFrame()

    # Define tier mappings
    TIER_MAPPINGS = {
        "Tier 1": ["Anomaly Presence"],
        "Tier 2": [
            "Anomaly Start",
            "Anomaly End",
            "Anomaly Categorization",
            "Anomaly Magnitude",
            "Anomaly Identification",
        ],
        "Tier 3": ["Anomaly Indicator", "Anomaly Correlation"],
    }

    stats = []

    # Use only first run for accuracy calculations
    first_run_df = results_df[results_df["run"] == 1]
    total_questions = first_run_df.groupby(["model", "task_category"])[
        "query_group"
    ].nunique()

    # Calculate statistics for each model
    for model in results_df["model"].unique():
        model_df = first_run_df[first_run_df["model"] == model]
        overall_correct = model_df["is_correct"].mean() * 100

        # Calculate tier-based accuracies
        tier_accuracies = {}
        for tier_name, categories in TIER_MAPPINGS.items():
            tier_df = model_df[model_df["task_category"].isin(categories)]
            if len(tier_df) > 0:
                tier_key = f"{tier_name.lower().replace(' ', '_')}_accuracy"
                tier_accuracies[tier_key] = tier_df["is_correct"].mean() * 100
            else:
                tier_key = f"{tier_name.lower().replace(' ', '_')}_accuracy"
                tier_accuracies[tier_key] = None

        # Per-category statistics using first run only
        for category in model_df["task_category"].unique():
            cat_df = model_df[model_df["task_category"] == category]
            cat_correct = cat_df["is_correct"].mean() * 100
            total_in_category = total_questions.get((model, category), 0)

            stats.append(
                {
                    "model": model,
                    "task_category": category,
                    "accuracy": cat_correct,
                    "overall_accuracy": overall_correct,
                    "total_questions_with_images": total_in_category,
                    **tier_accuracies,
                }
            )

    return pd.DataFrame(stats)


# ============================================================================
# Multiclass F1 Score Binning Functions
# ============================================================================


def count_channels(answer_choice: str) -> int:
    """
    Count the number of channels in an answer choice for Anomaly Identification.

    Channels can be separated by:
    1. ' and ' (e.g., 'method:1 and method:2')
    2. ', ' followed by dimension name (e.g., 'pod_name:614, pod_name:198')
    3. Repeating dimension names (e.g., 'service:2,track_type:15,service:1,track_type:41')

    Args:
        answer_choice (str): The answer choice string

    Returns:
        int: Number of channels (1, 2, 3, or more)

    Examples:
        >>> count_channels('pod_name:614')
        1
        >>> count_channels('pod_name:614, pod_name:198')
        2
        >>> count_channels('method:1 and method:2')
        2
        >>> count_channels('service:2,track_type:15,service:1,track_type:41')
        2
    """
    # Check for " and " delimiter
    and_count = answer_choice.count(" and ")
    if and_count > 0:
        return and_count + 1

    # Check for ", " delimiter followed by a word character
    comma_space_pattern = r", (?=[a-zA-Z_])"
    comma_space_matches = re.findall(comma_space_pattern, answer_choice)
    if len(comma_space_matches) > 0:
        return len(comma_space_matches) + 1

    # Check for repeating dimension names (e.g., multiple occurrences of "service:")
    dimension_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*):"
    dimensions = re.findall(dimension_pattern, answer_choice)

    if len(dimensions) > 0:
        # Count unique dimension sequences - if a dimension repeats, it indicates multiple channels
        first_dim = dimensions[0]
        first_dim_count = dimensions.count(first_dim)
        if first_dim_count > 1:
            return first_dim_count

    # Single channel
    return 1


def bin_anomaly_identification(answer: str, all_options: List[str]) -> Optional[str]:
    """
    Bin anomaly identification answers based on channel count.

    Classes:
    - one_channel_small: Single channel anomaly (lexicographically smaller)
    - one_channel_large: Single channel anomaly (lexicographically larger)
    - two_channel: Two channels anomaly
    - three_channel: Three channels anomaly
    - no_anomaly: No anomaly present

    Args:
        answer (str): The answer to classify
        all_options (List[str]): All available answer options

    Returns:
        Optional[str]: The bin classification or None if invalid
    """
    if pd.isna(answer) or answer == "":
        return None

    answer_str = str(answer).strip()

    def is_no_anomaly_identification_label(text: str) -> bool:
        normalized = text.strip().lower()
        return normalized in {
            "no anomaly",
            "no anomaly among the listed channels",
        }

    # Check for no-anomaly label first
    if is_no_anomaly_identification_label(answer_str):
        return "no_anomaly"

    # Count channels
    channels = count_channels(answer_str)

    if channels == 1:
        # Find all one-channel options (excluding no-anomaly labels)
        one_channel_options = []
        for opt in all_options:
            opt_str = str(opt).strip()
            if (
                not is_no_anomaly_identification_label(opt_str)
                and count_channels(opt_str) == 1
            ):
                one_channel_options.append(opt_str)

        # Sort lexicographically
        one_channel_options.sort()

        # Determine if current answer is the smaller or larger one
        if len(one_channel_options) >= 2:
            if answer_str < one_channel_options[1]:
                return "one_channel_small"
            else:
                return "one_channel_large"
        else:
            # Fallback if there aren't two distinct one-channel options
            return "one_channel_small"
    elif channels == 2:
        return "two_channel"
    elif channels >= 3:
        return "three_channel"

    return None


def bin_anomaly_magnitude(answer: str, all_options: List[str]) -> Optional[str]:
    """
    Bin anomaly magnitude answers into 5 classes based on sorted magnitude values.

    Classes (from smallest to largest):
    - smallest: Smallest magnitude in options
    - small: Second smallest magnitude
    - medium: Middle magnitude
    - large: Second largest magnitude
    - no_anomaly: No anomaly present

    Args:
        answer (str): The answer to classify
        all_options (List[str]): All available answer options

    Returns:
        Optional[str]: The bin classification or None if invalid
    """
    if pd.isna(answer) or answer == "":
        return None

    answer_str = str(answer).strip()

    # Check for "no anomaly"
    if answer_str.lower() == "no anomaly":
        return "no_anomaly"

    # Extract all numeric magnitudes from options (excluding "no anomaly")
    magnitudes = []
    for opt in all_options:
        opt_str = str(opt).strip()
        if opt_str.lower() == "no anomaly":
            continue
        # Extract numeric value (handles formats like "0.1", "1", "10", "40%", etc.)
        magnitude_match = re.search(r"(\d+\.?\d*)", opt_str)
        if magnitude_match:
            try:
                mag_value = float(magnitude_match.group(1))
                magnitudes.append(mag_value)
            except ValueError:
                pass

    # Sort and deduplicate magnitudes
    magnitudes = sorted(set(magnitudes))

    if len(magnitudes) == 0:
        return None

    # Extract magnitude from answer
    answer_magnitude_match = re.search(r"(\d+\.?\d*)", answer_str)
    if not answer_magnitude_match:
        return None

    try:
        answer_magnitude = float(answer_magnitude_match.group(1))

        # Find index in sorted magnitudes
        try:
            idx = magnitudes.index(answer_magnitude)
        except ValueError:
            return None

        # Bin into 4 magnitude classes (smallest, small, medium, large)
        # The 5th class is no_anomaly which is already handled
        n_magnitude_bins = min(4, len(magnitudes))

        if n_magnitude_bins == 1:
            return "smallest"
        elif n_magnitude_bins == 2:
            return "smallest" if idx == 0 else "large"
        elif n_magnitude_bins == 3:
            if idx == 0:
                return "smallest"
            elif idx == 1:
                return "medium"
            else:
                return "large"
        else:  # 4 or more magnitude values
            if idx == 0:
                return "smallest"
            elif idx == 1:
                return "small"
            elif idx == len(magnitudes) - 1:
                return "large"
            else:
                return "medium"

    except (ValueError, ZeroDivisionError):
        return None


def bin_anomaly_start(answer: str, all_options: List[str]) -> Optional[str]:
    """
    Bin anomaly start times into 5 classes based on sorted timestamps.

    Classes (from earliest to latest):
    - earliest: "Before the earliest timestamp" or first timestamp
    - early: Second timestamp
    - medium: Middle timestamp(s)
    - late: Second to last timestamp
    - no_anomaly: No anomaly present

    Args:
        answer (str): The answer to classify
        all_options (List[str]): All available answer options

    Returns:
        Optional[str]: The bin classification or None if invalid
    """
    if pd.isna(answer) or answer == "":
        return None

    answer_str = str(answer).strip()

    # Check for "no anomaly"
    if answer_str.lower() == "no anomaly":
        return "no_anomaly"

    has_before_earliest_option = any(
        "before the earliest" in str(opt).strip().lower() for opt in all_options
    )

    # Check for "Before the earliest timestamp" (earliest possible time)
    if "before the earliest" in answer_str.lower():
        return "earliest"

    # Extract all timestamps from options (excluding special cases and "no anomaly")
    timestamps = []
    for opt in all_options:
        opt_str = str(opt).strip()
        if opt_str.lower() == "no anomaly" or "before the earliest" in opt_str.lower():
            continue
        # Extract timestamp in format YYYY-MM-DD HH:MM:SS
        timestamp_match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", opt_str)
        if timestamp_match:
            timestamps.append(timestamp_match.group())

    # Sort and deduplicate timestamps
    timestamps = sorted(set(timestamps))

    if len(timestamps) == 0:
        return None

    # Find which timestamp the answer corresponds to
    answer_timestamp_match = re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", answer_str
    )
    if not answer_timestamp_match:
        return None

    answer_timestamp = answer_timestamp_match.group()

    try:
        idx = timestamps.index(answer_timestamp)
    except ValueError:
        return None

    # If a special "Before the earliest timestamp" option exists, reserve
    # the "earliest" class for that option and shift numeric timestamps to
    # early/medium/late.
    if has_before_earliest_option:
        if len(timestamps) == 1:
            return "early"
        if idx == 0:
            return "early"
        if idx == len(timestamps) - 1:
            return "late"
        return "medium"

    # Otherwise bin numeric timestamps into 4 time classes
    # (earliest, early, medium, late). The 5th class is no_anomaly.
    n_time_bins = min(4, len(timestamps))

    if n_time_bins == 1:
        return "earliest"
    elif n_time_bins == 2:
        return "earliest" if idx == 0 else "late"
    elif n_time_bins == 3:
        if idx == 0:
            return "earliest"
        elif idx == 1:
            return "medium"
        else:
            return "late"
    else:  # 4 or more timestamps
        if idx == 0:
            return "earliest"
        elif idx == 1:
            return "early"
        elif idx == len(timestamps) - 1:
            return "late"
        else:
            return "medium"


def bin_anomaly_end(answer: str, all_options: List[str]) -> Optional[str]:
    """
    Bin anomaly end times into 5 classes based on sorted timestamps.

    Classes (from earliest to latest):
    - early: First timestamp
    - medium: Middle timestamp(s)
    - late: Second to last timestamp
    - latest: "Not resolved" or last timestamp
    - no_anomaly: No anomaly present

    Args:
        answer (str): The answer to classify
        all_options (List[str]): All available answer options

    Returns:
        Optional[str]: The bin classification or None if invalid
    """
    if pd.isna(answer) or answer == "":
        return None

    answer_str = str(answer).strip()

    # Check for "no anomaly"
    if answer_str.lower() == "no anomaly":
        return "no_anomaly"

    has_not_resolved_option = any(
        "not resolved" in str(opt).strip().lower() for opt in all_options
    )

    # Check for "Not resolved" (latest possible time)
    if "not resolved" in answer_str.lower():
        return "latest"

    # Extract all timestamps from options (excluding special cases and "no anomaly")
    timestamps = []
    for opt in all_options:
        opt_str = str(opt).strip()
        if opt_str.lower() == "no anomaly" or "not resolved" in opt_str.lower():
            continue
        # Extract timestamp in format YYYY-MM-DD HH:MM:SS
        timestamp_match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", opt_str)
        if timestamp_match:
            timestamps.append(timestamp_match.group())

    # Sort and deduplicate timestamps
    timestamps = sorted(set(timestamps))

    if len(timestamps) == 0:
        return None

    # Find which timestamp the answer corresponds to
    answer_timestamp_match = re.search(
        r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", answer_str
    )
    if not answer_timestamp_match:
        return None

    answer_timestamp = answer_timestamp_match.group()

    try:
        idx = timestamps.index(answer_timestamp)
    except ValueError:
        return None

    # If a special "Not resolved" option exists, reserve "latest" for that
    # option and shift numeric timestamps so the last numeric timestamp maps
    # to "late" instead of "latest".
    if has_not_resolved_option:
        if len(timestamps) == 1:
            return "late"
        if idx == 0:
            return "early"
        if idx == len(timestamps) - 1:
            return "late"
        return "medium"

    # Otherwise bin numeric timestamps into 4 time classes
    # (early, medium, late, latest). The 5th class is no_anomaly.
    n_time_bins = min(4, len(timestamps))

    if n_time_bins == 1:
        return "latest"
    elif n_time_bins == 2:
        return "early" if idx == 0 else "latest"
    elif n_time_bins == 3:
        if idx == 0:
            return "early"
        elif idx == 1:
            return "medium"
        else:
            return "latest"
    else:  # 4 or more timestamps
        if idx == 0:
            return "early"
        elif idx == len(timestamps) - 1:
            return "latest"
        elif idx == len(timestamps) - 2:
            return "late"
        else:
            return "medium"


def bin_anomaly_categorization(answer: str, all_options: List[str]) -> Optional[str]:
    """
    Bin anomaly categorization answers, combining Change in Variance and
    Change in Seasonality into a single class.

    Classes:
    - Level Shift
    - Transient Spike
    - Change in Seasonality/Variance (combined class)
    - Change in Trend
    - No Anomaly

    Args:
        answer (str): The answer to classify
        all_options (List[str]): All available answer options

    Returns:
        Optional[str]: The bin classification or None if invalid
    """
    if pd.isna(answer) or answer == "":
        return None

    answer_str = str(answer).strip()

    # Normalize the answer to lowercase for comparison
    answer_lower = answer_str.lower()

    # Map answers to standardized bins
    if "no anomaly" in answer_lower:
        return "No Anomaly"
    elif "level shift" in answer_lower:
        return "Level Shift"
    elif "transient spike" in answer_lower:
        return "Transient Spike"
    elif (
        "change in seasonality" in answer_lower or "change in variance" in answer_lower
    ):
        # Combine these two categories
        return "Change in Seasonality/Variance"
    elif "change in trend" in answer_lower:
        return "Change in Trend"

    # If no match, return the original answer (might be a variant)
    return answer_str


def parse_options(options_str: str) -> List[str]:
    """
    Parse the options string which is in JSON format.

    Handles two formats:
    1. List of dicts: [{"value": "opt1"}, {"value": "opt2"}]
    2. Plain string list: ["opt1", "opt2"]

    Args:
        options_str (str): JSON string containing answer options

    Returns:
        List[str]: List of option values, or empty list if parsing fails
    """
    try:
        options = json.loads(options_str)
        if not isinstance(options, list) or not options:
            return []
        if isinstance(options[0], dict):
            return [opt["value"] for opt in options]
        return [str(opt) for opt in options]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


def apply_binning(answer: str, all_options: List[str], category: str) -> Optional[str]:
    """
    Apply appropriate binning function based on task category.

    Args:
        answer (str): The answer to classify
        all_options (List[str]): All available answer options
        category (str): Task category name

    Returns:
        Optional[str]: The bin classification or the original answer
    """
    category_lower = category.lower()

    if (
        "anomaly identification" in category_lower
        or "anomaly channel" in category_lower
    ):
        return bin_anomaly_identification(answer, all_options)
    elif "anomaly magnitude" in category_lower:
        return bin_anomaly_magnitude(answer, all_options)
    elif "anomaly start" in category_lower:
        return bin_anomaly_start(answer, all_options)
    elif "anomaly end" in category_lower or "anomaly recover" in category_lower:
        return bin_anomaly_end(answer, all_options)
    elif "anomaly categorization" in category_lower:
        return bin_anomaly_categorization(answer, all_options)
    else:
        # For other categories (Anomaly Presence, Anomaly Indicator, Correlation),
        # use the answer labels directly as classes
        return answer if not pd.isna(answer) and answer != "" else None


def compute_multiclass_f1_metrics(
    results_df: pd.DataFrame,
    correct_answer_col: str = "correct_answer",
    model_answer_col: str = "model_answer",
    options_col: str = "options",
    task_category_col: str = "task_category",
    random_seed: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Compute multiclass F1, precision, and recall metrics by task category.

    This function bins answers appropriately for each task category and
    calculates macro-averaged metrics across all classes.

    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results
        correct_answer_col (str): Column name for correct answers
        model_answer_col (str): Column name for model predictions
        options_col (str): Column name for answer options (JSON format)
        task_category_col (str): Column name for task categories
        random_seed (Optional[int]): Seed for invalid/OOV forced-choice
            assignment. Set to None to use nondeterministic global RNG.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - task_category: Task category name
            - macro_f1: Macro-averaged F1 score
            - macro_precision: Macro-averaged precision
            - macro_recall: Macro-averaged recall
            - n_samples: Number of valid samples
            - n_classes: Number of unique classes
            - classes: List of class labels
    """
    if results_df.empty:
        return pd.DataFrame()

    rng = random.Random(random_seed) if random_seed is not None else random
    results = []

    for category in results_df[task_category_col].unique():
        if pd.isna(category):
            continue

        category_df = results_df[results_df[task_category_col] == category].copy()

        # Parse options for each row
        category_df["parsed_options"] = category_df[options_col].apply(parse_options)

        # Apply binning to both correct and model answers
        category_df["binned_correct"] = category_df.apply(
            lambda row: apply_binning(
                row[correct_answer_col], row["parsed_options"], category
            ),
            axis=1,
        )
        category_df["binned_model"] = category_df.apply(
            lambda row: apply_binning(
                row[model_answer_col], row["parsed_options"], category
            ),
            axis=1,
        )

        # Remove rows with None correct answers (we need valid ground truth)
        # But keep rows with None model answers (empty answers should count as incorrect)
        valid_df = category_df[category_df["binned_correct"].notna()].copy()

        if len(valid_df) == 0:
            continue

        # Fixed label set: only classes that appear in ground truth (after binning).
        # Same set for all models/runs — no predicted-only labels.
        correct_classes = set(valid_df["binned_correct"].tolist())
        all_classes = sorted(correct_classes)

        # Count invalid predictions before mapping (for invalid_rate reporting)
        raw_binned = valid_df["binned_model"].copy()
        n_invalid = (raw_binned.isna() | ~raw_binned.isin(all_classes)).sum()

        # Forced-choice policy: invalid/OOV predictions are wrong — map into
        # the label set so they count as FN for the true class + FP for the
        # assigned (wrong) class.  We pick a uniformly random incorrect class
        # so that false positives are spread evenly instead of systematically
        # inflating one particular class.
        def assign_to_valid_class(row):
            model_answer = row["binned_model"]
            correct_class = row["binned_correct"]
            if pd.notna(model_answer) and model_answer in all_classes:
                return model_answer
            # Invalid: assign uniformly at random to an incorrect class
            wrong_classes = [c for c in all_classes if c != correct_class]
            if wrong_classes:
                return rng.choice(wrong_classes)
            # Fallback: only one class exists (shouldn't happen in practice)
            return all_classes[0]

        valid_df["binned_model"] = valid_df.apply(assign_to_valid_class, axis=1)

        # Calculate multiclass macro F1, Precision, and Recall
        try:
            f1_macro = f1_score(
                valid_df["binned_correct"],
                valid_df["binned_model"],
                average="macro",
                labels=all_classes,
                zero_division=0,
            )

            precision_macro = precision_score(
                valid_df["binned_correct"],
                valid_df["binned_model"],
                average="macro",
                labels=all_classes,
                zero_division=0,
            )

            recall_macro = recall_score(
                valid_df["binned_correct"],
                valid_df["binned_model"],
                average="macro",
                labels=all_classes,
                zero_division=0,
            )

            # Calculate per-class metrics for detailed analysis
            f1_per_class = f1_score(
                valid_df["binned_correct"],
                valid_df["binned_model"],
                average=None,
                labels=all_classes,
                zero_division=0,
            )

            precision_per_class = precision_score(
                valid_df["binned_correct"],
                valid_df["binned_model"],
                average=None,
                labels=all_classes,
                zero_division=0,
            )

            recall_per_class = recall_score(
                valid_df["binned_correct"],
                valid_df["binned_model"],
                average=None,
                labels=all_classes,
                zero_division=0,
            )

            invalid_rate = n_invalid / len(valid_df) if len(valid_df) > 0 else 0.0
            results.append(
                {
                    "task_category": category,
                    "macro_f1": f1_macro,
                    "macro_precision": precision_macro,
                    "macro_recall": recall_macro,
                    "n_samples": len(valid_df),
                    "n_classes": len(all_classes),
                    "classes": ", ".join(all_classes),
                    "invalid_rate": invalid_rate,
                    "n_invalid": n_invalid,
                    "per_class_f1": dict(zip(all_classes, f1_per_class)),
                    "per_class_precision": dict(zip(all_classes, precision_per_class)),
                    "per_class_recall": dict(zip(all_classes, recall_per_class)),
                }
            )

        except Exception as e:
            print(f"Error calculating F1 for {category}: {e}")

    category_f1_df = pd.DataFrame(results)

    # Append an overall row: weighted average of per-category macro-F1,
    # weighted by number of questions in each category.
    if not category_f1_df.empty:
        overall = compute_overall_weighted_f1(category_f1_df)
        if overall is not None:
            category_f1_df = pd.concat(
                [category_f1_df, pd.DataFrame([overall])], ignore_index=True
            )

    return category_f1_df


def compute_overall_weighted_f1(category_metrics_df: pd.DataFrame) -> Optional[dict]:
    """
    Compute overall weighted F1 from per-category metrics.

    Returns a single dict with task_category="Overall" containing the
    weighted average of macro_f1, macro_precision, and macro_recall across
    all categories, weighted by n_samples (number of questions).

    Args:
        category_metrics_df: DataFrame with columns macro_f1,
            macro_precision, macro_recall, n_samples (one row per category).

    Returns:
        Dict with overall metrics, or None if the input is empty.
    """
    if category_metrics_df.empty:
        return None

    # Exclude any existing "Overall" row to avoid double-counting
    df = category_metrics_df[category_metrics_df["task_category"] != "Overall"].copy()
    if df.empty:
        return None

    total = df["n_samples"].sum()
    if total == 0:
        return None

    weighted_f1 = (df["n_samples"] * df["macro_f1"]).sum() / total
    weighted_precision = (df["n_samples"] * df["macro_precision"]).sum() / total
    weighted_recall = (df["n_samples"] * df["macro_recall"]).sum() / total

    return {
        "task_category": "Overall",
        "macro_f1": weighted_f1,
        "macro_precision": weighted_precision,
        "macro_recall": weighted_recall,
        "n_samples": int(total),
        "n_classes": None,
        "classes": None,
        "invalid_rate": None,
        "n_invalid": None,
        "per_class_f1": None,
        "per_class_precision": None,
        "per_class_recall": None,
    }


def compute_tier_weighted_f1(
    results_df: pd.DataFrame,
    correct_answer_col: str = "correct_answer",
    model_answer_col: str = "model_answer",
    options_col: str = "options",
    task_category_col: str = "task_category",
    category_metrics_df: Optional[pd.DataFrame] = None,
    random_seed: Optional[int] = 0,
) -> pd.DataFrame:
    """
    Compute per-tier multiclass F1 scores using weighted average by category.

    For each tier, computes:
    tier_f1 = sum((num_questions_in_category * f1_in_category)) / total_questions_in_tier

    Args:
        results_df (pd.DataFrame): DataFrame containing evaluation results
        correct_answer_col (str): Column name for correct answers
        model_answer_col (str): Column name for model predictions
        options_col (str): Column name for answer options (JSON format)
        task_category_col (str): Column name for task categories
        category_metrics_df (Optional[pd.DataFrame]): Precomputed per-category
            metrics from compute_multiclass_f1_metrics. If provided, this
            function reuses the same F1 pass instead of recomputing category
            metrics.
        random_seed (Optional[int]): Seed forwarded to
            compute_multiclass_f1_metrics when category_metrics_df is not
            provided. Ignored if category_metrics_df is provided.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - tier: Tier name (Tier 1, Tier 2, Tier 3)
            - weighted_f1: Weighted average F1 score for the tier
            - weighted_precision: Weighted average precision for the tier
            - weighted_recall: Weighted average recall for the tier
            - total_questions: Total number of questions in the tier
            - categories: List of categories in the tier
            - category_f1_scores: Dictionary mapping categories to their F1 scores
            - category_question_counts: Dictionary mapping categories to question counts
    """
    if category_metrics_df is None and results_df.empty:
        return pd.DataFrame()

    # Define tier mappings
    TIER_MAPPINGS = {
        "Tier 1": ["Anomaly Presence"],
        "Tier 2": [
            "Anomaly Start",
            "Anomaly End",
            "Anomaly Categorization",
            "Anomaly Magnitude",
            "Anomaly Identification",
        ],
        "Tier 3": ["Anomaly Indicator", "Anomaly Correlation"],
    }

    # Reuse caller-provided category metrics when available so tier metrics
    # can be derived from the exact same F1 pass.
    if category_metrics_df is not None:
        category_metrics = category_metrics_df.copy()
    else:
        category_metrics = compute_multiclass_f1_metrics(
            results_df,
            correct_answer_col=correct_answer_col,
            model_answer_col=model_answer_col,
            options_col=options_col,
            task_category_col=task_category_col,
            random_seed=random_seed,
        )

    if category_metrics.empty:
        return pd.DataFrame()

    # Create a mapping of category to metrics
    category_to_metrics = {}
    for _, row in category_metrics.iterrows():
        category_to_metrics[row["task_category"]] = {
            "macro_f1": row["macro_f1"],
            "macro_precision": row["macro_precision"],
            "macro_recall": row["macro_recall"],
            "n_samples": row["n_samples"],
        }

    # Compute tier-level weighted metrics
    tier_results = []

    for tier_name, tier_categories in TIER_MAPPINGS.items():
        # Filter for categories in this tier
        tier_category_metrics = [
            (cat, category_to_metrics[cat])
            for cat in tier_categories
            if cat in category_to_metrics
        ]

        if not tier_category_metrics:
            continue

        # Calculate total questions in the tier
        total_questions = sum(
            metrics["n_samples"] for _, metrics in tier_category_metrics
        )

        if total_questions == 0:
            continue

        # Calculate weighted averages
        weighted_f1 = (
            sum(
                metrics["n_samples"] * metrics["macro_f1"]
                for _, metrics in tier_category_metrics
            )
            / total_questions
        )

        weighted_precision = (
            sum(
                metrics["n_samples"] * metrics["macro_precision"]
                for _, metrics in tier_category_metrics
            )
            / total_questions
        )

        weighted_recall = (
            sum(
                metrics["n_samples"] * metrics["macro_recall"]
                for _, metrics in tier_category_metrics
            )
            / total_questions
        )

        # Prepare detailed breakdown
        category_f1_scores = {
            cat: metrics["macro_f1"] for cat, metrics in tier_category_metrics
        }
        category_question_counts = {
            cat: metrics["n_samples"] for cat, metrics in tier_category_metrics
        }

        tier_results.append(
            {
                "tier": tier_name,
                "weighted_f1": weighted_f1,
                "weighted_precision": weighted_precision,
                "weighted_recall": weighted_recall,
                "total_questions": total_questions,
                "categories": ", ".join([cat for cat, _ in tier_category_metrics]),
                "category_f1_scores": category_f1_scores,
                "category_question_counts": category_question_counts,
            }
        )

    tier_f1_df = pd.DataFrame(tier_results)

    # Append an overall row.
    # Prefer the already-computed category "Overall" row when available so the
    # tier-level overall is identical to the category-level overall from the
    # same F1 pass.
    if not tier_f1_df.empty:
        overall_source = category_metrics[
            category_metrics["task_category"] == "Overall"
        ].copy()

        if not overall_source.empty:
            overall_row = overall_source.iloc[0]
            overall_f1 = float(overall_row["macro_f1"])
            overall_precision = float(overall_row["macro_precision"])
            overall_recall = float(overall_row["macro_recall"])
            grand_total = int(overall_row["n_samples"])
        else:
            non_overall = category_metrics[
                category_metrics["task_category"] != "Overall"
            ].copy()
            grand_total = int(non_overall["n_samples"].sum())
            if grand_total > 0:
                overall_f1 = (
                    non_overall["n_samples"] * non_overall["macro_f1"]
                ).sum() / grand_total
                overall_precision = (
                    non_overall["n_samples"] * non_overall["macro_precision"]
                ).sum() / grand_total
                overall_recall = (
                    non_overall["n_samples"] * non_overall["macro_recall"]
                ).sum() / grand_total
            else:
                overall_f1 = None
                overall_precision = None
                overall_recall = None

        if grand_total > 0:
            tier_f1_df = pd.concat(
                [
                    tier_f1_df,
                    pd.DataFrame(
                        [
                            {
                                "tier": "Overall",
                                "weighted_f1": overall_f1,
                                "weighted_precision": overall_precision,
                                "weighted_recall": overall_recall,
                                "total_questions": int(grand_total),
                                "categories": None,
                                "category_f1_scores": None,
                                "category_question_counts": None,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    return tier_f1_df
