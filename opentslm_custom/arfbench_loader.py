# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache-2.0 License.
#
# This product includes software developed at Datadog
# (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""ARFBench loader utilities for OpenTSLM evaluation."""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm


CSV_PATH = "arfbench-qa.csv"
DATA_DIR = "arfbench-ts-data"
INTERVALS = [10, 60, 300, 1800, 3600, 86400]


def ensure_arfbench_dataset(
    csv_path: str | None = None,
    data_dir: str | None = None,
):
    """Ensure ARFBench CSV and time-series directory exist."""
    csv_file = csv_path or CSV_PATH
    ts_dir = data_dir or DATA_DIR

    if not os.path.exists(csv_file):
        raise FileNotFoundError(
            f"ARFBench CSV not found at {csv_file}. "
            "Please ensure the file exists at this location."
        )

    if not os.path.exists(ts_dir):
        raise FileNotFoundError(
            f"ARFBench time series data directory not found at {ts_dir}. "
            "Please ensure the directory exists with parquet files."
        )

    print(f"ARFBench dataset found: {csv_file}")
    print(f"Time series data directory: {ts_dir}")


def estimate_token_count(text: str) -> int:
    """Rough estimate of token count using 4 chars/token."""
    return len(text) // 4


def find_optimal_interval(
    query_groups: str,
    data_dir: str,
    question: str = "",
    options: str = "",
    max_context_tokens: int = 30000,
) -> Optional[int]:
    """Find a time-series interval that fits expected context constraints."""
    base_tokens = estimate_token_count(question + options)
    reserved_tokens = 2000
    available_tokens = max_context_tokens - base_tokens - reserved_tokens

    groups = [g.strip() for g in query_groups.split(",")]

    for interval in INTERVALS:
        total_tokens = 0
        all_files_exist = True

        for group in groups:
            file_path = os.path.join(data_dir, f"{group}_{interval}.parquet")
            if not os.path.exists(file_path):
                all_files_exist = False
                break

            try:
                df = pd.read_parquet(file_path)
                csv_text = df.to_csv(index=False)
                total_tokens += estimate_token_count(csv_text)
            except Exception as exc:
                print(f"Error reading {file_path}: {exc}")
                all_files_exist = False
                break

        if all_files_exist and total_tokens <= available_tokens:
            return interval

    for interval in reversed(INTERVALS):
        all_files_exist = True
        for group in groups:
            file_path = os.path.join(data_dir, f"{group}_{interval}.parquet")
            if not os.path.exists(file_path):
                all_files_exist = False
                break

        if all_files_exist:
            print(f"Warning: using interval {interval}s which may exceed context")
            return interval

    return None


def load_time_series_for_query_group(
    query_group: str,
    interval: int,
    data_dir: str,
) -> Optional[np.ndarray]:
    """Load flattened numeric time-series values for a query group."""
    file_path = os.path.join(data_dir, f"{query_group}_{interval}.parquet")
    if not os.path.exists(file_path):
        return None

    try:
        df = pd.read_parquet(file_path)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        value_cols = [col for col in numeric_cols if col not in ["epoch", "timestamp"]]

        if value_cols:
            return df[value_cols].values.flatten()
        return df.select_dtypes(include=[np.number]).values.flatten()
    except Exception as exc:
        print(f"Error loading time series from {file_path}: {exc}")
        return None


def load_arfbench_with_time_series(
    csv_path: str,
    data_dir: str,
    max_context_tokens: int = 30000,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Load ARFBench and merge with time-series data.

    Returns (empty, empty, test) because ARFBench is test-only here.
    """
    _ = seed  # Kept for compatibility with existing OpenTSLM callsites.
    ensure_arfbench_dataset(csv_path, data_dir)

    print(f"Loading ARFBench CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples")
    print(f"Task categories: {df['task_category'].unique().tolist()}")

    processed_data = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Loading time series data",
    ):
        query_group = row["query_group"]
        question = str(row["question"])
        options_str = str(row["options_str"])

        interval = find_optimal_interval(
            query_group,
            data_dir,
            question,
            options_str,
            max_context_tokens,
        )
        if interval is None:
            print(
                "Warning: no suitable time-series data for query_group "
                f"{query_group}, skipping"
            )
            continue

        query_groups = [g.strip() for g in str(query_group).split(",")]
        time_series_list = []

        for qg in query_groups:
            ts = load_time_series_for_query_group(qg, interval, data_dir)
            if ts is not None:
                time_series_list.append(ts.tolist())

        if not time_series_list:
            print(f"Warning: could not load time series for {query_group}, " "skipping")
            continue

        processed_data.append(
            {
                "question": question,
                "task_category": row["task_category"],
                "difficulty": row["difficulty"],
                "options_str": options_str,
                "correct_answer": row["correct_answer"],
                "query_group": query_group,
                "time_series": json.dumps(time_series_list),
                "interval": interval,
            }
        )

    print(
        "Successfully processed " f"{len(processed_data)} samples with time-series data"
    )

    test_dataset = Dataset.from_list(processed_data)
    empty_dataset = Dataset.from_list([])
    print(
        "Dataset size - Test: "
        f"{len(test_dataset)} (train and val are empty; ARFBench is test-only)"
    )

    return empty_dataset, empty_dataset, test_dataset
