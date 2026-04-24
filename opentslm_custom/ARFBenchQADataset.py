# Unless explicitly stated otherwise all files in this repository are licensed
# under the Apache-2.0 License.
#
# This product includes software developed at Datadog
# (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""ARFBench QA dataset adapter for OpenTSLM."""

from __future__ import annotations

import json
from typing import List, Literal, Tuple

import numpy as np
from datasets import Dataset

from prompt.text_time_series_prompt import TextTimeSeriesPrompt
from time_series_datasets.QADataset import QADataset
from time_series_datasets.arfbench.arfbench_loader import (
    load_arfbench_with_time_series,
)


class ARFBenchQADataset(QADataset):
    """
    ARFBench Question-Answer dataset for time-series anomaly detection.
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        format_sample_str: bool = False,
        time_series_format_function=None,
        csv_path: str = "arfbench-qa.csv",
        data_dir: str = "arfbench-ts-data",
        max_context_tokens: int = 30000,
    ):
        self.csv_path = csv_path
        self.data_dir = data_dir
        self.max_context_tokens = max_context_tokens
        super().__init__(
            split,
            EOS_TOKEN,
            format_sample_str,
            time_series_format_function,
        )

    def _load_splits(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Load ARFBench splits with time-series data."""
        print("Loading ARFBench dataset with time series...")
        train, val, test = load_arfbench_with_time_series(
            csv_path=self.csv_path,
            data_dir=self.data_dir,
            max_context_tokens=self.max_context_tokens,
        )
        return train, val, test

    def _get_answer(self, row) -> str:
        return row["correct_answer"]

    def _get_pre_prompt(self, row) -> str:
        pre_prompt = """## Task Description
You are an expert at analyzing time-series data and answering questions about anomalies.
Your task is to answer the given question about time-series anomalies by selecting the most appropriate option.
Focus on the key aspects of the anomaly being analyzed and provide a clear explanation for your choice.

## Question Categories
There are 8 categories of questions:
1. Anomaly Presence - Yes/no question about whether an anomaly is present
2. Anomaly Identification - Identify the channel of the anomaly
3. Anomaly Start - Identify the start time of the anomaly
4. Anomaly End - Identify the end time of the anomaly
5. Anomaly Magnitude - Identify the magnitude of the anomaly
6. Anomaly Categorization - Identify the category
   (Level Shift, Transient Spike, Change in Seasonality,
   Change in Variance, Change in Trend, No Anomaly)
7. Anomaly Correlation - Whether anomalies in two time-series are correlated
8. Anomaly Indicator - Whether an anomaly in one series
   is a leading or lagging indicator

## Instructions
- Carefully analyze the time series data provided
- Consider patterns, trends, and deviations
- Answer based on the time series values and temporal characteristics
- Be precise and choose the most appropriate answer from the options
"""
        return pre_prompt

    def _get_post_prompt(self, row) -> str:
        question = row["question"]
        options_str = row["options_str"]
        post_prompt = f"""{question}

Options: {options_str}

The answer should match one of the options exactly.
Provide your answer by selecting the most appropriate option
based on the time series data.

Answer:"""
        return post_prompt

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        time_series_json = row["time_series"]
        if isinstance(time_series_json, str):
            time_series_list = json.loads(time_series_json)
        else:
            time_series_list = time_series_json

        interval = row.get("interval", "unknown")
        query_group = row.get("query_group", "")

        prompts: List[TextTimeSeriesPrompt] = []

        for idx, ts_data in enumerate(time_series_list):
            series = np.array(ts_data, dtype=np.float32)
            mean = float(np.mean(series))
            std = max(float(np.std(series)), 1e-6)
            series_norm = (series - mean) / std

            if len(time_series_list) > 1:
                text_prompt = (
                    f"Time series {idx + 1} (query_group: {query_group}, "
                    f"interval: {interval}s, mean: {mean:.4f}, std: {std:.4f}):"
                )
            else:
                text_prompt = (
                    f"Time series (query_group: {query_group}, "
                    f"interval: {interval}s, mean: {mean:.4f}, std: {std:.4f}):"
                )

            prompts.append(TextTimeSeriesPrompt(text_prompt, series_norm.tolist()))

        return prompts

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["task_category"] = row["task_category"]
        sample["difficulty"] = row.get("difficulty", "")
        sample["query_group"] = row.get("query_group", "")
        sample["interval"] = row.get("interval", 0)
        sample["options_str"] = row.get("options_str", "")
        return sample
