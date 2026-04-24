# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
This script evaluates ChatTS (https://github.com/NetManAIOps/ChatTS/tree/main/chatts) on ARFBench.
Follow the instructions in the ChatTS repository to download and install the model. This script does not use
vLLM, so the model can be downloaded on Huggingface.
"""

import os
import pandas as pd
import numpy as np
import time
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse
import traceback

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from utils.prompt_test_benchmark import get_test_benchmark_prompt
from utils.inference_utils import parse_response, shuffle_options
from utils.log_utils import save_results_and_statistics, print_evaluation_summary


def load_benchmark(csv_path: str) -> pd.DataFrame:
    """Load and prepare benchmark data."""
    df = pd.read_csv(csv_path)
    return df


def get_time_series_data(
    query_groups: List[str], interval=None
) -> Optional[np.ndarray]:
    """Load time series data from parquet file and return array of time series."""
    data_dir = Path("arfbench-data")
    time_series_1 = []
    time_series_2 = []
    # If no interval specified, find the best interval
    for i, query_group in enumerate(query_groups):
        if interval is None:
            interval = find_best_interval(query_group)

        # Construct filename
        filename = f"{query_group}_{interval}.parquet"
        file_path = data_dir / filename

        if not file_path.exists():
            print(f"Warning: Time series file {filename} not found")
            # Return empty lists if file not found
            if i == 0:
                time_series_1 = []
            else:
                time_series_2 = []
            continue

        try:
            # Load parquet file
            df = pd.read_parquet(file_path)
            df.fillna(0, inplace=True)

            # Check if required columns exist
            if "group" not in df.columns or "value" not in df.columns:
                print(
                    f"Warning: Required columns 'group' and 'value' not found in {filename}"
                )
                if i == 0:
                    time_series_1 = []
                else:
                    time_series_2 = []
                continue

            if "epoch" not in df.columns:
                # If no epoch column, assume the index represents time
                df = df.reset_index()
                df.rename(columns={"index": "epoch"}, inplace=True)

            # Group by epoch and group, taking mean of values if there are duplicates
            pivoted_df = (
                df.groupby(["epoch", "group"])["value"].mean().unstack(fill_value=0)
            )
            pivoted_df = pivoted_df.astype(np.float32)

            # Sort columns by their mean values (descending order)
            column_means = pivoted_df.mean().sort_values(ascending=False)
            pivoted_df = pivoted_df[column_means.index]

            # Get the time series as arrays
            time_series_list = []
            if len(query_groups) == 1:
                for group_name in pivoted_df.columns[:50]:  # Limit to first 50 series
                    series_data = pivoted_df[group_name].values
                    time_series_list.append(series_data.astype(np.float32))
            else:
                for group_name in pivoted_df.columns[:25]:
                    series_data = pivoted_df[group_name].values
                    time_series_list.append(series_data.astype(np.float32))
            if i == 0:
                time_series_1 = time_series_list
            else:
                time_series_2 = time_series_list
        except Exception as e:
            print(f"Error loading time series data from {filename}: {e}")
            if i == 0:
                time_series_1 = []
            else:
                time_series_2 = []
            continue
    return time_series_1, time_series_2


def find_best_interval(query_group: str) -> Optional[int]:
    """Find best interval for query group to get time series length 64-1024."""
    data_dir = Path("arfbench-data")

    # Common intervals to try (in ascending order of granularity)
    intervals = [10, 60, 300, 1800, 3600, 86400]

    for interval in intervals:
        filename = f"{query_group}_{interval}.parquet"
        file_path = data_dir / filename

        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)

                # Check if we can determine data length
                if "epoch" in df.columns:
                    data_length = df["epoch"].nunique()
                else:
                    data_length = len(df)

                # Check if length is in the desired range
                if 64 <= data_length <= 1024:
                    return interval
                elif data_length < 64:
                    # Too short, try a more granular interval
                    continue
                else:
                    # Too long, this interval might still work if we subsample
                    # For now, accept it as the best available
                    return interval

            except Exception as e:
                print(f"Error checking {filename}: {e}")
                continue

    # If no good interval found, try the first available file
    for interval in intervals:
        filename = f"{query_group}_{interval}.parquet"
        file_path = data_dir / filename
        if file_path.exists():
            return interval

    return None


def prepare_time_series_for_model(ts_data: List[np.ndarray]) -> np.ndarray:
    """Prepare time series data for ChatTS model."""
    if not ts_data:
        return np.array([], dtype=np.float32)
    # Ensure each time series length is within acceptable range
    processed_series = []
    for series in ts_data:
        series = series.astype(np.float32)

        if len(series) > 1024:
            # Subsample to get within range
            indices = np.linspace(0, len(series) - 1, 1024, dtype=int)
            series = series[indices]
        elif len(series) < 64:
            # Pad with zeros or repeat pattern
            if len(series) > 0:
                # Repeat the pattern to reach minimum length
                repeat_factor = (64 // len(series)) + 1
                series = np.tile(series, repeat_factor)[:64]
            else:
                # Create dummy data if no data available
                series = np.zeros(64, dtype=np.float32)

        processed_series.append(series)

    return (
        np.stack(processed_series, axis=0)
        if processed_series
        else np.array([], dtype=np.float32)
    )


def create_chatts_prompt(
    question: str, options: str, system_prompt: str, lens: List[int] = None
) -> str:
    """Create a prompt for ChatTS model."""

    # Handle time series tags based on lens parameter
    if lens[0] > 0 and lens[1] == 0:
        # Case for one set of time series
        ts_tags = "Time series 1: " + f"<ts><ts/>" * lens[0]
    elif lens[0] > 0 and lens[1] > 0:
        # Case for two sets of time series
        ts_tags = (
            f"Time series 1: "
            + f"<ts><ts/>" * lens[0]
            + f" Time series 2: "
            + f"<ts><ts/>" * lens[1]
        )
    else:
        # Case for no time series data
        ts_tags = "I do not have any time series data."

    prompt = f"""{ts_tags}

Question: {question}
Options: {options}"""

    # Apply chat template
    formatted_prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return formatted_prompt


def get_model_response(
    model,
    processor,
    tokenizer,
    question: str,
    options: str,
    time_series_data: List[np.ndarray],
    system_prompt: str,
    temperature: float = 0.05,
    max_new_tokens: int = 2000,
) -> Tuple[Optional[str], Optional[str]]:
    """Get response from ChatTS model."""

    try:
        ts1_prepared = prepare_time_series_for_model(time_series_data[0])
        ts2_prepared = prepare_time_series_for_model(time_series_data[1])

        # If we have both ts1 and ts2, pad them to the same length
        if ts1_prepared.size > 0 and ts2_prepared.size > 0:
            max_len = max(ts1_prepared.shape[1], ts2_prepared.shape[1])

            # Pad ts1 if needed
            if ts1_prepared.shape[1] < max_len:
                pad_width = ((0, 0), (0, max_len - ts1_prepared.shape[1]))
                ts1_prepared = np.pad(
                    ts1_prepared, pad_width, mode="constant", constant_values=0
                )

            # Pad ts2 if needed
            if ts2_prepared.shape[1] < max_len:
                pad_width = ((0, 0), (0, max_len - ts2_prepared.shape[1]))
                ts2_prepared = np.pad(
                    ts2_prepared, pad_width, mode="constant", constant_values=0
                )

            ts_data = np.concatenate((ts1_prepared, ts2_prepared), axis=0)
        elif ts1_prepared.size > 0:
            ts_data = ts1_prepared
        elif ts2_prepared.size > 0:
            ts_data = ts2_prepared
        else:
            ts_data = np.array([], dtype=np.float32)

        if ts_data.size == 0:
            ts_list = []
        else:
            ts_list = [ts.tolist() for ts in ts_data]

        # Create prompt
        prompt = create_chatts_prompt(
            question,
            options,
            system_prompt,
            lens=[ts1_prepared.shape[0], ts2_prepared.shape[0]],
        )

        # Process inputs
        inputs = processor(
            text=[prompt], timeseries=ts_list, padding=True, return_tensors="pt"
        )

        # Move to GPU (model.device will be the correct device with device_map="auto")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode only the generated tokens (skip the input)
        prediction = tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

        answer, reasoning = parse_response(prediction)
        return answer, reasoning

    except Exception as e:
        print(f"Error getting model response: {e}")

        traceback.print_exc()
        return None, None


def process_single_question(
    model,
    processor,
    tokenizer,
    row: pd.Series,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    run: int,
    completed_count: Dict,
    total_questions: int,
) -> Optional[Dict]:
    """Process a single question."""
    query_group = row["query_group"]
    groups = query_group.split(",")
    print(f"Starting question: {groups}")

    # Load time series data
    ts1, ts2 = get_time_series_data(groups)
    shuffled_options = shuffle_options(row.get("options_str", row["options"]))
    if ts1 is None and ts2 is None:
        print(f"Skipping question for {query_group} - " f"time series data not found")
        return None

    # Get model response
    answer, reasoning = get_model_response(
        model,
        processor,
        tokenizer,
        row["question"],
        shuffled_options,
        [ts1, ts2],
        system_prompt,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )

    # Update progress
    completed_count["count"] += 1
    progress = (completed_count["count"] / total_questions) * 100
    print(
        f"✓ Completed {completed_count['count']}/{total_questions} "
        f"({progress:.1f}%) - {query_group}"
    )

    return {
        "run": int(run + 1),
        "model": "chatts",
        "query_group": query_group,
        "question": row["question"],
        "options": shuffled_options,
        "correct_answer": row["correct_answer"],
        "task_category": row["task_category"],
        "model_answer": answer,
        "is_correct": bool(answer == row["correct_answer"] if answer else False),
        "has_time_series": True,
        "num_time_series": int(len(ts1) + len(ts2)),
        "time_series_length": int(len(ts1[0]) if ts1 else 0),
        "reasoning": reasoning,
    }


def evaluate_chatts(
    model,
    processor,
    tokenizer,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    temperature: float,
    max_new_tokens: int,
    num_runs: int,
) -> pd.DataFrame:
    """Evaluate ChatTS model on the benchmark."""
    results = []

    for run in range(num_runs):
        print(f"\nRun {run + 1} for ChatTS")

        completed_count = {"count": 0}
        total_questions = len(benchmark_df)
        run_results = []

        print(f"Processing {total_questions} questions sequentially...")
        for idx, row in benchmark_df.iterrows():
            result = process_single_question(
                model,
                processor,
                tokenizer,
                row,
                system_prompt,
                temperature,
                max_new_tokens,
                run,
                completed_count,
                total_questions,
            )
            if result:
                run_results.append(result)

        results.extend(run_results)

        print(
            f"Run {run + 1} completed: {len(run_results)}/{total_questions} "
            f"questions processed successfully"
        )

    return pd.DataFrame(results)


def load_chatts_model(model_path: str):
    """Load ChatTS model, tokenizer, and processor."""
    print("Loading ChatTS model...")
    print(
        "Note: GPU selection is controlled via CUDA_VISIBLE_DEVICES environment variable"
    )
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, tokenizer=tokenizer
    )

    model.eval()
    print(f"ChatTS model loaded successfully on device: {model.device}!")
    return model, tokenizer, processor


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChatTS on ARF benchmark")
    parser.add_argument(
        "--benchmark",
        default="arfbench-qa.csv",
        help="Path to benchmark CSV",
    )
    parser.add_argument(
        "--model_path",
        help="Path to ChatTS model. This can be the cache path to the huggingface model after downloading it or a local path.",
    )
    parser.add_argument(
        "--num_runs", type=int, default=1, help="Number of evaluation runs"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.05, help="Sampling temperature"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=512, help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="chatts_evaluation",
        help="Prefix for output files",
    )
    args = parser.parse_args()

    try:
        model, tokenizer, processor = load_chatts_model(args.model_path)
    except Exception as e:
        print(f"Error loading ChatTS model: {e}")
        traceback.print_exc()
        return

    # Load benchmark
    print(f"Loading benchmark from {args.benchmark}...")
    benchmark_df = load_benchmark(args.benchmark)
    print(f"Loaded {len(benchmark_df)} questions")

    # Get system prompt
    system_prompt = get_test_benchmark_prompt()

    # Evaluate the model
    print("Starting evaluation...")
    start_time = time.time()

    results_df = evaluate_chatts(
        model,
        processor,
        tokenizer,
        benchmark_df,
        system_prompt,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_runs=args.num_runs,
    )

    end_time = time.time()
    print(f"\nEvaluation completed in {end_time - start_time:.2f} seconds")

    stats_df, tier_f1_df = save_results_and_statistics(results_df, args.output_prefix)
    print_evaluation_summary(stats_df, tier_f1_df, results_df, benchmark_df)


if __name__ == "__main__":
    main()
