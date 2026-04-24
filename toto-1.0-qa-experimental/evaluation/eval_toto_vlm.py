# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
This script evaluates Toto + VLM models on ARFBench.
It combines time series embeddings from Toto with vision-language model inputs
for evaluating anomaly detection questions.

Key differences from eval.py:
- Loads TotoAnomalyQAModel instead of vanilla VLM
- Loads time series data from parquet files
- Extracts and projects Toto embeddings
- Combines TS embeddings with VLM inputs during inference
"""

import argparse
import os
import warnings
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoProcessor,
)

# Add parent directory and utils to path
toto_vlm_root = Path(__file__).parent.parent
sys.path.insert(0, str(toto_vlm_root))
sys.path.insert(0, str(toto_vlm_root.parent))

from utils.prompt_test_benchmark import get_test_benchmark_prompt
from utils.inference_utils import (
    parse_response,
    get_image_paths,
)
from utils.compute_statistics import (
    compute_statistics,
    compute_multiclass_f1_metrics,
    compute_tier_weighted_f1,
)
from utils.log_utils import log_progress
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Import model components
from model.load_checkpoint import load_checkpoint, load_checkpoint_for_tensor_parallel
from model.toto_vlm_components import TimeSeriesData, format_ts_metadata_for_prompt

warnings.filterwarnings("ignore", message="None of the inputs have requires_grad")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

# Default data directory, defaults to current directory
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", os.getcwd())

# Hugging Face ARFBench dataset (arfbench-qa.csv, arfbench-ts-data/, arfbench-images/)
HF_ARFBENCH_DATASET = "Datadog/ARFBench"


def resolve_benchmark_from_huggingface(
    hf_dataset: str = HF_ARFBENCH_DATASET,
    cache_dir: Optional[str] = None,
) -> Tuple[str, str, str]:
    """
    Download ARFBench from Hugging Face and return paths to CSV, TS data, and images.

    Expects the dataset to contain:
      - arfbench-qa.csv
      - arfbench-ts-data/  (parquet files)
      - arfbench-images/   (image files)

    Returns:
        Tuple of (benchmark_csv_path, data_dir, image_dir).
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to use --benchmark-source huggingface. "
            "Install with: pip install huggingface_hub"
        )
    kwargs = {"repo_type": "dataset"}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    print(f"Downloading ARFBench from Hugging Face: {hf_dataset}")
    local_dir = snapshot_download(repo_id=hf_dataset, **kwargs)
    root = Path(local_dir)
    benchmark_path = root / "arfbench-qa.csv"
    data_dir = root / "arfbench-ts-data"
    image_dir = root / "arfbench-images"
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Dataset missing arfbench-qa.csv at {benchmark_path}. "
            f"Ensure {hf_dataset} contains arfbench-qa.csv."
        )
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset missing arfbench-ts-data/ at {data_dir}. "
            f"Ensure {hf_dataset} contains arfbench-ts-data/."
        )
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset missing arfbench-images/ at {image_dir}. "
            f"Ensure {hf_dataset} contains arfbench-images/."
        )
    print(f"  Benchmark CSV: {benchmark_path}")
    print(f"  TS data dir:   {data_dir}")
    print(f"  Image dir:     {image_dir}")
    return str(benchmark_path), str(data_dir), str(image_dir)


# Available time series intervals (in seconds), ordered from highest to lowest resolution
AVAILABLE_INTERVALS = [10, 60, 300, 1800, 3600, 86400]

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
LONG_TS_TASK_CATEGORIES = frozenset(TIER_MAPPINGS["Tier 3"])


def select_max_ts_groups_for_task(
    task_category: Optional[str],
    short_max_ts_groups: int,
    long_max_ts_groups: int,
) -> int:
    """Return the TS group budget for a question category."""
    if task_category in LONG_TS_TASK_CATEGORIES:
        return long_max_ts_groups
    return short_max_ts_groups


def find_optimal_interval(
    query_group: str,
    data_dir: str,
    max_length: int = 4096,
    prefer_high_resolution: bool = True,
) -> Optional[int]:
    """
    Find the optimal time series interval that fits within Toto's context window.

    Strategy:
    1. Try intervals from highest to lowest resolution
    2. For each interval, check if the data fits within max_length
    3. Return the highest resolution that fits

    Args:
        query_group: Query group identifier (e.g., "35928_0")
        data_dir: Directory containing parquet files
        max_length: Maximum sequence length (Toto context window = 4096)
        prefer_high_resolution: If True, prefer higher resolution (lower interval)

    Returns:
        Optimal interval in seconds, or None if no suitable file found
    """
    intervals = (
        AVAILABLE_INTERVALS if prefer_high_resolution else reversed(AVAILABLE_INTERVALS)
    )

    for interval in intervals:
        file_path = Path(data_dir) / f"{query_group}_{interval}.parquet"

        if not file_path.exists():
            continue

        try:
            df = pd.read_parquet(file_path)

            # Pivot to get timesteps
            if "group" in df.columns and "epoch" in df.columns:
                ts_pivot = df.pivot_table(
                    index="epoch", columns="group", values="value", aggfunc="first"
                )
                n_timesteps = len(ts_pivot)
            else:
                n_timesteps = len(df)

            # Check if it fits within max_length
            if n_timesteps <= max_length:
                return interval
            else:
                print(
                    f"  Interval {interval}s too large: {n_timesteps} timesteps > {max_length}"
                )

        except Exception as e:
            print(f"  Error checking interval {interval}s: {e}")
            continue

    # If no interval fits, return the coarsest one and we'll chunk it
    print(f"  No interval fits within {max_length}, will use coarsest and chunk")
    for interval in reversed(AVAILABLE_INTERVALS):
        file_path = Path(data_dir) / f"{query_group}_{interval}.parquet"
        if file_path.exists():
            return interval

    return None


def load_and_preprocess_timeseries(
    query_group: str,
    data_dir: str,
    interval: int,
    max_length: int = 4096,
    max_groups: int = 20,
    interpolate: bool = False,
    remove_all_nan_groups: bool = True,
) -> Optional[TimeSeriesData]:
    """
    Load and preprocess time series data with interpolation and NaN handling.

    Args:
        query_group: Query group identifier (e.g., "35928_0")
        data_dir: Directory containing parquet files
        interval: Time series interval in seconds
        max_length: Maximum sequence length
        max_groups: Maximum number of groups/channels
        interpolate: If True, interpolate missing values
        remove_all_nan_groups: If True, remove groups that are all NaN

    Returns:
        TimeSeriesData object or None if file not found
    """
    file_path = Path(data_dir) / f"{query_group}_{interval}.parquet"

    if not file_path.exists():
        return None

    try:
        # Load parquet file
        df = pd.read_parquet(file_path)

        # Convert epoch to datetime if needed
        if df["epoch"].dtype != "datetime64[ns]":
            df["epoch"] = pd.to_datetime(df["epoch"])

        # Pivot to wide format: (timesteps, channels)
        ts_pivot = df.pivot_table(
            index="epoch", columns="group", values="value", aggfunc="first"
        ).sort_index()

        # Remove groups that are all NaN
        if remove_all_nan_groups:
            # Check which columns are all NaN
            all_nan_cols = ts_pivot.columns[ts_pivot.isna().all()]
            if len(all_nan_cols) > 0:
                print(
                    f"  Removing {len(all_nan_cols)} all-NaN groups: {list(all_nan_cols)}"
                )
                ts_pivot = ts_pivot.drop(columns=all_nan_cols)

        # Capture group names BEFORE limiting to max_groups (preserve order)
        group_names = ts_pivot.columns.tolist()

        # Get dimensions after removing all-NaN groups
        n_timesteps, n_channels = ts_pivot.shape
        n_channels = min(n_channels, max_groups)

        # Take only first max_groups channels if there are more
        if ts_pivot.shape[1] > max_groups:
            ts_pivot = ts_pivot.iloc[:, :max_groups]
            # Also limit group_names to match
            group_names = group_names[:max_groups]
            n_channels = max_groups

        # Convert to numpy then tensor: (channels, timesteps)
        ts_array = ts_pivot.values.T  # Transpose to (channels, timesteps)
        ts_tensor = torch.from_numpy(ts_array.astype(np.float32))

        # Handle NaN values
        for i in range(ts_tensor.shape[0]):
            channel = ts_tensor[i]
            mask = torch.isnan(channel)

            if mask.any():
                if interpolate:
                    # Use pandas for interpolation
                    series = pd.Series(channel.numpy())
                    # Linear interpolation, then forward fill, then backward fill, then 0
                    series = series.interpolate(method="linear", limit_direction="both")
                    series = series.ffill().bfill().fillna(0)
                    ts_tensor[i] = torch.from_numpy(series.values)
                else:
                    # Simple forward/backward fill without interpolation
                    series = pd.Series(channel.numpy())
                    series = series.ffill().bfill().fillna(0)
                    ts_tensor[i] = torch.from_numpy(series.values)

        # Pad or truncate to max_length
        current_length = ts_tensor.shape[1]
        if current_length < max_length:
            # Pad with zeros
            pad_length = max_length - current_length
            ts_tensor = F.pad(ts_tensor, (0, pad_length), value=0)
            padding_mask = torch.cat(
                [
                    torch.ones(n_channels, current_length, dtype=torch.bool),
                    torch.zeros(n_channels, pad_length, dtype=torch.bool),
                ],
                dim=1,
            )
        else:
            # Truncate
            ts_tensor = ts_tensor[:, :max_length]
            padding_mask = torch.ones(n_channels, max_length, dtype=torch.bool)

        # Pad channels if fewer than max_groups
        if n_channels < max_groups:
            channel_pad = max_groups - n_channels
            ts_tensor = F.pad(ts_tensor, (0, 0, 0, channel_pad), value=0)
            padding_mask = F.pad(padding_mask, (0, 0, 0, channel_pad), value=False)
            # Pad group_names with empty strings for padded channels
            group_names = group_names + [""] * channel_pad

        # Create timestamp and interval tensors
        timestamps = (
            torch.arange(max_length, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(max_groups, 1)
        )
        timestamps *= interval

        time_intervals = torch.full((max_groups,), interval, dtype=torch.float32)

        # ID mask (all zeros for now)
        id_mask = torch.zeros_like(ts_tensor)

        return TimeSeriesData(
            series=ts_tensor,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamps,
            time_interval_seconds=time_intervals,
            num_groups=n_channels,
            query_group=query_group,
            group_names=group_names,
        )

    except Exception as e:
        print(f"  Error loading time series: {e}")
        return None


def load_timeseries_adaptive(
    query_group: str,
    data_dir: str,
    max_length: int = 4096,
    max_groups: int = 20,
    auto_interval: bool = True,
    fixed_interval: Optional[int] = None,
    interpolate: bool = False,
    remove_all_nan_groups: bool = True,
) -> Optional[TimeSeriesData]:
    """
    Load time series data with adaptive interval selection and preprocessing.

    This function:
    1. Finds the optimal interval that fits in Toto's context window
    2. Loads the time series data at that interval
    3. Applies interpolation if specified
    4. Removes all-NaN groups if specified

    Args:
        query_group: Query group identifier (e.g., "35928_0")
        data_dir: Directory containing parquet files
        max_length: Maximum sequence length (Toto context window = 4096)
        max_groups: Maximum number of groups/channels
        auto_interval: If True, automatically select optimal interval
        fixed_interval: If provided, use this specific interval
        interpolate: If True, apply interpolation to missing values
        remove_all_nan_groups: If True, remove groups that are all NaN

    Returns:
        TimeSeriesData object or None if file not found
    """
    # Determine interval to use
    if auto_interval and fixed_interval is None:
        interval = find_optimal_interval(query_group, data_dir, max_length, max_groups)
        if interval is None:
            print(f"  No suitable interval found for {query_group}")
            return None
    else:
        interval = fixed_interval if fixed_interval is not None else 60  # default
        print(f"  Using fixed interval: {interval}s")

    # Load and preprocess the time series
    ts_data = load_and_preprocess_timeseries(
        query_group=query_group,
        data_dir=data_dir,
        interval=interval,
        max_length=max_length,
        max_groups=max_groups,
        interpolate=interpolate,
        remove_all_nan_groups=remove_all_nan_groups,
    )

    return ts_data


def load_model_and_processor(
    model_path: str,
    base_model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
    use_lora: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_timeseries: bool = True,
    toto_model_name: str = "Datadog/Toto-Open-Base-1.0",
    force_single_gpu: bool = True,
    ts_components_path: Optional[str] = None,
) -> Tuple[nn.Module, AutoProcessor]:
    """
    Load the trained Toto + VLM model and processor.

    model_path can be:
    - A Hugging Face Hub model id (e.g. "Datadog/Toto-1.0-QA-Experimental"): loads via
      load_checkpoint(repo_id), which uses TotoAnomalyQAModel.from_pretrained under the hood.
    - A local path to a mixin-format directory (vlm/, ts_modules.pt, config.json): same as Hub.
    - A local path to a legacy checkpoint (adapter/ + ts_modules.pt): uses base_model_name and use_lora.

    Args:
        model_path: Hub model id (e.g. Datadog/Toto-1.0-QA-Experimental) or path to checkpoint directory
        base_model_name: Base VLM model name (ignored when loading from Hub or mixin format)
        use_lora: Whether the model uses LoRA adapters (ignored for Hub/mixin format)
        load_in_8bit: Load base model in 8-bit quantization
        load_in_4bit: Load base model in 4-bit quantization
        use_timeseries: Whether to enable time series integration
        toto_model_name: Name of Toto model (ignored for Hub/mixin; Toto loaded from config)
        force_single_gpu: Force single GPU for TS mode (if False, enables tensor parallelism)

    Returns:
        Tuple of (model, processor)
    """
    # Use tensor parallel loader if multi-GPU is enabled and time series is used
    if not force_single_gpu and use_timeseries:
        print("[Model] Loading with tensor parallelism support")
        return load_checkpoint_for_tensor_parallel(
            checkpoint_dir=model_path,
            base_model_name=base_model_name,
            use_lora=use_lora,
            use_timeseries=use_timeseries,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            toto_model_name=toto_model_name,
            num_gpus=None,  # Auto-detect all available GPUs
            ts_components_path=ts_components_path,
        )
    else:
        # Use standard loader for single GPU or non-TS mode
        if not force_single_gpu:
            print("[Model] Loading with standard multi-GPU device_map='auto'")

        return load_checkpoint(
            checkpoint_dir=model_path,
            base_model_name=base_model_name,
            use_lora=use_lora,
            use_timeseries=use_timeseries,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            toto_model_name=toto_model_name,
            force_single_gpu=force_single_gpu,
            ts_components_path=ts_components_path,
        )


def get_model_prompt(
    question: str,
    image_paths: List[str],
    processor: AutoProcessor,
    system_prompt: Optional[str] = None,
    ts_data: Optional[TimeSeriesData] = None,
    add_ts_metadata: bool = False,
) -> Tuple[str, List[Image.Image], List[Dict]]:
    """
    Generate the appropriate prompt and image data for the model.

    Args:
        question: Question text with answer choices
        image_paths: List of paths to image files
        processor: Model processor
        system_prompt: Optional system prompt (uses benchmark prompt if None)
        ts_data: Optional time series data for adding metadata
        add_ts_metadata: Whether to add channel and timestamp information to prompt

    Returns:
        Tuple of (text_prompt, images, messages)
    """
    if system_prompt is None:
        system_prompt = get_test_benchmark_prompt()

    # Add time series channel and timestamp information to question if enabled
    if add_ts_metadata and ts_data is not None:
        ts_data_for_metadata = (
            ts_data[0] if isinstance(ts_data, list) and len(ts_data) > 0 else ts_data
        )
        ts_metadata_str = format_ts_metadata_for_prompt(ts_data_for_metadata)
        if ts_metadata_str is not None:
            question = f"{question}\n\n{ts_metadata_str}"

    # Create messages with image placeholders
    user_content = []
    for path in image_paths:
        user_content.append({"type": "image", "image": path})
    user_content.append({"type": "text", "text": question})

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Extract processed images using qwen_vl_utils
    processed_images, _ = process_vision_info(messages)

    # Apply chat template to get text prompt
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return text_prompt, processed_images, messages


def save_results_incrementally(results: List[Dict], output_file: str):
    """
    Save results to CSV file incrementally.

    Args:
        results: List of result dictionaries
        output_file: Path to output CSV file
    """
    if not output_file:
        return

    try:
        results_df = pd.DataFrame(results)
        # Create a temporary file first, then rename (atomic operation)
        temp_file = output_file + ".tmp"
        results_df.to_csv(temp_file, index=False)
        os.replace(temp_file, output_file)  # Atomic rename
    except Exception as e:
        print(f"[WARNING] Failed to save results incrementally: {e}")


def load_existing_results(output_file):
    """Load existing results for resume functionality.

    Returns:
        Tuple of (existing_results list, completed_questions set of (question_index, run) tuples)
    """
    existing_results = []
    completed_questions = set()

    print(f"\n[RESUME] Loading existing results from {output_file}")
    try:
        existing_results_df = pd.read_csv(output_file)
        existing_results = existing_results_df.to_dict("records")

        for result in existing_results:
            if "question_index" in result:
                key = (result["question_index"], result["run"])
                completed_questions.add(key)

        print(f"[RESUME] Found {len(completed_questions)} completed question-run pairs")
        if completed_questions:
            max_completed_idx = max(idx for idx, _ in completed_questions)
            print(
                f"[RESUME] Progress: up to question index {max_completed_idx} (line {max_completed_idx + 2} in CSV)"
            )
    except Exception as e:
        print(f"[RESUME] Warning: Could not load existing results: {e}")
        print(f"[RESUME] Starting fresh evaluation")
        existing_results = []
        completed_questions = set()

    return existing_results, completed_questions


def prepare_question_data(
    row,
    question_idx,
    image_dir,
    use_timeseries,
    data_dir,
    ts_interval,
    max_ts_length,
    short_max_ts_groups,
    long_max_ts_groups,
):
    """Prepare data for a single question including images and time series.

    Returns:
        Dict with question data, or None if the question should be skipped.
    """
    image_paths = get_image_paths(
        row["query_group"],
        image_dir=image_dir,
        task_category=row.get("task_category"),
    )

    if image_paths is None:
        print(f"Skipping question for {row['query_group']} - images not found")
        return None

    options = str(row["options_str"]).replace('"', "")
    question = f"{row['question']}\nAnswer Choices: {options}"

    ts_data = None
    if use_timeseries:
        selected_max_ts_groups = select_max_ts_groups_for_task(
            row.get("task_category"), short_max_ts_groups, long_max_ts_groups
        )
        query_group_str = row["query_group"]
        query_groups = [qg.strip() for qg in query_group_str.split(",")]

        interpolate_1 = bool(row.get("interpolate_1", 0))
        interpolate_2 = bool(row.get("interpolate_2", 0))

        num_ts = len(query_groups)
        max_groups_per_ts = (
            selected_max_ts_groups // num_ts
            if num_ts > 0
            else selected_max_ts_groups
        )
        if max_groups_per_ts == 0:
            max_groups_per_ts = 1

        ts_data_list = []
        for idx, query_group in enumerate(query_groups):
            interpolate_flag = (
                interpolate_1
                if idx == 0
                else (interpolate_2 if idx == 1 else interpolate_1)
            )

            if ts_interval is None or ts_interval == 0:
                ts_data_item = load_timeseries_adaptive(
                    query_group=query_group,
                    data_dir=data_dir,
                    max_length=max_ts_length,
                    max_groups=max_groups_per_ts,
                    auto_interval=True,
                    fixed_interval=None,
                    interpolate=interpolate_flag,
                    remove_all_nan_groups=True,
                )
            else:
                ts_data_item = load_and_preprocess_timeseries(
                    query_group=query_group,
                    data_dir=data_dir,
                    interval=ts_interval,
                    max_length=max_ts_length,
                    max_groups=max_groups_per_ts,
                    interpolate=interpolate_flag,
                    remove_all_nan_groups=True,
                )

            if ts_data_item is None:
                print(f"[WARNING] No time series data found for {query_group}")
            else:
                ts_data_list.append(ts_data_item)

        if len(ts_data_list) == 0:
            ts_data = None
        elif len(ts_data_list) == 1:
            ts_data = ts_data_list[0]
        else:
            ts_data = ts_data_list

    return {
        "row": row,
        "question_idx": question_idx,
        "image_paths": image_paths,
        "question": question,
        "ts_data": ts_data,
    }


def move_ts_data_to_device(ts_data, device):
    """Move time series data tensors to the specified device."""
    if ts_data is None:
        return
    if isinstance(ts_data, list):
        for ts_item in ts_data:
            if ts_item is not None:
                ts_item.series = ts_item.series.to(device)
                ts_item.padding_mask = ts_item.padding_mask.to(device)
                ts_item.id_mask = ts_item.id_mask.to(device)
                ts_item.timestamp_seconds = ts_item.timestamp_seconds.to(device)
                ts_item.time_interval_seconds = ts_item.time_interval_seconds.to(device)
    else:
        ts_data.series = ts_data.series.to(device)
        ts_data.padding_mask = ts_data.padding_mask.to(device)
        ts_data.id_mask = ts_data.id_mask.to(device)
        ts_data.timestamp_seconds = ts_data.timestamp_seconds.to(device)
        ts_data.time_interval_seconds = ts_data.time_interval_seconds.to(device)


def generate_and_decode(
    model, processor, inputs, ts_data, use_timeseries, temperature, max_new_tokens
):
    """Run model generation and decode the output to a prediction string."""
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }

    if temperature > 0:
        gen_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": 0.95,
            }
        )
    else:
        gen_kwargs["do_sample"] = False

    using_ts_mode = use_timeseries and ts_data is not None

    with torch.no_grad():
        if using_ts_mode:
            ts_data_list = ts_data if isinstance(ts_data, list) else [ts_data]
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                ts_data=ts_data_list,
                **gen_kwargs,
            )
        else:
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                **gen_kwargs,
            )

    output_length = outputs.shape[1]
    prompt_length = inputs["input_ids"].shape[1]

    if using_ts_mode and output_length < prompt_length:
        generated_ids = outputs[0]
        prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()
    elif output_length > prompt_length:
        generated_ids = outputs[0, prompt_length:]
        prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()
    else:
        generated_ids = outputs[0]
        prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

    if not prediction:
        print(f"⚠️  WARNING: Empty prediction!")
        print(f"  Prompt length: {prompt_length}, Output length: {output_length}")
        print(f"  Generated tokens: {generated_ids.shape[0]}")
        print(f"  TS mode: {using_ts_mode}")

    return prediction, generated_ids


def build_result_entry(row, item, answer, reasoning, ts_data, is_correct):
    """Build a result dictionary for a single question (single run only)."""
    return {
        "run": 1,
        "question_index": item["question_idx"],
        "model": "toto_vlm",
        "query_group": row["query_group"],
        "question": row["question"],
        "options": row["options"],
        "correct_answer": row["correct_answer"],
        "task_category": row["task_category"],
        "model_answer": answer,
        "is_correct": is_correct,
        "has_images": True,
        "has_timeseries": ts_data is not None,
        "num_images": len(item["image_paths"]),
        "reasoning": reasoning,
    }


def evaluate_toto_vlm_model(
    model: nn.Module,
    processor: AutoProcessor,
    benchmark_df: pd.DataFrame,
    image_dir: str = "arfbench-images-v2",
    use_timeseries: bool = True,
    data_dir: str = DEFAULT_DATA_DIR,
    ts_interval: Optional[int] = None,
    max_ts_length: int = 4096,
    short_max_ts_groups: int = 100,
    long_max_ts_groups: int = 1000,
    temperature: float = 0.05,
    max_new_tokens: int = 2000,
    batch_size: int = 1,  # Batch inference
    output_file: Optional[str] = None,  # For incremental saving
    resume: bool = True,  # Whether to resume from existing results
    add_ts_metadata: bool = False,  # Whether to add channel/timestamp info to prompt
) -> pd.DataFrame:
    """
    Evaluate Toto + VLM model on the benchmark with incremental saving.

    Args:
        model: Toto + VLM model
        processor: Model processor
        benchmark_df: Benchmark DataFrame
        use_timeseries: Whether to use time series data
        data_dir: Directory containing time series parquet files
        ts_interval: Time series interval in seconds (None or 0 for auto-selection)
        max_ts_length: Maximum time series length (Toto context window)
        short_max_ts_groups: Max TS groups for non-Tier-3 categories
        long_max_ts_groups: Max TS groups for Tier-3 categories
        synthetic: Whether using synthetic data (different file naming)
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate
        batch_size: Batch size for inference
        output_file: Path to output CSV file for incremental saving (enables resume)
        resume: Whether to resume from existing results file
        add_ts_metadata: Whether to add channel and timestamp information to prompt

    Returns:
        DataFrame with evaluation results
    """
    print(f"\nEvaluating model on {len(benchmark_df)} questions")
    print(f"Use time series: {use_timeseries}")
    print(f"Data directory: {data_dir}")
    print(f"Batch size: {batch_size}")
    print(
        f"TS group caps (short/long): {short_max_ts_groups}/{long_max_ts_groups} "
        f"(Tier-3 long categories: {sorted(LONG_TS_TASK_CATEGORIES)})"
    )

    # Load existing results if resuming
    existing_results = []
    completed_questions = set()

    if resume and output_file and os.path.exists(output_file):
        existing_results, completed_questions = load_existing_results(output_file)
    elif output_file:
        print(f"[RESUME] No existing results found, starting fresh evaluation")
        print(f"[RESUME] Results will be saved incrementally to {output_file}")

    results = existing_results.copy()
    correct_count = sum(1 for r in results if r.get("is_correct"))
    tier_counts = {tier: {"correct": 0, "total": 0} for tier in TIER_MAPPINGS}

    # Compute statistics over existing results
    for r in results:
        task_category = r.get("task_category")
        if task_category:
            for tier_name, categories in TIER_MAPPINGS.items():
                if task_category in categories:
                    tier_counts[tier_name]["total"] += 1
                    if r.get("is_correct"):
                        tier_counts[tier_name]["correct"] += 1

    # Prepare all data (skip already completed when resuming)
    all_data = []
    for question_idx, row in benchmark_df.iterrows():
        if (question_idx, 1) in completed_questions:
            continue
        item = prepare_question_data(
            row,
            question_idx,
            image_dir,
            use_timeseries,
            data_dir,
            ts_interval,
            max_ts_length,
            short_max_ts_groups,
            long_max_ts_groups,
        )
        if item is not None:
            all_data.append(item)

    # Process in batches
    for batch_start in tqdm(
        range(0, len(all_data), batch_size), desc="Processing batches"
    ):
        batch_end = min(batch_start + batch_size, len(all_data))
        batch = all_data[batch_start:batch_end]

        print(
            f"Processing batch {batch_start // batch_size + 1}/{(len(all_data) + batch_size - 1) // batch_size}"
        )

        try:
            batch_prompts = []
            batch_images_list = []
            batch_ts_data = []
            batch_metadata = []

            for item in batch:
                text_prompt, images, messages = get_model_prompt(
                    item["question"],
                    item["image_paths"],
                    processor,
                    ts_data=item["ts_data"],
                    add_ts_metadata=add_ts_metadata,
                )
                batch_prompts.append(text_prompt)
                batch_images_list.append(images)
                batch_ts_data.append(item["ts_data"])
                batch_metadata.append(item)

            for idx in range(len(batch)):
                text_prompt = batch_prompts[idx]
                images = batch_images_list[idx]
                ts_data = batch_ts_data[idx]
                item = batch_metadata[idx]
                row = item["row"]

                inputs = processor(
                    text=[text_prompt],
                    images=[images],
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    max_length=None,
                )

                device = next(model.parameters()).device
                inputs = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in inputs.items()
                }

                move_ts_data_to_device(ts_data, device)

                prediction, generated_ids = generate_and_decode(
                    model,
                    processor,
                    inputs,
                    ts_data,
                    use_timeseries,
                    temperature,
                    max_new_tokens,
                )

                answer, reasoning = parse_response(prediction)

                if answer is None or prediction == "":
                    print(
                        f"Warning: Empty or unparseable response for {row['question'][:100]}..."
                    )
                    print(f"Raw prediction: '{prediction}'")
                    print(f"Generated {generated_ids.shape[0]} tokens")
                    answer = None
                    reasoning = None

                print(f"Answer: {answer}, Correct answer: {row['correct_answer']}")
                is_correct = answer == row["correct_answer"]
                task_category = row["task_category"]

                result_entry = build_result_entry(
                    row,
                    item,
                    answer,
                    reasoning,
                    ts_data,
                    is_correct,
                )
                results.append(result_entry)

                if output_file:
                    save_results_incrementally(results, output_file)
                    print(f"[CHECKPOINT] Saved progress to {output_file}")

                if is_correct:
                    correct_count += 1

                for tier_name, categories in TIER_MAPPINGS.items():
                    if task_category in categories:
                        tier_counts[tier_name]["total"] += 1
                        if is_correct:
                            tier_counts[tier_name]["correct"] += 1

            log_progress(
                results,
                tier_counts,
                correct_count,
                batch_end,
                len(all_data),
            )

        except Exception as e:
            print(f"Error processing batch: {e}")
            import traceback

            traceback.print_exc()

            for item in batch:
                row = item["row"]
                result_entry = build_result_entry(
                    row,
                    item,
                    None,
                    None,
                    item["ts_data"],
                    False,
                )
                results.append(result_entry)

            if output_file:
                save_results_incrementally(results, output_file)
                print(f"[CHECKPOINT] Saved progress after error to {output_file}")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Toto + VLM models on ARFBench"
    )
    parser.add_argument(
        "--benchmark-source",
        choices=("local", "huggingface"),
        default="huggingface",
        help="Where to get the benchmark: 'local' (use --benchmark, --data-dir, --image-dir) "
        "or 'huggingface' (download from --hf-dataset: arfbench-qa.csv, arfbench-ts-data, arfbench-images)",
    )
    parser.add_argument(
        "--benchmark",
        default="arfbench-test/arfbench-qa.csv",
        help="Path to benchmark CSV (used only when --benchmark-source=local)",
    )
    parser.add_argument(
        "--model-path",
        default="Datadog/Toto-1.0-QA-Experimental",
        help="Path to checkpoint directory or Hugging Face Hub model id (e.g. Datadog/Toto-1.0-QA-Experimental). "
        "Use a Hub id to load via load_checkpoint(repo_id); use a local path for adapter/ + ts_modules.pt layout.",
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-VL-32B-Instruct",
        help="Base VLM model name (only used for legacy local checkpoints with adapter/; ignored for Hub or mixin-format)",
    )
    parser.add_argument(
        "--toto-model",
        default="Datadog/Toto-Open-Base-1.0",
        help="Toto model name",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Whether the model uses LoRA adapters",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (use full model)",
    )
    parser.add_argument(
        "--use-timeseries",
        action="store_true",
        default=True,
        help="Use time series integration",
    )
    parser.add_argument(
        "--no-timeseries",
        action="store_true",
        help="Disable time series integration",
    )
    parser.add_argument(
        "--hf-dataset",
        default=HF_ARFBENCH_DATASET,
        help="Hugging Face dataset id for ARFBench (e.g. Datadog/ARFBench). Used when --benchmark-source=huggingface.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional directory to cache Hugging Face dataset. If unset, uses default HF cache.",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Directory containing time series parquet files (used only when --benchmark-source=local)",
    )
    parser.add_argument(
        "--image-dir",
        default="arfbench-images-v2",
        help="Directory containing image files (used only when --benchmark-source=local)",
    )
    parser.add_argument(
        "--ts-interval",
        type=int,
        default=0,
        help="Time series interval in seconds (0 or None for auto-selection of optimal interval)",
    )
    parser.add_argument(
        "--max-ts-length",
        type=int,
        default=12800,
        help="Maximum time series length",
    )
    parser.add_argument(
        "--max-ts-groups",
        type=int,
        default=None,
        help="Legacy flag for a single TS group cap (applies to both short and long when used alone)",
    )
    parser.add_argument(
        "--short-max-ts-groups",
        type=int,
        default=100,
        help="Max TS groups/channels for non-Tier-3 categories",
    )
    parser.add_argument(
        "--long-max-ts-groups",
        type=int,
        default=1000,
        help="Max TS groups/channels for Tier-3 categories (Anomaly Indicator/Correlation)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (different file naming convention)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2000,
        help="Maximum tokens to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--allow-multi-gpu",
        action="store_true",
        default=True,
        help="Enable tensor parallelism across multiple GPUs (distributes VLM layers, keeps Toto on GPU 0)",
    )
    parser.add_argument(
        "--output-prefix",
        default="toto_vlm",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Enable incremental saving and resume from existing results if available (default: enabled)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable incremental saving and start fresh evaluation",
    )
    parser.add_argument(
        "--ts-components",
        default=None,
        type=str,
        help="Path to ts_modules.pt file from a previous run to load TS components (e.g., 120225-sft-then-labeled-sft/ts_modules.pt)",
    )
    parser.add_argument(
        "--add-ts-metadata",
        action="store_true",
        default=True,
        help="Add channel names and timestamp information to the evaluation prompt (like in training DataCollator)",
    )
    args = parser.parse_args()

    # Handle resume flag
    if args.no_resume:
        args.resume = False

    # Handle boolean flags
    use_lora = args.use_lora and not args.no_lora
    use_timeseries = args.use_timeseries and not args.no_timeseries
    if args.max_ts_groups is not None:
        using_default_split_caps = (
            args.short_max_ts_groups == 100 and args.long_max_ts_groups == 1000
        )
        if using_default_split_caps:
            args.short_max_ts_groups = args.max_ts_groups
            args.long_max_ts_groups = args.max_ts_groups
        else:
            print(
                "[CONFIG] --max-ts-groups is deprecated; explicit short/long caps take precedence."
            )

    # Resolve benchmark/data/image paths from Hugging Face if requested
    if args.benchmark_source == "huggingface":
        args.benchmark, args.data_dir, args.image_dir = (
            resolve_benchmark_from_huggingface(
                hf_dataset=args.hf_dataset,
                cache_dir=args.hf_cache_dir,
            )
        )

    # Load benchmark
    print(f"Loading benchmark from {args.benchmark}")
    benchmark_df = pd.read_csv(args.benchmark)
    print(f"Loaded {len(benchmark_df)} questions")

    # Load model and processor
    print("\n" + "=" * 80)
    print("MODEL LOADING CONFIGURATION")
    print("=" * 80)
    if (
        "/" in args.model_path
        and not os.path.isdir(args.model_path)
        and not os.path.isfile(args.model_path)
    ):
        print(f"Model source: Hugging Face Hub (repo_id={args.model_path})")
    else:
        print(f"Model source: local path ({args.model_path})")
    print(
        f"Multi-GPU Tensor Parallelism: {'ENABLED' if args.allow_multi_gpu else 'DISABLED'}"
    )
    if args.allow_multi_gpu:
        import torch

        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        print(f"Strategy: VLM distributed across GPUs, Toto/TS on GPU 0")
    else:
        print(f"Strategy: All components on single GPU")
    print("=" * 80 + "\n")

    model, processor = load_model_and_processor(
        model_path=args.model_path,
        base_model_name=args.base_model,
        use_lora=use_lora,
        load_in_8bit=args.load_in_8bit,
        load_in_4bit=args.load_in_4bit,
        use_timeseries=use_timeseries,
        toto_model_name=args.toto_model,
        force_single_gpu=not args.allow_multi_gpu,
        ts_components_path=args.ts_components,
    )

    # Handle ts_interval: treat 0 as None (auto-select)
    ts_interval = args.ts_interval if args.ts_interval > 0 else None

    # Determine output file for incremental saving
    output_suffix = "_results.csv"
    results_file = f"{args.output_prefix}{output_suffix}"

    # Run evaluation with incremental saving and resume support
    print(f"\n{'=' * 80}")
    print("EVALUATION MODE")
    print(f"{'=' * 80}")
    print(f"Incremental saving: {'ENABLED' if args.resume else 'DISABLED'}")
    if args.resume:
        print(f"Results file: {results_file}")
        print(f"Mode: Resume from existing results if available")
        print(f"OOM Recovery: Yes - can restart and continue from last checkpoint")
    print(f"{'=' * 80}\n")

    results_df = evaluate_toto_vlm_model(
        model=model,
        processor=processor,
        benchmark_df=benchmark_df,
        image_dir=args.image_dir,
        use_timeseries=use_timeseries,
        data_dir=args.data_dir,
        ts_interval=ts_interval,
        max_ts_length=args.max_ts_length,
        short_max_ts_groups=args.short_max_ts_groups,
        long_max_ts_groups=args.long_max_ts_groups,
        synthetic=args.synthetic,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        output_file=results_file if args.resume else None,
        resume=args.resume,
        add_ts_metadata=args.add_ts_metadata,
    )

    # Save results
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved raw results to {results_file}")

    # Compute and save statistics
    stats_df = compute_statistics(results_df)
    stats_file = f"{args.output_prefix}_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved statistics to {stats_file}")

    # Compute and save F1 metrics
    f1_df = compute_multiclass_f1_metrics(results_df)
    f1_file = f"{args.output_prefix}_f1.csv"
    f1_df.to_csv(f1_file, index=False)
    print(f"Saved F1 metrics to {f1_file}")

    # Compute and save tier F1 metrics
    tier_f1_df = compute_tier_weighted_f1(results_df, category_metrics_df=f1_df)
    tier_f1_file = f"{args.output_prefix}_tier_f1.csv"
    tier_f1_df.to_csv(tier_f1_file, index=False)
    print(f"Saved tier F1 metrics to {tier_f1_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print("\nOverall Statistics:")
    print(stats_df.to_string())

    print("\nTier-Level Weighted F1 Scores:")
    if not tier_f1_df.empty:
        for _, row in tier_f1_df.iterrows():
            print(f"\n{row['tier']}:")
            print(f"  Weighted F1:        {row['weighted_f1']:.4f}")
            print(f"  Weighted Precision: {row['weighted_precision']:.4f}")
            print(f"  Weighted Recall:    {row['weighted_recall']:.4f}")
            print(f"  Total Questions:    {row['total_questions']}")
            print(f"  Categories:         {row['categories']}")
            print(f"  Per-Category F1:")
            for cat, f1_score in row["category_f1_scores"].items():
                count = row["category_question_counts"][cat]
                print(f"    - {cat}: {f1_score:.4f} (n={count})")
    else:
        print("  No tier F1 data available.")

    # Print skipped questions summary
    skipped_questions = set(benchmark_df["query_group"]) - set(
        results_df["query_group"].unique()
    )
    if skipped_questions:
        print("\nSkipped Questions (missing data):")
        for q in sorted(skipped_questions):
            print(f"- {q}")

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
