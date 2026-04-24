# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Utility functions for ARFBench evaluation.
"""

import json
import random
import re
import io
import os
import pandas as pd
from PIL import Image
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any


@dataclass
class ModelConfig:
    """
    Class for model configurations for vLLM.

    Args:
        name (str): Name of the model
        hf_overrides (Optional[Dict[str, Any]]): Hugging Face overrides for the model
        max_model_len (int): Maximum length of the model's input
        max_num_seqs (int): Maximum number of sequences the model can process
        tensor_parallel_size (int): Number of tensor parallel processes
        trust_remote_code (bool): Whether to trust remote code
    """

    name: str
    hf_overrides: Optional[Dict[str, Any]] = None
    max_model_len: int = 30000
    max_num_seqs: int = 2
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True


def _normalize_text(value: Any) -> Optional[str]:
    """Convert values to stripped text while treating NaN as missing."""
    if value is None:
        return None

    if isinstance(value, float) and pd.isna(value):
        return None

    text = str(value).strip()
    return text or None


def _parse_options_str(options_str: Any) -> Optional[List[str]]:
    """Parse options_str into a list of option texts."""
    if isinstance(options_str, list):
        parsed_options = options_str
    else:
        options_text = _normalize_text(options_str)
        if not options_text:
            return None

        try:
            parsed_options = json.loads(options_text)
        except json.JSONDecodeError:
            return None

    if not isinstance(parsed_options, list):
        return None

    option_values = []
    for option in parsed_options:
        option_text = _extract_option_text(option)
        if option_text:
            option_values.append(option_text)

    return option_values if len(option_values) >= 2 else None


def _extract_option_text(option: Any) -> Optional[str]:
    """Extract text from scalar options or simple option dictionaries."""
    if isinstance(option, dict):
        for key in ("value", "text", "option", "answer"):
            option_text = _normalize_text(option.get(key))
            if option_text:
                return option_text
        return None

    return _normalize_text(option)


def shuffle_options(
    options_str: Any,
) -> str:
    f"""
    Shuffle options from options_str and return them as JSON text.
    options_str is a column in arfbench-qa.csv, but this function can also take
    in a list of options in JSON format, such as the options column in arfbench-qa.csv.
    Returns the original options text if options_str cannot be parsed.
    """
    fallback_options = _normalize_text(options_str) or ""
    option_values = _parse_options_str(options_str)
    if not option_values:
        return fallback_options

    shuffled_values = option_values.copy()
    random.shuffle(shuffled_values)

    shuffled_options = json.dumps(shuffled_values, ensure_ascii=False)
    return shuffled_options


def shuffle_options_in_question(question: str) -> str:
    """
    Shuffle the answer options embedded in a question string.

    Training-data questions embed options as:
        '... Answer Choices: [opt1, opt2, opt3]'
    This function parses the bracket-delimited list, shuffles it, and
    reconstructs the question with the shuffled options.

    Returns the original question unchanged if no parseable options are found.
    """
    for delimiter in ["Answer Choices:", "Options:"]:
        if delimiter not in question:
            continue

        prefix, opts_part = question.split(delimiter, 1)
        opts_part = opts_part.strip()

        # Try JSON parse first (handles quoted strings)
        try:
            opts_list = json.loads(opts_part)
            if isinstance(opts_list, list) and len(opts_list) >= 2:
                random.shuffle(opts_list)
                return (
                    f"{prefix}{delimiter} {json.dumps(opts_list, ensure_ascii=False)}"
                )
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: parse bracket-delimited, comma-separated plain text
        # Format: [opt1, opt2, opt3]
        if opts_part.startswith("[") and "]" in opts_part:
            bracket_end = opts_part.index("]")
            inner = opts_part[1:bracket_end]
            items = [item.strip() for item in inner.split(",")]
            if len(items) >= 2:
                random.shuffle(items)
                shuffled = "[" + ", ".join(items) + "]"
                rest = opts_part[bracket_end + 1 :]
                return f"{prefix}{delimiter} {shuffled}{rest}"

    return question


def parse_response(prediction: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse model response to extract answer and reasoning.

    Tries multiple strategies to extract structured information from model
    responses, from most specific to most general:
    1. JSON format with "answer" and "reasoning" fields
    2. "Answer:" prefix format
    3. Simple sentence format (answer before first period)
    4. Raw prediction as fallback

    Args:
        prediction (str): Raw model prediction string

    Returns:
        Tuple[Optional[str], Optional[str]]: (answer, reasoning) where answer
        is the extracted answer and reasoning is the extracted reasoning. Both
        can be None if parsing fails.

    Examples:
        >>> parse_response('{"answer": "A", "reasoning": "Because X"}')
        ('A', 'Because X')

        >>> parse_response('Answer: B\\nThe data shows a clear trend.')
        ('B', 'The data shows a clear trend.')

        >>> parse_response('C. This is the correct choice.')
        ('C', ' This is the correct choice.')

        >>> parse_response('Invalid response')
        ('Invalid response', None)
    """
    if not prediction or not isinstance(prediction, str):
        return None, None

    prediction = prediction.strip()
    if not prediction:
        return None, None

    # Strategy 1: Try to extract JSON from response
    try:
        if "{" in prediction and "}" in prediction:
            start = prediction.find("{")
            end = prediction.rfind("}") + 1
            json_str = prediction[start:end]
            prediction_json = json.loads(json_str)
            answer = prediction_json.get("answer")
            reasoning = prediction_json.get("reasoning")
            return answer, reasoning
        elif "{" in prediction:
            prediction = prediction + "}"
            start = prediction.find("{")
            end = prediction.rfind("}") + 1
            json_str = prediction[start:end]
            prediction_json = json.loads(json_str)
            answer = prediction_json.get("answer")
            reasoning = prediction_json.get("reasoning")
            return answer, reasoning
    except json.JSONDecodeError:
        # Continue to other strategies if JSON parsing fails
        pass

    # Strategy 2: Look for "Answer:" prefix format
    answer_prefix = "Answer:"
    if answer_prefix in prediction:
        start = prediction.find(answer_prefix) + len(answer_prefix)
        end = prediction.find("\n", start)
        if end == -1:  # No newline found, take rest of string
            answer = prediction[start:].strip()
            reasoning = None
        else:
            answer = prediction[start:end].strip()
            reasoning = prediction[end + 1 :].strip()
            reasoning = reasoning if reasoning else None
        return answer, reasoning

    # Strategy 3: Simple sentence format (answer before first period)
    if "." in prediction:
        end = prediction.find(".")
        answer = prediction[:end].strip()
        reasoning = prediction[end + 1 :].strip()
        reasoning = reasoning if reasoning else None
        return answer, reasoning

    # Strategy 4: Return raw prediction as answer with no reasoning
    return prediction, None


def load_benchmark(csv_path: str) -> pd.DataFrame:
    """Load and prepare benchmark data."""
    df = pd.read_csv(csv_path)
    return df


def resize_image(image_path: str, max_size: int = 1500) -> bytes:
    """
    Resize image to be under max_size pixels on each side while maintaining
    aspect ratio.

    Args:
        image_path (str): Path to the image file
        max_size (int): Maximum size for width or height

    Returns:
        bytes: Resized image as PNG bytes
    """
    with Image.open(image_path) as img:
        width, height = img.size

        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()


def resize_image_pil(image_path: str, max_size: int = 1500) -> Image.Image:
    """
    Resize image to be under max_size pixels on each side while maintaining
    aspect ratio.

    Args:
        image_path (str): Path to the image file
        max_size (int): Maximum size for width or height

    Returns:
        Image.Image: Resized PIL Image object
    """
    with Image.open(image_path) as img:
        width, height = img.size

        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        return img.copy()


def format_incident_context(incident_1: str, incident_2: str) -> str:
    """
    Format incident data as context for the model.

    Args:
        incident_1 (str): First incident description
        incident_2 (str): Second incident description

    Returns:
        str: Formatted incident context string
    """
    context_parts = []

    if pd.notna(incident_1) and str(incident_1).strip():
        context_parts.append(f"Incident Context 1: {str(incident_1).strip()}")

    if pd.notna(incident_2) and str(incident_2).strip():
        context_parts.append(f"Incident Context 2: {str(incident_2).strip()}")

    if context_parts:
        return "\n\n" + "\n\n".join(context_parts) + "\n\n"
    else:
        return ""


def remove_timeseries_caption(question: str) -> str:
    """
    Remove the time-series caption from the question text.

    The caption is the text after and including "Time-series:" or "Time-series 1:".
    Only used for ablation experiments.

    Args:
        question (str): Original question text

    Returns:
        str: Question text without the time-series caption
    """
    if not question:
        return question

    # Find the position of "Time-series:" or "Time-series 1:" etc.
    patterns = [r"Time-series\s*\d*:", r"Time-series:"]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            # Return everything before the match
            return question[: match.start()].strip()

    # If no pattern found, return original question
    return question


def get_image_paths(
    query_groups: str,
    image_dir: str = "arfbench-images-v2",
    task_category: Optional[str] = None,
) -> Optional[List[str]]:
    """
    Find all image files for given query groups.

    Args:
        query_groups (str): Comma-separated query group identifiers
        image_dir (str): Directory containing image files
        task_category (str, optional): Task category (for special handling
            of Anomaly Indicator/Correlation)

    Returns:
        Optional[List[str]]: List of image paths if all found,
        None if any missing
    """
    image_dir = Path(image_dir)
    image_paths = []
    groups = [g.strip() for g in query_groups.split(",")]

    # Special handling for Anomaly Indicator and Anomaly Correlation
    # These use both a combined image AND separate images:
    # {query_group1}-{query_group2}.png + individual images
    if task_category and (
        "Indicator" in task_category or "Correlation" in task_category
    ):
        if len(groups) == 2:
            # First, try to find the combined image
            combined_filename = f"{groups[0]}-{groups[1]}.png"
            combined_path = image_dir / combined_filename
            combined_found = False

            if combined_path.exists():
                image_paths.append(str(combined_path))
                combined_found = True

            # Then find the individual images for each group
            # Look for exact matches first (e.g., "36504_0.png"), then fallback to prefix matches
            # but exclude paired images (files containing a dash)
            for group in groups:
                found = False
                # First try exact match: {group}.png
                exact_match = image_dir / f"{group}.png"
                if exact_match.exists():
                    image_paths.append(str(exact_match))
                    found = True
                else:
                    # Fallback to prefix match, but exclude paired images (containing dash)
                    for file in image_dir.glob(f"{group}*"):
                        file_name = file.name
                        # Skip paired images (contain dash) and the combined image we already added
                        if "-" not in file_name and str(file) not in image_paths:
                            image_paths.append(str(file))
                            found = True
                            break

                if not found:
                    return None

            # Return the combined list if we found the combined image
            if combined_found:
                return image_paths
            else:
                # If no combined image found, fallback to default behavior
                image_paths = []
        else:
            # Fallback to original behavior if not exactly 2 groups
            pass

    # Default behavior: find separate images for each query group
    # Prefer exact matches (e.g., "36504_0.png") and skip paired images (containing dash)
    for group in groups:
        found = False
        # First try exact match: {group}.png
        exact_match = image_dir / f"{group}.png"
        if exact_match.exists():
            image_paths.append(str(exact_match))
            found = True
        else:
            # Fallback to prefix match, but exclude paired images (containing dash)
            for file in image_dir.glob(f"{group}*"):
                file_name = file.name
                # Skip paired images (contain dash)
                if "-" not in file_name:
                    image_paths.append(str(file))
                    found = True
                    break
        if not found:
            return None

    return image_paths


def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Uses a rough approximation of 4 characters per token.

    Args:
        text (str): Text to estimate tokens for

    Returns:
        int: Estimated number of tokens
    """
    return len(text) // 4


def find_optimal_time_series_interval(
    query_groups: str,
    data_dir: str = "../../arfbench-data",
    question: str = "",
    options: str = "",
    max_context_tokens: int = 120000,
) -> Optional[int]:
    """
    Find the optimal time series interval that fits within the context window.

    Args:
        query_groups (str): Comma-separated query group identifiers
        data_dir (str): Directory containing parquet files
        question (str): Question text for token estimation
        options (str): Options text for token estimation
        max_context_tokens (int): Maximum context window size in tokens

    Returns:
        Optional[int]: Optimal interval in seconds, or None if no suitable
        interval found
    """
    # Available intervals in order of preference (smallest to largest)
    intervals = [10, 60, 300, 1800, 3600, 86400]

    # Estimate tokens for question and options
    base_tokens = estimate_token_count(question + options)

    # Reserve some tokens for system prompt and response
    reserved_tokens = 2000
    available_tokens = max_context_tokens - base_tokens - reserved_tokens

    groups = [g.strip() for g in query_groups.split(",")]

    for interval in intervals:
        total_tokens = 0
        all_files_exist = True

        for group in groups:
            file_path = os.path.join(data_dir, f"{group}_{interval}.parquet")

            if not os.path.exists(file_path):
                all_files_exist = False
                break

            try:
                # Load the parquet file and estimate tokens
                df = pd.read_parquet(file_path)
                # Format as CSV-like text for token estimation
                csv_text = df.to_csv(index=False)
                file_tokens = estimate_token_count(csv_text)
                total_tokens += file_tokens

            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                all_files_exist = False
                break

        if all_files_exist and total_tokens <= available_tokens:
            return interval

    # If no interval fits, return the largest available interval
    # and let the model handle truncation
    for interval in reversed(intervals):
        all_files_exist = True
        for group in groups:
            file_path = os.path.join(data_dir, f"{group}_{interval}.parquet")
            if not os.path.exists(file_path):
                all_files_exist = False
                break
        if all_files_exist:
            return interval

    return None


def load_time_series_data(
    query_groups: str, interval: int, data_dir: str = "../../arfbench-data"
) -> Optional[List[str]]:
    """
    Load time series data from parquet files and format as text.

    Args:
        query_groups (str): Comma-separated query group identifiers
        interval (int): Time interval in seconds
        data_dir (str): Directory containing parquet files

    Returns:
        Optional[List[str]]: List of formatted time series data strings,
        or None if error
    """
    groups = [g.strip() for g in query_groups.split(",")]
    time_series_data = []

    for group in groups:
        file_path = os.path.join(data_dir, f"{group}_{interval}.parquet")

        if not os.path.exists(file_path):
            print(f"Time series file not found: {file_path}")
            return None

        try:
            df = pd.read_parquet(file_path)

            # Format the data as a readable text representation
            formatted_data = f"Time Series Data for {group} (interval: {interval}s):\n"
            formatted_data += f"Columns: {', '.join(df.columns.tolist())}\n"
            formatted_data += f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
            formatted_data += (
                f"Time range: {df['epoch'].min()} to {df['epoch'].max()}\n\n"
            )

            # Convert to CSV format for the actual data
            csv_data = df.to_csv(index=False)
            formatted_data += csv_data

            time_series_data.append(formatted_data)

        except Exception as e:
            print(f"Error loading time series data from {file_path}: {e}")
            return None

    return time_series_data


def get_time_series_paths(
    query_groups: str,
    data_dir: str = "../../arfbench-data",
    question: str = "",
    options: str = "",
    max_context_tokens: int = 120000,
) -> Optional[Tuple[List[str], int]]:
    """
    Find time series data paths and optimal interval for given query groups.

    Args:
        query_groups (str): Comma-separated query group identifiers
        data_dir (str): Directory containing parquet files
        question (str): Question text for context estimation
        options (str): Options text for context estimation
        max_context_tokens (int): Maximum context window size

    Returns:
        Optional[Tuple[List[str], int]]: Tuple of (time_series_data_list,
        interval) or None if no suitable data found
    """
    # Find optimal interval
    optimal_interval = find_optimal_time_series_interval(
        query_groups, data_dir, question, options, max_context_tokens
    )

    if optimal_interval is None:
        return None

    # Load the time series data
    time_series_data = load_time_series_data(query_groups, optimal_interval, data_dir)

    if time_series_data is None:
        return None

    return time_series_data, optimal_interval
