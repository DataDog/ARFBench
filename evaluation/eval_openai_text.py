# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
This script evaluates OpenAI models on ARFBench with time series data as text.
This assumes you have ARFBench time series data in parquet format in a
directory called arfbench-data. The script automatically selects the optimal
time interval (10, 60, 300, 1800, 3600, 86400 seconds) that fits within the
context window.
"""

import pandas as pd
from openai import AsyncOpenAI
import argparse
import asyncio
import re
from typing import Dict, List, Optional, Tuple

from utils.prompt_test_benchmark import get_test_benchmark_prompt
from utils.inference_utils import (
    parse_response,
    load_benchmark,
    get_time_series_paths,
    load_time_series_data,
    shuffle_options,
)
from utils.log_utils import save_results_and_statistics, print_evaluation_summary


TIME_SERIES_INTERVALS = [10, 60, 300, 1800, 3600, 86400]
MAX_CONTEXT_RECOVERY_ATTEMPTS = 4


class ContextLengthExceededError(Exception):
    """Raised when API request exceeds model input context length."""

    def __init__(
        self,
        message: str,
        limit_tokens: Optional[int] = None,
        used_tokens: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.limit_tokens = limit_tokens
        self.used_tokens = used_tokens


def _extract_context_limit_and_usage(
    error_message: str,
) -> Tuple[Optional[int], Optional[int]]:
    """Extract token limit and usage from OpenAI context-length errors."""
    limit_match = re.search(r"limit of (\d+) tokens", error_message)
    used_match = re.search(r"resulted in (\d+) tokens", error_message)
    limit_tokens = int(limit_match.group(1)) if limit_match else None
    used_tokens = int(used_match.group(1)) if used_match else None
    return limit_tokens, used_tokens


def _build_context_length_error(exc: Exception) -> Optional[ContextLengthExceededError]:
    """Return a structured context-length error when present."""
    body = getattr(exc, "body", None)
    code = None
    if isinstance(body, dict):
        error_payload = body.get("error")
        if isinstance(error_payload, dict):
            code = error_payload.get("code")

    message = str(exc)
    is_context_error = (
        code == "context_length_exceeded"
        or "context_length_exceeded" in message
        or "Input tokens exceed the configured limit" in message
    )
    if not is_context_error:
        return None

    limit_tokens, used_tokens = _extract_context_limit_and_usage(message)
    return ContextLengthExceededError(
        message=message,
        limit_tokens=limit_tokens,
        used_tokens=used_tokens,
    )


def _next_coarser_intervals(current_interval: int) -> List[int]:
    """Return candidate coarser intervals than current interval."""
    if current_interval not in TIME_SERIES_INTERVALS:
        return []
    current_index = TIME_SERIES_INTERVALS.index(current_interval)
    return TIME_SERIES_INTERVALS[current_index + 1 :]


def _compute_truncation_ratio(
    context_error: ContextLengthExceededError, truncation_attempt: int
) -> float:
    """
    Compute truncation ratio for text payload retries.

    The ratio is applied to each time-series text independently.
    """
    # Default progressive backoff when token counts are unavailable.
    fallback_ratios = [0.75, 0.60, 0.45, 0.30]
    fallback_ratio = fallback_ratios[
        min(truncation_attempt - 1, len(fallback_ratios) - 1)
    ]

    if (
        context_error.limit_tokens is not None
        and context_error.used_tokens is not None
        and context_error.used_tokens > 0
    ):
        # Keep a 10% safety margin under hard model limits.
        model_ratio = (context_error.limit_tokens / context_error.used_tokens) * 0.9
        model_ratio = min(max(model_ratio, 0.20), 0.95)
        return min(fallback_ratio, model_ratio)

    return fallback_ratio


def _truncate_time_series_data(
    time_series_data: List[str], truncation_ratio: float
) -> List[str]:
    """Truncate each series text while preserving header and tail context."""
    truncated_data = []
    changed = False

    for ts_data in time_series_data:
        target_chars = max(500, int(len(ts_data) * truncation_ratio))
        if target_chars >= len(ts_data):
            truncated_data.append(ts_data)
            continue

        changed = True
        head_chars = max(300, int(target_chars * 0.85))
        tail_chars = max(0, target_chars - head_chars)
        truncation_marker = "\n\n[...TRUNCATED FOR CONTEXT WINDOW...]\n\n"

        if head_chars + tail_chars + len(truncation_marker) >= len(ts_data):
            truncated_data.append(ts_data[:target_chars])
            continue

        if tail_chars > 0:
            truncated_data.append(
                ts_data[:head_chars] + truncation_marker + ts_data[-tail_chars:]
            )
        else:
            truncated_data.append(ts_data[:head_chars] + truncation_marker.strip())

    return truncated_data if changed else time_series_data


async def get_model_response(
    client: AsyncOpenAI,
    model: str,
    question: str,
    options: str,
    time_series_data: List[str],
    system_prompt: str,
    global_temperature: float,
    use_reasoning: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """Get a single response from a model asynchronously."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": []},
    ]

    # Add time series data as text
    for i, ts_data in enumerate(time_series_data):
        messages[1]["content"].append(
            {"type": "text", "text": f"Time Series {i+1}:\n{ts_data}\n"}
        )

    # Add question and options after time series data
    messages[1]["content"].append(
        {"type": "text", "text": f"Question: {question}\nOptions: {options}"}
    )

    try:
        if not use_reasoning:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=global_temperature,
                max_tokens=2000,
            )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=1,
                max_completion_tokens=2000,
                reasoning_effort="medium",
            )

        # Get the raw response text
        response_text = response.choices[0].message.content.strip()

        answer, reasoning = parse_response(response_text)

        if answer is None:
            print(f"Warning: Could not parse response for {question}")
            print(f"Raw response: {response_text}")
            return None, None

        return answer, reasoning

    except Exception as e:
        context_error = _build_context_length_error(e)
        if context_error is not None:
            raise context_error
        print(f"Error getting response from {model}: {e}")
        return None, None


async def process_single_question(
    client: AsyncOpenAI,
    model: str,
    row: pd.Series,
    system_prompt: str,
    run: int,
    global_temperature: float,
    use_reasoning: bool,
    max_context_tokens: int,
    data_dir: str = "arfbench-data",
) -> Optional[Dict]:
    """Process a single question asynchronously."""
    shuffled_options = shuffle_options(row.get("options_str", row["options"]))
    time_series_result = get_time_series_paths(
        row["query_group"],
        data_dir,
        row["question"],
        shuffled_options,
        max_context_tokens,
    )

    # Skip questions where time series data doesn't exist or doesn't fit
    if time_series_result is None:
        print(
            f"Skipping question for {row['query_group']} - "
            f"time series data not found or doesn't fit in context window"
        )
        return None

    time_series_data, interval = time_series_result
    current_interval = interval
    current_time_series_data = time_series_data
    attempted_intervals = {current_interval}
    truncation_attempts = 0

    while True:
        try:
            answer, reasoning = await get_model_response(
                client,
                model,
                row["question"],
                shuffled_options,
                current_time_series_data,
                system_prompt,
                global_temperature,
                use_reasoning,
            )
            interval = current_interval
            time_series_data = current_time_series_data
            break
        except ContextLengthExceededError as context_error:
            moved_to_coarser_interval = False
            for next_interval in _next_coarser_intervals(current_interval):
                if next_interval in attempted_intervals:
                    continue

                candidate_data = load_time_series_data(
                    row["query_group"],
                    next_interval,
                    data_dir,
                )
                attempted_intervals.add(next_interval)
                if candidate_data is None:
                    continue

                print(
                    f"Context too large for {row['query_group']} at {current_interval}s; "
                    f"retrying with coarser interval {next_interval}s."
                )
                current_interval = next_interval
                current_time_series_data = candidate_data
                moved_to_coarser_interval = True
                break

            if moved_to_coarser_interval:
                continue

            if truncation_attempts >= MAX_CONTEXT_RECOVERY_ATTEMPTS:
                print(
                    f"Skipping question for {row['query_group']} after "
                    "context-limit retries and truncation attempts"
                )
                return None

            truncation_attempts += 1
            truncation_ratio = _compute_truncation_ratio(
                context_error, truncation_attempts
            )
            truncated_time_series_data = _truncate_time_series_data(
                current_time_series_data, truncation_ratio
            )
            if truncated_time_series_data == current_time_series_data:
                print(
                    f"Skipping question for {row['query_group']} - "
                    "unable to reduce context further"
                )
                return None

            print(
                f"Context too large for {row['query_group']} at {current_interval}s; "
                f"truncating payload (attempt {truncation_attempts}, "
                f"ratio={truncation_ratio:.2f}) and retrying."
            )
            current_time_series_data = truncated_time_series_data

    return {
        "run": run + 1,
        "model": model,
        "query_group": row["query_group"],
        "question": row["question"],
        "options": shuffled_options,
        "correct_answer": row["correct_answer"],
        "task_category": row["task_category"],
        "model_answer": answer,
        "is_correct": answer == row["correct_answer"],
        "has_time_series": True,
        "time_series_interval": interval,
        "num_time_series": len(time_series_data) if time_series_data else 0,
        "reasoning": reasoning,
    }


async def run_single_evaluation_run(
    client: AsyncOpenAI,
    model: str,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    run: int,
    global_temperature: float,
    use_reasoning: bool,
    max_context_tokens: int,
    max_concurrent: int = 10,
    data_dir: str = "arfbench-data",
) -> List[Dict]:
    """Run one evaluation pass over all questions with async concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    completed_count = 0
    total_questions = len(benchmark_df)

    async def process_with_semaphore(row, question_idx):
        nonlocal completed_count
        async with semaphore:
            print(f"  Starting question {question_idx + 1}: " f"{row['query_group']}")

            result = await process_single_question(
                client,
                model,
                row,
                system_prompt,
                run,
                global_temperature,
                use_reasoning,
                max_context_tokens,
                data_dir,
            )
            await asyncio.sleep(0.1)

            completed_count += 1
            progress_pct = completed_count / total_questions * 100
            print(
                f"  ✓ Completed {completed_count}/{total_questions} "
                f"({progress_pct:.1f}%) - {row['query_group']}"
            )

            return result

    tasks = [
        process_with_semaphore(row, i)
        for i, (_, row) in enumerate(benchmark_df.iterrows())
    ]

    print(
        f"Processing {len(tasks)} questions concurrently "
        f"(max {max_concurrent} at once)..."
    )
    run_results = await asyncio.gather(*tasks, return_exceptions=True)

    valid_results = [
        result
        for result in run_results
        if result is not None and not isinstance(result, Exception)
    ]

    error_count = 0
    for i, result in enumerate(run_results):
        if isinstance(result, Exception):
            error_count += 1
            print(f"  ❌ Error processing question {i+1}: {result}")

    success_rate = len(valid_results) / total_questions * 100
    print(
        f"✅ Run {run + 1} completed: {len(valid_results)} successful "
        f"({success_rate:.1f}%), {error_count} errors"
    )
    return valid_results


async def evaluate_model(
    client: AsyncOpenAI,
    model: str,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    global_temperature: float,
    use_reasoning: bool,
    max_context_tokens: int,
    num_runs: int = 1,
    max_concurrent: int = 10,
    data_dir: str = "arfbench-data",
) -> pd.DataFrame:
    """Evaluate a model on the benchmark with multiple runs asynchronously."""
    results = []

    for run in range(num_runs):
        print(f"\nRun {run + 1} for {model}")
        run_results = await run_single_evaluation_run(
            client,
            model,
            benchmark_df,
            system_prompt,
            run,
            global_temperature,
            use_reasoning,
            max_context_tokens,
            max_concurrent,
            data_dir,
        )
        results.extend(run_results)

    return pd.DataFrame(results)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate models on ARFBench")
    parser.add_argument(
        "--benchmark",
        default="arfbench-qa.csv",
        help="Path to benchmark CSV",
    )
    parser.add_argument(
        "--data-dir",
        default="arfbench-data",
        help="Path to time series data directory",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=200,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--global-temperature",
        type=float,
        default=0.05,
        help="Global temperature for all models",
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Use reasoning-model request parameters",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs for each model",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=120000,
        help=(
            "Maximum input context budget used when selecting time-series "
            "intervals."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default="openai",
        help="Name of the experiment",
    )
    args = parser.parse_args()
    system_prompt = get_test_benchmark_prompt()

    # Initialize AsyncOpenAI client
    client = AsyncOpenAI()

    # Load benchmark
    benchmark_df = load_benchmark(args.benchmark)

    models = [args.model]

    # Evaluate each model
    all_results = []
    for model in models:
        print(f"\nEvaluating {model}...")
        results_df = await evaluate_model(
            client,
            model,
            benchmark_df,
            system_prompt,
            args.global_temperature,
            args.reasoning,
            args.max_context_tokens,
            num_runs=args.num_runs,
            max_concurrent=args.max_concurrent,
            data_dir=args.data_dir,
        )
        results_df.to_csv(f"{model}_{args.experiment_name}_results.csv", index=False)
        all_results.append(results_df)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    stats_df, tier_f1_df = save_results_and_statistics(
        combined_results,
        args.experiment_name,
    )
    print_evaluation_summary(stats_df, tier_f1_df, combined_results, benchmark_df)


if __name__ == "__main__":
    asyncio.run(main())
