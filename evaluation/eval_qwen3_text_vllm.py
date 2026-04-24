# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
This script evaluates Qwen3 text models on ARFBench time-series data using vLLM.
It assumes ARFBench time-series parquet data is available in a directory.
"""

import argparse
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.prompt_test_benchmark import get_test_benchmark_prompt
from utils.inference_utils import (
    get_time_series_paths,
    load_benchmark,
    load_time_series_data,
    parse_response,
    shuffle_options,
)
from utils.log_utils import save_results_and_statistics, print_evaluation_summary


TIME_SERIES_INTERVALS = [10, 60, 300, 1800, 3600, 86400]
MAX_CONTEXT_RECOVERY_ATTEMPTS = 4


class ContextLengthExceededError(Exception):
    """Raised when generation exceeds model input context length."""

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
    """Extract token limit and token usage from context-length errors."""
    limit_patterns = [
        r"maximum context length is\s+(\d+)\s+tokens",
        r"max_model_len(?:=|\s+is\s+)(\d+)",
        r"limit of\s+(\d+)\s+tokens",
        r"configured limit(?: of)?\s+(\d+)\s+tokens",
    ]
    used_patterns = [
        r"you requested\s+(\d+)\s+tokens",
        r"resulted in\s+(\d+)\s+tokens",
        r"input (?:length|tokens).{0,20}?(\d+)\s+tokens",
    ]

    limit_tokens = None
    used_tokens = None

    for pattern in limit_patterns:
        match = re.search(pattern, error_message, flags=re.IGNORECASE)
        if match:
            limit_tokens = int(match.group(1))
            break

    for pattern in used_patterns:
        match = re.search(pattern, error_message, flags=re.IGNORECASE)
        if match:
            used_tokens = int(match.group(1))
            break

    return limit_tokens, used_tokens


def _build_context_length_error(exc: Exception) -> Optional[ContextLengthExceededError]:
    """Return structured context-length error when present."""
    message = str(exc)
    lower = message.lower()
    is_context_error = any(
        marker in lower
        for marker in [
            "context length",
            "maximum context",
            "max_model_len",
            "input too long",
            "prompt is too long",
            "exceeds the model",
            "requested tokens",
        ]
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
    """Compute truncation ratio for text payload retries."""
    fallback_ratios = [0.75, 0.60, 0.45, 0.30]
    fallback_ratio = fallback_ratios[
        min(truncation_attempt - 1, len(fallback_ratios) - 1)
    ]

    if (
        context_error.limit_tokens is not None
        and context_error.used_tokens is not None
        and context_error.used_tokens > 0
    ):
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


def _build_chat_prompt(
    tokenizer,
    question: str,
    options: str,
    time_series_data: List[str],
    system_prompt: str,
) -> str:
    """Build a text-only chat prompt for Qwen3-style instruct models."""
    user_sections = []
    for i, ts_data in enumerate(time_series_data, start=1):
        user_sections.append(f"Time Series {i}:\n{ts_data}")
    user_sections.append(f"Question: {question}\nOptions: {options}")
    user_content = "\n\n".join(user_sections)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_model_response(
    llm: LLM,
    tokenizer,
    question: str,
    options: str,
    time_series_data: List[str],
    system_prompt: str,
    global_temperature: float,
    max_tokens: int,
) -> Tuple[Optional[str], Optional[str]]:
    """Get a single response from a vLLM model."""
    prompt = _build_chat_prompt(
        tokenizer=tokenizer,
        question=question,
        options=options,
        time_series_data=time_series_data,
        system_prompt=system_prompt,
    )

    sampling_params = SamplingParams(
        temperature=global_temperature,
        max_tokens=max_tokens,
        top_p=0.95 if global_temperature > 0 else 1.0,
    )

    try:
        outputs = llm.generate([prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()
        answer, reasoning = parse_response(response_text)

        if answer is None:
            print(f"Warning: Could not parse response for question: {question}")
            print(f"Raw response: {response_text}")
            return None, None

        return answer, reasoning

    except Exception as exc:  # noqa: BLE001
        context_error = _build_context_length_error(exc)
        if context_error is not None:
            raise context_error
        print(f"Error getting response: {exc}")
        return None, None


def get_model_responses_batch(
    llm: LLM,
    tokenizer,
    batch_requests: List[Dict],
    system_prompt: str,
    global_temperature: float,
    max_tokens: int,
) -> List[Tuple[Optional[str], Optional[str]]]:
    """Get batched responses from a vLLM model."""
    prompts = []
    for request in batch_requests:
        prompts.append(
            _build_chat_prompt(
                tokenizer=tokenizer,
                question=request["row"]["question"],
                options=request["options"],
                time_series_data=request["time_series_data"],
                system_prompt=system_prompt,
            )
        )

    sampling_params = SamplingParams(
        temperature=global_temperature,
        max_tokens=max_tokens,
        top_p=0.95 if global_temperature > 0 else 1.0,
    )

    outputs = llm.generate(prompts, sampling_params)
    parsed_results: List[Tuple[Optional[str], Optional[str]]] = []

    for output, request in zip(outputs, batch_requests):
        response_text = output.outputs[0].text.strip() if output.outputs else ""
        answer, reasoning = parse_response(response_text)
        if answer is None:
            print(
                f"Warning: Could not parse batched response for question: "
                f"{request['row']['question']}"
            )
            print(f"Raw response: {response_text}")
        parsed_results.append((answer, reasoning))

    return parsed_results


def _build_result_record(
    run: int,
    model_name: str,
    row: pd.Series,
    options: str,
    answer: Optional[str],
    reasoning: Optional[str],
    interval: int,
    time_series_data: List[str],
) -> Dict:
    """Build standardized output row for one evaluated question."""
    return {
        "run": run + 1,
        "model": model_name,
        "query_group": row["query_group"],
        "question": row["question"],
        "options": options,
        "correct_answer": row["correct_answer"],
        "task_category": row["task_category"],
        "model_answer": answer,
        "is_correct": answer == row["correct_answer"],
        "has_time_series": True,
        "time_series_interval": interval,
        "num_time_series": len(time_series_data) if time_series_data else 0,
        "reasoning": reasoning,
    }


def prepare_question_inputs(
    row: pd.Series,
    max_context_tokens: int,
    data_dir: str,
) -> Optional[Dict]:
    """Prepare options and time-series payload for one question."""
    shuffled_options = shuffle_options(row.get("options_str", row["options"]))
    time_series_result = get_time_series_paths(
        row["query_group"],
        data_dir,
        row["question"],
        shuffled_options,
        max_context_tokens,
    )

    if time_series_result is None:
        print(
            f"Skipping question for {row['query_group']} - "
            "time series data not found or does not fit context budget"
        )
        return None

    time_series_data, interval = time_series_result
    return {
        "row": row,
        "options": shuffled_options,
        "time_series_data": time_series_data,
        "interval": interval,
    }


def process_prepared_question(
    llm: LLM,
    tokenizer,
    model_name: str,
    prepared_question: Dict,
    system_prompt: str,
    run: int,
    global_temperature: float,
    max_tokens: int,
    data_dir: str,
) -> Optional[Dict]:
    """
    Process one pre-prepared question with runtime context-recovery retries.

    This path is used both for normal single-question execution and as a
    fallback when batched generation fails (for example due to one oversized
    prompt in the batch).
    """
    row = prepared_question["row"]
    shuffled_options = prepared_question["options"]
    current_time_series_data = prepared_question["time_series_data"]
    current_interval = prepared_question["interval"]
    attempted_intervals = {current_interval}
    truncation_attempts = 0

    while True:
        try:
            answer, reasoning = get_model_response(
                llm=llm,
                tokenizer=tokenizer,
                question=row["question"],
                options=shuffled_options,
                time_series_data=current_time_series_data,
                system_prompt=system_prompt,
                global_temperature=global_temperature,
                max_tokens=max_tokens,
            )
            return _build_result_record(
                run=run,
                model_name=model_name,
                row=row,
                options=shuffled_options,
                answer=answer,
                reasoning=reasoning,
                interval=current_interval,
                time_series_data=current_time_series_data,
            )
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


def process_single_question(
    llm: LLM,
    tokenizer,
    model_name: str,
    row: pd.Series,
    system_prompt: str,
    run: int,
    global_temperature: float,
    max_tokens: int,
    max_context_tokens: int,
    data_dir: str,
) -> Optional[Dict]:
    """Process one question (single-item execution path)."""
    prepared_question = prepare_question_inputs(row, max_context_tokens, data_dir)
    if prepared_question is None:
        return None

    return process_prepared_question(
        llm=llm,
        tokenizer=tokenizer,
        model_name=model_name,
        prepared_question=prepared_question,
        system_prompt=system_prompt,
        run=run,
        global_temperature=global_temperature,
        max_tokens=max_tokens,
        data_dir=data_dir,
    )


def evaluate_model(
    llm: LLM,
    tokenizer,
    model_name: str,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    global_temperature: float,
    max_tokens: int,
    max_context_tokens: int,
    data_dir: str,
    num_runs: int = 1,
    batch_size: int = 8,
) -> pd.DataFrame:
    """Evaluate vLLM model on benchmark with true prompt batching."""
    results = []
    total_questions = len(benchmark_df)
    benchmark_rows = list(benchmark_df.iterrows())
    total_batches = (total_questions + batch_size - 1) // batch_size

    for run in range(num_runs):
        print(f"\nRun {run + 1} for {model_name}")
        completed_count = 0
        run_results = []

        for batch_start in range(0, total_questions, batch_size):
            batch_end = min(batch_start + batch_size, total_questions)
            current_rows = benchmark_rows[batch_start:batch_end]
            print(
                f"Processing batch {batch_start // batch_size + 1}/{total_batches} "
                f"({batch_end - batch_start} questions)"
            )

            prepared_batch = []
            for question_idx, (_, row) in enumerate(current_rows, start=batch_start + 1):
                print(
                    f"  Starting question {question_idx}/{total_questions}: "
                    f"{row['query_group']}"
                )
                prepared_question = prepare_question_inputs(
                    row=row,
                    max_context_tokens=max_context_tokens,
                    data_dir=data_dir,
                )
                if prepared_question is None:
                    completed_count += 1
                    progress_pct = completed_count / total_questions * 100
                    print(
                        f"  ✓ Completed {completed_count}/{total_questions} "
                        f"({progress_pct:.1f}%) - {row['query_group']}"
                    )
                    continue
                prepared_batch.append(prepared_question)

            if not prepared_batch:
                continue

            try:
                batched_responses = get_model_responses_batch(
                    llm=llm,
                    tokenizer=tokenizer,
                    batch_requests=prepared_batch,
                    system_prompt=system_prompt,
                    global_temperature=global_temperature,
                    max_tokens=max_tokens,
                )

                for prepared_question, (answer, reasoning) in zip(
                    prepared_batch, batched_responses
                ):
                    row = prepared_question["row"]
                    run_results.append(
                        _build_result_record(
                            run=run,
                            model_name=model_name,
                            row=row,
                            options=prepared_question["options"],
                            answer=answer,
                            reasoning=reasoning,
                            interval=prepared_question["interval"],
                            time_series_data=prepared_question["time_series_data"],
                        )
                    )
                    completed_count += 1
                    progress_pct = completed_count / total_questions * 100
                    print(
                        f"  ✓ Completed {completed_count}/{total_questions} "
                        f"({progress_pct:.1f}%) - {row['query_group']}"
                    )
            except Exception as exc:  # noqa: BLE001
                context_error = _build_context_length_error(exc)
                if context_error is not None:
                    print(
                        "Batch generation hit context length limits; "
                        "falling back to sequential retry for this batch."
                    )
                else:
                    print(
                        f"Batch generation error ({exc}); "
                        "falling back to sequential retry for this batch."
                    )

                for prepared_question in prepared_batch:
                    row = prepared_question["row"]
                    result = process_prepared_question(
                        llm=llm,
                        tokenizer=tokenizer,
                        model_name=model_name,
                        prepared_question=prepared_question,
                        system_prompt=system_prompt,
                        run=run,
                        global_temperature=global_temperature,
                        max_tokens=max_tokens,
                        data_dir=data_dir,
                    )
                    if result is not None:
                        run_results.append(result)

                    completed_count += 1
                    progress_pct = completed_count / total_questions * 100
                    print(
                        f"  ✓ Completed {completed_count}/{total_questions} "
                        f"({progress_pct:.1f}%) - {row['query_group']}"
                    )

        success_rate = len(run_results) / total_questions * 100 if total_questions else 0
        print(
            f"✅ Run {run + 1} completed: {len(run_results)} successful "
            f"({success_rate:.1f}%)"
        )
        results.extend(run_results)

    return pd.DataFrame(results)


def load_vllm_model(
    model_path: str,
    tensor_parallel_size: int,
    max_num_seqs: int,
    dtype: str,
    max_model_len: Optional[int],
    gpu_memory_utilization: Optional[float],
):
    """Load tokenizer and vLLM model."""
    print(f"Loading tokenizer for {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm_kwargs = {
        "model": model_path,
        "tensor_parallel_size": tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "trust_remote_code": True,
        "dtype": dtype,
    }
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if gpu_memory_utilization is not None:
        llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    print(
        "Loading vLLM model with "
        f"tensor_parallel_size={tensor_parallel_size}, max_num_seqs={max_num_seqs}"
    )
    llm = LLM(**llm_kwargs)
    return llm, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3 text models on ARFBench with vLLM"
    )
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
        "--model-path",
        default="Qwen/Qwen3-32B-Instruct",
        help="Model path or Hugging Face model id",
    )
    parser.add_argument(
        "--model-name",
        default="qwen3-32b-instruct-vllm",
        help="Model label used in output rows and filenames",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Tensor parallel size for vLLM",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=8,
        help="Max number of concurrent sequences in vLLM scheduler",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["auto", "half", "float16", "bfloat16", "float32"],
        help="Model dtype for vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional max model length override for vLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory to use for vLLM",
    )
    parser.add_argument(
        "--global-temperature",
        type=float,
        default=0.05,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum generated tokens per answer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="True vLLM prompt batch size per generate() call",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of full evaluation passes",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=30000,
        help=(
            "Context token budget used when selecting time-series interval "
            "before generation."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default="qwen3_text_vllm",
        help="Prefix name for saved aggregate CSV outputs",
    )
    args = parser.parse_args()

    system_prompt = get_test_benchmark_prompt()
    benchmark_df = load_benchmark(args.benchmark)
    print(f"Loaded benchmark with {len(benchmark_df)} questions")

    llm, tokenizer = load_vllm_model(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    print(f"\nEvaluating {args.model_name}...")
    results_df = evaluate_model(
        llm=llm,
        tokenizer=tokenizer,
        model_name=args.model_name,
        benchmark_df=benchmark_df,
        system_prompt=system_prompt,
        global_temperature=args.global_temperature,
        max_tokens=args.max_tokens,
        max_context_tokens=args.max_context_tokens,
        data_dir=args.data_dir,
        num_runs=args.num_runs,
        batch_size=args.batch_size,
    )

    if results_df.empty:
        print("No results were generated. Exiting without statistics export.")
        return

    per_model_results_path = (
        f"{args.model_name}_{args.experiment_name}_results.csv"
    )
    results_df.to_csv(per_model_results_path, index=False)
    print(f"Per-model results saved to {per_model_results_path}")

    stats_df, tier_f1_df = save_results_and_statistics(
        results_df,
        args.experiment_name,
    )
    print_evaluation_summary(stats_df, tier_f1_df, results_df, benchmark_df)


if __name__ == "__main__":
    main()
