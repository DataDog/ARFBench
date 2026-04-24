# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Inference Context Ablation experiment.

This script evaluates GPT-4o or any OpenAI model under three different conditions:
1. With incident context (incident_1 and incident_2 columns). Note: This is not provided in the public release of ARFBench.
2. Without time-series captions (removing text after "Time-series:")
3. Without images (text-only evaluation)

This assumes you have ARFBench plot data in a directory called arfbench-images.
"""

import pandas as pd
from openai import AsyncOpenAI
import argparse
import base64
import asyncio
from typing import Dict, List, Optional, Tuple

from utils.prompt_test_benchmark import get_test_benchmark_prompt
from utils.inference_utils import (
    parse_response,
    load_benchmark,
    resize_image,
    get_image_paths,
    format_incident_context,
    remove_timeseries_caption,
    shuffle_options,
)
from utils.compute_statistics import (
    compute_statistics,
    compute_multiclass_f1_metrics,
)


async def get_model_response_with_incident(
    client: AsyncOpenAI,
    model: str,
    question: str,
    options: str,
    image_paths: List[str],
    system_prompt: str,
    global_temperature: float,
    use_reasoning: bool,
    incident_context: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """Get a single response from a model with optional incident context."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": []},
    ]

    for image_path in image_paths:
        try:
            # Resize and encode image
            image_bytes = resize_image(image_path)
            image_data = base64.b64encode(image_bytes).decode("utf-8")

            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                }
            )
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None

    # Add text after images, including incident context if provided
    full_text = f"{incident_context}Question: {question}\nOptions: {options}"

    messages[1]["content"].append({"type": "text", "text": full_text})

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
        print(f"Error getting response from {model}: {e}")
        return None, None


async def get_model_response_no_caption(
    client: AsyncOpenAI,
    model: str,
    question: str,
    options: str,
    image_paths: List[str],
    system_prompt: str,
    global_temperature: float,
    use_reasoning: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """Get a response for no-caption ablation with image pipeline behavior."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": []},
    ]

    for image_path in image_paths:
        try:
            # Resize and encode image
            image_bytes = resize_image(image_path)
            image_data = base64.b64encode(image_bytes).decode("utf-8")

            messages[1]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"},
                }
            )
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None

    # Add text after images
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
        print(f"Error getting response from {model}: {e}")
        return None, None


async def get_model_response_text_only(
    client: AsyncOpenAI,
    model: str,
    question: str,
    options: str,
    system_prompt: str,
    global_temperature: float,
    use_reasoning: bool,
    incident_context: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """Get a response from model using only text (no images)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ""},
    ]

    # Add text content only, including incident context if provided
    full_text = f"{incident_context}Question: {question}\nOptions: {options}"

    messages[1]["content"] = full_text

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
        print(f"Error getting response from {model}: {e}")
        return None, None


async def process_single_question_ablation(
    client: AsyncOpenAI,
    model: str,
    row: pd.Series,
    system_prompt: str,
    run: int,
    global_temperature: float,
    use_reasoning: bool,
    experiment_type: str,
    image_dir: str = "arfbench-images",
) -> Optional[Dict]:
    """Process a single question for ablation experiment."""

    # Prepare incident context if available
    incident_context = ""
    if experiment_type == "with_incident":
        incident_1 = row.get("incident_1", "")
        incident_2 = row.get("incident_2", "")
        incident_context = format_incident_context(incident_1, incident_2)

    # Prepare question — strip caption for the no_caption ablation
    question = row["question"]
    if experiment_type == "no_caption":
        question = remove_timeseries_caption(question)

    shuffled_options = shuffle_options(row.get("options_str", row["options"]))

    # Handle different experiment types
    if experiment_type == "text_only":
        # Text-only evaluation (no images)
        answer, reasoning = await get_model_response_text_only(
            client,
            model,
            question,
            shuffled_options,
            system_prompt,
            global_temperature,
            use_reasoning,
            incident_context,
        )
        num_images = 0
        has_images = False
    elif experiment_type == "no_caption":
        # Dedicated no-caption image pipeline (caption already truncated above).
        image_paths = get_image_paths(
            row["query_group"], image_dir, row.get("task_category")
        )

        # Skip questions where any image doesn't exist
        if image_paths is None:
            print(
                f"Skipping question for {row['query_group']} - "
                f"one or more images not found"
            )
            return None

        answer, reasoning = await get_model_response_no_caption(
            client,
            model,
            question,
            shuffled_options,
            image_paths,
            system_prompt,
            global_temperature,
            use_reasoning,
        )
        num_images = len(image_paths) if image_paths else 0
        has_images = True
    elif experiment_type == "with_incident":
        # Image-based evaluation with incident context.
        image_paths = get_image_paths(
            row["query_group"], image_dir, row.get("task_category")
        )

        # Skip questions where any image doesn't exist
        if image_paths is None:
            print(
                f"Skipping question for {row['query_group']} - "
                f"one or more images not found"
            )
            return None

        answer, reasoning = await get_model_response_with_incident(
            client,
            model,
            question,
            shuffled_options,
            image_paths,
            system_prompt,
            global_temperature,
            use_reasoning,
            incident_context,
        )
        num_images = len(image_paths) if image_paths else 0
        has_images = True
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    is_correct = answer == row["correct_answer"]

    return {
        "run": run + 1,
        "model": model,
        "experiment_type": experiment_type,
        "query_group": row["query_group"],
        "question": row["question"],
        "processed_question": question,
        "options": shuffled_options,
        "correct_answer": row["correct_answer"],
        "task_category": row["task_category"],
        "model_answer": answer,
        "is_correct": is_correct,
        "has_images": has_images,
        "num_images": num_images,
        "reasoning": reasoning,
        "incident_context_used": bool(incident_context.strip()),
    }


async def evaluate_model_ablation(
    client: AsyncOpenAI,
    model: str,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    global_temperature: float,
    use_reasoning: bool,
    experiment_type: str,
    num_runs: int = 1,
    max_concurrent: int = 10,
    image_dir: str = "arfbench-images",
) -> pd.DataFrame:
    """Evaluate a model on the benchmark for ablation experiment."""
    results = []

    for run in range(num_runs):
        print(f"\nRun {run + 1} for {model} ({experiment_type})")

        semaphore = asyncio.Semaphore(max_concurrent)
        completed_count = 0
        total_questions = len(benchmark_df)

        async def process_with_semaphore(row, question_idx):
            nonlocal completed_count
            async with semaphore:
                print(
                    f"  Starting question {question_idx + 1}: " f"{row['query_group']}"
                )

                result = await process_single_question_ablation(
                    client,
                    model,
                    row,
                    system_prompt,
                    run,
                    global_temperature,
                    use_reasoning,
                    experiment_type,
                    image_dir,
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

        # Filter out None results and exceptions
        valid_results = [
            result
            for result in run_results
            if result is not None and not isinstance(result, Exception)
        ]

        # Log any exceptions
        error_count = 0
        for i, result in enumerate(run_results):
            if isinstance(result, Exception):
                error_count += 1
                print(f"  ❌ Error processing question {i+1}: {result}")

        results.extend(valid_results)
        success_rate = len(valid_results) / total_questions * 100
        print(
            f"✅ Run {run + 1} completed: {len(valid_results)} successful "
            f"({success_rate:.1f}%), {error_count} errors"
        )

    return pd.DataFrame(results)


async def main():
    parser = argparse.ArgumentParser(
        description="Ablation experiment for GPT-4o on ARFBench"
    )
    parser.add_argument(
        "--benchmark",
        default="arfbench-qa.csv",
        help="Path to benchmark CSV (should contain incident_1 and incident_2 columns for incident experiment)",
    )
    parser.add_argument(
        "--image-dir",
        default="arfbench-images",
        help="Path to image directory",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=100,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--global-temperature",
        type=float,
        default=0.05,
        help="Global temperature for the model",
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
        help="Number of runs for each experiment",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["with_incident", "no_caption", "text_only"],
        choices=["with_incident", "no_caption", "text_only"],
        help="Which ablation experiments to run",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model to use for evaluation",
    )
    args = parser.parse_args()

    system_prompt = get_test_benchmark_prompt()

    # Initialize AsyncOpenAI client
    client = AsyncOpenAI()

    # Load benchmark
    benchmark_df = load_benchmark(args.benchmark)

    # Check if incident columns exist for incident experiment
    if "with_incident" in args.experiments:
        if (
            "incident_1" not in benchmark_df.columns
            and "incident_2" not in benchmark_df.columns
        ):
            print("Warning: incident_1 and incident_2 columns not found in benchmark.")
            print("The 'with_incident' experiment will run without incident context.")

    # Evaluate each experiment type
    all_results = []
    for experiment_type in args.experiments:
        print(f"\n{'='*60}")
        print(f"Running ablation experiment: {experiment_type}")
        print(f"{'='*60}")

        results_df = await evaluate_model_ablation(
            client,
            args.model,
            benchmark_df,
            system_prompt,
            args.global_temperature,
            args.reasoning,
            experiment_type,
            num_runs=args.num_runs,
            max_concurrent=args.max_concurrent,
            image_dir=args.image_dir,
        )

        # Save individual experiment results
        results_df.to_csv(
            f"{args.model}_ablation_{experiment_type}_results.csv", index=False
        )
        all_results.append(results_df)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save combined raw results
    combined_results.to_csv(f"{args.model}_ablation_combined_results.csv", index=False)

    # Compute and save statistics for each experiment
    for experiment_type in args.experiments:
        exp_results = combined_results[
            combined_results["experiment_type"] == experiment_type
        ]

        # Compute statistics
        stats_df = compute_statistics(exp_results)
        stats_df.to_csv(
            f"{args.model}_ablation_{experiment_type}_statistics.csv", index=False
        )

        # Compute F1 metrics
        f1_df = compute_multiclass_f1_metrics(exp_results)
        f1_df.to_csv(
            f"{args.model}_ablation_{experiment_type}_f1_statistics.csv", index=False
        )

    # Print summary comparison
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT SUMMARY")
    print("=" * 80)

    summary_stats = []
    for experiment_type in args.experiments:
        exp_results = combined_results[
            combined_results["experiment_type"] == experiment_type
        ]
        if not exp_results.empty:
            first_run_results = exp_results[exp_results["run"] == 1]
            overall_accuracy = first_run_results["is_correct"].mean() * 100

            summary_stats.append(
                {
                    "experiment_type": experiment_type,
                    "overall_accuracy": overall_accuracy,
                    "total_questions": len(first_run_results),
                }
            )

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f"{args.model}_ablation_summary.csv", index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
