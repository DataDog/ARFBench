# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
This script evaluates Anthropic models on ARFBench.
This assumes you have ARFBench plot data in a directory called arfbench-images.
"""

import pandas as pd
import anthropic
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
    shuffle_options,
)
from utils.log_utils import save_results_and_statistics, print_evaluation_summary


async def get_model_response(
    client: anthropic.AsyncAnthropic,
    model: str,
    question: str,
    options: str,
    image_paths: List[str],
    system_prompt: str,
    global_temperature: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Get a single response from Claude model asynchronously."""
    # Prepare the message content
    content = []

    for image_path in image_paths:
        try:
            image_bytes = resize_image(image_path)
            image_data = base64.b64encode(image_bytes).decode("utf-8")

            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_data,
                    },
                }
            )
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None

    # Add text after images
    content.append(
        {"type": "text", "text": f"Question: {question}\nOptions: {options}"}
    )

    try:
        response = await client.messages.create(
            model=model,
            system=system_prompt,
            temperature=global_temperature,
            messages=[{"role": "user", "content": content}],
            max_tokens=2000,
        )

        response_text = response.content[0].text.strip()
        answer, reasoning = parse_response(response_text)

        if answer is None:
            print(f"Warning: Could not parse response for {question}")
            print(f"Raw response: {response_text}")
            return None, None

        return answer, reasoning

    except Exception as e:
        print(f"Error getting response from {model}: {e}")
        return None, None


async def process_single_question(
    client: anthropic.AsyncAnthropic,
    model: str,
    row: pd.Series,
    system_prompt: str,
    run: int,
    global_temperature: float,
    image_dir: str = "arfbench-images",
) -> Optional[Dict]:
    """Process a single question asynchronously."""
    image_paths = get_image_paths(
        row["query_group"], image_dir, row.get("task_category")
    )
    shuffled_options = shuffle_options(row.get("options_str", row["options"]))

    if image_paths is None:
        print(
            f"Skipping question for {row['query_group']} - "
            f"one or more images not found"
        )
        return None

    answer, reasoning = await get_model_response(
        client,
        model,
        row["question"],
        shuffled_options,
        image_paths,
        system_prompt,
        global_temperature,
    )

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
        "has_images": True,
        "num_images": len(image_paths) if image_paths else 0,
        "reasoning": reasoning,
    }


async def run_single_evaluation_run(
    client: anthropic.AsyncAnthropic,
    model: str,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    run: int,
    global_temperature: float,
    max_concurrent: int = 5,
    image_dir: str = "arfbench-images",
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
                image_dir,
            )
            await asyncio.sleep(0.2)

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
    client: anthropic.AsyncAnthropic,
    model: str,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    global_temperature: float,
    num_runs: int = 1,
    max_concurrent: int = 5,
    image_dir: str = "arfbench-images",
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
            max_concurrent,
            image_dir,
        )
        results.extend(run_results)

    return pd.DataFrame(results)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate Claude models on ARFBench")
    parser.add_argument(
        "--benchmark",
        default="arfbench-qa.csv",
        help="Path to benchmark CSV",
    )
    parser.add_argument(
        "--image-dir",
        default="arfbench-images",
        help="Path to image directory",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=50,
        help="Maximum number of concurrent API requests (lower for Anthropic)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--global-temperature",
        type=float,
        default=0.05,
        help="Global temperature for all models",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        help="Number of runs for each model",
    )
    parser.add_argument(
        "--experiment-name",
        default="claude",
        help="Name of the experiment",
    )

    args = parser.parse_args()
    system_prompt = get_test_benchmark_prompt()

    # Initialize AsyncAnthropic client
    client = anthropic.AsyncAnthropic()

    # Load benchmark
    benchmark_df = load_benchmark(args.benchmark)

    # Models to evaluate
    if args.model is None:
        models = ["claude-haiku-4-5-20251001"]
    else:
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
            num_runs=args.num_runs,
            max_concurrent=args.max_concurrent,
            image_dir=args.image_dir,
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
