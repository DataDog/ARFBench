# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
This script evaluates Google Gemini models on ARFBench.
This assumes you have ARFBench plot data in a directory called arfbench-images.
"""

import pandas as pd
from google import genai
from google.genai import types
import argparse
import asyncio
import os
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from utils.prompt_test_benchmark import get_test_benchmark_prompt
from utils.inference_utils import (
    load_benchmark,
    resize_image,
    get_image_paths,
    shuffle_options,
)
from utils.log_utils import save_results_and_statistics, print_evaluation_summary


class ARFBenchResponse(BaseModel):
    """Response model for ARFBench evaluation."""

    reasoning: str = Field(description="The reasoning behind the answer.")
    answer: str = Field(description="The selected answer option.")


async def get_model_response(
    client: genai.Client,
    model_name: str,
    question: str,
    options: str,
    image_paths: List[str],
    system_prompt: str,
    global_temperature: float,
) -> Tuple[Optional[str], Optional[str]]:
    """Get a single response from a Gemini model with retry logic."""
    # Prepare the prompt
    user_prompt = f"Question: {question}\nOptions: {options}"

    # Combine system prompt and user prompt
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    # Create content list starting with text
    contents = [full_prompt]

    # Add images as inline data using types.Part.from_bytes
    for image_path in image_paths:
        try:
            # Resize and get image bytes
            image_bytes = resize_image(image_path)

            # Determine MIME type from file extension
            if image_path.lower().endswith(".png"):
                mime_type = "image/png"
            elif image_path.lower().endswith((".jpg", ".jpeg")):
                mime_type = "image/jpeg"
            else:
                mime_type = "image/png"  # default to PNG

            # Add image as Part
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            )
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None, None

    # Retry logic with exponential backoff (max 5 retries)
    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Generate response with structured output using Pydantic schema
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=model_name,
                contents=contents,
                config={
                    "temperature": global_temperature,
                    "response_mime_type": "application/json",
                    "response_json_schema": (ARFBenchResponse.model_json_schema()),
                    "max_output_tokens": 2000,
                },
            )

            # Parse the response using Pydantic model
            try:
                result = ARFBenchResponse.model_validate_json(response.text)
                return result.answer, result.reasoning
            except Exception as parse_error:
                print(f"Warning: Could not parse response for {question}")
                print(f"Parse error: {parse_error}")
                print(f"Raw response: {response.text}")
                return None, None

        except Exception as e:
            error_str = str(e)
            is_rate_limit = (
                "429" in error_str
                or "RESOURCE_EXHAUSTED" in error_str
                or "quota" in error_str.lower()
                or "rate limit" in error_str.lower()
                or "API_KEY_INVALID" in error_str
            )

            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff: 2, 4, 8, 16, 32 seconds
                wait_time = 2 ** (attempt + 1)
                print(
                    f"⚠️  Rate limit/API error, retrying in {wait_time}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(wait_time)
                continue
            else:
                print(f"❌ Error getting response from model: {e}")
                return None, None

    return None, None


async def process_single_question(
    client: genai.Client,
    model_name: str,
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

    # Skip questions where any image doesn't exist
    if image_paths is None:
        print(
            f"Skipping question for {row['query_group']} - "
            f"one or more images not found"
        )
        return None

    answer, reasoning = await get_model_response(
        client,
        model_name,
        row["question"],
        shuffled_options,
        image_paths,
        system_prompt,
        global_temperature,
    )

    return {
        "run": run + 1,
        "model": model_name,
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
    client: genai.Client,
    model_name: str,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    run: int,
    global_temperature: float,
    max_concurrent: int = 10,
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
                model_name,
                row,
                system_prompt,
                run,
                global_temperature,
                image_dir,
            )
            # Longer delay to avoid rate limits (4 seconds = 15 RPM max)
            await asyncio.sleep(4.0)

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
    client: genai.Client,
    model_name: str,
    benchmark_df: pd.DataFrame,
    system_prompt: str,
    global_temperature: float,
    num_runs: int = 1,
    max_concurrent: int = 10,
    image_dir: str = "arfbench-images",
) -> pd.DataFrame:
    """Evaluate a model on the benchmark with multiple runs asynchronously."""
    results = []

    for run in range(num_runs):
        print(f"\nRun {run + 1} for {model_name}")
        run_results = await run_single_evaluation_run(
            client,
            model_name,
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
    parser = argparse.ArgumentParser(description="Evaluate Gemini models on ARFBench")
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
        default=5,
        help="Maximum number of concurrent API requests",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to evaluate (e.g., gemini-2.0-flash-exp, gemini-1.5-pro)",
    )
    parser.add_argument(
        "--global-temperature",
        type=float,
        default=1.0,
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
        default="gemini",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Google API key (or set GOOGLE_API_KEY environment variable)",
    )
    args = parser.parse_args()

    # Configure API key
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Google API key must be provided via --api-key argument "
            "or GOOGLE_API_KEY environment variable"
        )

    # Initialize the Gemini client
    client = genai.Client(api_key=api_key)

    system_prompt = get_test_benchmark_prompt()

    # Load benchmark
    benchmark_df = load_benchmark(args.benchmark)

    # Models to evaluate
    if args.model is None:
        models = ["gemini-3-pro-preview"]
    else:
        models = [args.model]

    # Evaluate each model
    all_results = []
    for model_name in models:
        print(f"\nEvaluating {model_name}...")

        results_df = await evaluate_model(
            client,
            model_name,
            benchmark_df,
            system_prompt,
            args.global_temperature,
            num_runs=args.num_runs,
            max_concurrent=args.max_concurrent,
            image_dir=args.image_dir,
        )
        results_df.to_csv(
            f"{model_name}_{args.experiment_name}_results.csv", index=False
        )
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
