# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
This script evaluates open source vLLM models on ARFBench, mainly
vision-language models using vLLM.
This assumes you have ARFBench plot data in a directory called arfbench-images.
"""

import argparse

import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from utils.prompt_test_benchmark import get_test_benchmark_prompt
from typing import Optional, List, Tuple
from utils.inference_utils import (
    parse_response,
    get_image_paths,
    ModelConfig,
    shuffle_options,
)
from utils.compute_statistics import compute_statistics
from utils.log_utils import (
    save_results_and_statistics,
    print_evaluation_summary,
    log_progress,
)
from qwen_vl_utils import process_vision_info

# Add more models here if you want to evaluate them.
MODEL_CONFIGS = {
    "qwen3-vl-8b": ModelConfig(
        name="Qwen/Qwen3-VL-8B-Instruct",
        tensor_parallel_size=4,
        max_num_seqs=128,
    ),
    "qwen3-vl-32b": ModelConfig(
        name="Qwen/Qwen3-VL-32B-Instruct",
        tensor_parallel_size=8,
        max_num_seqs=128,
    ),
}


def get_model_prompt(
    model_name: str,
    question: str,
    image_paths: List[str],
    processor: AutoProcessor,
) -> Tuple[str, List[Image.Image]]:
    """Generate the appropriate prompt and image data for each model in vLLM format."""
    system_prompt = get_test_benchmark_prompt()

    placeholders = [{"type": "image", "image": path} for path in image_paths]
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [*placeholders, {"type": "text", "text": question}],
        },
    ]

    images, _ = process_vision_info(messages)

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    return text_prompt, images


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


def load_vllm_model(
    model_name: str,
    model_path: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
):
    """Load vLLM model and processor."""
    print(f"Loading model: {model_name}")

    if model_path:
        print(f"Using custom model path: {model_path}")
        config = ModelConfig(
            name=model_path,
            tensor_parallel_size=tensor_parallel_size or 4,
            max_num_seqs=max_num_seqs or 128,
        )
    else:
        config = MODEL_CONFIGS[model_name]

    print(f"Loading processor for {config.name}")
    processor = AutoProcessor.from_pretrained(config.name)

    llm_kwargs = {
        "model": config.name,
        "tensor_parallel_size": config.tensor_parallel_size,
        "max_num_seqs": config.max_num_seqs,
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "limit_mm_per_prompt": {"image": 3},
        "max_model_len": 30000,
    }

    llm = LLM(**llm_kwargs)
    return llm, processor, config


def prepare_batch_data(
    benchmark_df, model_name, processor, image_dir: str = "arfbench-images"
):
    """Prepare all batch items from benchmark data."""
    batch_data = []
    batch_indices = []

    for i, row in benchmark_df.iterrows():
        image_paths = get_image_paths(
            row["query_group"], image_dir, task_category=row.get("task_category")
        )
        shuffled_options = shuffle_options(row.get("options_str", row["options"]))
        options = shuffled_options.replace('"', "")

        if image_paths is None:
            print(f"Skipping question for {row['query_group']} - " "images not found")
            continue

        question = f"{row['question']}\nAnswer Choices: {options}"
        text_prompt, images = get_model_prompt(
            model_name, question, image_paths, processor
        )

        batch_data.append(
            {
                "text_prompt": text_prompt,
                "images": images,
                "row": row,
                "image_paths": image_paths,
                "question": row["question"],
                "options": shuffled_options,
            }
        )
        batch_indices.append(i)

    return batch_data, batch_indices


def process_vllm_batch(llm, current_batch, sampling_params):
    """Run vLLM generation on a single batch and parse responses."""
    prompts = []
    multi_modal_data = []
    for item in current_batch:
        prompts.append(item["text_prompt"])
        multi_modal_data.append({"image": item["images"]})

    inputs = [
        {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        for prompt, mm_data in zip(prompts, multi_modal_data)
    ]

    outputs = llm.generate(inputs, sampling_params)

    parsed_results = []
    for output, item in zip(outputs, current_batch):
        prediction = output.outputs[0].text.strip()
        answer, reasoning = parse_response(prediction)

        if answer is None:
            print(f"Warning: Could not parse response for {item['question']}")
            print(f"Raw prediction: {prediction}")
            answer = None
            reasoning = None

        parsed_results.append((answer, reasoning, item))

    return parsed_results


def evaluate_vllm_model(
    model_name: str,
    benchmark_df: pd.DataFrame,
    global_temperature: float,
    num_runs: int = 1,
    batch_size: int = 8,
    model_path: Optional[str] = None,
    tensor_parallel_size: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
    image_dir: str = "arfbench-images",
) -> pd.DataFrame:
    """Evaluate a vLLM model on the benchmark with batched inference."""
    llm, processor, config = load_vllm_model(
        model_name,
        model_path,
        tensor_parallel_size,
        max_num_seqs,
    )

    results = []

    for run in range(num_runs):
        correct_count = 0
        tier_counts = {tier: {"correct": 0, "total": 0} for tier in TIER_MAPPINGS}
        print(f"\nRun {run + 1} for {model_name}")
        run_results = []

        batch_data, batch_indices = prepare_batch_data(
            benchmark_df,
            model_name,
            processor,
            image_dir,
        )

        for batch_start in range(0, len(batch_data), batch_size):
            batch_end = min(batch_start + batch_size, len(batch_data))
            current_batch = batch_data[batch_start:batch_end]

            print(
                f"Processing batch {batch_start//batch_size + 1}/{(len(batch_data) + batch_size - 1)//batch_size}"
            )

            sampling_params = SamplingParams(
                temperature=global_temperature,
                max_tokens=2000,
                top_p=0.95 if global_temperature > 0 else 1.0,
            )

            try:
                parsed_results = process_vllm_batch(llm, current_batch, sampling_params)

                for answer, reasoning, item in parsed_results:
                    row = item["row"]
                    is_correct = answer == row["correct_answer"]
                    task_category = row["task_category"]

                    run_results.append(
                        {
                            "run": run + 1,
                            "model": model_name,
                            "query_group": row["query_group"],
                            "question": row["question"],
                            "options": item["options"],
                            "correct_answer": row["correct_answer"],
                            "task_category": task_category,
                            "model_answer": answer,
                            "is_correct": is_correct,
                            "has_images": True,
                            "num_images": len(item["image_paths"]),
                            "reasoning": reasoning,
                        }
                    )

                    if is_correct:
                        correct_count += 1

                    for tier_name, categories in TIER_MAPPINGS.items():
                        if task_category in categories:
                            tier_counts[tier_name]["total"] += 1
                            if is_correct:
                                tier_counts[tier_name]["correct"] += 1

                log_progress(
                    run_results,
                    tier_counts,
                    correct_count,
                    batch_end,
                    len(batch_data),
                )

            except Exception as e:
                print(f"Error processing batch: {e}")
                for item in current_batch:
                    row = item["row"]
                    run_results.append(
                        {
                            "run": run + 1,
                            "model": model_name,
                            "query_group": row["query_group"],
                            "question": row["question"],
                            "options": item["options"],
                            "correct_answer": row["correct_answer"],
                            "task_category": row["task_category"],
                            "model_answer": None,
                            "is_correct": False,
                            "has_images": True,
                            "num_images": len(item["image_paths"]),
                            "reasoning": None,
                        }
                    )

        results.extend(run_results)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vision-language models using vLLM on ARFBench"
    )
    parser.add_argument(
        "--benchmark",
        default="arfbench-qa.csv",
        help="Path to benchmark CSV",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        default=None,
        help="Models to evaluate from predefined configs (mutually exclusive with --model-path)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to custom model directory (mutually exclusive with --models)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name to use for custom model in output files (only used with --model-path)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel size for custom model (only used with --model-path)",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=128,
        help="Max number of sequences for custom model (only used with --model-path)",
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
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for vLLM inference",
    )
    parser.add_argument(
        "--image-dir",
        default="arfbench-images",
        help="Path to image directory",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.model_path and args.models:
        parser.error(
            "--model-path and --models are mutually exclusive. Please use only one."
        )

    if not args.model_path and not args.models:
        # Default to all predefined models if none is specified
        args.models = list(MODEL_CONFIGS.keys())

    # Load benchmark
    benchmark_df = pd.read_csv(args.benchmark)

    # Evaluate each model
    all_results = []

    # Determine which models to evaluate
    if args.model_path:
        # Evaluate custom model
        models_to_eval = [(args.model_name, args.model_path)]
    else:
        # Evaluate predefined models
        models_to_eval = [(model_name, None) for model_name in args.models]

    for model_name, model_path in models_to_eval:
        print(f"\nEvaluating {model_name}...")

        results_df = evaluate_vllm_model(
            model_name,
            benchmark_df,
            args.global_temperature,
            num_runs=args.num_runs,
            batch_size=args.batch_size,
            model_path=model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            max_num_seqs=args.max_num_seqs,
            image_dir=args.image_dir,
        )

        results_df.to_csv(f"{model_name}_results.csv", index=False)
        statistics_df = compute_statistics(results_df)
        statistics_df.to_csv(f"{model_name}_statistics.csv", index=False)
        all_results.append(results_df)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    stats_df, tier_f1_df = save_results_and_statistics(
        combined_results,
        f"{model_name}_evaluation",
    )
    print_evaluation_summary(stats_df, tier_f1_df, combined_results, benchmark_df)


if __name__ == "__main__":
    main()
