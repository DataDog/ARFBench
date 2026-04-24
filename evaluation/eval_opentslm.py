# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""Evaluate OpenTSLM models on ARFBench."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from utils.compute_statistics import (
    compute_multiclass_f1_metrics,
    compute_statistics,
    compute_tier_weighted_f1,
)
from utils.inference_utils import parse_response, shuffle_options

if TYPE_CHECKING:
    from model.llm.OpenTSLMSP import OpenTSLMSP


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_OPENTSLM_SRC = REPO_ROOT / "third_party" / "OpenTSLM" / "src"
DEFAULT_BENCHMARK_PATH = REPO_ROOT / "data" / "arfbench-qa.csv"
DEFAULT_DATA_DIR = REPO_ROOT / "arfbench-data"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "opentslm_eval_results"

EOS_MARKERS = ("<|end_of_text|>", "</s>", "<|im_end|>", "<|eot_id|>")


def ensure_opentslm_import_path(opentslm_src: Path | None = None) -> Path:
    """Ensure OpenTSLM source and repository root are importable."""
    configured_path = (
        opentslm_src
        if opentslm_src is not None
        else Path(os.environ.get("ARFBENCH_OPENTSLM_SRC", str(DEFAULT_OPENTSLM_SRC)))
    )
    resolved_path = configured_path.expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            "OpenTSLM source not found at "
            f"{resolved_path}. Clone OpenTSLM under third_party/OpenTSLM "
            "and copy ARFBench loader files as described in README.md."
        )

    for path in (resolved_path, REPO_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return resolved_path


def normalize_answer_text(value: Any) -> str:
    """Normalize model and ground-truth answers for consistent matching."""
    text = str(value).strip() if value is not None else ""
    for marker in EOS_MARKERS:
        text = text.replace(marker, "")
    return text.strip()


def is_prediction_correct(prediction: str, gold_answer: str) -> bool:
    """Match OpenTSLM prediction to ARFBench answer using prefix logic."""
    pred_clean = normalize_answer_text(prediction)
    gold_clean = normalize_answer_text(gold_answer)
    if not pred_clean or not gold_clean:
        return False
    return gold_clean.startswith(pred_clean) or pred_clean == gold_clean


def maybe_shuffle_batch_options(
    batch: List[Dict[str, Any]],
    shuffle_option_order: bool,
) -> List[Dict[str, Any]]:
    """Shuffle options for each sample and update post-prompt accordingly."""
    if not shuffle_option_order:
        return batch

    shuffled_batch: List[Dict[str, Any]] = []
    for sample in batch:
        sample_copy = dict(sample)
        original_options = str(sample_copy.get("options_str", ""))
        shuffled_options = shuffle_options(original_options)
        sample_copy["options_str"] = shuffled_options

        post_prompt = sample_copy.get("post_prompt", "")
        if (
            isinstance(post_prompt, str)
            and original_options
            and shuffled_options
            and original_options in post_prompt
        ):
            sample_copy["post_prompt"] = post_prompt.replace(
                original_options, shuffled_options, 1
            )

        shuffled_batch.append(sample_copy)

    return shuffled_batch


def load_pretrained_model(
    repo_id: str,
    device: str = "cuda",
    opentslm_src: Path | None = None,
) -> "OpenTSLMSP":
    """
    Load a pretrained OpenTSLM model from Hugging Face Hub.
    """
    ensure_opentslm_import_path(opentslm_src)

    from huggingface_hub import hf_hub_download
    from model.encoder.TransformerCNNEncoder import TransformerCNNEncoder
    from model.llm.OpenTSLMSP import OpenTSLMSP
    from model.projector.MLPProjector import MLPProjector
    from model_config import ENCODER_OUTPUT_DIM

    print(f"Downloading checkpoint from Hugging Face Hub: {repo_id}")

    checkpoint_names = ["model_checkpoint.pt", "checkpoint.pt", "model.pt"]
    checkpoint_path = None

    for checkpoint_name in checkpoint_names:
        try:
            checkpoint_path = hf_hub_download(
                repo_id=repo_id,
                filename=checkpoint_name,
            )
            print(f"Downloaded checkpoint: {checkpoint_name}")
            break
        except Exception:
            continue

    if checkpoint_path is None:
        attempted = ", ".join(checkpoint_names)
        raise FileNotFoundError(
            f"Could not find checkpoint in {repo_id}. Tried: {attempted}"
        )

    print("Loading checkpoint...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "encoder_state" in ckpt and "pos_embed" in ckpt["encoder_state"]:
        pos_embed_shape = ckpt["encoder_state"]["pos_embed"].shape
        max_patches = pos_embed_shape[1]
        print(f"Detected max_patches: {max_patches}")
    else:
        max_patches = 1024
        print(f"Using default max_patches: {max_patches}")

    llm_mapping = {
        "OpenTSLM/llama-3.2-1b-ecg-sp": "meta-llama/Llama-3.2-1B",
        "OpenTSLM/llama-3.2-1b-tsqa-sp": "meta-llama/Llama-3.2-1B",
        "OpenTSLM/llama-3.2-3b-ecg-sp": "meta-llama/Llama-3.2-3B",
    }

    base_llm_id = llm_mapping.get(repo_id, "meta-llama/Llama-3.2-1B")
    print(f"Loading base LLM: {base_llm_id}")
    model = OpenTSLMSP(llm_id=base_llm_id, device=device)

    print(f"Initializing encoder with max_patches={max_patches}")
    model.encoder = TransformerCNNEncoder(max_patches=max_patches).to(device)
    model.projector = MLPProjector(
        ENCODER_OUTPUT_DIM, model.llm.config.hidden_size, device=device
    ).to(device)

    print("Loading encoder and projector weights...")
    model.encoder.load_state_dict(ckpt["encoder_state"])
    model.projector.load_state_dict(ckpt["projector_state"])

    checkpoint_has_lora = ckpt.get("lora_enabled", False)
    if checkpoint_has_lora:
        print("Enabling LoRA adapters...")
        model.enable_lora()
        model.load_lora_state_from_checkpoint(ckpt, allow_missing=False)
        print("LoRA adapters loaded")

    model.eval()
    epoch = ckpt.get("epoch", "unknown")
    print(f"Model loaded successfully (epoch: {epoch})")
    return model


def evaluate_arfbench(
    model: "OpenTSLMSP",
    batch_size: int = 4,
    benchmark_path: str = str(DEFAULT_BENCHMARK_PATH),
    data_dir: str = str(DEFAULT_DATA_DIR),
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    max_new_tokens: int = 50,
    shuffle_option_order: bool = True,
    opentslm_src: Path | None = None,
) -> Dict[str, Any]:
    """Evaluate the model on the ARFBench dataset."""
    ensure_opentslm_import_path(opentslm_src)

    from time_series_datasets.arfbench.ARFBenchQADataset import (
        ARFBenchQADataset,
    )
    from time_series_datasets.util import (
        extend_time_series_to_match_patch_size_and_aggregate,
    )

    print("\n" + "=" * 80)
    print("ARFBench Evaluation")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    print("\nLoading ARFBench test dataset...")
    test_dataset = ARFBenchQADataset(
        split="test",
        EOS_TOKEN=model.get_eos_token(),
        csv_path=benchmark_path,
        data_dir=data_dir,
    )
    print(f"Test set size: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=(
            lambda batch: extend_time_series_to_match_patch_size_and_aggregate(
                batch, patch_size=4
            )
        ),
    )

    print("\nRunning evaluation...")
    results_data = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = maybe_shuffle_batch_options(batch, shuffle_option_order)

            try:
                outputs = model.generate(
                    batch,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=model.tokenizer.pad_token_id,
                    eos_token_id=model.tokenizer.eos_token_id,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    repetition_penalty=1.0,
                )
            except Exception as exc:
                print(f"Warning: error generating for batch: {exc}")
                outputs = [""] * len(batch)

            for idx, output in enumerate(outputs):
                sample = batch[idx]
                pre_prompt = sample["pre_prompt"]
                post_prompt = sample["post_prompt"]
                gold = sample["answer"]

                if "Answer:" in output:
                    candidate = output.split("Answer:", 1)[-1].strip()
                else:
                    combined_prompt = pre_prompt + post_prompt
                    if combined_prompt in output:
                        candidate = output.replace(combined_prompt, "", 1).strip()
                    else:
                        candidate = output.strip()

                parsed_answer, _ = parse_response(candidate)
                pred = (
                    str(parsed_answer).split("\n")[0].strip()
                    if parsed_answer is not None
                    else candidate.split("\n")[0].strip()
                )

                gold_clean = normalize_answer_text(gold)
                pred_clean = normalize_answer_text(pred)
                is_correct = is_prediction_correct(pred_clean, gold_clean)
                results_data.append(
                    {
                        "run": 1,
                        "model": "opentslm",
                        "query_group": sample.get("query_group", ""),
                        "question": (
                            sample.get("post_prompt", "")
                            .split("Options:", 1)[0]
                            .strip()
                            if isinstance(sample.get("post_prompt", ""), str)
                            else ""
                        ),
                        "options": sample.get("options_str", ""),
                        "correct_answer": gold_clean,
                        "task_category": sample.get("task_category", ""),
                        "difficulty": sample.get("difficulty", ""),
                        "model_answer": pred_clean,
                        "is_correct": is_correct,
                        "generated": pred,
                        "gold": gold_clean,
                        "options_str": sample.get("options_str", ""),
                    }
                )

    results_file = os.path.join(output_dir, "results.jsonl")
    print(f"\nSaving results to: {results_file}")
    with open(results_file, "w", encoding="utf-8") as handle:
        for item in results_data:
            handle.write(json.dumps(item) + "\n")

    print("\nCalculating metrics...")
    (
        metrics,
        results_df,
        stats_df,
        f1_df,
        tier_f1_df,
    ) = calculate_arfbench_metrics(results_data)

    metrics_file = os.path.join(output_dir, "metrics.json")
    print(f"Saving metrics to: {metrics_file}")
    with open(metrics_file, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    results_csv = os.path.join(output_dir, "results.csv")
    results_df.to_csv(results_csv, index=False)
    print(f"Saving tabular results to: {results_csv}")

    stats_file = os.path.join(output_dir, "statistics.csv")
    stats_df.to_csv(stats_file, index=False)
    print(f"Saving statistics to: {stats_file}")

    f1_file = os.path.join(output_dir, "f1.csv")
    f1_df.to_csv(f1_file, index=False)
    print(f"Saving F1 metrics to: {f1_file}")

    tier_f1_file = os.path.join(output_dir, "tier_f1.csv")
    tier_f1_df.to_csv(tier_f1_file, index=False)
    print(f"Saving tier F1 metrics to: {tier_f1_file}")

    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Multiclass F1 (macro): {metrics['multiclass_f1_macro']:.4f}")
    print(f"Tier-weighted F1: {metrics['tier_weighted_f1']:.4f}")
    print("\nPer-tier metrics:")
    for tier_name, tier_f1 in metrics.get("per_tier_f1", {}).items():
        print(f"  {tier_name}: {tier_f1:.4f}")
    print("=" * 80)

    return metrics


def calculate_arfbench_metrics(
    results_data: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate ARFBench metrics including multiclass F1 and tier-weighted F1.
    """
    results_df = pd.DataFrame(results_data)

    accuracy = float(results_df["is_correct"].mean()) if not results_df.empty else 0.0
    stats_df = compute_statistics(results_df)
    f1_df = compute_multiclass_f1_metrics(results_df)
    tier_f1_df = compute_tier_weighted_f1(results_df)

    if not f1_df.empty and "task_category" in f1_df.columns:
        f1_without_overall = f1_df[f1_df["task_category"] != "Overall"]
        multiclass_f1_macro = (
            float(f1_without_overall["macro_f1"].mean())
            if not f1_without_overall.empty
            else float(f1_df["macro_f1"].mean())
        )
    else:
        multiclass_f1_macro = 0.0

    if not tier_f1_df.empty:
        tier_rows = tier_f1_df[tier_f1_df["tier"] != "Overall"]
        if tier_rows.empty:
            tier_rows = tier_f1_df
        tier_weighted_f1 = float(tier_rows["weighted_f1"].mean())
        per_tier_f1 = {
            str(row["tier"]): float(row["weighted_f1"])
            for _, row in tier_rows.iterrows()
        }
    else:
        tier_weighted_f1 = 0.0
        per_tier_f1 = {}

    metrics = {
        "accuracy": accuracy,
        "multiclass_f1_macro": multiclass_f1_macro,
        "tier_weighted_f1": tier_weighted_f1,
        "per_tier_f1": per_tier_f1,
        "num_samples": int(len(results_df)),
        "detailed_statistics": (
            stats_df.to_dict("records") if not stats_df.empty else []
        ),
        "detailed_multiclass_metrics": (
            f1_df.to_dict("records") if not f1_df.empty else []
        ),
        "detailed_tier_metrics": (
            tier_f1_df.to_dict("records") if not tier_f1_df.empty else []
        ),
    }
    return metrics, results_df, stats_df, f1_df, tier_f1_df


def main() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained OpenTSLM model on ARFBench"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="OpenTSLM/llama-3.2-1b-ecg-sp",
        help="Hugging Face repository ID",
    )
    parser.add_argument(
        "--opentslm-src",
        type=str,
        default=str(DEFAULT_OPENTSLM_SRC),
        help="Path to cloned OpenTSLM src directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=str(DEFAULT_BENCHMARK_PATH),
        help="Path to ARFBench CSV file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DEFAULT_DATA_DIR),
        help="Path to ARFBench time-series parquet directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Maximum number of generated tokens",
    )
    parser.add_argument(
        "--shuffle-options",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to shuffle answer options before inference",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help="Random seed for reproducible option shuffling",
    )
    args = parser.parse_args()

    opentslm_src = Path(args.opentslm_src).expanduser()
    resolved_src = ensure_opentslm_import_path(opentslm_src)

    print("=" * 80)
    print("OpenTSLM ARFBench Evaluation")
    print("=" * 80)
    print(f"Model ID: {args.model_id}")
    print(f"OpenTSLM src: {resolved_src}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    shuffle_status = "enabled" if args.shuffle_options else "disabled"
    print(f"Option shuffling: {shuffle_status}")
    if args.shuffle_options:
        random.seed(args.shuffle_seed)
        print(f"Option shuffle seed: {args.shuffle_seed}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")

    model = load_pretrained_model(
        repo_id=args.model_id,
        device=args.device,
        opentslm_src=resolved_src,
    )

    metrics = evaluate_arfbench(
        model=model,
        batch_size=args.batch_size,
        benchmark_path=args.benchmark,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        shuffle_option_order=args.shuffle_options,
        opentslm_src=resolved_src,
    )

    print("\nEvaluation complete")
    print(f"Results saved to: {args.output_dir}")
    return metrics


if __name__ == "__main__":
    main()
