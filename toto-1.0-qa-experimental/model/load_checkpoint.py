# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Unified checkpoint loader for VLM models with optional time series integration.

This module provides a clean interface for loading trained models with:
- LoRA adapters
- Time series components (Toto + projectors)
- Quantization support
- Multi-GPU tensor parallelism

Loading entry points:
- Hub or mixin-format dir: use load_checkpoint(checkpoint_dir) where checkpoint_dir is
  a Hub repo id (e.g. "Datadog/Toto-1.0-QA-Experimental") or a local directory saved with
  TotoAnomalyQAModel.save_pretrained() (contains vlm/, ts_modules.pt, config.json).
  This uses TotoAnomalyQAModel.from_pretrained() under the hood (ModelHubMixin).
- Legacy layout (adapter/ + ts_modules.pt): use load_checkpoint(checkpoint_dir, ...)
  with base_model_name, use_lora, etc. as needed.
"""

import json
import os
import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Optional, Any
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel

from model.toto_vlm_components import TotoAnomalyQAModel


def _is_mixin_format(checkpoint_dir: str) -> bool:
    """True if directory or Hub repo is in TotoAnomalyQAModel save_pretrained format."""
    path = Path(checkpoint_dir)
    if path.is_dir():
        config_file = path / "config.json"
        if not config_file.exists():
            return False
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            return (
                config.get("model_type") == "vlm_with_timeseries"
                and (path / "vlm").is_dir()
            )
        except Exception:
            return False
    return "/" in checkpoint_dir and not os.path.exists(checkpoint_dir)


def load_checkpoint(
    checkpoint_dir: str,
    base_model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
    use_lora: bool = True,
    use_timeseries: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    toto_model_name: str = "Datadog/Toto-Open-Base-1.0",
    force_single_gpu: bool = False,
    device_map: Optional[str] = "auto",
    ts_components_path: Optional[str] = None,
) -> Tuple[nn.Module, AutoProcessor]:
    """
    Load a trained VLM model with optional time series integration.

    Args:
        checkpoint_dir: Path to checkpoint directory containing:
            - adapter/ (if using LoRA)
            - ts_modules.pt (if using time series)
        base_model_name: Base VLM model name
        use_lora: Whether to load LoRA adapters
        use_timeseries: Whether to load time series components
        load_in_8bit: Load model in 8-bit quantization
        load_in_4bit: Load model in 4-bit quantization
        toto_model_name: Toto model name for time series
        force_single_gpu: Force model onto single GPU (disables tensor parallelism)
        device_map: Device map strategy ("auto", "balanced", or explicit dict)
                   Only used if force_single_gpu=False
        ts_components_path: Optional path to ts_modules.pt file from a previous run.
                           If provided, loads TS components from this path instead of
                           looking in checkpoint_dir. Can be a file path or directory
                           containing ts_modules.pt or ts_components/ts_modules.pt

    Returns:
        Tuple of (model, processor)
    """
    print(f"[Checkpoint] Loading from {checkpoint_dir}")

    # Hub or mixin-format: use TotoAnomalyQAModel.from_pretrained (ModelHubMixin)
    if _is_mixin_format(checkpoint_dir):
        if use_lora:
            print(
                "[Checkpoint] WARNING: use_lora is ignored for Hub/mixin format (VLM is already merged)"
            )
        model = TotoAnomalyQAModel.from_pretrained(
            checkpoint_dir,
            device_map="auto" if not force_single_gpu else {"": 0},
            torch_dtype=(
                torch.bfloat16 if not (load_in_8bit or load_in_4bit) else torch.float16
            ),
        )
        processor = AutoProcessor.from_pretrained(checkpoint_dir)
        print("[Checkpoint] Loaded via from_pretrained (Hub/mixin format)")
        return model, processor

    print(f"[Checkpoint] Base model: {base_model_name}")
    print(f"[Checkpoint] LoRA: {use_lora}, Time Series: {use_timeseries}")

    checkpoint_path = Path(checkpoint_dir)

    # Configure quantization
    quantization_config = None
    if load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("[Checkpoint] Using 8-bit quantization")
    elif load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        print("[Checkpoint] Using 4-bit quantization")

    if force_single_gpu:
        actual_device_map = {"": 0}  # Put everything on GPU 0
        print("[Checkpoint] Using single GPU mode (device 0)")
    else:
        # Multi-GPU tensor parallelism
        actual_device_map = device_map
        print(f"[Checkpoint] Using device_map: {device_map}")

    # Determine dtype
    if load_in_8bit or load_in_4bit:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    # Load processor
    print("[Checkpoint] Loading processor...")
    # Try to load from checkpoint root first, then adapter, then fall back to base model
    processor = None

    # Option 1: Checkpoint root directory (if model was saved there)
    if (checkpoint_path / "preprocessor_config.json").exists():
        try:
            processor = AutoProcessor.from_pretrained(checkpoint_path)
            print(
                f"[Checkpoint] Loaded processor from checkpoint root: {checkpoint_path}"
            )
        except Exception as e:
            print(f"[Checkpoint] Could not load processor from checkpoint root: {e}")

    # Option 2: Adapter directory (if processor was saved with adapter)
    if processor is None and use_lora:
        adapter_path = checkpoint_path / "adapter"
        if (adapter_path / "preprocessor_config.json").exists():
            try:
                processor = AutoProcessor.from_pretrained(adapter_path)
                print(f"[Checkpoint] Loaded processor from adapter: {adapter_path}")
            except Exception as e:
                print(f"[Checkpoint] Could not load processor from adapter: {e}")

    if processor is None:
        processor = AutoProcessor.from_pretrained(base_model_name)
        print(f"[Checkpoint] Loaded processor from base model: {base_model_name}")

    # Load base VLM
    print("[Checkpoint] Loading base VLM...")
    vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=actual_device_map,
        attn_implementation="sdpa",
    )
    print("[Checkpoint] Base VLM loaded")

    if use_lora:
        adapter_path = checkpoint_path / "adapter"
        if adapter_path.exists():
            print(f"[Checkpoint] Loading LoRA adapter from {adapter_path}")
            vlm_model = PeftModel.from_pretrained(
                vlm_model,
                str(adapter_path),
                is_trainable=False,
            )
            print("[Checkpoint] LoRA adapter loaded")
        else:
            print(
                f"[Checkpoint] WARNING: LoRA requested but adapter not found at {adapter_path}"
            )

    if use_timeseries:
        print("[Checkpoint] Loading time series components...")

        config = {
            "use_timeseries": True,
            "toto_model_name": toto_model_name,
            "freeze_toto": True,
        }

        model = TotoAnomalyQAModel(vlm_model, config)

        if ts_components_path:
            print(
                f"[Checkpoint] Loading TS components from custom path: {ts_components_path}"
            )
            model.load_ts_components(ts_components_path)

            # Move TS components to GPU
            ts_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if hasattr(model, "variate_embedding"):
                model.variate_embedding = model.variate_embedding.to(ts_device)
            if hasattr(model, "ts_projector"):
                model.ts_projector = model.ts_projector.to(ts_device)
            print(f"[Checkpoint] Moved TS components to {ts_device}")
        else:
            # Look for TS components in checkpoint directory
            ts_modules_path = checkpoint_path / "ts_modules.pt"
            if not ts_modules_path.exists():
                # Try alternate location
                ts_modules_path = checkpoint_path / "ts_components" / "ts_modules.pt"

            if ts_modules_path.exists():
                print(f"[Checkpoint] Loading TS modules from {ts_modules_path}")
                ts_state = torch.load(ts_modules_path, map_location="cpu")

                if (
                    hasattr(model, "variate_embedding")
                    and "variate_embedding" in ts_state
                ):
                    model.variate_embedding.load_state_dict(
                        ts_state["variate_embedding"]
                    )
                    print("[Checkpoint] Loaded variate_embedding")

                if hasattr(model, "ts_projector") and "ts_projector" in ts_state:
                    model.ts_projector.load_state_dict(ts_state["ts_projector"])
                    print("[Checkpoint] Loaded ts_projector")

                # move TS components to GPU (they stay on GPU 0 for both single and multi-GPU)
                # In tensor parallel mode, Toto/TS stay on GPU 0 while VLM is distributed
                ts_device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )
                model.variate_embedding = model.variate_embedding.to(ts_device)
                model.ts_projector = model.ts_projector.to(ts_device)
                print(f"[Checkpoint] Moved TS components to {ts_device}")

                print("[Checkpoint] TS components loaded successfully")
            else:
                print(
                    f"[Checkpoint] WARNING: TS modules not found at {ts_modules_path}"
                )
                print("[Checkpoint] Using randomly initialized TS components")

        # Set to eval mode
        model.eval()
        if hasattr(model, "variate_embedding"):
            model.variate_embedding.eval()
        if hasattr(model, "ts_projector"):
            model.ts_projector.eval()
        if hasattr(model, "toto_extractor"):
            model.toto_extractor.eval()

        print("[Checkpoint] Model ready for inference with time series")
        return model, processor

    else:
        # Return just the VLM without TS wrapper
        vlm_model.eval()
        print("[Checkpoint] Model ready for inference (no time series)")
        return vlm_model, processor


def load_checkpoint_for_tensor_parallel(
    checkpoint_dir: str,
    base_model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
    use_lora: bool = True,
    use_timeseries: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    toto_model_name: str = "Datadog/Toto-Open-Base-1.0",
    num_gpus: Optional[int] = None,
    ts_components_path: Optional[str] = None,
) -> Tuple[nn.Module, AutoProcessor]:
    """
    Load checkpoint with explicit tensor parallelism across multiple GPUs.

    This function handles device placement for the VLM + Toto architecture:
    - VLM is split across GPUs using device_map="auto"
    - Toto and TS components stay on GPU 0
    - Embeddings are moved to appropriate devices during forward pass

    Args:
        checkpoint_dir: Path to checkpoint
        base_model_name: Base VLM model name
        use_lora: Load LoRA adapters
        use_timeseries: Load time series components
        load_in_8bit: Load model in 8-bit quantization
        load_in_4bit: Load model in 4-bit quantization
        toto_model_name: Toto model name
        num_gpus: Number of GPUs to use (None = auto-detect all available)

    Returns:
        Tuple of (model, processor)
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    print(f"[TensorParallel] Using {num_gpus} GPUs for inference")

    if num_gpus <= 1:
        print("[TensorParallel] Only 1 GPU available, falling back to single GPU mode")
        return load_checkpoint(
            checkpoint_dir=checkpoint_dir,
            base_model_name=base_model_name,
            use_lora=use_lora,
            use_timeseries=use_timeseries,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            toto_model_name=toto_model_name,
            force_single_gpu=True,
            ts_components_path=ts_components_path,
        )

    return load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        base_model_name=base_model_name,
        use_lora=use_lora,
        use_timeseries=use_timeseries,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        toto_model_name=toto_model_name,
        force_single_gpu=False,
        device_map="auto",
        ts_components_path=ts_components_path,
    )


def load_checkpoint_from_hub(
    hf_model_id: str,
    use_lora: bool = False,
    use_timeseries: bool = True,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    toto_model_name: str = "Datadog/Toto-Open-Base-1.0",
    force_single_gpu: bool = False,
    device_map: Optional[str] = "auto",
    **kwargs: Any,
) -> Tuple[nn.Module, AutoProcessor]:
    """
    Load a model from a Hugging Face Hub repo (merged VLM + ts_modules.pt layout).

    Downloads the repo with snapshot_download, then calls load_checkpoint with
    the local path. Use this for repos uploaded with upload_model_to_hub.py
    (no custom modeling_toto_anomaly_qa).

    Args:
        hf_model_id: Hugging Face repo id (e.g. "Datadog/Toto-1.0-QA-Experimental")
        use_lora: False for merged-Hub repos
        use_timeseries: True to load TS components from ts_modules.pt
        load_in_8bit: 8-bit quantization
        load_in_4bit: 4-bit quantization
        toto_model_name: Toto model name (loaded at runtime by TotoAnomalyQAModel)
        force_single_gpu: Force single GPU
        device_map: Device map when not force_single_gpu
        **kwargs: Passed through to load_checkpoint

    Returns:
        Tuple of (model, processor)
    """
    from huggingface_hub import snapshot_download

    print(f"[Checkpoint] Downloading Hub model: {hf_model_id}")
    local_dir = snapshot_download(
        repo_id=hf_model_id,
        **(kwargs.pop("snapshot_download_kwargs", {})),
    )
    return load_checkpoint(
        checkpoint_dir=local_dir,
        base_model_name=local_dir,
        use_lora=use_lora,
        use_timeseries=use_timeseries,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        toto_model_name=toto_model_name,
        force_single_gpu=force_single_gpu,
        device_map=device_map,
        ts_components_path=None,
    )
