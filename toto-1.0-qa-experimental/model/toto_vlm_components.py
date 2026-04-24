#!/usr/bin/env python3
# Unless explicitly stated otherwise all files in this repository are licensed under the Apache-2.0 License.
#
# This product includes software developed at Datadog (https://www.datadoghq.com/)
# Copyright 2026 Datadog, Inc.
"""
Shared Toto-VLM Integration Components

This module contains all the shared components for integrating Toto time series
embeddings with Vision-Language Models (VLMs). These components are used by both
SFT and GRPO training scripts.

Components:
- TimeSeriesData: Container for processed time series data
- TotoEmbeddingExtractor: Extracts embeddings from Toto backbone
- TotoVariateEmbedding: Aggregates TS embeddings over time dimension
- TimeSeriesProjector: Projects TS embeddings to VLM space
- TotoAnomalyQAModel: Wrapper combining VLM + Toto + projection layers
- TSComponentsCheckpointCallback: Saves TS components during training
- DataCollatorForVLMWithTimeSeries: Custom data collator for SFT training
"""

import os
import warnings
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Type, TypeVar
from dataclasses import dataclass

try:
    from huggingface_hub import ModelHubMixin
    from huggingface_hub import snapshot_download
except ImportError:
    ModelHubMixin = None  # type: ignore[misc, assignment]
    snapshot_download = None

from transformers import (
    AutoProcessor,
    PreTrainedModel,
    TrainerCallback,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

# Suppress gradient checkpointing warnings for frozen layers
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

# Toto imports
try:
    from toto.model.toto import Toto

    TOTO_AVAILABLE = True
except ImportError:
    print("[WARNING] Toto not available. Install with: uv add toto-ts")
    TOTO_AVAILABLE = False

# ============================================================================
# Qwen3-VL Patch for TS Embeddings Support
# ============================================================================


def patch_qwen3vl_for_ts_embeddings():
    """
    Patch Qwen3-VL's get_placeholder_mask to handle TS embeddings prepended to inputs_embeds.

    When TS embeddings are prepended, image token detection needs to skip TS positions.
    This patch checks for a temporary _ts_offset attribute on the model and skips
    those positions when searching for image tokens.
    """
    try:

        original_get_placeholder_mask = Qwen3VLModel.get_placeholder_mask

        def patched_get_placeholder_mask(
            self,
            input_ids,
            inputs_embeds,
            image_features=None,
            video_features=None,
        ):
            """
            Patched version that handles TS embeddings prepended to inputs_embeds.

            When input_ids is None and inputs_embeds is used with TS prepended,
            we need to skip TS positions when detecting image tokens.
            """
            # If input_ids is provided, use original method (no TS offset needed)
            if input_ids is not None:
                return original_get_placeholder_mask(
                    self, input_ids, inputs_embeds, image_features, video_features
                )

            # When input_ids is None, check if TS offset is stored
            ts_offset = getattr(self, "_ts_offset", 0)

            if ts_offset > 0 and inputs_embeds is not None:
                # TS embeddings are prepended, skip them when searching for image tokens
                # Only search in positions after ts_offset
                text_embeds = inputs_embeds[:, ts_offset:, :]

                # Get embedding layer to find image token embedding
                embed_layer = self.get_input_embeddings()

                # Get image token embedding
                image_token_id = self.config.image_token_id
                image_token_embed = embed_layer(
                    torch.tensor(
                        image_token_id,
                        dtype=torch.long,
                        device=embed_layer.weight.device,
                    )
                ).to(inputs_embeds.device)

                # Search for image tokens ONLY in text_embeds (skip TS positions)
                # Compare each position in text_embeds with image_token_embed
                # Shape: text_embeds is (batch, text_seq_len, hidden_dim)
                #        image_token_embed is (hidden_dim,)
                # We need to compare along the last dimension: (batch, text_seq_len, hidden_dim) == (hidden_dim,)
                # Broadcasting: (batch, text_seq_len, hidden_dim) == (1, 1, hidden_dim)
                image_token_embed_expanded = image_token_embed.unsqueeze(0).unsqueeze(
                    0
                )  # (1, 1, hidden_dim)
                image_token_mask_text = (text_embeds == image_token_embed_expanded).all(
                    dim=-1
                )  # (batch, text_seq_len)
                n_image_tokens = image_token_mask_text.sum().item()

                # Expand mask to full sequence length (with TS positions set to False)
                batch_size, text_seq_len, hidden_dim = text_embeds.shape
                full_seq_len = inputs_embeds.shape[1]

                # Create full mask: False for TS positions, then image_token_mask for text positions
                full_image_mask = torch.zeros(
                    batch_size,
                    full_seq_len,
                    dtype=image_token_mask_text.dtype,
                    device=inputs_embeds.device,
                )
                full_image_mask[:, ts_offset:] = image_token_mask_text

                # Expand to match inputs_embeds shape for mask (required by Qwen3-VL)
                full_image_mask_expanded = full_image_mask.unsqueeze(-1).expand_as(
                    inputs_embeds
                )

                # Validate against image_features
                if image_features is not None:
                    if n_image_tokens != image_features.shape[0]:
                        raise ValueError(
                            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
                        )

                # Handle video tokens similarly if needed
                full_video_mask_expanded = None
                if (
                    hasattr(self.config, "video_token_id")
                    and self.config.video_token_id is not None
                ):
                    video_token_id = self.config.video_token_id
                    video_token_embed = embed_layer(
                        torch.tensor(
                            video_token_id,
                            dtype=torch.long,
                            device=embed_layer.weight.device,
                        )
                    ).to(inputs_embeds.device)
                    video_token_embed_expanded = video_token_embed.unsqueeze(
                        0
                    ).unsqueeze(0)
                    video_token_mask_text = (
                        text_embeds == video_token_embed_expanded
                    ).all(dim=-1)
                    n_video_tokens = video_token_mask_text.sum().item()

                    full_video_mask = torch.zeros(
                        batch_size,
                        full_seq_len,
                        dtype=video_token_mask_text.dtype,
                        device=inputs_embeds.device,
                    )
                    full_video_mask[:, ts_offset:] = video_token_mask_text
                    full_video_mask_expanded = full_video_mask.unsqueeze(-1).expand_as(
                        inputs_embeds
                    )

                    if video_features is not None:
                        if n_video_tokens != video_features.shape[0]:
                            raise ValueError(
                                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
                            )

                return full_image_mask_expanded, full_video_mask_expanded

            # No TS offset, use original method
            return original_get_placeholder_mask(
                self, input_ids, inputs_embeds, image_features, video_features
            )

        # Apply the patch
        Qwen3VLModel.get_placeholder_mask = patched_get_placeholder_mask
        print(
            "[Patch] ✓ Applied Qwen3-VL get_placeholder_mask patch for TS embeddings support"
        )

    except ImportError:
        print(
            "[WARNING] Could not import Qwen3VLModel, skipping get_placeholder_mask patch"
        )
    except Exception as e:
        print(f"[WARNING] Failed to patch Qwen3-VL get_placeholder_mask: {e}")


# Apply patch on module import
patch_qwen3vl_for_ts_embeddings()


# ============================================================================
# Time Series Data Loading Utilities
# ============================================================================


@dataclass
class TimeSeriesData:
    """Container for processed time series data."""

    series: torch.Tensor  # (n_channels, n_timesteps)
    padding_mask: torch.Tensor  # (n_channels, n_timesteps)
    id_mask: torch.Tensor  # (n_channels, n_timesteps)
    timestamp_seconds: torch.Tensor  # (n_channels, n_timesteps)
    time_interval_seconds: torch.Tensor  # (n_channels,)
    num_groups: int
    query_group: str
    group_names: List[
        str
    ]  # Names of groups/channels in order (matches series dimension 0)


def format_ts_timestamp_metadata(ts_data: Any) -> Optional[str]:
    """Build Time Series Metadata string (interval, first/last timestamp, duration) from ts_data.

    Returns the formatted string to append to ts_info_parts, or None if timestamp/interval
    information is not available.
    """
    if not hasattr(ts_data, "timestamp_seconds") or not hasattr(
        ts_data, "time_interval_seconds"
    ):
        return None
    if not hasattr(ts_data, "padding_mask") or ts_data.padding_mask is None:
        return None

    first_channel_mask = (
        ts_data.padding_mask[0] if ts_data.padding_mask.shape[0] > 0 else None
    )
    if first_channel_mask is None:
        return None

    valid_indices = torch.where(first_channel_mask)[0]
    if len(valid_indices) == 0:
        return None

    first_idx = valid_indices[0].item()
    last_idx = valid_indices[-1].item()
    first_timestamp = ts_data.timestamp_seconds[0, first_idx].item()
    last_timestamp = ts_data.timestamp_seconds[0, last_idx].item()
    interval_seconds = ts_data.time_interval_seconds[0].item()

    def _format_unix_timestamp(unix_seconds: float) -> str:
        from datetime import datetime

        try:
            dt = datetime.fromtimestamp(unix_seconds)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return f"Unix timestamp: {unix_seconds:.0f}"

    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            return f"{seconds / 60:.1f} minutes"
        elif seconds < 86400:
            return f"{seconds / 3600:.2f} hours"
        else:
            return f"{seconds / 86400:.2f} days"

    interval_text = _format_duration(interval_seconds)
    first_time_text = _format_unix_timestamp(first_timestamp)
    last_time_text = _format_unix_timestamp(last_timestamp)
    duration_seconds = last_timestamp - first_timestamp + interval_seconds

    return (
        f"Time Series Metadata:\n"
        f"- Time interval: {interval_text}\n"
        f"- First timestamp: {first_time_text}\n"
        f"- Last timestamp: {last_time_text}\n"
        f"- Total duration: {_format_duration(duration_seconds)}"
    )


def format_ts_metadata_for_prompt(ts_data: Any) -> Optional[str]:
    """Build full time series metadata string (channel names + timestamp info) to append to a prompt.

    Returns the formatted string to append, or None if ts_data is None or has no metadata to add.
    Used by both the eval script (get_model_prompt) and the DataCollator.
    """
    if ts_data is None:
        return None
    ts_info_parts = []
    if hasattr(ts_data, "group_names") and ts_data.group_names:
        num_groups = getattr(ts_data, "num_groups", len(ts_data.group_names))
        valid_group_names = [name for name in ts_data.group_names[:num_groups] if name]
        if valid_group_names:
            group_names_text = "\n".join(
                f"Channel {idx + 1}: {name}"
                for idx, name in enumerate(valid_group_names)
            )
            ts_info_parts.append(
                f"Time Series Channels (in order):\n{group_names_text}"
            )
    ts_metadata_str = format_ts_timestamp_metadata(ts_data)
    if ts_metadata_str is not None:
        ts_info_parts.append(ts_metadata_str)
    if not ts_info_parts:
        return None
    return "\n\n".join(ts_info_parts)


# ============================================================================
# Toto Embedding Extraction Module
# ============================================================================


class TotoEmbeddingExtractor(nn.Module):
    """
    Extracts embeddings from Toto model before the prediction head.

    Toto's standard API only provides forecasting outputs. This module
    hooks into the backbone to extract intermediate representations.
    """

    def __init__(
        self,
        toto_model_name: str = "Datadog/Toto-Open-Base-1.0",
        freeze: bool = True,
        device: str = "cuda",  # Keep parameter for backward compatibility
        dtype: torch.dtype = torch.bfloat16,  # Allow custom dtype
    ):
        super().__init__()

        if not TOTO_AVAILABLE:
            raise ImportError("Toto not available. Install with: uv add toto-ts")

        # Load Toto model
        print(f"[Toto] Loading model: {toto_model_name}")
        self.toto = Toto.from_pretrained(toto_model_name)

        # Convert to specified dtype and move to CUDA explicitly
        # Frozen Toto model needs to stay on GPU to avoid DeepSpeed CPU offloading issues
        dtype_name = (
            "FP16"
            if dtype == torch.float16
            else "BF16" if dtype == torch.bfloat16 else str(dtype)
        )
        if torch.cuda.is_available():
            self.toto = self.toto.to(device="cuda", dtype=dtype)
            print(f"[Toto] Moved to CUDA and converted to {dtype_name}")
        else:
            self.toto = self.toto.to(dtype=dtype)
            print(f"[Toto] Converted to {dtype_name} (no CUDA available)")

        # Freeze Toto weights if specified
        if freeze:
            print("[Toto] Freezing Toto model weights")
            for param in self.toto.parameters():
                param.requires_grad = False
            self.toto.eval()

        # Don't store device - get it dynamically in forward pass
        self.embed_dim = 768  # Toto embedding dimension

    def forward(self, ts_data: TimeSeriesData) -> torch.Tensor:
        """
        Extract embeddings from Toto model.

        Args:
            ts_data: TimeSeriesData object with preprocessed time series

        Returns:
            Embeddings tensor of shape (batch, n_channels, n_timesteps, embed_dim)
            After the backbone but before the distribution head.
        """
        # Get device and dtype dynamically from model parameters (DDP-safe)
        device = next(self.toto.parameters()).device
        dtype = next(self.toto.parameters()).dtype

        # Move tensors to the correct device AND dtype (bfloat16)
        inputs = ts_data.series.unsqueeze(0).to(
            device=device, dtype=dtype
        )  # Add batch dimension
        padding_mask = ts_data.padding_mask.unsqueeze(0).to(
            device=device
        )  # Keep bool dtype
        id_mask = ts_data.id_mask.unsqueeze(0).to(device=device, dtype=dtype)

        # Forward pass through Toto backbone to get embeddings
        with torch.set_grad_enabled(not self.toto.training):
            try:
                # Extract embeddings using the backbone method
                flattened, loc, scale = self.toto.model.backbone(
                    inputs,
                    padding_mask,
                    id_mask,
                    kv_cache=None,
                    scaling_prefix_length=None,
                )
                # flattened shape: (batch, n_channels, time_steps, 768)
                embeddings = flattened

            except Exception as e:
                print(f"[ERROR] Failed to extract embeddings from Toto: {e}")
                print("[WARNING] Returning zero embeddings as fallback")
                # Return zero embeddings as fallback
                batch_size = 1
                n_channels = ts_data.num_groups
                # Approximate number of time steps after patching
                n_timesteps = ts_data.series.shape[1]
                embeddings = torch.zeros(
                    batch_size,
                    n_channels,
                    n_timesteps,
                    self.embed_dim,
                    device=device,
                    dtype=dtype,
                )

        return embeddings


# ============================================================================
# Variate Embedding Module (based on Toto paper approach)
# ============================================================================


class TotoVariateEmbedding(nn.Module):
    """
    Creates variate embeddings from Toto backbone outputs.

    This follows the approach from the Toto paper:
    1. Extract embeddings from backbone (batch, variates, timesteps, 768)
    2. Average over timesteps dimension
    3. Apply RMSNorm
    4. Project through MLP
    """

    def __init__(
        self,
        embed_dim: int = 768,
        mlp_hidden_dim: int = 2048,
        out_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # RMSNorm for normalization (as used in Toto)
        try:
            from toto.model.util import RMSNorm

            self.norm = RMSNorm(embed_dim)
        except ImportError:
            print("[WARNING] Could not import RMSNorm from toto, using LayerNorm")
            self.norm = nn.LayerNorm(embed_dim)

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, out_dim),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate embeddings per variate by averaging over time.

        Args:
            embeddings: (batch, num_variates, time_steps, embed_dim)
                        Output from Toto backbone

        Returns:
            Variate embeddings: (batch, num_variates, out_dim)
        """
        # Check for NaNs in input embeddings and replace with 0
        if torch.isnan(embeddings).any():
            print("[WARNING] NaN values detected in Toto embeddings, replacing with 0")
            embeddings = torch.nan_to_num(embeddings, nan=0.0)

        # Average over time dimension (dim=-2)
        # (batch, variates, timesteps, embed_dim) -> (batch, variates, embed_dim)
        average_embeddings = embeddings.mean(dim=-2)

        # Check for NaNs after averaging
        if torch.isnan(average_embeddings).any():
            print("[WARNING] NaN values after averaging, replacing with 0")
            average_embeddings = torch.nan_to_num(average_embeddings, nan=0.0)

        # Apply normalization
        normalized_embeddings = self.norm(average_embeddings)

        # Check for NaNs after normalization
        if torch.isnan(normalized_embeddings).any():
            print("[WARNING] NaN values after normalization, replacing with 0")
            normalized_embeddings = torch.nan_to_num(normalized_embeddings, nan=0.0)

        # Project through MLP
        output_embeddings = self.mlp(normalized_embeddings)

        # Final check for NaNs
        if torch.isnan(output_embeddings).any():
            print("[WARNING] NaN values in output embeddings, replacing with 0")
            output_embeddings = torch.nan_to_num(output_embeddings, nan=0.0)

        return output_embeddings


# ============================================================================
# Time Series to VLM Projector
# ============================================================================


class TimeSeriesProjector(nn.Module):
    """
    Projects time series embeddings to VLM embedding space.

    Maps from Toto's 768-dimensional embeddings to the VLM's hidden dimension
    (typically 4096+ for large VLMs).
    """

    def __init__(
        self,
        ts_dim: int = 768,
        vlm_dim: int = 4096,
        hidden_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(ts_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vlm_dim),
            nn.LayerNorm(vlm_dim),
        )

    def forward(self, ts_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Project time series embeddings.

        Args:
            ts_embeddings: (batch, num_groups, ts_dim)

        Returns:
            Projected embeddings: (batch, num_groups, vlm_dim)
        """
        # Check for NaNs in input
        if torch.isnan(ts_embeddings).any():
            print(
                "[WARNING] NaN values in input to TimeSeriesProjector, replacing with 0"
            )
            ts_embeddings = torch.nan_to_num(ts_embeddings, nan=0.0)

        projected = self.projection(ts_embeddings)

        # Check for NaNs in output
        if torch.isnan(projected).any():
            print(
                "[WARNING] NaN values in TimeSeriesProjector output, replacing with 0"
            )
            projected = torch.nan_to_num(projected, nan=0.0)

        return projected


# ============================================================================
# Integrated Model Wrapper
# ============================================================================

_T = TypeVar("_T", bound="TotoAnomalyQAModel")

_TotoAnomalyQAModelBases = (
    (nn.Module, ModelHubMixin) if ModelHubMixin is not None else (nn.Module,)
)  # type: tuple


class TotoAnomalyQAModel(*_TotoAnomalyQAModelBases):  # type: ignore[misc]
    """
    Wrapper that combines VLM + Toto + Projection layers (Toto-1.0-QA-Experimental).

    This model:
    1. Extracts TS embeddings from Toto
    2. Aggregates per group
    3. Projects to VLM space
    4. Passes through VLM with concatenated embeddings

    When ``huggingface_hub`` is installed, supports Hub upload/download via
    ``save_pretrained`` / ``from_pretrained`` (ModelHubMixin). Saved repos contain
    merged VLM weights in ``vlm/``, ``ts_modules.pt``, and config; Toto is loaded
    from Hub at load time (e.g. ``Datadog/Toto-Open-Base-1.0``).
    """

    def __init__(
        self,
        vlm_model: PreTrainedModel,
        config: Dict[str, Any],
    ):
        super().__init__()

        self.vlm = vlm_model
        # Store wrapper config for Hub save_pretrained (toto_model_name, etc.)
        self._wrapper_config = dict(config)
        # Expose VLM config for compatibility with Trainer
        self.config = vlm_model.config
        # Expose warnings_issued for compatibility with GRPOTrainer
        self.warnings_issued = vlm_model.warnings_issued
        # Expose name_or_path for model identification
        self.name_or_path = getattr(
            vlm_model, "name_or_path", config.get("model_name", "unknown")
        )
        # Expose is_gradient_checkpointing for unwrap_model_for_generation
        self.is_gradient_checkpointing = getattr(
            vlm_model, "is_gradient_checkpointing", False
        )

        # Initialize Toto components if available
        if TOTO_AVAILABLE and config.get("use_timeseries", False):
            self.use_timeseries = True

            # Toto embedding extractor
            # Use the VLM's dtype for consistency
            toto_dtype = next(vlm_model.parameters()).dtype
            self.toto_extractor = TotoEmbeddingExtractor(
                toto_model_name=config.get(
                    "toto_model_name", "Datadog/Toto-Open-Base-1.0"
                ),
                freeze=config.get("freeze_toto", True),
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=toto_dtype,
            )

            # Variate embedding module (averages over time)
            self.variate_embedding = TotoVariateEmbedding(
                embed_dim=768,
                mlp_hidden_dim=config.get("variate_mlp_hidden_dim", 768 * 4),
                out_dim=768,  # Keep at 768, will project to VLM dim later
                dropout=config.get("variate_dropout", 0.1),
            )

            # Projector to VLM space
            vlm_hidden_dim = self._get_vlm_hidden_dim(vlm_model)
            self.ts_projector = TimeSeriesProjector(
                ts_dim=768,
                vlm_dim=vlm_hidden_dim,
                hidden_dim=config.get("projector_hidden_dim", 2048),
            )

            # Convert TS components to VLM dtype and move to GPU 0
            self.vlm_dtype = next(vlm_model.parameters()).dtype
            ts_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.variate_embedding = self.variate_embedding.to(
                device=ts_device, dtype=self.vlm_dtype
            )
            self.ts_projector = self.ts_projector.to(
                device=ts_device, dtype=self.vlm_dtype
            )
            print(
                f"[Model] TS components converted to dtype: {self.vlm_dtype} and moved to {ts_device}"
            )

            # Load TS components from checkpoint if specified
            ts_checkpoint_path = config.get("ts_checkpoint_path", None)
            if ts_checkpoint_path:
                self.load_ts_components(ts_checkpoint_path)

            print(f"[Model] Initialized VLM with Time Series integration")
            print(f"[Model] VLM hidden dim: {vlm_hidden_dim}")
        else:
            self.use_timeseries = False
            print("[Model] Initialized VLM without Time Series integration")

    def load_ts_components(self, checkpoint_path: str):
        """
        Load time series components (variate_embedding, ts_projector) from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file (ts_modules.pt) or directory containing ts_modules.pt
        """
        # Handle both file paths and directory paths
        if os.path.isdir(checkpoint_path):
            # If it's a directory, look for ts_modules.pt inside
            checkpoint_file = os.path.join(checkpoint_path, "ts_modules.pt")
            # Also check for ts_components subdirectory
            if not os.path.exists(checkpoint_file):
                checkpoint_file = os.path.join(
                    checkpoint_path, "ts_components", "ts_modules.pt"
                )
        else:
            checkpoint_file = checkpoint_path

        if not os.path.exists(checkpoint_file):
            print(
                f"[WARNING] TS checkpoint not found at {checkpoint_file}, using randomly initialized TS components"
            )
            return

        try:
            print(f"[Model] Loading TS components from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location="cpu")

            # Load variate_embedding if present
            if "variate_embedding" in checkpoint:
                self.variate_embedding.load_state_dict(
                    checkpoint["variate_embedding"], strict=False
                )
                print("[Model] ✅ Loaded variate_embedding state dict")
            else:
                print("[WARNING] variate_embedding not found in checkpoint")

            # Load ts_projector if present
            if "ts_projector" in checkpoint:
                self.ts_projector.load_state_dict(
                    checkpoint["ts_projector"], strict=False
                )
                print("[Model] ✅ Loaded ts_projector state dict")
            else:
                print("[WARNING] ts_projector not found in checkpoint")

            # Load Toto model weights if present (for unfrozen Toto models)
            if "toto_model" in checkpoint and hasattr(self, "toto_extractor"):
                try:
                    self.toto_extractor.toto.load_state_dict(
                        checkpoint["toto_model"], strict=False
                    )
                    print("[Model] ✅ Loaded Toto model weights from checkpoint")
                except Exception as e:
                    print(f"[WARNING] Failed to load Toto model weights: {e}")

            # Move to correct device and dtype after loading
            ts_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.variate_embedding = self.variate_embedding.to(
                device=ts_device, dtype=self.vlm_dtype
            )
            self.ts_projector = self.ts_projector.to(
                device=ts_device, dtype=self.vlm_dtype
            )
            print(
                f"[Model] TS components moved to {ts_device} with dtype {self.vlm_dtype}"
            )

        except Exception as e:
            print(f"[ERROR] Failed to load TS components from {checkpoint_file}: {e}")
            print("[WARNING] Continuing with randomly initialized TS components")
            import traceback

            traceback.print_exc()

    # -------------------------------------------------------------------------
    # ModelHubMixin: save/load for Hub (merged VLM + ts_modules; Toto at load time)
    # -------------------------------------------------------------------------

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save merged VLM, ts_modules.pt, and config. Toto is not stored."""
        if ModelHubMixin is None:
            raise ImportError(
                "huggingface_hub is required for save_pretrained. pip install huggingface_hub"
            )
        import json

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Config for from_pretrained (and for load_checkpoint to detect format)
        w = getattr(self, "_wrapper_config", None) or {}
        config = {
            "model_type": "vlm_with_timeseries",
            "use_timeseries": getattr(self, "use_timeseries", True),
            "toto_model_name": w.get("toto_model_name", "Datadog/Toto-Open-Base-1.0"),
            "freeze_toto": w.get("freeze_toto", True),
            "variate_mlp_hidden_dim": w.get("variate_mlp_hidden_dim", 3072),
            "projector_hidden_dim": w.get("projector_hidden_dim", 2048),
            "variate_dropout": w.get("variate_dropout", 0.1),
        }
        with open(save_directory / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # VLM (merged) into vlm/
        vlm_dir = save_directory / "vlm"
        vlm_dir.mkdir(parents=True, exist_ok=True)
        self.vlm.save_pretrained(str(vlm_dir), safe_serialization=True)

        # TS heads only (no Toto)
        ts_dict = {
            "variate_embedding": self.variate_embedding.state_dict(),
            "ts_projector": self.ts_projector.state_dict(),
        }
        torch.save(ts_dict, save_directory / "ts_modules.pt")

    @classmethod
    def _from_pretrained(
        cls: Type[_T],
        *,
        model_id: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        proxies: Optional[Dict] = None,
        resume_download: Optional[bool] = None,
        local_files_only: bool = False,
        token: Optional[str] = None,
        **model_kwargs: Any,
    ) -> _T:
        """Load from Hub or local dir (mixin format: vlm/ + ts_modules.pt + config)."""
        if ModelHubMixin is None or snapshot_download is None:
            raise ImportError(
                "huggingface_hub is required for from_pretrained. pip install huggingface_hub"
            )
        from transformers import Qwen3VLForConditionalGeneration
        import json

        # Resolve to local path
        if os.path.isdir(model_id):
            local_dir = Path(model_id)
        else:
            local_dir = Path(
                snapshot_download(
                    repo_id=model_id,
                    revision=revision or "main",
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                )
            )

        config_path = local_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Expected config.json in {local_dir}. Use load_checkpoint() for legacy adapter/ + ts_modules.pt layout."
            )
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if config.get("model_type") != "vlm_with_timeseries":
            raise ValueError(
                f"config.model_type must be 'vlm_with_timeseries', got {config.get('model_type')}. Use load_checkpoint() for legacy layout."
            )

        vlm_dir = local_dir / "vlm"
        if not vlm_dir.is_dir():
            raise FileNotFoundError(f"Expected vlm/ directory in {local_dir}")

        device_map = model_kwargs.pop("device_map", "auto")
        torch_dtype = model_kwargs.pop("torch_dtype", torch.bfloat16)
        attn_implementation = model_kwargs.pop("attn_implementation", "sdpa")

        vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(vlm_dir),
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

        # Build wrapper config (same shape as load_checkpoint)
        wrapper_config = {
            "use_timeseries": config.get("use_timeseries", True),
            "toto_model_name": config.get(
                "toto_model_name", "Datadog/Toto-Open-Base-1.0"
            ),
            "freeze_toto": config.get("freeze_toto", True),
            "variate_mlp_hidden_dim": config.get("variate_mlp_hidden_dim", 3072),
            "projector_hidden_dim": config.get("projector_hidden_dim", 2048),
            "variate_dropout": config.get("variate_dropout", 0.1),
            "ts_checkpoint_path": str(
                local_dir
            ),  # so load_ts_components finds ts_modules.pt
        }

        model = cls(vlm_model, wrapper_config)
        model.eval()
        if getattr(model, "variate_embedding", None) is not None:
            model.variate_embedding.eval()
        if getattr(model, "ts_projector", None) is not None:
            model.ts_projector.eval()
        if getattr(model, "toto_extractor", None) is not None:
            model.toto_extractor.eval()
        return model

    def _get_vlm_hidden_dim(self, model: PreTrainedModel) -> int:
        """
        Extract hidden dimension from VLM config.

        For Qwen3-VL, this is typically in model.config.hidden_size
        """
        # Try multiple possible attributes
        if hasattr(model.config, "hidden_size"):
            hidden_dim = model.config.hidden_size
            print(f"[Model] Found hidden_size in config: {hidden_dim}")
            return hidden_dim
        elif hasattr(model.config, "d_model"):
            hidden_dim = model.config.d_model
            print(f"[Model] Found d_model in config: {hidden_dim}")
            return hidden_dim
        else:
            # Try to get from the actual embedding layer
            try:
                embed_layer = model.get_input_embeddings()
                hidden_dim = embed_layer.embedding_dim
                print(f"[Model] Extracted hidden_dim from embeddings: {hidden_dim}")
                return hidden_dim
            except Exception as e:
                print(f"[ERROR] Could not extract embedding dim: {e}")
                # Default fallback
                print("[WARNING] Could not determine VLM hidden dim, using 4096")
                return 4096

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for the VLM.
        """
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            if gradient_checkpointing_kwargs is not None:
                self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                self.vlm.gradient_checkpointing_enable()
            # Update our cached attribute
            self.is_gradient_checkpointing = getattr(
                self.vlm, "is_gradient_checkpointing", True
            )
        else:
            print("[WARNING] VLM does not support gradient checkpointing")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the VLM."""
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
            # Update our cached attribute
            self.is_gradient_checkpointing = getattr(
                self.vlm, "is_gradient_checkpointing", False
            )

    def enable_input_require_grads(self):
        """
        Enable gradients for inputs.

        This is needed when using gradient checkpointing with embeddings,
        as the input embeddings need to have gradients enabled.
        """
        if hasattr(self.vlm, "enable_input_require_grads"):
            self.vlm.enable_input_require_grads()
        else:
            # Fallback: manually enable for embedding layer
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            if hasattr(self.vlm, "get_input_embeddings"):
                self.vlm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

    def add_model_tags(self, tags):
        """
        Add model tags for tracking training method.

        Delegates to the underlying VLM model.
        """
        if hasattr(self.vlm, "add_model_tags"):
            self.vlm.add_model_tags(tags)
        else:
            pass

    def get_input_embeddings(self):
        """
        Get the input embeddings layer from the VLM.

        Required for some trainer operations.
        """
        return self.vlm.get_input_embeddings()

    def _compute_ts_projected(
        self,
        ts_data: Optional[List[TimeSeriesData]],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        """
        Compute projected TS embeddings without materializing a large (B, variates, timesteps, 768) tensor.
        Processes each sample through Toto -> variate -> projector, then concats only the small outputs.
        Returns (B, n_variates, vlm_dim) or None if no ts_data.
        """
        if not self.use_timeseries or not ts_data or len(ts_data) == 0:
            return None
        ts_device = next(self.variate_embedding.parameters()).device
        vlm_dtype = next(self.vlm.parameters()).dtype
        parts = []
        for idx in range(batch_size):
            if idx < len(ts_data) and ts_data[idx] is not None:
                ts_emb = self.toto_extractor(
                    ts_data[idx]
                )  # (1, variates, timesteps, 768)
                ts_emb = ts_emb.to(device=ts_device, dtype=vlm_dtype)
            else:
                ts_emb = torch.zeros(
                    1,
                    10,
                    512,
                    self.toto_extractor.embed_dim,
                    device=ts_device,
                    dtype=vlm_dtype,
                )
            var_emb = self.variate_embedding(ts_emb)  # (1, variates, 768)
            proj = self.ts_projector(var_emb)  # (1, variates, vlm_dim)
            parts.append(proj)
        return torch.cat(parts, dim=0)

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        ts_data: Optional[List[TimeSeriesData]] = None,
        **kwargs,
    ):
        """
        Generate text with optional time series integration.

        For inference with time series, we pre-compute the TS embeddings once
        and prepend them to the input embeddings, then use standard generation.

        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask for text
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions for Qwen3-VL
            ts_data: Time series data (if use_timeseries=True)
            **kwargs: Additional generation arguments

        Returns:
            Generated token IDs
        """
        # If time series is enabled and data is provided
        if self.use_timeseries and ts_data is not None and len(ts_data) > 0:
            batch_size = input_ids.size(0) if input_ids is not None else 1
            # Process one sample at a time through Toto -> variate -> projector to avoid OOM
            # (never materialize large (B, variates, timesteps, 768) tensor)
            ts_projected = self._compute_ts_projected(ts_data, batch_size)
            if ts_projected is None:
                return self.vlm.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kwargs,
                )

            # Store TS offset for get_placeholder_mask patch to use
            ts_offset = ts_projected.size(1)  # Number of TS embedding positions
            self.vlm.model._ts_offset = ts_offset  # Store on the VLM's model submodule

            # Get token embeddings
            token_embed = self.vlm.get_input_embeddings()
            input_embeds = token_embed(input_ids)  # (B, L, hidden_size)
            # Match device for concat (e.g. when device_map="auto")
            ts_projected = ts_projected.to(input_embeds.device)

            # Concatenate TS embeddings before text
            combined_embeds = torch.cat(
                [ts_projected, input_embeds], dim=1
            )  # (B, V+L, hidden_size)

            # Build matching attention mask for TS tokens
            ts_mask = torch.ones(
                ts_projected.size(0),  # batch_size
                ts_projected.size(1),  # num_variates
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            combined_mask = torch.cat([ts_mask, attention_mask], dim=1)  # (B, V+L)

            # Generate using inputs_embeds
            generated_ids = self.vlm.generate(
                input_ids=None,  # Explicitly None to use inputs_embeds
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )

            # Clean up temporary TS offset attribute
            if hasattr(self.vlm.model, "_ts_offset"):
                delattr(self.vlm.model, "_ts_offset")

            outputs = torch.cat([input_ids, generated_ids], dim=1)

            return outputs
        else:
            # Standard VLM generation without time series
            return self.vlm.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ts_data: Optional[List[TimeSeriesData]] = None,
        **kwargs,
    ):
        """
        Forward pass with optional time series integration.

        During training, this concatenates TS embeddings with text embeddings
        before passing through the VLM.

        Args:
            input_ids: Text token IDs
            attention_mask: Attention mask
            pixel_values: Image pixel values
            image_grid_thw: Image grid dimensions
            labels: Target labels for loss computation
            ts_data: List of TimeSeriesData objects (one per batch example)
            **kwargs: Additional model arguments

        Returns:
            Model outputs (loss, logits, etc.)
        """
        # If time series is enabled and data is provided
        if self.use_timeseries and ts_data is not None and len(ts_data) > 0:
            batch_size = input_ids.size(0)
            # Process one sample at a time to avoid OOM (no large (B, variates, timesteps, 768) tensor)
            ts_projected = self._compute_ts_projected(ts_data, batch_size)
            if ts_projected is None:
                return self.vlm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    labels=labels,
                    **kwargs,
                )

            # Check for NaNs in projected embeddings
            if torch.isnan(ts_projected).any():
                print(
                    "[WARNING] NaN values in projected TS embeddings, replacing with 0"
                )
                ts_projected = torch.nan_to_num(ts_projected, nan=0.0)

            # Store TS offset for get_placeholder_mask patch to use
            ts_offset = ts_projected.size(1)  # Number of TS embedding positions
            self.vlm.model._ts_offset = ts_offset  # Store on the VLM's model submodule

            # Get text embeddings
            token_embed = self.vlm.get_input_embeddings()
            input_embeds = token_embed(input_ids)

            # Check for NaNs in text embeddings
            if torch.isnan(input_embeds).any():
                print("[WARNING] NaN values in input text embeddings, replacing with 0")
                input_embeds = torch.nan_to_num(input_embeds, nan=0.0)

            # Concatenate TS before text
            combined_embeds = torch.cat([ts_projected, input_embeds], dim=1)

            # Final check for NaNs in combined embeddings
            if torch.isnan(combined_embeds).any():
                print("[WARNING] NaN values in combined embeddings, replacing with 0")
                combined_embeds = torch.nan_to_num(combined_embeds, nan=0.0)

            # Build matching attention mask
            ts_mask = torch.ones(
                ts_projected.size(0),
                ts_projected.size(1),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            combined_mask = torch.cat([ts_mask, attention_mask], dim=1)

            # Adjust labels if provided (shift for TS tokens)
            if labels is not None:
                # Prepend -100 for TS tokens (don't compute loss on them)
                ts_labels = torch.full(
                    (labels.size(0), ts_projected.size(1)),
                    -100,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                combined_labels = torch.cat([ts_labels, labels], dim=1)
            else:
                combined_labels = None

            # Forward through VLM with combined embeddings
            # CRITICAL: Set input_ids=None to force use of inputs_embeds
            # This prevents Qwen3-VL from counting image tokens from input_ids
            # The get_placeholder_mask patch will use _ts_offset to skip TS positions
            outputs = self.vlm(
                input_ids=None,  # Explicitly None to use inputs_embeds
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=combined_labels,
                **kwargs,
            )

            # Clean up temporary TS offset attribute
            if hasattr(self.vlm.model, "_ts_offset"):
                delattr(self.vlm.model, "_ts_offset")

            return outputs
        else:
            # Standard VLM forward pass
            return self.vlm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                **kwargs,
            )


# ============================================================================
# Training Callback for Saving TS Components
# ============================================================================


class TSComponentsCheckpointCallback(TrainerCallback):
    """
    Custom callback to save time series components (variate_embedding, ts_projector)
    and LoRA adapter whenever a checkpoint is saved during training.

    This ensures that TS components and LoRA adapters are saved alongside model checkpoints,
    not just at the end of training.
    """

    def __init__(self, use_timeseries: bool = False, use_lora: bool = False):
        self.use_timeseries = use_timeseries
        self.use_lora = use_lora

    def on_save(self, args, state, control, model=None, **kwargs):
        """Called when a checkpoint is saved."""
        if model is None:
            return

        # Get the checkpoint directory (most recent one)
        checkpoint_folder = f"checkpoint-{state.global_step}"
        output_dir = os.path.join(args.output_dir, checkpoint_folder)

        if not os.path.exists(output_dir):
            # Fallback to output_dir if checkpoint folder doesn't exist
            output_dir = args.output_dir

        # Save TS components if using time series
        if (
            self.use_timeseries
            and hasattr(model, "variate_embedding")
            and hasattr(model, "ts_projector")
        ):
            ts_components_path = os.path.join(output_dir, "ts_modules.pt")

            try:
                # Prepare TS components dict
                ts_components_dict = {
                    "variate_embedding": model.variate_embedding.state_dict(),
                    "ts_projector": model.ts_projector.state_dict(),
                }

                # Save Toto weights if Toto is unfrozen (trainable)
                # Check if freeze_toto is False by checking if Toto parameters require grad
                if hasattr(model, "toto_extractor"):
                    toto_model = model.toto_extractor.toto
                    # Check if any Toto parameters require gradients (i.e., unfrozen)
                    toto_trainable = any(
                        p.requires_grad for p in toto_model.parameters()
                    )
                    if toto_trainable:
                        ts_components_dict["toto_model"] = toto_model.state_dict()
                        print("[Callback] Saving Toto model weights (unfrozen)")

                torch.save(
                    ts_components_dict,
                    ts_components_path,
                )
                print(f"[Callback] Saved TS components to {ts_components_path}")
            except Exception as e:
                print(f"[Callback] WARNING: Failed to save TS components: {e}")

        # Save LoRA adapter if using LoRA
        if self.use_lora:
            try:
                # For TotoAnomalyQAModel wrapper, the LoRA is on model.vlm
                # For standard training, the model is directly the VLM with LoRA
                if hasattr(model, "vlm"):
                    lora_model = model.vlm
                else:
                    lora_model = model

                # Check if this is actually a PEFT model
                if hasattr(lora_model, "save_pretrained"):
                    adapter_dir = os.path.join(output_dir, "adapter")
                    os.makedirs(adapter_dir, exist_ok=True)
                    lora_model.save_pretrained(adapter_dir)
                    print(f"[Callback] Saved LoRA adapter to {adapter_dir}")
            except Exception as e:
                print(f"[Callback] WARNING: Failed to save LoRA adapter: {e}")


# ============================================================================
# Custom Data Collator with Time Series Support (for SFT)
# ============================================================================


class DataCollatorForVLMWithTimeSeries:
    """
    Data collator that handles time series data alongside vision-language inputs.

    This collator:
    1. Processes images, text using the VLM processor
    2. Preserves ts_data from the dataset
    3. Creates proper batches that can be passed to TotoAnomalyQAModel
    """

    def __init__(
        self,
        processor: AutoProcessor,
        use_timeseries: bool = False,
    ):
        self.processor = processor
        self.use_timeseries = use_timeseries

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a list of features into a batch.

        Args:
            features: List of dicts from dataset, each containing:
                - prompt: str
                - completion: str
                - images: List[PIL.Image]
                - ts_data: Optional[TimeSeriesData]

        Returns:
            Batch dict with all inputs ready for model
        """
        # Extract and remove ts_data before processing
        ts_data_list = []
        has_ts_data = False

        if self.use_timeseries:
            for feature in features:
                if "ts_data" in feature:
                    ts_data_list.append(feature["ts_data"])
                    if feature["ts_data"] is not None:
                        has_ts_data = True
                else:
                    ts_data_list.append(None)

        try:
            # Process each example independently
            batch_input_ids = []
            batch_attention_mask = []
            batch_pixel_values = []
            batch_image_grid_thw = []
            batch_labels = []

            for i, feature in enumerate(features):
                # Get raw components
                images = feature.get("images", [])  # List of PIL Images
                system_prompt = feature.get("system_prompt", "")
                user_question = feature.get("user_question", "")
                completion = feature.get("completion", "")

                # Add time series channel and timestamp information to user question if available
                if (
                    self.use_timeseries
                    and i < len(ts_data_list)
                    and ts_data_list[i] is not None
                ):
                    ts_metadata_str = format_ts_metadata_for_prompt(ts_data_list[i])
                    if ts_metadata_str is not None:
                        user_question = f"{user_question}\n\n{ts_metadata_str}"

                # Build messages with image placeholders for Qwen-VL
                user_content = []

                # Add images first
                if images:
                    for img in images:
                        user_content.append({"type": "image", "image": img})

                # Then add text
                user_content.append({"type": "text", "text": user_question})

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ]

                # Use process_vision_info to extract just the image data
                from qwen_vl_utils import process_vision_info

                image_inputs, video_inputs = process_vision_info(messages)

                # Apply chat template - this inserts image token placeholders
                prompt_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Process prompt only (to get its length for masking)
                prompt_inputs = self.processor(
                    text=prompt_text,
                    images=image_inputs,
                    return_tensors="pt",
                    padding=False,
                )
                prompt_length = prompt_inputs["input_ids"].shape[1]

                # Combine with completion
                full_text = prompt_text + completion

                # Now process: text has image tokens, we pass actual images
                inputs = self.processor(
                    text=full_text,
                    images=image_inputs,  # Processed images from process_vision_info
                    return_tensors="pt",
                    padding=False,
                )

                # Extract tensors (all have batch dim of 1, so squeeze)
                input_ids = inputs["input_ids"].squeeze(0)
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(inputs["attention_mask"].squeeze(0))

                # Create labels with prompt masked out (completion_only_loss behavior)
                labels = input_ids.clone()
                # Mask out the prompt portion with -100 (ignore_index)
                labels[:prompt_length] = -100
                batch_labels.append(labels)

                # Handle vision inputs if present
                if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                    batch_pixel_values.append(inputs["pixel_values"])
                if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
                    batch_image_grid_thw.append(inputs["image_grid_thw"])

            # Pad text sequences to same length

            batch_input_ids = pad_sequence(
                batch_input_ids,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            )
            batch_attention_mask = pad_sequence(
                batch_attention_mask, batch_first=True, padding_value=0
            )
            batch_labels = pad_sequence(
                batch_labels, batch_first=True, padding_value=-100
            )

            # Build output batch
            batch = {
                "input_ids": batch_input_ids,
                "attention_mask": batch_attention_mask,
                "labels": batch_labels,
            }

            # Add vision tensors if present
            if batch_pixel_values:
                batch["pixel_values"] = torch.cat(batch_pixel_values, dim=0)

            if batch_image_grid_thw:
                batch["image_grid_thw"] = torch.cat(batch_image_grid_thw, dim=0)

            # Add ts_data if we have it
            if has_ts_data and self.use_timeseries:
                batch["ts_data"] = ts_data_list

            return batch

        except Exception as e:
            import traceback

            print(f"[ERROR] Collator failed: {e}")
            print(f"[ERROR] Full traceback:")
            traceback.print_exc()
            print(f"[ERROR] Number of features: {len(features)}")
            print(f"[ERROR] Features keys: {features[0].keys() if features else 'N/A'}")
            raise  # Re-raise to see the actual error
