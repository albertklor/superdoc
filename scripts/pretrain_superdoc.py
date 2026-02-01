"""
Pretraining script for LayoutLMv3 with three objectives:
- MLM: Masked Language Modeling with whole word masking
- IRR: Image Region Replacement (predict which 16x16 patches were replaced)
- WPA: Word Patch Alignment (predict if word center overlaps replaced patch)

Supports mixed synthetic/real data training with configurable ratios.
"""

import argparse
import glob
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import torch
import wandb
from datasets import load_dataset
from huggingface_hub import list_repo_files
from torch.utils.data import IterableDataset as TorchIterableDataset
from transformers import AutoProcessor, Trainer, TrainingArguments, TrainerCallback

from superdoc import LayoutLMv3Config, LayoutLMv3ForPreTraining
from superdoc.muon import create_muon_optimizer
from superdoc.synthetic_generator import DocumentGenerator


# =============================================================================
# Mixed Data Source
# =============================================================================

class MixedDocumentDataset(TorchIterableDataset):
    """
    Dataset that mixes synthetic documents (from FineWeb) with real documents (from SafeDocs).

    Supports configurable mixing ratio and on-the-fly synthetic generation.
    Uses memory-mapped Arrow files for real data to share memory across workers.
    """

    def __init__(
        self,
        processor: Any,
        max_2d_position_embeddings: int = 1024,
        synthetic_ratio: float = 0.95,
        real_dataset_name: str = "albertklorer/safedocs",
        synthetic_dataset_name: str = "HuggingFaceFW/fineweb",
        synthetic_dataset_config: str = "sample-10BT",
        max_seq_length: int = 512,
        seed: int = 42,
        safedocs_percentage: float = 100.0,
    ):
        """
        Args:
            processor: LayoutLMv3 processor for tokenization
            max_2d_position_embeddings: Max value for bbox coordinates (from config)
            synthetic_ratio: Ratio of synthetic data (0.0 = all real, 1.0 = all synthetic)
            real_dataset_name: HuggingFace dataset for real documents
            synthetic_dataset_name: HuggingFace dataset for text (to render synthetically)
            synthetic_dataset_config: Config/subset name for synthetic dataset
            max_seq_length: Maximum sequence length for tokenization
            seed: Random seed
            safedocs_percentage: Percentage of SafeDocs dataset to download (1-100)
        """
        self.processor = processor
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.bbox_max = max_2d_position_embeddings - 1  # 0 to 1023 for 1024
        self.synthetic_ratio = synthetic_ratio
        self.real_dataset_name = real_dataset_name
        self.synthetic_dataset_name = synthetic_dataset_name
        self.synthetic_dataset_config = synthetic_dataset_config
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.safedocs_percentage = safedocs_percentage
        self.global_step_offset = 0  # For resumption: adjusts RNG seeding

        # Initialize document generator for synthetic data
        canvas_sizes = [
            (224, 224), (336, 336), (448, 448),
            (200, 400), (250, 500), (300, 600),
            (400, 200), (500, 250), (600, 300),
            (340, 440), (400, 520),
            (500, 350), (600, 400),
        ]

        self.generator = DocumentGenerator(
            output_size=(224, 224),
            font_sizes=list(range(8, 16)),
            canvas_sizes=canvas_sizes,
            background_color=(220, 255),
            text_color=(0, 80),
            horizontal_shift=(-0.1, 0.1),
            num_columns=[1, 1, 1, 2, 2, 3],
            paragraph_spacing=(0.5, 2.0),
            indent_ratio=(0.0, 0.12),
            title_prob=0.3,
            list_prob=0.2,
            bbox_max=self.bbox_max,
        )

        # Load real dataset in __init__ with memory-mapping.
        # Arrow format uses mmap, so workers share physical memory pages.
        # The dataset object is pickle-friendly - workers re-open the same mmap.
        if self.synthetic_ratio < 1.0:
            if self.safedocs_percentage < 100.0:
                # List parquet files and download only a subset
                all_files = list_repo_files(real_dataset_name, repo_type="dataset")
                train_files = sorted([f for f in all_files if f.startswith("data/train/") and f.endswith(".parquet")])
                num_files = max(1, math.ceil(len(train_files) * self.safedocs_percentage / 100.0))
                selected_files = train_files[:num_files]
                print(f"Loading real dataset {real_dataset_name} ({self.safedocs_percentage:.1f}%, {num_files}/{len(train_files)} files)...")
                self._real_dataset = load_dataset(
                    real_dataset_name,
                    data_files={"train": selected_files},
                    split="train",
                    keep_in_memory=False,
                )
            else:
                print(f"Loading real dataset {real_dataset_name} (memory-mapped)...")
                self._real_dataset = load_dataset(
                    real_dataset_name,
                    split="train",
                    keep_in_memory=False,  # Use memory-mapping, not RAM
                )
            self._real_dataset_len = len(self._real_dataset)
            print(f"  Loaded {self._real_dataset_len} examples (shared via mmap)")
        else:
            self._real_dataset = None
            self._real_dataset_len = 0

        # Synthetic dataset is streamed, loaded lazily per worker
        self._synthetic_dataset = None

    def _load_synthetic_dataset(self):
        """Lazy load synthetic text dataset (FineWeb) - streaming."""
        if self._synthetic_dataset is None:
            self._synthetic_dataset = load_dataset(
                self.synthetic_dataset_name,
                name=self.synthetic_dataset_config,
                split="train",
                streaming=True,
            )
        return self._synthetic_dataset

    def _process_real_example(self, example: dict) -> Optional[dict]:
        """Process a real document example with proper bbox normalization."""
        try:
            # Get image dimensions for bbox normalization
            image = example["image"]
            width = example.get("width") or getattr(image, "width", None)
            height = example.get("height") or getattr(image, "height", None)

            if not width or not height:
                return None  # Skip examples without valid dimensions

            # Normalize bboxes from pixel coordinates to 0-(max_2d_position_embeddings-1) range
            bbox_max = self.bbox_max
            raw_bboxes = example["bounding_boxes"]
            normalized_bboxes = []
            for bbox in raw_bboxes:
                x0, y0, x1, y1 = bbox
                # Normalize and clamp to valid range
                normalized_bboxes.append([
                    max(0, min(bbox_max, int(x0 * bbox_max / width))),
                    max(0, min(bbox_max, int(y0 * bbox_max / height))),
                    max(0, min(bbox_max, int(x1 * bbox_max / width))),
                    max(0, min(bbox_max, int(y1 * bbox_max / height))),
                ])

            processed = self.processor(
                image,
                example["words"],
                boxes=normalized_bboxes,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "bbox": processed["bbox"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
            }
        except Exception:
            return None

    def _process_synthetic_example(self, example: dict) -> Optional[dict]:
        """Generate and process a synthetic document from text."""
        try:
            text = example.get("text", "")
            if not text or len(text) < 50:
                return None

            # Generate synthetic document
            doc = self.generator.generate(text)

            if not doc["tokens"]:
                return None

            # Process with LayoutLMv3 processor
            processed = self.processor(
                doc["image"],
                doc["tokens"],
                boxes=doc["bboxes"],
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "bbox": processed["bbox"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
            }
        except Exception:
            return None

    def set_global_step_offset(self, global_step: int):
        """Set global step offset for RNG seeding when resuming."""
        self.global_step_offset = global_step

    def __iter__(self) -> Iterator[dict]:
        """Iterate over mixed synthetic and real examples."""
        worker_info = torch.utils.data.get_worker_info()

        # Adjust seed for worker + global_step_offset for resumption
        # This ensures different randomness after resuming vs starting fresh
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id + self.global_step_offset
        else:
            worker_seed = self.seed + self.global_step_offset

        rng = random.Random(worker_seed)

        # Real dataset was loaded in __init__ (memory-mapped, shared across workers)
        real_dataset = self._real_dataset
        real_dataset_len = self._real_dataset_len

        # Create streaming iterator for synthetic data (per-worker)
        synthetic_iter = iter(self._load_synthetic_dataset()) if self.synthetic_ratio > 0.0 else None

        while True:
            try:
                # Decide synthetic vs real
                use_synthetic = rng.random() < self.synthetic_ratio

                if use_synthetic and synthetic_iter is not None:
                    example = next(synthetic_iter)
                    processed = self._process_synthetic_example(example)
                elif real_dataset is not None:
                    # Random sample from memory-mapped real dataset
                    idx = rng.randint(0, real_dataset_len - 1)
                    example = real_dataset[idx]
                    processed = self._process_real_example(example)
                else:
                    break

                if processed is not None:
                    yield processed

            except StopIteration:
                # Restart synthetic iterator (real dataset doesn't need restart - we sample randomly)
                synthetic_iter = iter(self._load_synthetic_dataset())


# =============================================================================
# Evaluation Dataset
# =============================================================================

class SafeDocsEvalDataset(torch.utils.data.Dataset):
    """
    Fixed-size evaluation dataset from SafeDocs validation split.
    """

    def __init__(
        self,
        processor: Any,
        max_2d_position_embeddings: int = 1024,
        dataset_name: str = "albertklorer/safedocs",
        max_seq_length: int = 512,
        safedocs_percentage: float = 100.0,
    ):
        self.processor = processor
        self.bbox_max = max_2d_position_embeddings - 1
        self.max_seq_length = max_seq_length

        # Always use data_files to avoid downloading entire dataset (including train split)
        all_files = list_repo_files(dataset_name, repo_type="dataset")
        val_files = sorted([f for f in all_files if f.startswith("data/validation/") and f.endswith(".parquet")])
        num_files = max(1, math.ceil(len(val_files) * safedocs_percentage / 100.0))
        selected_files = val_files[:num_files]
        print(f"Loading evaluation dataset {dataset_name} ({safedocs_percentage:.1f}%, {num_files}/{len(val_files)} files)...")
        self._dataset = load_dataset(
            dataset_name,
            data_files={"validation": selected_files},
            split="validation",
            keep_in_memory=False,
        )
        print(f"  Loaded {len(self._dataset)} eval examples")

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> Optional[dict]:
        example = self._dataset[idx]
        try:
            image = example["image"]
            width = example.get("width") or getattr(image, "width", None)
            height = example.get("height") or getattr(image, "height", None)

            if not width or not height:
                # Return a dummy example that will be filtered by collator
                return None

            bbox_max = self.bbox_max
            raw_bboxes = example["bounding_boxes"]
            normalized_bboxes = []
            for bbox in raw_bboxes:
                x0, y0, x1, y1 = bbox
                normalized_bboxes.append([
                    max(0, min(bbox_max, int(x0 * bbox_max / width))),
                    max(0, min(bbox_max, int(y0 * bbox_max / height))),
                    max(0, min(bbox_max, int(x1 * bbox_max / width))),
                    max(0, min(bbox_max, int(y1 * bbox_max / height))),
                ])

            processed = self.processor(
                image,
                example["words"],
                boxes=normalized_bboxes,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )

            return {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "bbox": processed["bbox"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
            }
        except Exception:
            return None


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class LayoutLMv3PreTrainingCollator:
    """
    Data collator for LayoutLMv3 pretraining with MLM, IRR, and WPA objectives.
    """

    processor: Any
    mlm_probability: float = 0.15
    irr_probability: float = 0.15
    mask_token_id: int = None
    vocab_size: int = None
    pad_token_id: int = None
    num_patches: int = 196
    patch_grid_size: int = 14
    max_2d_position_embeddings: int = 1024

    def __post_init__(self):
        if self.mask_token_id is None:
            self.mask_token_id = self.processor.tokenizer.mask_token_id
        if self.vocab_size is None:
            # Use len() not .vocab_size - they differ due to added_tokens
            self.vocab_size = len(self.processor.tokenizer)
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # Filter out None examples (from failed processing)
        features = [f for f in features if f is not None]
        if not features:
            raise ValueError("All examples in batch failed processing")

        batch_size = len(features)

        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        bbox = torch.stack([f["bbox"] for f in features])
        pixel_values = torch.stack([f["pixel_values"] for f in features])

        original_pixel_values = pixel_values.clone()

        mlm_labels = torch.full_like(input_ids, -100)
        irr_labels = torch.zeros(batch_size, self.num_patches)
        wpa_labels = torch.full_like(input_ids, -100, dtype=torch.float)

        for i in range(batch_size):
            mlm_labels[i], input_ids[i] = self._apply_whole_word_masking(
                input_ids[i].clone(),
                bbox[i],
                attention_mask[i],
            )

            pixel_values[i], irr_labels[i], replaced_patches = self._apply_irr(
                pixel_values[i],
                original_pixel_values,
                i,
            )

            wpa_labels[i] = self._compute_wpa_labels(
                bbox[i],
                attention_mask[i],
                replaced_patches,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "pixel_values": pixel_values,
            "mlm_labels": mlm_labels,
            "irr_labels": irr_labels,
            "wpa_labels": wpa_labels,
        }

    def _apply_whole_word_masking(
        self,
        input_ids: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full_like(input_ids, -100)
        seq_len = input_ids.size(0)

        word_groups = []
        current_word = []
        prev_bbox = None

        for idx in range(seq_len):
            if attention_mask[idx] == 0:
                if current_word:
                    word_groups.append(current_word)
                    current_word = []
                prev_bbox = None
                continue

            current_bbox = tuple(bbox[idx].tolist())

            if current_bbox == (0, 0, 0, 0):
                if current_word:
                    word_groups.append(current_word)
                    current_word = []
                prev_bbox = None
                continue

            if prev_bbox is not None and current_bbox == prev_bbox:
                current_word.append(idx)
            else:
                if current_word:
                    word_groups.append(current_word)
                current_word = [idx]
                prev_bbox = current_bbox

        if current_word:
            word_groups.append(current_word)

        if not word_groups:
            return labels, input_ids

        num_words_to_mask = max(1, int(len(word_groups) * self.mlm_probability))
        words_to_mask = random.sample(
            range(len(word_groups)),
            min(num_words_to_mask, len(word_groups))
        )

        for word_idx in words_to_mask:
            word_tokens = word_groups[word_idx]
            for token_idx in word_tokens:
                labels[token_idx] = input_ids[token_idx]

                rand_val = random.random()
                if rand_val < 0.8:
                    input_ids[token_idx] = self.mask_token_id
                elif rand_val < 0.9:
                    input_ids[token_idx] = random.randint(0, self.vocab_size - 1)

        return labels, input_ids

    def _apply_irr(
        self,
        pixel_values: torch.Tensor,
        all_pixel_values: torch.Tensor,
        current_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, set]:
        irr_labels = torch.zeros(self.num_patches)
        replaced_patches = set()

        num_patches_to_replace = max(1, int(self.num_patches * self.irr_probability))
        patches_to_replace = random.sample(range(self.num_patches), num_patches_to_replace)

        batch_size = all_pixel_values.size(0)
        patch_size = 16

        for patch_idx in patches_to_replace:
            row = patch_idx // self.patch_grid_size
            col = patch_idx % self.patch_grid_size

            y_start = row * patch_size
            y_end = y_start + patch_size
            x_start = col * patch_size
            x_end = x_start + patch_size

            if batch_size > 1:
                source_idx = random.choice([j for j in range(batch_size) if j != current_idx])
            else:
                source_idx = current_idx

            source_patch_idx = random.randint(0, self.num_patches - 1)
            source_row = source_patch_idx // self.patch_grid_size
            source_col = source_patch_idx % self.patch_grid_size
            source_y_start = source_row * patch_size
            source_y_end = source_y_start + patch_size
            source_x_start = source_col * patch_size
            source_x_end = source_x_start + patch_size

            pixel_values[:, y_start:y_end, x_start:x_end] = \
                all_pixel_values[source_idx, :, source_y_start:source_y_end, source_x_start:source_x_end].clone()

            irr_labels[patch_idx] = 1.0
            replaced_patches.add(patch_idx)

        return pixel_values, irr_labels, replaced_patches

    def _compute_wpa_labels(
        self,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        replaced_patches: set,
    ) -> torch.Tensor:
        seq_len = bbox.size(0)
        wpa_labels = torch.full((seq_len,), -100, dtype=torch.float)
        bbox_max = self.max_2d_position_embeddings - 1  # 1023 for 1024

        for idx in range(seq_len):
            if attention_mask[idx] == 0:
                continue

            box = bbox[idx].tolist()

            if box == [0, 0, 0, 0]:
                continue

            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2

            # Map normalized coords [0, bbox_max] to pixel coords [0, 223]
            pixel_x = center_x * 223 / bbox_max
            pixel_y = center_y * 223 / bbox_max

            patch_col = int(pixel_x / 16)
            patch_row = int(pixel_y / 16)

            patch_col = min(patch_col, self.patch_grid_size - 1)
            patch_row = min(patch_row, self.patch_grid_size - 1)

            patch_idx = patch_row * self.patch_grid_size + patch_col

            wpa_labels[idx] = 1.0 if patch_idx in replaced_patches else 0.0

        return wpa_labels


# =============================================================================
# Callbacks
# =============================================================================

class PushToHubCallback(TrainerCallback):
    """Push model and processor to HuggingFace Hub after each evaluation."""

    def __init__(self, hub_model_id: str, processor):
        self.hub_model_id = hub_model_id
        self.processor = processor

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is not None and self.hub_model_id:
            print(f"\nPushing model to {self.hub_model_id}...")
            model.push_to_hub(self.hub_model_id, commit_message=f"Step {state.global_step}")
            self.processor.push_to_hub(self.hub_model_id, commit_message=f"Step {state.global_step}")
            print(f"Pushed to https://huggingface.co/{self.hub_model_id}")


# =============================================================================
# Trainer
# =============================================================================

class LayoutLMv3PreTrainer(Trainer):
    """Custom trainer to handle multiple label types."""

    def __init__(self, mlm_weight=1.0, irr_weight=1.0, wpa_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.mlm_weight = mlm_weight
        self.irr_weight = irr_weight
        self.wpa_weight = wpa_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        mlm_labels = inputs.pop("mlm_labels", None)
        irr_labels = inputs.pop("irr_labels", None)
        wpa_labels = inputs.pop("wpa_labels", None)

        outputs = model(
            **inputs,
            mlm_labels=mlm_labels,
            irr_labels=irr_labels,
            wpa_labels=wpa_labels,
            mlm_weight=self.mlm_weight,
            irr_weight=self.irr_weight,
            wpa_weight=self.wpa_weight,
        )

        loss = outputs.loss

        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            if outputs.mlm_loss is not None:
                self.log({"mlm_loss": outputs.mlm_loss.item()})
            if outputs.irr_loss is not None:
                self.log({"irr_loss": outputs.irr_loss.item()})
            if outputs.wpa_loss is not None:
                self.log({"wpa_loss": outputs.wpa_loss.item()})

        return (loss, outputs) if return_outputs else loss


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain LayoutLMv3 model")

    # Model arguments
    parser.add_argument("--hf_path", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--from_scratch", action="store_true")

    # Data source arguments
    parser.add_argument(
        "--synthetic_ratio",
        type=float,
        default=0.95,
        help="Ratio of synthetic data (0.0 = all real, 1.0 = all synthetic)",
    )
    parser.add_argument(
        "--real_dataset",
        type=str,
        default="albertklorer/safedocs",
        help="HuggingFace dataset for real documents",
    )
    parser.add_argument(
        "--synthetic_dataset",
        type=str,
        default="HuggingFaceFW/fineweb",
        help="HuggingFace dataset for synthetic text source",
    )
    parser.add_argument(
        "--synthetic_dataset_config",
        type=str,
        default="sample-10BT",
        help="Config/subset name for synthetic dataset",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )
    parser.add_argument(
        "--safedocs_percentage",
        type=float,
        default=100.0,
        help="Percentage of SafeDocs dataset to download (1-100). Useful for testing with smaller data.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum training steps (for streaming datasets)",
    )

    # Pretraining objectives
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--irr_probability", type=float, default=0.15)
    parser.add_argument("--mlm_weight", type=float, default=1.0)
    parser.add_argument("--irr_weight", type=float, default=1.0)
    parser.add_argument("--wpa_weight", type=float, default=1.0)

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output/pretrain")
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3, help="Keep only last N checkpoints")
    parser.add_argument("--eval_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # Optimizer arguments
    parser.add_argument("--optimizer", type=str, default="muon", choices=["adamw", "muon"])
    parser.add_argument("--muon_lr", type=float, default=0.02)
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_weight_decay", type=float, default=0.1)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default="layoutlmv3-pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None, help="W&B run ID for resuming a run")
    parser.add_argument("--no_wandb", action="store_true")

    # Hub
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to HuggingFace Hub after each eval")

    # Resumption (for spot instances)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory or 'latest' to auto-detect latest checkpoint in output_dir",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Check if we're the main process (for distributed training)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main_process = local_rank == 0

    # Initialize wandb with resumption support (only on main process)
    if not args.no_wandb and is_main_process:
        wandb_kwargs = {
            "project": args.wandb_project,
            "name": args.wandb_run_name,
            "config": vars(args),
        }
        if args.wandb_run_id:
            # Resume existing run
            wandb_kwargs["id"] = args.wandb_run_id
            wandb_kwargs["resume"] = "must"  # Fail if run doesn't exist
            print(f"Resuming W&B run: {args.wandb_run_id}")
        wandb.init(**wandb_kwargs)
        # Print run ID for future resumption
        print(f"W&B run ID: {wandb.run.id} (use --wandb_run_id={wandb.run.id} to resume)")

    # Load config and processor
    config = LayoutLMv3Config.from_pretrained(args.hf_path)
    processor = AutoProcessor.from_pretrained(args.hf_path, apply_ocr=False)

    # Update config for sequence length if needed
    if args.max_seq_length != 512:
        print(f"Setting max_position_embeddings to {args.max_seq_length + 2}")
        config.max_position_embeddings = args.max_seq_length + 2

    # Load or initialize model
    if args.from_scratch:
        print("Initializing model from scratch...")
        model = LayoutLMv3ForPreTraining(config)
    else:
        print(f"Loading pretrained model from {args.hf_path}...")
        model = LayoutLMv3ForPreTraining.from_pretrained(args.hf_path, config=config)

    # Resize embeddings if tokenizer vocab differs from model vocab
    tokenizer_vocab_size = len(processor.tokenizer)
    if model.config.vocab_size != tokenizer_vocab_size:
        print(f"Resizing model embeddings: {model.config.vocab_size} -> {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)

    # Create mixed dataset
    print(f"Creating mixed dataset (synthetic_ratio={args.synthetic_ratio})...")
    print(f"  - Real data: {args.real_dataset}")
    if args.safedocs_percentage < 100.0:
        print(f"  - SafeDocs percentage: {args.safedocs_percentage:.1f}%")
    print(f"  - Synthetic text: {args.synthetic_dataset}/{args.synthetic_dataset_config}")
    print(f"  - Max sequence length: {args.max_seq_length}")

    train_dataset = MixedDocumentDataset(
        processor=processor,
        max_2d_position_embeddings=config.max_2d_position_embeddings,
        synthetic_ratio=args.synthetic_ratio,
        real_dataset_name=args.real_dataset,
        synthetic_dataset_name=args.synthetic_dataset,
        synthetic_dataset_config=args.synthetic_dataset_config,
        max_seq_length=args.max_seq_length,
        seed=args.seed,
        safedocs_percentage=args.safedocs_percentage,
    )

    # Create eval dataset from SafeDocs validation split
    eval_dataset = SafeDocsEvalDataset(
        processor=processor,
        max_2d_position_embeddings=config.max_2d_position_embeddings,
        dataset_name=args.real_dataset,
        max_seq_length=args.max_seq_length,
        safedocs_percentage=args.safedocs_percentage,
    )

    # Create data collator
    data_collator = LayoutLMv3PreTrainingCollator(
        processor=processor,
        mlm_probability=args.mlm_probability,
        irr_probability=args.irr_probability,
        max_2d_position_embeddings=config.max_2d_position_embeddings,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps if args.max_steps else -1,
        num_train_epochs=args.num_train_epochs if not args.max_steps else 1,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio if args.warmup_steps is None else 0,
        warmup_steps=args.warmup_steps if args.warmup_steps else 0,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb" if not args.no_wandb else "none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=False,
    )

    # Create optimizer
    optimizer = None
    if args.optimizer == "muon":
        print("Using Muon+AdamW optimizer...")
        optimizer = create_muon_optimizer(
            model,
            lr=args.learning_rate,
            muon_lr=args.muon_lr,
            weight_decay=args.weight_decay,
            muon_weight_decay=args.muon_weight_decay,
            momentum=args.muon_momentum,
        )

    # Create callbacks
    callbacks = []
    if args.push_to_hub:
        callbacks.append(PushToHubCallback(hub_model_id=args.hf_path, processor=processor))

    # Create trainer
    trainer = LayoutLMv3PreTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        mlm_weight=args.mlm_weight,
        irr_weight=args.irr_weight,
        wpa_weight=args.wpa_weight,
        optimizers=(optimizer, None) if optimizer else (None, None),
        callbacks=callbacks if callbacks else None,
    )

    # Resolve checkpoint path
    resume_checkpoint = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "latest":
            # Auto-detect latest checkpoint in output_dir
            checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
            if checkpoint_dirs:
                # Sort by step number
                checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
                resume_checkpoint = checkpoint_dirs[-1]
                print(f"Auto-detected latest checkpoint: {resume_checkpoint}")
            else:
                print("No checkpoints found in output_dir, starting from scratch")
        else:
            resume_checkpoint = args.resume_from_checkpoint
            print(f"Resuming from checkpoint: {resume_checkpoint}")

    # Set global_step_offset for dataset RNG seeding when resuming
    if resume_checkpoint:
        trainer_state_path = os.path.join(resume_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path) as f:
                trainer_state = json.load(f)
            global_step = trainer_state.get("global_step", 0)
            train_dataset.set_global_step_offset(global_step)
            print(f"  Set dataset RNG offset to global_step={global_step}")

    # Train
    print("Starting training...")
    print(f"  - Synthetic ratio: {args.synthetic_ratio:.0%}")
    print(f"  - Sequence length: {args.max_seq_length}")
    print(f"  - Eval every {args.eval_steps} steps ({len(eval_dataset)} samples)")
    if args.max_steps:
        print(f"  - Max steps: {args.max_steps}")
    if args.push_to_hub:
        print(f"  - Push to hub: {args.hf_path} (after each eval)")
    if resume_checkpoint:
        print(f"  - Resuming from: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if not args.no_wandb and is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
