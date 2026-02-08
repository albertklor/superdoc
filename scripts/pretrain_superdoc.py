"""
Pretraining script for LayoutLMv3 with four objectives:
- MLM: Masked Language Modeling with whole word masking
- ROP: Reading Order Prediction (first subword predicts word reading-order index)
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
import shutil
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


def reorder_words_top_down_left_right(
    words: list[str],
    bboxes: list[list[int]],
) -> tuple[list[str], list[list[int]], list[int]]:
    """
    Reorder words by top/down then left/right position.

    Returns:
        - sorted_words: words in spatial reading order
        - sorted_bboxes: boxes aligned with sorted_words
        - reading_order_indices: original word-order index for each sorted word
    """
    entries = []
    for idx, (word, bbox) in enumerate(zip(words, bboxes)):
        if not word or bbox is None or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = (int(v) for v in bbox)
        entries.append((idx, word, [x0, y0, x1, y1]))

    entries.sort(key=lambda item: (item[2][1], item[2][0], item[0]))

    sorted_words = [item[1] for item in entries]
    sorted_bboxes = [item[2] for item in entries]
    reading_order_indices = [item[0] for item in entries]
    return sorted_words, sorted_bboxes, reading_order_indices


def build_rop_labels_from_word_ids(
    word_ids: list[Optional[int]],
    word_reading_order: list[int],
    max_seq_length: int,
) -> torch.LongTensor:
    """
    Build ROP labels for tokenized inputs.

    Only the first subword token of each word gets a label.
    Label space is [0, max_seq_length), and all other positions are -100.
    """
    seq_len = len(word_ids)
    labels = torch.full((seq_len,), -100, dtype=torch.long)
    if not word_ids or not word_reading_order:
        return labels

    first_token_by_word: dict[int, int] = {}
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id < 0 or word_id >= len(word_reading_order):
            continue
        if word_id not in first_token_by_word:
            first_token_by_word[word_id] = token_idx

    if not first_token_by_word:
        return labels

    sorted_word_ids = sorted(
        first_token_by_word.keys(),
        key=lambda wid: (word_reading_order[wid], wid),
    )
    word_to_rank = {
        wid: rank
        for rank, wid in enumerate(sorted_word_ids[:max_seq_length])
    }

    for wid, token_idx in first_token_by_word.items():
        rank = word_to_rank.get(wid)
        if rank is not None:
            labels[token_idx] = rank

    return labels


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
            (320, 420),
            (360, 520), (420, 600), (480, 700), (540, 780),
            (600, 840), (680, 960), (760, 1080), (840, 1200),
            (420, 300), (600, 420), (700, 480), (840, 600),
            (900, 640), (980, 700), (1100, 780), (1280, 900),
        ]

        self.generator = DocumentGenerator(
            output_size=(224, 224),
            font_sizes=list(range(8, 19)),
            canvas_sizes=canvas_sizes,
            canvas_area_weight_power=1.25,
            line_spacing=1.35,
            font_scale_power=0.72,
            background_color=(220, 255),
            text_color=(0, 80),
            horizontal_shift=(-0.1, 0.1),
            num_columns=[1, 1, 1, 2, 2, 3],
            paragraph_spacing=(1.1, 1.9),
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

    def _build_rop_labels(
        self,
        processed,
        word_reading_order: list[int],
    ) -> torch.LongTensor:
        seq_len = processed["input_ids"].shape[-1]
        default_labels = torch.full((seq_len,), -100, dtype=torch.long)

        try:
            word_ids = processed.word_ids(batch_index=0)
        except Exception:
            return default_labels
        if word_ids is None:
            return default_labels

        labels = build_rop_labels_from_word_ids(
            list(word_ids),
            word_reading_order,
            self.max_seq_length,
        )
        if labels.shape[0] != seq_len:
            return default_labels
        return labels

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
            rop_labels = self._build_rop_labels(
                processed,
                list(range(len(example["words"]))),
            )

            return {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "bbox": processed["bbox"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
                "rop_labels": rop_labels,
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

            # Keep original text order as label target, then feed spatially ordered words.
            words_spatial, bboxes_spatial, reading_order_indices = reorder_words_top_down_left_right(
                doc["tokens"],
                doc["bboxes"],
            )
            if not words_spatial:
                return None

            # Process with LayoutLMv3 processor
            processed = self.processor(
                doc["image"],
                words_spatial,
                boxes=bboxes_spatial,
                truncation=True,
                max_length=self.max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            rop_labels = self._build_rop_labels(processed, reading_order_indices)

            return {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "bbox": processed["bbox"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
                "rop_labels": rop_labels,
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

    def _build_rop_labels(self, processed, num_words: int) -> torch.LongTensor:
        seq_len = processed["input_ids"].shape[-1]
        default_labels = torch.full((seq_len,), -100, dtype=torch.long)

        try:
            word_ids = processed.word_ids(batch_index=0)
        except Exception:
            return default_labels
        if word_ids is None:
            return default_labels

        labels = build_rop_labels_from_word_ids(
            list(word_ids),
            list(range(num_words)),
            self.max_seq_length,
        )
        if labels.shape[0] != seq_len:
            return default_labels
        return labels

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
            rop_labels = self._build_rop_labels(processed, len(example["words"]))

            return {
                "input_ids": processed["input_ids"].squeeze(0),
                "attention_mask": processed["attention_mask"].squeeze(0),
                "bbox": processed["bbox"].squeeze(0),
                "pixel_values": processed["pixel_values"].squeeze(0),
                "rop_labels": rop_labels,
            }
        except Exception:
            return None


# =============================================================================
# Data Collator
# =============================================================================

@dataclass
class LayoutLMv3PreTrainingCollator:
    """
    Data collator for LayoutLMv3 pretraining with MLM, ROP, IRR, and WPA objectives.
    """

    processor: Any
    mlm_probability: float = 0.15
    irr_probability: float = 0.15
    mask_token_id: int = None
    vocab_size: int = None
    pad_token_id: int = None
    irr_grid_size: int = 4  # 4x4 = 16 regions for IRR
    num_irr_regions: int = 16
    max_2d_position_embeddings: int = 1024
    alignment_check_batches: int = 0

    def __post_init__(self):
        if self.mask_token_id is None:
            self.mask_token_id = self.processor.tokenizer.mask_token_id
        if self.vocab_size is None:
            # Use len() not .vocab_size - they differ due to added_tokens
            self.vocab_size = len(self.processor.tokenizer)
        if self.pad_token_id is None:
            self.pad_token_id = self.processor.tokenizer.pad_token_id

    def _group_tokens_by_contiguous_bbox(
        self,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> list[list[int]]:
        """Group token indices into words using contiguous identical bbox spans."""
        seq_len = bbox.size(0)
        groups: list[list[int]] = []
        current: list[int] = []
        prev_bbox: Optional[tuple[int, int, int, int]] = None

        for idx in range(seq_len):
            if attention_mask[idx] == 0:
                if current:
                    groups.append(current)
                    current = []
                prev_bbox = None
                continue

            current_bbox = tuple(int(v) for v in bbox[idx].tolist())
            if current_bbox == (0, 0, 0, 0):
                if current:
                    groups.append(current)
                    current = []
                prev_bbox = None
                continue

            if prev_bbox is not None and current_bbox == prev_bbox:
                current.append(idx)
            else:
                if current:
                    groups.append(current)
                current = [idx]
                prev_bbox = current_bbox

        if current:
            groups.append(current)
        return groups

    def _validate_sample_alignment(
        self,
        sample_idx: int,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        bbox: torch.Tensor,
        pixel_values_before_wpa: torch.Tensor,
        pixel_values_after_wpa: torch.Tensor,
        mlm_labels: torch.Tensor,
        rop_labels: torch.Tensor,
        irr_labels: torch.Tensor,
        wpa_labels: torch.Tensor,
        replaced_groups: list[list[int]],
    ) -> None:
        """Validate token-level and word-level objective alignment invariants."""
        seq_len = input_ids.size(0)
        if seq_len == 0:
            raise ValueError(f"sample {sample_idx}: empty sequence")

        valid_tokens = attention_mask == 1
        invalid_tokens = ~valid_tokens
        zero_bbox = bbox.sum(dim=1) == 0

        # Token-level checks.
        if not torch.all(mlm_labels[invalid_tokens] == -100):
            raise ValueError(f"sample {sample_idx}: mlm_labels must be -100 on padding tokens")
        if not torch.all(rop_labels[invalid_tokens] == -100):
            raise ValueError(f"sample {sample_idx}: rop_labels must be -100 on padding tokens")
        if not torch.all(wpa_labels[invalid_tokens] == -100):
            raise ValueError(f"sample {sample_idx}: wpa_labels must be -100 on padding tokens")
        if not torch.all(rop_labels[zero_bbox] == -100):
            raise ValueError(f"sample {sample_idx}: rop_labels must be -100 for zero bbox tokens")
        if not torch.all(wpa_labels[zero_bbox] == -100):
            raise ValueError(f"sample {sample_idx}: wpa_labels must be -100 for zero bbox tokens")

        valid_rop = rop_labels[rop_labels != -100]
        if valid_rop.numel() and (valid_rop.min() < 0 or valid_rop.max() >= seq_len):
            raise ValueError(
                f"sample {sample_idx}: rop_labels out of range [0, {seq_len})"
            )

        if not torch.all((irr_labels == 0.0) | (irr_labels == 1.0)):
            raise ValueError(f"sample {sample_idx}: irr_labels must be binary")

        valid_wpa = wpa_labels[wpa_labels != -100]
        if not torch.all((valid_wpa == 0.0) | (valid_wpa == 1.0)):
            raise ValueError(f"sample {sample_idx}: wpa_labels must be in {{-100,0,1}}")

        # Word-level checks.
        groups = self._group_tokens_by_contiguous_bbox(bbox, attention_mask)
        rop_seen: list[int] = []

        for group in groups:
            group_t = torch.tensor(group, dtype=torch.long)

            # MLM is whole-word: either all tokens masked or none.
            group_mlm = mlm_labels[group_t]
            group_mlm_valid = group_mlm != -100
            if not (bool(group_mlm_valid.all()) or bool((~group_mlm_valid).all())):
                raise ValueError(f"sample {sample_idx}: MLM labels not word-aligned")

            # WPA labels should be uniform across a word.
            group_wpa = wpa_labels[group_t]
            if not torch.all(group_wpa == group_wpa[0]):
                raise ValueError(f"sample {sample_idx}: WPA labels not word-aligned")

            # ROP: only first subword gets label.
            group_rop = rop_labels[group_t]
            labeled_positions = (group_rop != -100).nonzero(as_tuple=True)[0]
            if labeled_positions.numel() > 1:
                raise ValueError(f"sample {sample_idx}: multiple ROP labels in one word")
            if labeled_positions.numel() == 1:
                if int(labeled_positions[0].item()) != 0:
                    raise ValueError(
                        f"sample {sample_idx}: ROP label must be on first subword token"
                    )
                rop_seen.append(int(group_rop[labeled_positions[0]].item()))

        if rop_seen:
            if len(set(rop_seen)) != len(rop_seen):
                raise ValueError(f"sample {sample_idx}: duplicate ROP labels in sequence")
            sorted_rop = sorted(rop_seen)
            if sorted_rop != list(range(len(sorted_rop))):
                raise ValueError(
                    f"sample {sample_idx}: ROP labels must be dense ranks 0..{len(sorted_rop)-1}"
                )

        # Transformation alignment check for WPA-positive groups.
        bbox_max = self.max_2d_position_embeddings - 1
        expected_positive = 0
        for group in replaced_groups:
            if not group:
                continue
            expected_positive += len(group)
            token_idx = group[0]
            box = bbox[token_idx]
            x0 = int(box[0].item() * 223 / bbox_max)
            y0 = int(box[1].item() * 223 / bbox_max)
            x1 = int(box[2].item() * 223 / bbox_max)
            y1 = int(box[3].item() * 223 / bbox_max)
            x0, x1 = max(0, x0), min(223, x1)
            y0, y1 = max(0, y0), min(223, y1)
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"sample {sample_idx}: WPA replaced group has degenerate bbox")

            delta = (
                pixel_values_after_wpa[:, y0:y1 + 1, x0:x1 + 1]
                - pixel_values_before_wpa[:, y0:y1 + 1, x0:x1 + 1]
            ).abs().sum().item()
            if delta <= 0:
                raise ValueError(
                    f"sample {sample_idx}: WPA-positive group has no pixel change in its bbox"
                )

        actual_positive = int((wpa_labels == 1.0).sum().item())
        if actual_positive != expected_positive:
            raise ValueError(
                f"sample {sample_idx}: WPA-positive token mismatch "
                f"(expected {expected_positive}, got {actual_positive})"
            )

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
        rop_labels = (
            torch.stack([f["rop_labels"] for f in features])
            if all("rop_labels" in f and f["rop_labels"] is not None for f in features)
            else torch.full_like(input_ids, -100)
        )

        mlm_labels = torch.full_like(input_ids, -100)
        irr_labels = torch.zeros(batch_size, self.num_irr_regions)
        wpa_labels = torch.full_like(input_ids, -100, dtype=torch.float)

        # Clone original pixel values BEFORE any modifications for IRR source patches
        # This prevents cascading contamination where modified images provide patches
        original_pixel_values = pixel_values.clone()
        run_alignment_checks = self.alignment_check_batches > 0

        for i in range(batch_size):
            mlm_labels[i], input_ids[i] = self._apply_whole_word_masking(
                input_ids[i].clone(),
                bbox[i],
                attention_mask[i],
            )

            pixel_values[i], irr_labels[i], _ = self._apply_irr(
                pixel_values[i],
                original_pixel_values,  # Use ORIGINAL unmodified patches
                i,  # Current sample index
            )

            pre_wpa_pixel_values = pixel_values[i].clone() if run_alignment_checks else None
            wpa_result = self._apply_wpa(
                pixel_values[i],
                bbox[i],
                attention_mask[i],
                return_replaced_groups=run_alignment_checks,
            )
            if run_alignment_checks:
                pixel_values[i], wpa_labels[i], replaced_groups = wpa_result
                self._validate_sample_alignment(
                    sample_idx=i,
                    input_ids=input_ids[i],
                    attention_mask=attention_mask[i],
                    bbox=bbox[i],
                    pixel_values_before_wpa=pre_wpa_pixel_values,
                    pixel_values_after_wpa=pixel_values[i],
                    mlm_labels=mlm_labels[i],
                    rop_labels=rop_labels[i],
                    irr_labels=irr_labels[i],
                    wpa_labels=wpa_labels[i],
                    replaced_groups=replaced_groups,
                )
            else:
                pixel_values[i], wpa_labels[i] = wpa_result

        if run_alignment_checks:
            self.alignment_check_batches -= 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox": bbox,
            "pixel_values": pixel_values,
            "mlm_labels": mlm_labels,
            "rop_labels": rop_labels,
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
        word_groups = self._group_tokens_by_contiguous_bbox(bbox, attention_mask)

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
        """Apply IRR using 4x4 grid (16 regions of 56x56 pixels each).

        Replaces regions with patches from OTHER images in the batch.
        The model learns to detect visual seams/discontinuities at region boundaries.
        """
        irr_labels = torch.zeros(self.num_irr_regions)
        replaced_regions = set()

        batch_size = all_pixel_values.shape[0]
        if batch_size < 2:
            # Can't do cross-image replacement with batch size 1
            return pixel_values, irr_labels, replaced_regions

        num_regions_to_replace = max(1, int(self.num_irr_regions * self.irr_probability))
        regions_to_replace = random.sample(range(self.num_irr_regions), num_regions_to_replace)

        image_size = pixel_values.shape[-1]  # 224
        region_size = image_size // self.irr_grid_size  # 56 pixels per region

        for region_idx in regions_to_replace:
            row = region_idx // self.irr_grid_size
            col = region_idx % self.irr_grid_size

            y_start = row * region_size
            y_end = y_start + region_size
            x_start = col * region_size
            x_end = x_start + region_size

            # Pick a different image from the batch
            source_idx = current_idx
            while source_idx == current_idx:
                source_idx = random.randint(0, batch_size - 1)

            # Pick a random region from the source image (can be same or different region)
            source_region_idx = random.randint(0, self.num_irr_regions - 1)
            source_row = source_region_idx // self.irr_grid_size
            source_col = source_region_idx % self.irr_grid_size
            source_y_start = source_row * region_size
            source_y_end = source_y_start + region_size
            source_x_start = source_col * region_size
            source_x_end = source_x_start + region_size

            # Replace with patch from another image - this creates visible seams
            pixel_values[:, y_start:y_end, x_start:x_end] = \
                all_pixel_values[source_idx, :, source_y_start:source_y_end, source_x_start:source_x_end].clone()

            irr_labels[region_idx] = 1.0
            replaced_regions.add(region_idx)

        return pixel_values, irr_labels, replaced_regions

    def _apply_wpa(
        self,
        pixel_values: torch.Tensor,
        bbox: torch.Tensor,
        attention_mask: torch.Tensor,
        wpa_probability: float = 0.10,
        return_replaced_groups: bool = False,
    ):
        """
        Apply WPA: replace 10% of word bounding boxes with random noise.

        Works at the WORD level (tokens grouped by bbox) to ensure all tokens
        sharing a bbox get the same label.

        Returns:
            pixel_values: Updated with noise in selected word regions
            wpa_labels: 1 if word's region was replaced with noise, 0 otherwise
        """
        seq_len = bbox.size(0)
        bbox_max = self.max_2d_position_embeddings - 1

        # Group tokens by contiguous bbox spans (word-level grouping).
        token_groups = self._group_tokens_by_contiguous_bbox(bbox, attention_mask)
        word_groups = [
            (tuple(int(v) for v in bbox[token_indices[0]].tolist()), token_indices)
            for token_indices in token_groups
            if token_indices
        ]

        # Initialize labels: -100 for invalid, 0 for valid (not replaced)
        wpa_labels = torch.full((seq_len,), -100, dtype=torch.float)
        for _, token_indices in word_groups:
            for idx in token_indices:
                wpa_labels[idx] = 0.0

        if not word_groups:
            if return_replaced_groups:
                return pixel_values, wpa_labels, []
            return pixel_values, wpa_labels

        # Select 10% of word groups to replace
        num_to_replace = max(1, int(len(word_groups) * wpa_probability))
        groups_to_replace = random.sample(range(len(word_groups)), num_to_replace)

        # Replace each selected word's bbox region with noise
        replaced_groups: list[list[int]] = []
        for group_idx in groups_to_replace:
            bbox_tuple, token_indices = word_groups[group_idx]

            # Map bbox coords [0, bbox_max] to pixel coords [0, 223]
            x0 = int(bbox_tuple[0] * 223 / bbox_max)
            y0 = int(bbox_tuple[1] * 223 / bbox_max)
            x1 = int(bbox_tuple[2] * 223 / bbox_max)
            y1 = int(bbox_tuple[3] * 223 / bbox_max)

            # Clamp to image bounds
            x0, x1 = max(0, x0), min(223, x1)
            y0, y1 = max(0, y0), min(223, y1)

            # Replace with random noise
            if x1 > x0 and y1 > y0:
                pixel_values[:, y0:y1+1, x0:x1+1] = torch.randn_like(
                    pixel_values[:, y0:y1+1, x0:x1+1]
                )
                replaced_groups.append(token_indices)

                # Label ALL tokens in this word group
                for idx in token_indices:
                    wpa_labels[idx] = 1.0

        if return_replaced_groups:
            return pixel_values, wpa_labels, replaced_groups
        return pixel_values, wpa_labels


# =============================================================================
# Callbacks
# =============================================================================

class PushToHubCallback(TrainerCallback):
    """Push model and processor to HuggingFace Hub on checkpoints (and eval if enabled)."""

    def __init__(self, hub_model_id: str, processor):
        self.hub_model_id = hub_model_id
        self.processor = processor
        self._last_pushed_step = None

    def _push(self, state, model) -> None:
        if model is None or not self.hub_model_id:
            return
        if not state.is_world_process_zero:
            return

        # Avoid duplicate pushes for the same global step (can happen with multiple callback hooks).
        if self._last_pushed_step == state.global_step:
            return

        step = state.global_step
        print(f"\nPushing checkpoint to {self.hub_model_id} at step {step}...")
        model.push_to_hub(self.hub_model_id, commit_message=f"Step {step}")
        self.processor.push_to_hub(self.hub_model_id, commit_message=f"Step {step}")
        print(f"Pushed to https://huggingface.co/{self.hub_model_id}")
        self._last_pushed_step = step

    def on_save(self, args, state, control, model=None, **kwargs):
        # Called at each Trainer checkpoint save (save_steps).
        self._push(state, model)

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        # Keep eval hook for compatibility when eval is enabled.
        self._push(state, model)


def _checkpoint_step(path: str) -> int:
    name = os.path.basename(path.rstrip("/"))
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.split("-")[-1])
    except ValueError:
        return -1


def prune_old_checkpoints(
    output_dir: str,
    keep_last_n: int,
    protected_checkpoint: Optional[str] = None,
) -> int:
    """
    Keep only the most recent `keep_last_n` checkpoint directories on local disk.

    Returns number of removed checkpoint directories.
    """
    if keep_last_n <= 0:
        return 0

    checkpoint_dirs = [
        p
        for p in glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if os.path.isdir(p) and _checkpoint_step(p) >= 0
    ]
    if len(checkpoint_dirs) <= keep_last_n:
        return 0

    checkpoint_dirs.sort(key=_checkpoint_step)
    protected_abs = os.path.abspath(protected_checkpoint) if protected_checkpoint else None

    removable = []
    for ckpt in checkpoint_dirs:
        if protected_abs is not None and os.path.abspath(ckpt) == protected_abs:
            continue
        removable.append(ckpt)

    # Remove oldest removable checkpoints, keeping the newest keep_last_n overall.
    to_remove_count = max(0, len(checkpoint_dirs) - keep_last_n)
    to_remove = removable[:to_remove_count]

    removed = 0
    for ckpt in to_remove:
        shutil.rmtree(ckpt, ignore_errors=True)
        removed += 1
    return removed


class CheckpointPruneCallback(TrainerCallback):
    """Prune old local checkpoints on each save to cap disk usage."""

    def __init__(self, output_dir: str, keep_last_n: int):
        self.output_dir = output_dir
        self.keep_last_n = keep_last_n

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        removed = prune_old_checkpoints(self.output_dir, self.keep_last_n)
        if removed > 0:
            print(
                f"Pruned {removed} old checkpoints. "
                f"Keeping latest {self.keep_last_n} in {self.output_dir}."
            )


def _collect_features_for_batch(
    data_iter: Iterator[dict],
    batch_size: int,
) -> list[dict]:
    features: list[dict] = []
    while len(features) < batch_size:
        try:
            example = next(data_iter)
        except StopIteration:
            break
        if example is not None:
            features.append(example)
    return features


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=(device.type == "cuda"))
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def run_preflight_checks(
    model: LayoutLMv3ForPreTraining,
    train_dataset: MixedDocumentDataset,
    data_collator: LayoutLMv3PreTrainingCollator,
    args,
) -> None:
    """
    Run small, fail-fast sanity checks before long training jobs.

    Validates:
    - Objective labels exist and are numerically valid.
    - Forward/backward pass is finite on the target device.
    - Each objective head receives gradients.
    """
    if args.preflight_batches <= 0:
        return

    preflight_batch_size = max(1, args.preflight_batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Running preflight checks: {args.preflight_batches} batches x "
        f"{preflight_batch_size} samples on {device}..."
    )

    was_training = model.training
    model.to(device)
    model.train()

    use_autocast = device.type == "cuda" and (args.fp16 or args.bf16)
    autocast_dtype = torch.float16 if args.fp16 else torch.bfloat16

    saved_alignment_batches = data_collator.alignment_check_batches
    data_collator.alignment_check_batches = max(saved_alignment_batches, args.preflight_batches)

    try:
        data_iter = iter(train_dataset)

        for batch_idx in range(args.preflight_batches):
            features = _collect_features_for_batch(data_iter, preflight_batch_size)
            if len(features) < preflight_batch_size:
                raise RuntimeError(
                    f"Preflight failed: expected {preflight_batch_size} samples, got {len(features)}"
                )

            batch = data_collator(features)

            mlm_count = int((batch["mlm_labels"] != -100).sum().item())
            rop_count = int((batch["rop_labels"] != -100).sum().item())
            irr_count = int((batch["irr_labels"] == 1.0).sum().item())
            wpa_count = int((batch["wpa_labels"] == 1.0).sum().item())

            if mlm_count <= 0:
                raise ValueError(f"Preflight batch {batch_idx}: MLM has no supervised tokens")
            if rop_count <= 0:
                raise ValueError(f"Preflight batch {batch_idx}: ROP has no supervised tokens")
            if irr_count <= 0:
                raise ValueError(f"Preflight batch {batch_idx}: IRR has no positive regions")
            if wpa_count <= 0:
                raise ValueError(f"Preflight batch {batch_idx}: WPA has no positive tokens")

            batch = _move_batch_to_device(batch, device)
            model.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=use_autocast,
            ):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    bbox=batch["bbox"],
                    pixel_values=batch["pixel_values"],
                    mlm_labels=batch["mlm_labels"],
                    rop_labels=batch["rop_labels"],
                    irr_labels=batch["irr_labels"],
                    wpa_labels=batch["wpa_labels"],
                    mlm_weight=1.0,
                    rop_weight=1.0,
                    irr_weight=1.0,
                    wpa_weight=1.0,
                )

            if outputs.loss is None or not torch.isfinite(outputs.loss):
                raise ValueError(f"Preflight batch {batch_idx}: total loss is non-finite")
            for name, loss in (
                ("mlm", outputs.mlm_loss),
                ("rop", outputs.rop_loss),
                ("irr", outputs.irr_loss),
                ("wpa", outputs.wpa_loss),
            ):
                if loss is None or not torch.isfinite(loss):
                    raise ValueError(f"Preflight batch {batch_idx}: {name} loss is non-finite")

            outputs.loss.backward()

            # Head-level gradient checks: ensure each objective contributes finite gradients.
            for head_name in ("mlm_head", "rop_head", "irr_head", "wpa_head"):
                grads = [
                    p.grad for n, p in model.named_parameters()
                    if head_name in n and p.grad is not None
                ]
                if not grads:
                    raise ValueError(f"Preflight batch {batch_idx}: no gradients for {head_name}")
                for grad in grads:
                    if not torch.isfinite(grad).all():
                        raise ValueError(f"Preflight batch {batch_idx}: non-finite gradients in {head_name}")

            model.zero_grad(set_to_none=True)
            print(
                f"  Preflight batch {batch_idx + 1}/{args.preflight_batches}: "
                f"loss={outputs.loss.item():.4f}, mlm={mlm_count}, rop={rop_count}, irr={irr_count}, wpa={wpa_count}"
            )
    finally:
        data_collator.alignment_check_batches = saved_alignment_batches
        if not was_training:
            model.eval()

    print("Preflight checks passed.")


# =============================================================================
# Trainer
# =============================================================================

class LayoutLMv3PreTrainer(Trainer):
    """Custom trainer to handle multiple label types."""

    def __init__(
        self,
        mlm_weight=1.0,
        rop_weight=1.0,
        irr_weight=1.0,
        wpa_weight=1.0,
        rop_start_step=0,
        wpa_start_step=50_000,
        irr_start_step=100_000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mlm_weight = mlm_weight
        self.rop_weight = rop_weight
        self.irr_weight = irr_weight
        self.wpa_weight = wpa_weight
        self.rop_start_step = rop_start_step
        self.wpa_start_step = wpa_start_step
        self.irr_start_step = irr_start_step
        self._last_objective_phase = None

        if self.rop_start_step < 0 or self.wpa_start_step < 0 or self.irr_start_step < 0:
            raise ValueError("rop_start_step, wpa_start_step, and irr_start_step must be >= 0")
        if self.irr_start_step < self.wpa_start_step:
            raise ValueError("irr_start_step must be >= wpa_start_step")

    def _get_objective_weights(self, global_step: int) -> tuple[float, float, float, float]:
        mlm_weight = self.mlm_weight
        rop_weight = self.rop_weight if global_step >= self.rop_start_step else 0.0
        wpa_weight = self.wpa_weight if global_step >= self.wpa_start_step else 0.0
        irr_weight = self.irr_weight if global_step >= self.irr_start_step else 0.0
        return mlm_weight, rop_weight, irr_weight, wpa_weight

    def _get_objective_phase(self, global_step: int) -> str:
        active = ["mlm"]
        if global_step >= self.rop_start_step:
            active.append("rop")
        if global_step >= self.wpa_start_step:
            active.append("wpa")
        if global_step >= self.irr_start_step:
            active.append("irr")
        return "_".join(active)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        mlm_labels = inputs.pop("mlm_labels", None)
        rop_labels = inputs.pop("rop_labels", None)
        irr_labels = inputs.pop("irr_labels", None)
        wpa_labels = inputs.pop("wpa_labels", None)

        global_step = self.state.global_step
        mlm_weight, rop_weight, irr_weight, wpa_weight = self._get_objective_weights(global_step)
        phase = self._get_objective_phase(global_step)

        if phase != self._last_objective_phase:
            print(
                f"[objective schedule] step={global_step}: phase={phase} "
                f"(mlm={mlm_weight}, rop={rop_weight}, wpa={wpa_weight}, irr={irr_weight})"
            )
            self._last_objective_phase = phase

        outputs = model(
            **inputs,
            mlm_labels=mlm_labels,
            rop_labels=rop_labels,
            irr_labels=irr_labels,
            wpa_labels=wpa_labels,
            mlm_weight=mlm_weight,
            rop_weight=rop_weight,
            irr_weight=irr_weight,
            wpa_weight=wpa_weight,
        )

        loss = outputs.loss

        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "mlm_weight_active": mlm_weight,
                    "rop_weight_active": rop_weight,
                    "wpa_weight_active": wpa_weight,
                    "irr_weight_active": irr_weight,
                }
            )
            if outputs.mlm_loss is not None:
                self.log({"mlm_loss": outputs.mlm_loss.item()})
            if outputs.rop_loss is not None:
                self.log({"rop_loss": outputs.rop_loss.item()})
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
        default=1.0,
        help="Deprecated in this script; synthetic-only mode forces this to 1.0.",
    )
    parser.add_argument(
        "--real_dataset",
        type=str,
        default="albertklorer/safedocs",
        help="Deprecated (unused in synthetic-only mode).",
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
        default=10.0,
        help="Deprecated (unused in synthetic-only mode).",
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
    parser.add_argument("--rop_weight", type=float, default=1.0)
    parser.add_argument("--irr_weight", type=float, default=1.0)
    parser.add_argument("--wpa_weight", type=float, default=1.0)
    parser.add_argument(
        "--alignment_check_batches",
        type=int,
        default=8,
        help="Run strict word/token alignment checks in collator for the first N batches.",
    )
    parser.add_argument(
        "--preflight_batches",
        type=int,
        default=3,
        help="Run N preflight batches (forward+backward) before training. Set 0 to disable.",
    )
    parser.add_argument(
        "--preflight_batch_size",
        type=int,
        default=2,
        help="Batch size used for preflight checks.",
    )
    parser.add_argument(
        "--rop_start_step",
        type=int,
        default=0,
        help="Enable ROP loss at this global step (0 = enabled from start).",
    )
    parser.add_argument(
        "--wpa_start_step",
        type=int,
        default=50_000,
        help="Enable WPA loss at this global step (0 = enabled from start).",
    )
    parser.add_argument(
        "--irr_start_step",
        type=int,
        default=100_000,
        help="Enable IRR loss at this global step (must be >= wpa_start_step).",
    )

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
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Keep only last N local checkpoints (older ones are pruned).",
    )
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
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push checkpoint saves regularly to HuggingFace Hub (and final model at end).",
    )

    # Resumption (for spot instances)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory or 'latest' to auto-detect latest checkpoint in output_dir",
    )
    parser.add_argument(
        "--auto_resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-resume from latest checkpoint in output_dir when available. Disable with --no-auto_resume.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.rop_start_step < 0 or args.wpa_start_step < 0 or args.irr_start_step < 0:
        raise ValueError("--rop_start_step, --wpa_start_step, and --irr_start_step must be >= 0")
    if args.irr_start_step < args.wpa_start_step:
        raise ValueError("--irr_start_step must be >= --wpa_start_step")
    if args.alignment_check_batches < 0:
        raise ValueError("--alignment_check_batches must be >= 0")
    if args.preflight_batches < 0:
        raise ValueError("--preflight_batches must be >= 0")
    if args.preflight_batch_size <= 0:
        raise ValueError("--preflight_batch_size must be > 0")
    # Enforce synthetic-only training/eval pipeline.
    if args.synthetic_ratio != 1.0:
        print(f"Synthetic-only mode: overriding --synthetic_ratio={args.synthetic_ratio} to 1.0")
        args.synthetic_ratio = 1.0

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
    # ROP uses an MLM-style classifier with class count equal to sequence length.
    config.rop_vocab_size = args.max_seq_length

    # Load or initialize model
    if args.from_scratch:
        print("Initializing model from scratch...")
        model = LayoutLMv3ForPreTraining(config)
    else:
        print(f"Loading pretrained model from {args.hf_path}...")
        model = LayoutLMv3ForPreTraining.from_pretrained(args.hf_path, config=config)

    # Verify vocab sizes match
    tokenizer_vocab_size = len(processor.tokenizer)
    assert model.config.vocab_size == tokenizer_vocab_size, (
        f"Vocab size mismatch: model={model.config.vocab_size}, tokenizer={tokenizer_vocab_size}. "
        f"Regenerate your model with the correct vocab size."
    )

    # Create synthetic training dataset
    print("Creating synthetic training dataset...")
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
        safedocs_percentage=0.0,
    )

    # Synthetic-only mode: disable SafeDocs eval loading.
    eval_dataset = None

    # Create data collator
    data_collator = LayoutLMv3PreTrainingCollator(
        processor=processor,
        mlm_probability=args.mlm_probability,
        irr_probability=args.irr_probability,
        max_2d_position_embeddings=config.max_2d_position_embeddings,
        alignment_check_batches=args.alignment_check_batches,
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
        eval_strategy="no",
        seed=args.seed,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="wandb" if not args.no_wandb else "none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        load_best_model_at_end=False,
        ignore_data_skip=True,  # Don't skip examples on resume - we use RNG offset instead
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
    if args.save_total_limit and args.save_total_limit > 0:
        callbacks.append(
            CheckpointPruneCallback(
                output_dir=args.output_dir,
                keep_last_n=args.save_total_limit,
            )
        )
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
        rop_weight=args.rop_weight,
        irr_weight=args.irr_weight,
        wpa_weight=args.wpa_weight,
        rop_start_step=args.rop_start_step,
        wpa_start_step=args.wpa_start_step,
        irr_start_step=args.irr_start_step,
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
                checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split("-")[-1]))
                resume_checkpoint = checkpoint_dirs[-1]
                print(f"Auto-detected latest checkpoint: {resume_checkpoint}")
            else:
                print("No checkpoints found in output_dir, starting from scratch")
        else:
            resume_checkpoint = args.resume_from_checkpoint
            print(f"Resuming from checkpoint: {resume_checkpoint}")
    elif args.auto_resume:
        checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(os.path.basename(x).split("-")[-1]))
            resume_checkpoint = checkpoint_dirs[-1]
            print(f"Auto-resuming from latest checkpoint: {resume_checkpoint}")

    # Set global_step_offset for dataset RNG seeding when resuming
    if resume_checkpoint:
        trainer_state_path = os.path.join(resume_checkpoint, "trainer_state.json")
        if os.path.exists(trainer_state_path):
            with open(trainer_state_path) as f:
                trainer_state = json.load(f)
            global_step = trainer_state.get("global_step", 0)
            train_dataset.set_global_step_offset(global_step)
            print(f"  Set dataset RNG offset to global_step={global_step}")

    # Startup cleanup: enforce checkpoint cap immediately (protect active resume checkpoint).
    if args.save_total_limit and args.save_total_limit > 0:
        removed = prune_old_checkpoints(
            args.output_dir,
            args.save_total_limit,
            protected_checkpoint=resume_checkpoint,
        )
        if removed > 0:
            print(
                f"Startup prune removed {removed} old checkpoints. "
                f"Keeping latest {args.save_total_limit}."
            )

    # Fail-fast sanity checks before long GPU runs.
    run_preflight_checks(
        model=trainer.model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=args,
    )

    # Train
    print("Starting training...")
    print("  - Data source: synthetic only")
    print(f"  - Sequence length: {args.max_seq_length}")
    print("  - Evaluation: disabled (no SafeDocs usage)")
    print(f"  - Alignment checks: first {args.alignment_check_batches} collator batches")
    if args.preflight_batches > 0:
        print(
            f"  - Preflight: {args.preflight_batches} batches x "
            f"{args.preflight_batch_size} samples"
        )
    else:
        print("  - Preflight: disabled")
    print(f"  - Objective schedule:")
    print("    - MLM enabled from step 0")
    print(f"    - ROP enabled from step {args.rop_start_step}")
    print(f"    - WPA enabled from step {args.wpa_start_step}")
    print(f"    - IRR enabled from step {args.irr_start_step}")
    if args.max_steps:
        print(f"  - Max steps: {args.max_steps}")
    if args.push_to_hub:
        print(f"  - Push to hub: {args.hf_path} (every save step + final)")
    if resume_checkpoint:
        print(f"  - Resuming from: {resume_checkpoint}")

    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save final model
    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if args.push_to_hub:
        print(f"Pushing final model to {args.hf_path}...")
        trainer.model.push_to_hub(args.hf_path, commit_message="Final model")
        processor.push_to_hub(args.hf_path, commit_message="Final processor")

    if not args.no_wandb and is_main_process:
        wandb.finish()


if __name__ == "__main__":
    main()
