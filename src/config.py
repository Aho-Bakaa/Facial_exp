"""
Global configuration and environment parameters for the Autonomous VLM pipeline.
Supports automated Kaggle environment detection, multi-GPU scaling, and memory constraints.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Explicitly seed all RNGs for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class PathConfig:
    """Centralized path management with Kaggle vs local path resolution."""
    # Determine base directory
    base_dir: Path = field(default_factory=lambda: Path(os.environ.get("KAGGLE_WORKING_DIR", Path.cwd())).resolve())
    
    # Input data directories
    kaggle_input_dir: Path = Path("/kaggle/input/celeba-dataset")
    kaggle_alt_input_dir: Path = Path("/kaggle/input/celeba")
    local_input_dir: Path = field(init=False)
    
    # Active data directory (resolved dynamically)
    data_dir: Path = field(init=False)
    images_dir: Path = field(init=False)
    attr_file: Path = field(init=False)
    landmarks_file: Path = field(init=False)
    partition_file: Path = field(init=False)
    
    # Output and experiment directories
    output_dir: Path = field(init=False)
    adapter_dir: Path = field(init=False)
    results_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.local_input_dir = self.base_dir / "input_data"
        self.output_dir = self.base_dir / "experiments"
        self.adapter_dir = self.output_dir / "qlora_adapter"
        self.results_file = self.output_dir / "benchmark_results.json"
        
        self.local_input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Dynamic search for mounted CelebA dataset across /kaggle/input and local dirs
        candidate_dirs = []
        kaggle_root = Path("/kaggle/input")
        if kaggle_root.exists():
            try:
                for item in kaggle_root.iterdir():
                    if item.is_dir():
                        candidate_dirs.append(item)
            except Exception:
                pass
        candidate_dirs.extend([
            self.kaggle_input_dir,
            self.kaggle_alt_input_dir,
            self.local_input_dir,
        ])

        # 1. Resolve annotation data directory and files
        resolved_data_dir = self.local_input_dir
        resolved_attr = None
        for cand in candidate_dirs:
            if not cand.exists():
                continue
            csv_f = cand / "list_attr_celeba.csv"
            txt_f = cand / "list_attr_celeba.txt"
            if csv_f.exists():
                resolved_attr = csv_f
                resolved_data_dir = cand
                break
            elif txt_f.exists():
                resolved_attr = txt_f
                resolved_data_dir = cand
                break

        self.data_dir = resolved_data_dir
        self.attr_file = resolved_attr if resolved_attr else (self.data_dir / "list_attr_celeba.csv")

        csv_lm = self.data_dir / "list_landmarks_align_celeba.csv"
        txt_lm = self.data_dir / "list_landmarks_align_celeba.txt"
        self.landmarks_file = csv_lm if csv_lm.exists() else txt_lm

        csv_part = self.data_dir / "list_eval_partition.csv"
        txt_part = self.data_dir / "list_eval_partition.txt"
        self.partition_file = csv_part if csv_part.exists() else txt_part

        # 2. Resolve real images directory
        image_candidates = [
            self.data_dir / "img_align_celeba" / "img_align_celeba",
            self.data_dir / "img_align_celeba",
            self.data_dir,
            self.local_input_dir / "img_align_celeba",
        ]
        if kaggle_root.exists():
            try:
                for item in kaggle_root.iterdir():
                    if item.is_dir():
                        image_candidates.append(item / "img_align_celeba" / "img_align_celeba")
                        image_candidates.append(item / "img_align_celeba")
                        image_candidates.append(item)
            except Exception:
                pass

        resolved_img = self.data_dir / "img_align_celeba"
        for img_cand in image_candidates:
            if img_cand.exists() and (img_cand / "000001.jpg").exists():
                resolved_img = img_cand
                break

        self.images_dir = resolved_img


@dataclass
class ModelConfig:
    """Model identifiers and architecture targets."""
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    fallback_model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    max_new_tokens: int = 512
    temperature: float = 0.0  # Greedy decoding for deterministic schema parsing
    top_p: float = 0.95
    use_flash_attention_2: bool = False  # Set dynamically based on hardware support


@dataclass
class HardwareConfig:
    """Hardware detection, memory optimization, and dtype allocation."""
    device: str = field(init=False)
    num_gpus: int = field(init=False)
    torch_dtype: torch.dtype = field(init=False)
    compute_dtype: torch.dtype = field(init=False)
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    def __post_init__(self) -> None:
        if torch.cuda.is_available():
            self.device = "cuda"
            self.num_gpus = torch.cuda.device_count()
            # T4 GPUs (compute capability 7.5) do not support native bfloat16 efficiently
            # Check native support before assigning bfloat16
            if torch.cuda.is_bf16_supported():
                self.torch_dtype = torch.bfloat16
                self.compute_dtype = torch.bfloat16
            else:
                self.torch_dtype = torch.float16
                self.compute_dtype = torch.float16
        else:
            self.device = "cpu"
            self.num_gpus = 0
            self.torch_dtype = torch.float32
            self.compute_dtype = torch.float32
            self.use_4bit = False  # CPU does not support bitsandbytes 4-bit CUDA kernels


@dataclass
class QLoRAConfig:
    """Hyperparameters for Parameter-Efficient Fine-Tuning."""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ])
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    num_train_epochs: int = 2
    max_train_samples: int = 8000
    val_sample_size: int = 1000
    test_sample_size: int = 1000
    gradient_checkpointing: bool = True
    logging_steps: int = 20
    save_strategy: str = "epoch"


@dataclass
class GDriveConfig:
    """Google Drive identifiers for CelebA and CelebA Dialog."""
    celeba_folder_id: str = "0B7EVK8r0v71pWEZsZE9oNnFzTm8"
    celeba_dialog_folder_id: str = "18nejI_hrwNzWyoF6SW8bL27EYnM4STAs"
    # Individual file Google Drive IDs for reliable gdown retrieval
    file_ids: dict = field(default_factory=lambda: {
        "list_attr_celeba.txt": "0B7EVK8r0v71pM0phNjhjdndJdZM",
        "list_landmarks_align_celeba.txt": "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
        "list_eval_partition.txt": "0B7EVK8r0v71pY0NSMzRuN1ZZR28",
        "img_align_celeba.zip": "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
    })


@dataclass
class SystemConfig:
    """Master configuration encompassing all subsystems."""
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    qlora: QLoRAConfig = field(default_factory=QLoRAConfig)
    gdrive: GDriveConfig = field(default_factory=GDriveConfig)
    seed: int = 42

    def __post_init__(self) -> None:
        seed_everything(self.seed)


# Global default configuration instance
DEFAULT_CONFIG = SystemConfig()
