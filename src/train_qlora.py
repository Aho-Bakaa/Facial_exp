"""
Resilient QLoRA Trainer with Intermediate Step-Wise Checkpointing,
Validation Auditing, Early Stopping, and Auto-Resumption Protocol.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from peft import PeftModel, get_peft_model, LoraConfig, prepare_model_for_kbit_training
except ImportError:
    PeftModel = Any
    get_peft_model = None
    LoraConfig = None
    prepare_model_for_kbit_training = None

from src.config import SystemConfig, DEFAULT_CONFIG
from src.data_loader import CelebAVLMDataset, export_curated_training_dataset
from src.prompt_builder import SYSTEM_PROMPT, USER_ANALYSIS_PROMPT, build_sft_conversation

logger = logging.getLogger(__name__)


# ==============================================================================
# Resilient Step-Wise QLoRA Trainer
# ==============================================================================

class ResilientQLoRATrainer:
    """
    Resilient Parameter-Efficient Fine-Tuning (QLoRA) Trainer.
    Decouples checkpoint saving from epoch boundaries, performs step-wise validation auditing,
    implements early stopping, and provides atomic auto-resumption from crashes/timeouts.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        train_dataset: CelebAVLMDataset,
        val_dataset: Optional[CelebAVLMDataset] = None,
        checkpoint_dir: Union[str, Path] = "experiments/checkpoints",
        final_adapter_dir: Union[str, Path] = "experiments/qlora_adapter_final",
        lr: float = 2e-4,
        grad_accum_steps: int = 4,
        save_steps: int = 50,
        eval_steps: int = 50,
        early_stopping_patience: int = 3,
        min_loss_delta: float = 0.005,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.checkpoint_dir = Path(checkpoint_dir)
        self.final_adapter_dir = Path(final_adapter_dir)
        self.lr = lr
        self.grad_accum_steps = grad_accum_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.patience = early_stopping_patience
        self.min_loss_delta = min_loss_delta
        self.device = device

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.final_adapter_dir.mkdir(parents=True, exist_ok=True)

        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=self.lr,
            weight_decay=0.01,
        )
        self.scheduler = None

        self.global_step = 0
        self.start_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Reads atomic checkpoint state pointer to find latest valid checkpoint."""
        state_file = self.checkpoint_dir / "checkpoint_state.json"
        if not state_file.exists():
            return None
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            latest_dir = Path(data.get("latest_checkpoint_path", ""))
            if latest_dir.exists() and (latest_dir / "adapter_model.safetensors").exists():
                return latest_dir
            elif latest_dir.exists() and (latest_dir / "adapter_config.json").exists():
                return latest_dir
        except Exception as e:
            logger.warning("Error reading checkpoint state: %s", e)
        return None

    def save_checkpoint(self, step: int, epoch: int, val_loss: float) -> None:
        """Atomically saves PEFT weights, optimizer state, scheduler state, and metadata."""
        step_dir = self.checkpoint_dir / f"step_{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save PEFT Adapter Weights
        self.model.save_pretrained(str(step_dir))
        self.processor.save_pretrained(str(step_dir))

        # 2. Save Optimizer & Scheduler State
        torch.save(self.optimizer.state_dict(), step_dir / "optimizer.pt")
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), step_dir / "scheduler.pt")

        # 3. Save Metadata
        meta = {
            "global_step": step,
            "epoch": epoch,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(step_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # 4. Atomic Pointer Update
        temp_state = self.checkpoint_dir / "checkpoint_state.tmp"
        with open(temp_state, "w", encoding="utf-8") as f:
            json.dump({
                "latest_checkpoint_path": str(step_dir),
                "global_step": step,
                "epoch": epoch,
            }, f, indent=2)
        temp_state.replace(self.checkpoint_dir / "checkpoint_state.json")

        logger.info("[Checkpoint] Step %d committed successfully to %s (Val Loss: %.4f)", step, step_dir, val_loss)

    def resume_if_available(self) -> bool:
        """Inspects checkpoint_state.json and restores runtime state if valid checkpoint exists."""
        latest = self.get_latest_checkpoint()
        if not latest:
            logger.info("[Resume] No previous checkpoints located. Starting fresh training run.")
            return False

        logger.info("[Resume] Found valid checkpoint at %s. Restoring runtime state...", latest)
        try:
            # 1. Reload adapter weights
            from peft import PeftModel
            base_model = getattr(self.model, "base_model", self.model)
            if hasattr(base_model, "model"):
                base_model = base_model.model
            self.model = PeftModel.from_pretrained(
                base_model,
                str(latest),
                is_trainable=True,
            )

            # 2. Reload optimizer
            opt_path = latest / "optimizer.pt"
            if opt_path.exists():
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))

            # 3. Reload scheduler
            sched_path = latest / "scheduler.pt"
            if sched_path.exists() and self.scheduler is not None:
                self.scheduler.load_state_dict(torch.load(sched_path, map_location=self.device))

            # 4. Restore step counters and early stopping state
            with open(latest / "metadata.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.global_step = meta.get("global_step", 0)
                self.start_epoch = meta.get("epoch", 0)
                self.best_val_loss = meta.get("best_val_loss", float("inf"))
                self.patience_counter = meta.get("patience_counter", 0)

            logger.info("[Resume] Successfully restored to Step %d (Epoch %d).", self.global_step, self.start_epoch)
            return True
        except Exception as e:
            logger.error("[Resume] Failed to restore checkpoint: %s. Continuing fresh.", e)
            return False

    def evaluate_validation(self, max_eval_samples: int = 50) -> float:
        """Computes average cross-entropy validation loss on val_dataset."""
        if not self.val_dataset or len(self.val_dataset) == 0:
            return 0.0

        from qwen_vl_utils import process_vision_info

        self.model.eval()
        total_loss = 0.0
        val_samples = 0
        limit = min(len(self.val_dataset), max_eval_samples)

        with torch.no_grad():
            for idx in range(limit):
                try:
                    sample = self.val_dataset[idx]
                    rois = sample.get("rois", sample.get("image"))
                    target_json_str = sample["structured_target"].to_json_str()

                    messages = build_sft_conversation(
                        image=rois,
                        target_json_str=target_json_str,
                        user_prompt=USER_ANALYSIS_PROMPT,
                        system_prompt=SYSTEM_PROMPT,
                    )
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)

                    labels = inputs.input_ids.clone()
                    assistant_marker = "<|im_start|>assistant\n"
                    prompt_text = text.split(assistant_marker)[0] + assistant_marker
                    prompt_len = len(self.processor.tokenizer.encode(prompt_text))
                    labels[:, :prompt_len] = -100

                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    if loss is not None and not torch.isnan(loss):
                        total_loss += loss.item()
                        val_samples += 1
                except Exception as e:
                    logger.debug("Validation sample %d error: %s", idx, e)

        self.model.train()
        return total_loss / max(1, val_samples)

    def train(self, epochs: int = 1, max_train_samples: Optional[int] = None) -> Path:
        """
        Executes robust training loop with gradient accumulation, intermediate checkpointing,
        validation auditing, and early stopping.
        """
        from qwen_vl_utils import process_vision_info

        limit = min(len(self.train_dataset), max_train_samples or len(self.train_dataset))
        total_steps = (limit // self.grad_accum_steps) * epochs

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=max(1, int(total_steps * 0.05)),
            num_training_steps=max(1, total_steps),
        )

        # Attempt recovery before starting
        self.resume_if_available()

        logger.info(
            "[Training] Commencing Resilient QLoRA: %d samples, %d epochs, %d target steps",
            limit, epochs, total_steps
        )
        self.model.train()
        running_loss = 0.0

        for epoch in range(self.start_epoch, epochs):
            logger.info(">>> Epoch %d / %d", epoch + 1, epochs)
            for idx in tqdm(range(limit), desc=f"QLoRA Epoch {epoch+1}/{epochs}"):
                # Fast forward to resumed global step
                current_step_estimate = (epoch * limit + idx) // self.grad_accum_steps
                if current_step_estimate < self.global_step:
                    continue

                sample = self.train_dataset[idx]
                rois = sample.get("rois", sample.get("image"))
                target_json_str = sample["structured_target"].to_json_str()

                messages = build_sft_conversation(
                    image=rois,
                    target_json_str=target_json_str,
                    user_prompt=USER_ANALYSIS_PROMPT,
                    system_prompt=SYSTEM_PROMPT,
                )

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)

                # Label masking: compute loss ONLY on assistant prediction tokens
                labels = inputs.input_ids.clone()
                assistant_marker = "<|im_start|>assistant\n"
                prompt_text = text.split(assistant_marker)[0] + assistant_marker
                prompt_len = len(self.processor.tokenizer.encode(prompt_text))
                labels[:, :prompt_len] = -100

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss / self.grad_accum_steps
                loss.backward()

                running_loss += loss.item() * self.grad_accum_steps

                # Optimizer step upon gradient accumulation
                if (idx + 1) % self.grad_accum_steps == 0 or (idx + 1) == limit:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, max_norm=1.0)
                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Logging
                    if self.global_step % 10 == 0:
                        cur_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.lr
                        print(
                            f"[QLoRA Step {self.global_step}/{total_steps}] Train Loss: {(running_loss/10):.4f} | LR: {cur_lr:.2e}",
                            flush=True,
                        )
                        running_loss = 0.0

                    # Validation & Early Stopping Audit
                    if self.global_step % self.eval_steps == 0 and self.val_dataset:
                        val_loss = self.evaluate_validation()
                        delta = self.best_val_loss - val_loss
                        print(f"\n[Audit Step {self.global_step}] Val Loss: {val_loss:.4f} (Best: {self.best_val_loss:.4f})", flush=True)

                        if delta > self.min_loss_delta:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            self.save_checkpoint(self.global_step, epoch, val_loss)
                        else:
                            self.patience_counter += 1
                            print(f"[Patience] No significant improvement. Counter: {self.patience_counter}/{self.patience}", flush=True)

                        if self.patience_counter >= self.patience:
                            print(f"[Early Stopping] Triggered at step {self.global_step}. Convergence reached.", flush=True)
                            self._finalize_and_save()
                            return self.final_adapter_dir

                    # Periodic Safety Checkpoint
                    elif self.global_step % self.save_steps == 0:
                        self.save_checkpoint(self.global_step, epoch, val_loss=loss.item() * self.grad_accum_steps)

        self._finalize_and_save()
        return self.final_adapter_dir

    def _finalize_and_save(self) -> None:
        """Exports the final fine-tuned adapter weights to experiments/qlora_adapter_final."""
        logger.info("[Export] Committing final fine-tuned adapter to %s", self.final_adapter_dir)
        self.model.save_pretrained(str(self.final_adapter_dir))
        self.processor.save_pretrained(str(self.final_adapter_dir))

        meta = {
            "status": "trained_and_finalized",
            "global_steps": self.global_step,
            "best_val_loss": self.best_val_loss,
            "learning_rate": self.lr,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(self.final_adapter_dir / "training_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# ==============================================================================
# High-Level Pipeline Interface
# ==============================================================================

class QLoRATrainerPipeline:
    """Wrapper pipeline managing hardware setup, model loading, and training orchestration."""

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.adapter_output_dir = config.paths.output_dir / "qlora_adapter_final"
        self.device = config.hardware.device

    def configure_lora(self, model: Any) -> Any:
        lora_cfg = self.config.qlora
        logger.info(
            "Configuring LoRA: r=%d, alpha=%d, dropout=%.2f, targets=%s",
            lora_cfg.r,
            lora_cfg.lora_alpha,
            lora_cfg.lora_dropout,
            lora_cfg.target_modules,
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=lora_cfg.gradient_checkpointing,
        )
        peft_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.lora_alpha,
            lora_dropout=lora_cfg.lora_dropout,
            target_modules=lora_cfg.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()
        return peft_model

    def train(
        self,
        train_dataset: CelebAVLMDataset,
        val_dataset: Optional[CelebAVLMDataset] = None,
        max_samples: Optional[int] = None,
        epochs: int = 1,
    ) -> Path:
        """Executes resilient training with intermediate checkpoints."""
        self.adapter_output_dir.mkdir(parents=True, exist_ok=True)
        limit = min(len(train_dataset), max_samples or len(train_dataset))

        # Export curated SFT pairs
        curated_jsonl = self.config.paths.output_dir / "sft_training_curated.jsonl"
        export_curated_training_dataset(train_dataset, curated_jsonl, max_samples=limit)

        if self.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("QLoRA training requires CUDA GPU environment.")

        from transformers import (
            AutoProcessor,
            BitsAndBytesConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=self.config.hardware.compute_dtype,
            bnb_4bit_quant_type=self.config.hardware.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.config.hardware.bnb_4bit_use_double_quant,
        )

        logger.info("Loading base Qwen2.5-VL model with 4-bit NF4 quantization...")
        max_mem = {i: "14GiB" for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else None
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.config.model.model_id,
            quantization_config=quant_config,
            torch_dtype=self.config.hardware.torch_dtype,
            device_map="auto",
            max_memory=max_mem,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(
            self.config.model.model_id,
            min_pixels=112 * 112,
            max_pixels=224 * 224,
            trust_remote_code=True,
        )

        model = self.configure_lora(model)

        resilient_trainer = ResilientQLoRATrainer(
            model=model,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            checkpoint_dir=self.config.paths.output_dir / "checkpoints",
            final_adapter_dir=self.adapter_output_dir,
            lr=self.config.qlora.learning_rate,
            grad_accum_steps=self.config.qlora.gradient_accumulation_steps,
            save_steps=50,
            eval_steps=50,
            early_stopping_patience=3,
            device=self.device,
        )

        return resilient_trainer.train(epochs=epochs, max_train_samples=limit)
