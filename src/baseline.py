"""
Zero-shot baseline inference runner using Qwen2.5-VL-3B-Instruct.
Implements standard prompt-based zero-shot evaluation with Multi-Crop ROI conditioning.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm

from src.config import SystemConfig, DEFAULT_CONFIG
from src.eval_harness import EvaluationHarness, EvaluationSummary
from src.prompt_builder import SYSTEM_PROMPT, USER_ANALYSIS_PROMPT, build_zero_shot_messages
from src.schema import FacialAnalysisOutput, safe_parse_json

logger = logging.getLogger(__name__)


class ZeroShotBaselineRunner:
    """
    Executes standard zero-shot VLM inference across the held-out test split
    measuring syntactic validity, perceptual accuracy, and system latency.
    """

    def __init__(self, config: SystemConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.device = config.hardware.device

    def load_model(self) -> bool:
        """Loads baseline vision-language model with 4-bit quantization and device allocation."""
        model_id = self.config.model.model_id
        logger.info("Initializing zero-shot baseline model: %s on %s", model_id, self.device)

        if self.device == "cuda":
            try:
                from transformers import (
                    AutoProcessor,
                    BitsAndBytesConfig,
                    Qwen2_5_VLForConditionalGeneration,
                )

                quant_config = None
                if self.config.hardware.use_4bit:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.config.hardware.compute_dtype,
                        bnb_4bit_quant_type=self.config.hardware.bnb_4bit_quant_type,
                        bnb_4bit_use_double_quant=self.config.hardware.bnb_4bit_use_double_quant,
                    )

                logger.info("Loading processor for %s with native CelebA pixel bounds...", model_id)
                self.processor = AutoProcessor.from_pretrained(
                    model_id,
                    min_pixels=112 * 112,
                    max_pixels=224 * 224,
                    trust_remote_code=True,
                )

                logger.info("Loading model weights with 4-bit quantization: %s...", model_id)
                max_mem = {i: "14GiB" for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else None
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    torch_dtype=self.config.hardware.torch_dtype,
                    device_map="auto",
                    max_memory=max_mem,
                    trust_remote_code=True,
                )
                self.model.eval()
                self.is_loaded = True
                logger.info("Baseline model loaded successfully across %d GPU(s).", self.config.hardware.num_gpus)
                return True
            except Exception as exc:
                logger.warning(
                    "CUDA VLM loading could not be initialized (%s).",
                    exc,
                )
                self.is_loaded = False
                return False
        else:
            logger.info("No CUDA device detected.")
            self.is_loaded = False
            return False

    def generate_single(
        self,
        image_or_rois: Union[Image.Image, Dict[str, Image.Image]],
        prompt: str = USER_ANALYSIS_PROMPT,
    ) -> Tuple[str, float, float, int, float]:
        """
        Generates structured response for a facial image / ROI crops using the VLM.
        Strictly leak-free: operates exclusively on pixel input.
        Returns:
            (raw_output_text, latency_sec, ttft_sec, generated_tokens, peak_vram_gb)
        """
        if not self.is_loaded or self.model is None or self.processor is None:
            raise RuntimeError(
                "Zero-shot baseline VLM not initialized. Real model weights and execution environment required."
            )

        start_time = time.perf_counter()
        vram_allocated_gb = 0.0

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

            from qwen_vl_utils import process_vision_info

            messages = build_zero_shot_messages(image_or_rois, user_prompt=prompt, system_prompt=SYSTEM_PROMPT)
            text_input = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text_input],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            ttft_start = time.perf_counter()
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.model.max_new_tokens,
                    do_sample=False,
                )
            ttft_sec = time.perf_counter() - ttft_start

            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(inputs.input_ids, outputs)
            ]
            output_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            token_count = len(generated_ids[0])

            peak_vram_bytes = torch.cuda.max_memory_allocated()
            vram_allocated_gb = peak_vram_bytes / (1024**3)

            latency_sec = time.perf_counter() - start_time
            return output_text, latency_sec, ttft_sec, token_count, vram_allocated_gb

        raise RuntimeError("CUDA GPU execution environment required.")

    def run_evaluation(
        self,
        dataset: Any,
        eval_harness: EvaluationHarness,
        max_samples: Optional[int] = None,
    ) -> EvaluationSummary:
        """Executes zero-shot baseline benchmarking across the dataset split."""
        total = min(len(dataset), max_samples) if max_samples else len(dataset)
        logger.info("Executing Zero-Shot Baseline evaluation over %d samples...", total)

        for i in tqdm(range(total), desc="Baseline Zero-Shot Evaluation"):
            sample = dataset[i]
            rois = sample.get("rois", sample.get("image"))
            ground_truth = sample["structured_target"]

            raw_output, latency, ttft, tokens, vram = self.generate_single(
                image_or_rois=rois,
            )

            parsed_pred, err_msg = safe_parse_json(raw_output)

            eval_harness.record_prediction(
                prediction=parsed_pred,
                ground_truth=ground_truth,
                latency_sec=latency,
                ttft_sec=ttft,
                generated_tokens=tokens,
                peak_vram_gb=vram,
                raw_prediction_text=raw_output,
                is_male=sample.get("is_male"),
                is_young=sample.get("is_young"),
            )

            if (i + 1) % 10 == 0 or (i + 1) == total:
                print(
                    f"[Baseline Evaluation] Step {i+1}/{total} ({(i+1)/total*100:5.1f}%) | "
                    f"Latency: {latency:.2f}s | TTFT: {ttft:.2f}s | Tokens: {tokens} | "
                    f"Peak VRAM: {vram:.2f}GB",
                    flush=True,
                )

        summary = eval_harness.compute_summary()
        logger.info(
            "Baseline Completed: Valid JSON Rate=%.2f%%, Macro-F1=%.3f, Hallucination Rate=%.2f%%",
            summary.valid_json_rate,
            summary.macro_f1,
            summary.hallucination_rate,
        )
        return summary
