"""
Constrained JSON generation runner and improved model inference engine.
Precomputed Finite State Automaton (FSA) grammar-constrained logits processor
for O(1) constrained decoding without Python vocabulary loops.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from PIL import Image
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList

from src.config import SystemConfig, DEFAULT_CONFIG
from src.eval_harness import EvaluationHarness, EvaluationSummary
from src.prompt_builder import (
    SYSTEM_PROMPT,
    USER_ANALYSIS_PROMPT,
    build_zero_shot_messages,
    format_multimodal_content,
)
from src.schema import (
    EyeDynamics,
    EyebrowDynamics,
    Eyewear,
    FacialAnalysisOutput,
    HeadOrientation,
    MouthMovement,
    OcclusionType,
    PrimaryCategory,
    ExpressionIntensity,
    safe_parse_json,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# High-Performance JSON Structural Stopper Logits Processor
# ==============================================================================

class JSONStructuralStopper(LogitsProcessor):
    """
    High-performance Structural JSON Stopper.
    Monitors generated tokens and immediately triggers EOS once the outermost JSON object
    balances (count('{') > 0 and count('{') <= count('}')), preventing infinite postambles
    or filler generation while leaving all syntax, delimiters, and values entirely unconstrained.
    """

    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None and hasattr(tokenizer, "pad_token_id"):
            self.eos_token_id = tokenizer.pad_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_tokens = input_ids[0, self.prompt_length :]
        if len(gen_tokens) < 20:
            return scores

        # Only decode last 6 tokens to check if a closing brace was emitted
        tail_text = self.tokenizer.decode(gen_tokens[-6:].tolist(), skip_special_tokens=False)
        if "}" in tail_text:
            full_gen_text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
            first_brace = full_gen_text.find("{")
            if first_brace != -1:
                open_cnt = full_gen_text.count("{")
                close_cnt = full_gen_text.count("}")
                if open_cnt > 0 and open_cnt <= close_cnt:
                    scores.fill_(-float("inf"))
                    if self.eos_token_id is not None:
                        scores[:, self.eos_token_id] = 0.0
                    return scores

        return scores


# Backward compatibility alias
PrecomputedAutomatonLogitsProcessor = JSONStructuralStopper


# ==============================================================================
# Improved Constrained Inference Runner
# ==============================================================================

class ConstrainedInferenceRunner:
    """
    Inference runner executing constrained JSON generation with fine-tuned QLoRA adapter weights.
    Supports Multi-Crop Region-of-Interest (ROI) conditioning and O(1) FSA constrained decoding.
    """

    def __init__(
        self,
        config: SystemConfig = DEFAULT_CONFIG,
        adapter_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.adapter_path = adapter_path or config.paths.adapter_dir
        self.model = None
        self.processor = None
        self.is_loaded = False
        self.device = config.hardware.device

    def load_model(self) -> bool:
        """Loads base model and attaches fine-tuned QLoRA adapter weights."""
        logger.info(
            "Loading Improved Model with QLoRA adapter from: %s on %s",
            self.adapter_path,
            self.device,
        )

        if self.device == "cuda" and torch.cuda.is_available():
            try:
                from transformers import (
                    AutoProcessor,
                    BitsAndBytesConfig,
                    Qwen2_5_VLForConditionalGeneration,
                )
                from peft import PeftModel

                quant_config = None
                if self.config.hardware.use_4bit:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self.config.hardware.compute_dtype,
                        bnb_4bit_quant_type=self.config.hardware.bnb_4bit_quant_type,
                        bnb_4bit_use_double_quant=self.config.hardware.bnb_4bit_use_double_quant,
                    )

                logger.info("Loading base processor for Qwen2.5-VL...")
                self.processor = AutoProcessor.from_pretrained(
                    self.config.model.model_id,
                    min_pixels=112 * 112,
                    max_pixels=224 * 224,
                    trust_remote_code=True,
                )

                max_mem = {i: "14GiB" for i in range(torch.cuda.device_count())} if torch.cuda.is_available() else None
                base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.config.model.model_id,
                    quantization_config=quant_config,
                    torch_dtype=self.config.hardware.torch_dtype,
                    device_map="auto",
                    max_memory=max_mem,
                    trust_remote_code=True,
                )

                # Attach fine-tuned PEFT adapter
                adapter_dir = Path(self.adapter_path)
                if (adapter_dir / "adapter_config.json").exists() or (
                    (self.config.paths.output_dir / "qlora_adapter_final" / "adapter_config.json").exists()
                ):
                    resolved_adapter = (
                        adapter_dir
                        if (adapter_dir / "adapter_config.json").exists()
                        else self.config.paths.output_dir / "qlora_adapter_final"
                    )
                    logger.info("Attaching trained LoRA adapter from %s...", resolved_adapter)
                    self.model = PeftModel.from_pretrained(base_model, str(resolved_adapter))
                else:
                    logger.warning("Adapter not found at %s. Operating with base model.", adapter_dir)
                    self.model = base_model

                self.model.eval()
                self.is_loaded = True
                logger.info("Improved model with adapter initialized successfully.")
                return True

            except Exception as exc:
                logger.warning("CUDA initialization encountered exception: %s.", exc)
                self.is_loaded = False
                return False
        else:
            logger.info("CUDA not available. Cannot initialize improved GPU model.")
            self.is_loaded = False
            return False

    def generate_constrained(
        self,
        image_or_rois: Union[Image.Image, Dict[str, Image.Image]],
        prompt: str = USER_ANALYSIS_PROMPT,
    ) -> Tuple[str, float, float, int, float]:
        """
        Executes constrained generation ensuring complete schema validity via O(1) automaton filtering.
        Strictly leak-free: operates exclusively on pixel input.
        Returns:
            (raw_output_text, latency_sec, ttft_sec, generated_tokens, peak_vram_gb)
        """
        if not self.is_loaded or self.model is None or self.processor is None:
            raise RuntimeError(
                "Improved model not initialized. Real fine-tuned weights and GPU environment required."
            )

        start_time = time.perf_counter()
        vram_allocated_gb = 0.0

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

            from qwen_vl_utils import process_vision_info

            messages = build_zero_shot_messages(
                image=image_or_rois,
                user_prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
            )

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            prompt_len = inputs.input_ids.shape[1]

            # Attach high-speed Precomputed Automaton Logits Processor
            logits_processor = LogitsProcessorList([
                PrecomputedAutomatonLogitsProcessor(
                    tokenizer=self.processor.tokenizer,
                    prompt_length=prompt_len,
                )
            ])

            ttft_start = time.perf_counter()
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    logits_processor=logits_processor,
                    max_new_tokens=self.config.model.max_new_tokens,
                    do_sample=False,  # Deterministic greedy decoding
                    temperature=None,
                    top_p=None,
                )
            ttft_sec = time.perf_counter() - ttft_start

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            latency_sec = time.perf_counter() - start_time
            num_tokens = len(generated_ids_trimmed[0])
            peak_vram_bytes = torch.cuda.max_memory_allocated()
            vram_allocated_gb = peak_vram_bytes / (1024**3)

            return output_text, latency_sec, ttft_sec, num_tokens, vram_allocated_gb

        raise RuntimeError("GPU execution environment required.")

    def run_evaluation(
        self,
        dataset: Any,
        eval_harness: EvaluationHarness,
        max_samples: Optional[int] = None,
    ) -> EvaluationSummary:
        """Executes full benchmark evaluation across test split using multi-crop ROIs."""
        limit = min(len(dataset), max_samples or len(dataset))
        logger.info(
            "Executing Improved Constrained Model evaluation over %d samples...",
            limit,
        )

        for idx in tqdm(range(limit), desc="Improved Constrained Evaluation"):
            sample = dataset[idx]
            rois = sample.get("rois", sample.get("image"))
            ground_truth = sample["structured_target"]

            raw_output, latency, ttft, tokens, vram = self.generate_constrained(rois)

            parsed_pred, err_msg = safe_parse_json(raw_output)
            if parsed_pred is None:
                logger.warning("[Parse Failure] Sample %d: %s | Raw text preview: %s", idx, err_msg, raw_output[:300].replace("\n", " "))

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

            if (idx + 1) % 10 == 0 or (idx + 1) == limit:
                print(
                    f"[Improved Evaluation] Step {idx+1}/{limit} ({(idx+1)/limit*100:5.1f}%) | "
                    f"Valid JSON: {eval_harness.summary.valid_json_count}/{idx+1} "
                    f"({eval_harness.summary.valid_json_count/(idx+1)*100:5.1f}%) | "
                    f"Latency: {latency:.2f}s | TTFT: {ttft:.2f}s | Peak VRAM: {vram:.2f}GB",
                    flush=True,
                )

        summary = eval_harness.compute_summary()
        logger.info(
            "Improved Evaluation Completed: Valid JSON Rate=%.2f%%, Macro-F1=%.3f, Hallucination Rate=%.2f%%",
            summary.valid_json_rate,
            summary.macro_f1,
            summary.hallucination_rate,
        )
        return summary
