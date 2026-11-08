"""
Master execution script for the Autonomous VLM Facial Attribute & Causal Analysis Benchmark.
Orchestrates Phase 1 (Dataset Integrity & Prior Audit) -> Phase 2 (Baseline Zero-Shot) ->
Phase 3 (Resilient QLoRA with Intermediate Checkpoints) -> Phase 4 (Improved Constrained Automaton) ->
Phase 5 (Multi-Dimensional Synthesis & Persistence).
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import torch

from src.baseline import ZeroShotBaselineRunner
from src.config import SystemConfig, DEFAULT_CONFIG, seed_everything
from src.data_loader import (
    CelebAVLMDataset,
    parse_celeba_attributes,
    parse_celeba_landmarks,
    parse_celeba_partitions,
    create_deterministic_splits,
    compute_continuous_landmarks,
)
from src.eval_harness import EvaluationHarness, EvaluationSummary
from src.inference import ConstrainedInferenceRunner
from src.train_qlora import QLoRATrainerPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("BenchmarkOrchestrator")


def cleanup_gpu_memory() -> None:
    """Explicitly releases PyTorch CUDA allocations and forces Python garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ==============================================================================
# Phase 1: Dataset Verification & Biometric Prior Auditing
# ==============================================================================

def execute_phase1_dataset_verification(config: SystemConfig) -> Dict[str, Any]:
    """
    Phase 1: Validates CelebA dataset files, partition allocations, demographic distributions,
    and continuous landmark geometry priors.
    """
    logger.info("Verifying CelebA dataset files at: %s", config.paths.data_dir)
    attr_names, attributes = parse_celeba_attributes(config.paths.attr_file)
    landmarks = parse_celeba_landmarks(config.paths.landmarks_file)
    partitions = parse_celeba_partitions(config.paths.partition_file)

    if not partitions:
        logger.warning("No partition file found. Generating deterministic 80/10/10 split...")
        partitions = create_deterministic_splits(list(attributes.keys()))

    train_count = sum(1 for p in partitions.values() if p == 0)
    val_count = sum(1 for p in partitions.values() if p == 1)
    test_count = sum(1 for p in partitions.values() if p == 2)

    # Compute target attribute prevalence
    target_attrs = [
        "Smiling", "Male", "Young", "Eyeglasses",
        "Narrow_Eyes", "Arched_Eyebrows", "Mouth_Slightly_Open", "Bangs"
    ]
    prevalence = {}
    for attr in target_attrs:
        pos = sum(1 for a in attributes.values() if a.get(attr, False))
        prevalence[attr] = {
            "positive_samples": pos,
            "total_samples": len(attributes),
            "prevalence_pct": round((pos / max(1, len(attributes))) * 100.0, 2),
        }

    # Audit continuous biometric landmark geometry (MAR, EAR, head roll tilt)
    mars, ears, rolls = [], [], []
    sample_keys = list(landmarks.keys())[:1500]
    for k in sample_keys:
        geom = compute_continuous_landmarks(landmarks[k])
        mars.append(geom.get("mouth_aspect_ratio", geom.get("mar", 1.15)))
        ears.append(geom.get("eye_aspect_ratio", geom.get("ear", 0.45)))
        rolls.append(geom.get("head_roll_deg", geom.get("roll_deg", 0.0)))

    summary = {
        "status": "verified",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": len(attributes),
        "partitions": {
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
        },
        "target_attribute_prevalence": prevalence,
        "landmark_biometric_priors": {
            "mar_mean": round(float(np.mean(mars)), 4) if mars else 0.0,
            "mar_std": round(float(np.std(mars)), 4) if mars else 0.0,
            "ear_mean": round(float(np.mean(ears)), 4) if ears else 0.0,
            "ear_std": round(float(np.std(ears)), 4) if ears else 0.0,
            "head_roll_mean_deg": round(float(np.mean(rolls)), 4) if rolls else 0.0,
            "head_roll_std_deg": round(float(np.std(rolls)), 4) if rolls else 0.0,
        },
        "multi_crop_rois": ["global_portrait_224x224", "periorbital_112x112", "oral_112x112"],
    }
    return summary


# ==============================================================================
# Phase-Level State Serialization Orchestrator
# ==============================================================================

class BenchmarkPhaseOrchestrator:
    """
    Phase-Level Benchmark Orchestrator with Persistent Boundary Artifacts.
    Serializes state immediately at each boundary to eliminate lost compute upon interruption:
      - phase1_dataset_summary.json
      - phase2_baseline_results.json
      - checkpoints/ & qlora_adapter_final/
      - phase4_improved_results.json
      - benchmark_results.json
    """

    def __init__(
        self,
        experiment_dir: Union[str, Path] = "experiments",
        config: SystemConfig = DEFAULT_CONFIG,
    ) -> None:
        self.config = config
        self.exp_dir = Path(experiment_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.p1_file = self.exp_dir / "phase1_dataset_summary.json"
        self.p2_file = self.exp_dir / "phase2_baseline_results.json"
        self.p4_file = self.exp_dir / "phase4_improved_results.json"
        self.p5_file = self.exp_dir / "benchmark_results.json"
        self.final_adapter = self.exp_dir / "qlora_adapter_final"

    def run_phase_1(self, data_builder_fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 1: Dataset Verification & Biometric Auditing."""
        if self.p1_file.exists():
            logger.info("[Phase 1] Cached artifact found at %s. Skipping Phase 1.", self.p1_file)
            with open(self.p1_file, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info("[Phase 1] Executing dataset caching, biometric prior auditing & split partition...")
        summary = data_builder_fn()
        with open(self.p1_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("[Phase 1] State successfully serialized to %s", self.p1_file)
        return summary

    def run_phase_2(
        self,
        baseline_runner_fn: Callable[[CelebAVLMDataset], EvaluationSummary],
        test_split: CelebAVLMDataset,
    ) -> Dict[str, Any]:
        """Phase 2: Zero-Shot Baseline Evaluation."""
        if self.p2_file.exists():
            logger.info("[Phase 2] Baseline evaluation cached at %s. Skipping Phase 2.", self.p2_file)
            with open(self.p2_file, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info("[Phase 2] Executing Zero-Shot Baseline inference across test split (%d samples)...", len(test_split))
        baseline_summary = baseline_runner_fn(test_split)

        summary_dict = baseline_summary.to_dict() if hasattr(baseline_summary, "to_dict") else baseline_summary
        baseline_metrics = {
            "summary_metrics": summary_dict,
            "sample_size": len(test_split),
            "model_name": "Baseline (Qwen2.5-VL-3B Zero-Shot)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.p2_file, "w", encoding="utf-8") as f:
            json.dump(baseline_metrics, f, indent=2)
        logger.info("[Phase 2] Baseline metrics successfully serialized to %s", self.p2_file)
        return baseline_metrics

    def run_phase_3(
        self,
        trainer_fn: Callable[[int], Path],
        epochs: int = 1,
    ) -> Path:
        """Phase 3: Resilient QLoRA Fine-Tuning with Step-Wise Checkpoints."""
        safetensors_file = self.final_adapter / "adapter_model.safetensors"
        config_file = self.final_adapter / "adapter_config.json"
        if safetensors_file.exists() or config_file.exists():
            logger.info("[Phase 3] Final trained adapter located at %s. Skipping fine-tuning.", self.final_adapter)
            return self.final_adapter

        logger.info("[Phase 3] Commencing Resilient Fine-Tuning Loop...")
        adapter_path = trainer_fn(epochs)
        logger.info("[Phase 3] QLoRA adapter successfully converged and saved to: %s", adapter_path)
        return adapter_path

    def run_phase_4(
        self,
        improved_runner_fn: Callable[[CelebAVLMDataset], EvaluationSummary],
        test_split: CelebAVLMDataset,
    ) -> Dict[str, Any]:
        """Phase 4: Improved Constrained Automaton Evaluation."""
        if self.p4_file.exists():
            logger.info("[Phase 4] Improved model evaluation cached at %s. Skipping Phase 4.", self.p4_file)
            with open(self.p4_file, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.info("[Phase 4] Executing Improved Model inference across test split (%d samples)...", len(test_split))
        improved_summary = improved_runner_fn(test_split)

        summary_dict = improved_summary.to_dict() if hasattr(improved_summary, "to_dict") else improved_summary
        improved_metrics = {
            "summary_metrics": summary_dict,
            "sample_size": len(test_split),
            "model_name": "Improved (QLoRA + Prefix-Trie Automaton)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.p4_file, "w", encoding="utf-8") as f:
            json.dump(improved_metrics, f, indent=2)
        logger.info("[Phase 4] Improved model metrics successfully serialized to %s", self.p4_file)
        return improved_metrics

    def run_phase_5_synthesis(self) -> Dict[str, Any]:
        """Phase 5: Multi-Dimensional Cross-Phase Synthesis & Comparative Reporting."""
        logger.info("[Phase 5] Synthesizing cross-phase benchmark comparison...")
        with open(self.p2_file, "r", encoding="utf-8") as f:
            b_data = json.load(f)
        with open(self.p4_file, "r", encoding="utf-8") as f:
            i_data = json.load(f)

        b = b_data.get("summary_metrics", b_data)
        i = i_data.get("summary_metrics", i_data)

        # Baseline and Improved metric normalization
        b_p50 = b.get("p50_latency_ms", b.get("mean_latency_sec", 0.0) * 1000.0)
        i_p50 = i.get("p50_latency_ms", i.get("mean_latency_sec", 0.0) * 1000.0)
        latency_p50_red = ((b_p50 - i_p50) / max(1e-5, b_p50)) * 100.0 if b_p50 > 0 else 0.0

        b_conf = b.get("conformance_rate", b.get("valid_json_rate", 0.0) / 100.0)
        i_conf = i.get("conformance_rate", i.get("valid_json_rate", 0.0) / 100.0)

        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hardware": {
                "device": self.config.hardware.device,
                "num_gpus": self.config.hardware.num_gpus,
                "torch_dtype": str(self.config.hardware.torch_dtype),
            },
            "baseline_model": b,
            "improved_model": i,
            "deltas": {
                "macro_f1_delta": round(i.get("macro_f1", 0.0) - b.get("macro_f1", 0.0), 4),
                "schema_conformance_delta": round(i_conf - b_conf, 4),
                "latency_p50_reduction_pct": round(latency_p50_red, 2),
                "gender_disparity_delta_change": round(
                    i.get("gender_disparity_delta", 0.0) - b.get("gender_disparity_delta", 0.0), 4
                ),
                "age_disparity_delta_change": round(
                    i.get("age_disparity_delta", 0.0) - b.get("age_disparity_delta", 0.0), 4
                ),
                "faithfulness_score_delta": round(
                    i.get("faithfulness_score", 0.0) - b.get("faithfulness_score", 0.0), 2
                ),
                "hallucination_rate_delta": round(
                    i.get("hallucination_rate", 0.0) - b.get("hallucination_rate", 0.0), 2
                ),
                "causal_rouge_l_f1_delta": round(
                    i.get("mean_causal_rouge_f1", 0.0) - b.get("mean_causal_rouge_f1", 0.0), 4
                ),
                "vram_delta_gb": round(
                    i.get("peak_vram_gb", 0.0) - b.get("peak_vram_gb", 0.0), 2
                ),
            },
        }

        with open(self.p5_file, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)

        logger.info("[Phase 5] Benchmark synthesis serialized to %s", self.p5_file)
        self._print_markdown_table(b, i, comparison["deltas"])
        return comparison

    def _print_markdown_table(
        self,
        b: Dict[str, Any],
        i: Dict[str, Any],
        d: Dict[str, Any],
    ) -> None:
        """Prints the industry-grade comparative Markdown table."""
        b_conf = b.get("conformance_rate", b.get("valid_json_rate", 0.0) / 100.0) * 100.0
        i_conf = i.get("conformance_rate", i.get("valid_json_rate", 0.0) / 100.0) * 100.0
        d_conf = d.get("schema_conformance_delta", 0.0) * 100.0

        b_f1 = b.get("macro_f1", 0.0)
        i_f1 = i.get("macro_f1", 0.0)
        d_f1 = d.get("macro_f1_delta", 0.0)

        b_p50 = b.get("p50_latency_ms", b.get("mean_latency_sec", 0.0) * 1000.0)
        i_p50 = i.get("p50_latency_ms", i.get("mean_latency_sec", 0.0) * 1000.0)
        d_p50_pct = d.get("latency_p50_reduction_pct", 0.0)

        b_p95 = b.get("p95_latency_ms", 0.0)
        i_p95 = i.get("p95_latency_ms", 0.0)

        b_rouge = b.get("mean_causal_rouge_f1", 0.0)
        i_rouge = i.get("mean_causal_rouge_f1", 0.0)
        d_rouge = d.get("causal_rouge_l_f1_delta", 0.0)

        b_gender = b.get("gender_disparity_delta", 0.0)
        i_gender = i.get("gender_disparity_delta", 0.0)
        d_gender = d.get("gender_disparity_delta_change", 0.0)

        b_age = b.get("age_disparity_delta", 0.0)
        i_age = i.get("age_disparity_delta", 0.0)
        d_age = d.get("age_disparity_delta_change", 0.0)

        b_faith = b.get("faithfulness_score", 0.0)
        i_faith = i.get("faithfulness_score", 0.0)
        d_faith = d.get("faithfulness_score_delta", 0.0)

        b_halluc = b.get("hallucination_rate", 0.0)
        i_halluc = i.get("hallucination_rate", 0.0)
        d_halluc = d.get("hallucination_rate_delta", 0.0)

        b_vram = b.get("peak_vram_gb", 0.0)
        i_vram = i.get("peak_vram_gb", 0.0)
        d_vram = d.get("vram_delta_gb", 0.0)

        print("\n" + "=" * 90)
        print("               FACIAL ANALYSIS BENCHMARK: FINAL COMPARISON TABLE")
        print("=" * 90)
        print(f"| Evaluation Metric               | Zero-Shot Baseline | Fine-Tuned + Constrained | Relative Delta   |")
        print(f"| :------------------------------ | :----------------- | :----------------------- | :--------------- |")
        print(f"| **Schema Conformance Rate**     | {b_conf:6.2f}%            | {i_conf:6.2f}%                 | **{d_conf:+6.2f}%**        |")
        print(f"| **CelebA Attribute Macro-F1**   | {b_f1:6.4f}             | {i_f1:6.4f}                  | **{d_f1:+6.4f}**         |")
        print(f"| **Semantic Causal ROUGE-L F1**  | {b_rouge:6.4f}             | {i_rouge:6.4f}                  | **{d_rouge:+6.4f}**         |")
        print(f"| **Gender Disparity (|M - F|)**  | {b_gender:6.4f}             | {i_gender:6.4f}                  | {d_gender:+6.4f} (Parity) |")
        print(f"| **Age Disparity (|Y - O|)**     | {b_age:6.4f}             | {i_age:6.4f}                  | {d_age:+6.4f} (Parity) |")
        print(f"| **Causal Faithfulness Score**   | {b_faith:6.2f}%            | {i_faith:6.2f}%                 | {d_faith:+6.2f}%        |")
        print(f"| **Hallucination Rate**          | {b_halluc:6.2f}%            | {i_halluc:6.2f}%                 | **{d_halluc:+6.2f}%** (Lower) |")
        print(f"| **Latency Median (p50)**        | {b_p50:6.1f} ms           | {i_p50:6.1f} ms                | **{d_p50_pct:-6.2f}%**        |")
        print(f"| **Latency 95th Pct (p95)**      | {b_p95:6.1f} ms           | {i_p95:6.1f} ms                | {i_p95 - b_p95:+6.1f} ms        |")
        print(f"| **Peak VRAM Allocated**         | {b_vram:6.2f} GB           | {i_vram:6.2f} GB                | {d_vram:+6.2f} GB        |")
        print("=" * 90 + "\n")


# ==============================================================================
# Master Pipeline Execution
# ==============================================================================

def run_full_benchmark(
    config: SystemConfig = DEFAULT_CONFIG,
    test_sample_size: int = 150,
    train_sample_size: int = 400,
    val_sample_size: int = 100,
    epochs: int = 1,
    skip_training: bool = False,
) -> Dict[str, Any]:
    """
    Executes the autonomous benchmark pipeline across all 5 phases with full boundary serialization.
    """
    seed_everything(config.seed)
    logger.info("================================================================================")
    logger.info("AUTONOMOUS VLM FACIAL ATTRIBUTE & CAUSAL ANALYSIS BENCHMARK")
    logger.info(
        "Device: %s (Count: %d), Precision: %s, Compute: %s",
        config.hardware.device,
        config.hardware.num_gpus,
        config.hardware.torch_dtype,
        config.hardware.compute_dtype,
    )
    logger.info("================================================================================")

    orchestrator = BenchmarkPhaseOrchestrator(
        experiment_dir=config.paths.output_dir,
        config=config,
    )

    # -------------------------------------------------------------------------
    # Phase 1: Dataset Verification & Biometric Auditing
    # -------------------------------------------------------------------------
    p1_summary = orchestrator.run_phase_1(
        data_builder_fn=lambda: execute_phase1_dataset_verification(config)
    )

    # Prepare Held-out Test Dataset
    test_dataset = CelebAVLMDataset(
        image_dir=config.paths.images_dir,
        attr_file=config.paths.attr_file,
        landmarks_file=config.paths.landmarks_file,
        partition_file=config.paths.partition_file,
        split="test",
        sample_limit=test_sample_size,
        seed=config.seed,
    )
    logger.info("Test split initialized with %d samples.", len(test_dataset))

    # -------------------------------------------------------------------------
    # Phase 2: Zero-Shot Baseline Evaluation
    # -------------------------------------------------------------------------
    def execute_baseline(dataset: CelebAVLMDataset) -> EvaluationSummary:
        runner = ZeroShotBaselineRunner(config=config)
        runner.load_model()
        harness = EvaluationHarness(model_name="Baseline (Qwen2.5-VL-3B Zero-Shot)")
        summary = runner.run_evaluation(dataset=dataset, eval_harness=harness, max_samples=test_sample_size)
        del runner
        cleanup_gpu_memory()
        return summary

    orchestrator.run_phase_2(baseline_runner_fn=execute_baseline, test_split=test_dataset)
    cleanup_gpu_memory()

    # -------------------------------------------------------------------------
    # Phase 3: Resilient QLoRA Fine-Tuning
    # -------------------------------------------------------------------------
    def execute_training(num_epochs: int) -> Path:
        train_dataset = CelebAVLMDataset(
            image_dir=config.paths.images_dir,
            attr_file=config.paths.attr_file,
            landmarks_file=config.paths.landmarks_file,
            partition_file=config.paths.partition_file,
            split="train",
            sample_limit=train_sample_size,
            seed=config.seed,
        )
        val_dataset = CelebAVLMDataset(
            image_dir=config.paths.images_dir,
            attr_file=config.paths.attr_file,
            landmarks_file=config.paths.landmarks_file,
            partition_file=config.paths.partition_file,
            split="val",
            sample_limit=val_sample_size,
            seed=config.seed + 1,
        )
        trainer = QLoRATrainerPipeline(config=config)
        adapter_path = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            max_samples=train_sample_size,
            epochs=num_epochs,
        )
        del trainer
        cleanup_gpu_memory()
        return adapter_path

    if not skip_training:
        adapter_dir = orchestrator.run_phase_3(trainer_fn=execute_training, epochs=epochs)
    else:
        logger.info("[Phase 3] Skipping training upon user request.")
        adapter_dir = orchestrator.final_adapter

    cleanup_gpu_memory()

    # -------------------------------------------------------------------------
    # Phase 4: Improved Constrained Automaton Evaluation
    # -------------------------------------------------------------------------
    def execute_improved(dataset: CelebAVLMDataset) -> EvaluationSummary:
        runner = ConstrainedInferenceRunner(config=config, adapter_path=adapter_dir)
        runner.load_model()
        harness = EvaluationHarness(model_name="Improved (QLoRA + Constrained JSON)")
        summary = runner.run_evaluation(dataset=dataset, eval_harness=harness, max_samples=test_sample_size)
        del runner
        cleanup_gpu_memory()
        return summary

    orchestrator.run_phase_4(improved_runner_fn=execute_improved, test_split=test_dataset)
    cleanup_gpu_memory()

    # -------------------------------------------------------------------------
    # Phase 5: Synthesis & Comparative Reporting
    # -------------------------------------------------------------------------
    results = orchestrator.run_phase_5_synthesis()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous VLM Benchmark Suite")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=150,
        help="Number of held-out test samples (default: 150)",
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=400,
        help="Number of curated training samples for QLoRA (default: 400)",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=100,
        help="Number of validation auditing samples (default: 100)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Fine-tuning epochs (default: 1)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip QLoRA training phase and evaluate existing adapter",
    )
    args = parser.parse_args()

    run_full_benchmark(
        config=DEFAULT_CONFIG,
        test_sample_size=args.sample_size,
        train_sample_size=args.train_size,
        val_sample_size=args.val_size,
        epochs=args.epochs,
        skip_training=args.skip_train,
    )
