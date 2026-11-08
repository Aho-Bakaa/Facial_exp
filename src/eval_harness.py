"""
Comprehensive statistical Evaluation Harness for Hierarchical Facial Attribute extraction,
Disaggregated Demographic Fairness, Semantic Causal Similarity (ROUGE-L),
Landmark Geometric Consistency, Counterfactual Faithfulness, and System Profiling.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from src.data_loader import (
    classify_geometric_head_orientation,
    compute_yaw_symmetry_ratio,
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
    safe_parse_json,
)

logger = logging.getLogger(__name__)


# ==============================================================================
# ROUGE-L Metric Computation (Longest Common Subsequence)
# ==============================================================================

def compute_lcs(a: List[str], b: List[str]) -> int:
    """Computes the length of Longest Common Subsequence between two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[m][n]


def compute_rouge_l(pred_text: str, ref_text: str) -> Dict[str, float]:
    """Computes token-level ROUGE-L precision, recall, and F1."""
    pred_tokens = pred_text.lower().split()
    ref_tokens = ref_text.lower().split()

    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs_len = compute_lcs(pred_tokens, ref_tokens)
    prec = lcs_len / len(pred_tokens)
    rec = lcs_len / len(ref_tokens)
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


# ==============================================================================
# Metric Data Structures
# ==============================================================================

@dataclass
class BinaryClassificationMetric:
    attribute_name: str
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    f1_score: float = 0.0
    balanced_accuracy: float = 0.0

    def compute(self) -> None:
        tp, fp = self.true_positives, self.false_positives
        fn, tn = self.false_negatives, self.true_negatives

        self.precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        self.recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        self.specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        if (self.precision + self.recall) > 0:
            self.f1_score = float(2 * (self.precision * self.recall) / (self.precision + self.recall))
        else:
            self.f1_score = 0.0

        self.balanced_accuracy = float(0.5 * (self.recall + self.specificity))


@dataclass
class EvaluationSummary:
    model_name: str
    total_samples: int = 0

    # 1. Syntactic & Structural Conformance
    valid_json_count: int = 0
    valid_json_rate: float = 0.0
    conformance_rate: float = 0.0  # Normalized [0, 1] for orchestrator tables
    complete_fields_count: int = 0
    field_completeness_rate: float = 0.0
    parsing_errors: List[str] = field(default_factory=list)

    # 2. Perceptual Alignment Metrics
    perceptual_metrics: Dict[str, BinaryClassificationMetric] = field(default_factory=dict)
    macro_f1: float = 0.0
    macro_precision: float = 0.0
    macro_recall: float = 0.0
    macro_balanced_accuracy: float = 0.0

    # 3. Disaggregated Demographic Fairness
    male_macro_f1: float = 0.0
    female_macro_f1: float = 0.0
    gender_disparity_delta: float = 0.0
    young_macro_f1: float = 0.0
    older_macro_f1: float = 0.0
    age_disparity_delta: float = 0.0

    # 4. Geometric Landmark & Head Pose Consistency
    pose_evaluated_samples: int = 0
    pose_exact_matches: int = 0
    pose_consistency_rate: float = 0.0
    mean_yaw_error: float = 0.0

    # 5. Semantic Causal Similarity (Level 4 Alignment)
    rouge_l_evidence_f1: float = 0.0
    rouge_l_cause_f1: float = 0.0
    mean_causal_rouge_f1: float = 0.0

    # 6. Counterfactual Faithfulness & Hallucination Audit
    causal_evaluated_samples: int = 0
    faithfulness_score: float = 0.0
    causal_faithfulness_ratio: float = 0.0
    hallucination_rate: float = 0.0
    hallucination_instances: List[Dict[str, Any]] = field(default_factory=list)

    # 7. System & Latency Profiling
    total_inference_time_sec: float = 0.0
    mean_latency_sec: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    mean_ttft_sec: float = 0.0
    tokens_per_second: float = 0.0
    peak_vram_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


# ==============================================================================
# Multi-Dimensional Evaluation Harness
# ==============================================================================

class EvaluationHarness:
    """
    Industry-grade multi-dimensional evaluation harness benchmarking:
      1. Syntactic JSON Schema Conformance
      2. Perceptual Ground-Truth Alignment (CelebA Attributes)
      3. Disaggregated Demographic Fairness Audit (Gender & Age)
      4. Geometric Head Pose & Landmark Ratio Consistency
      5. Semantic Causal Similarity (ROUGE-L Alignment)
      6. Counterfactual Faithfulness & Hallucination Detection
      7. System Profiling (p50/p95 Latency, VRAM, Throughput)
    """

    TARGET_ATTRIBUTES = [
        "Narrow_Eyes",
        "Smiling",
        "Eyeglasses",
        "Arched_Eyebrows",
        "Mouth_Slightly_Open",
        "Bangs",
    ]

    def __init__(self, model_name: str = "Baseline-VLM") -> None:
        self.model_name = model_name
        self.summary = EvaluationSummary(model_name=model_name)

        # Main perceptual metrics
        for attr in self.TARGET_ATTRIBUTES:
            self.summary.perceptual_metrics[attr] = BinaryClassificationMetric(attribute_name=attr)

        # Demographic subgroups
        self._male_metrics: Dict[str, BinaryClassificationMetric] = {
            attr: BinaryClassificationMetric(attribute_name=attr) for attr in self.TARGET_ATTRIBUTES
        }
        self._female_metrics: Dict[str, BinaryClassificationMetric] = {
            attr: BinaryClassificationMetric(attribute_name=attr) for attr in self.TARGET_ATTRIBUTES
        }
        self._young_metrics: Dict[str, BinaryClassificationMetric] = {
            attr: BinaryClassificationMetric(attribute_name=attr) for attr in self.TARGET_ATTRIBUTES
        }
        self._older_metrics: Dict[str, BinaryClassificationMetric] = {
            attr: BinaryClassificationMetric(attribute_name=attr) for attr in self.TARGET_ATTRIBUTES
        }

        self._yaw_errors: List[float] = []
        self._faithfulness_scores: List[float] = []
        self._evidence_rouge_scores: List[float] = []
        self._cause_rouge_scores: List[float] = []
        self._latencies: List[float] = []
        self._ttfts: List[float] = []
        self._token_counts: List[int] = []

    def record_prediction(
        self,
        prediction: Optional[FacialAnalysisOutput],
        ground_truth: FacialAnalysisOutput,
        latency_sec: float = 0.0,
        ttft_sec: float = 0.0,
        generated_tokens: int = 0,
        peak_vram_gb: float = 0.0,
        raw_prediction_text: str = "",
        is_male: Optional[bool] = None,
        is_young: Optional[bool] = None,
        image_name: str = "sample",
    ) -> None:
        """Records a single prediction sample and performs multi-dimensional evaluations."""
        self.summary.total_samples += 1
        self._latencies.append(latency_sec)
        self._ttfts.append(ttft_sec)
        self._token_counts.append(generated_tokens)
        if peak_vram_gb > self.summary.peak_vram_gb:
            self.summary.peak_vram_gb = peak_vram_gb

        if prediction is None:
            self.summary.parsing_errors.append(f"{image_name}: Failed JSON parsing")
            return

        self.summary.valid_json_count += 1
        if self._check_field_completeness(prediction):
            self.summary.complete_fields_count += 1

        # 1. Perceptual Ground-Truth Alignment
        gt_attrs = self._extract_attributes_from_schema(ground_truth)
        self._evaluate_perceptual_alignment(
            prediction,
            gt_attrs,
            is_male=is_male,
            is_young=is_young,
        )

        # 2. Geometric Head Pose Consistency
        if ground_truth.level1_anatomy.head_orientation:
            self.summary.pose_evaluated_samples += 1
            if prediction.level1_anatomy.head_orientation == ground_truth.level1_anatomy.head_orientation:
                self.summary.pose_exact_matches += 1

        # 3. Semantic Causal Similarity (ROUGE-L)
        if ground_truth.level4_causal:
            ev_rouge = compute_rouge_l(
                prediction.level4_causal.visual_evidence,
                ground_truth.level4_causal.visual_evidence,
            )
            cause_rouge = compute_rouge_l(
                prediction.level4_causal.inferred_cause,
                ground_truth.level4_causal.inferred_cause,
            )
            self._evidence_rouge_scores.append(ev_rouge["f1"])
            self._cause_rouge_scores.append(cause_rouge["f1"])

        # 4. Counterfactual Faithfulness & Hallucination Audit
        self._evaluate_causal_faithfulness(image_name, prediction, gt_attrs)

    def evaluate_sample(
        self,
        image_name: str,
        raw_model_output: str,
        ground_truth_attrs: Dict[str, bool],
        landmarks: Optional[Dict[str, float]] = None,
        latency_sec: float = 0.0,
        ttft_sec: float = 0.0,
        generated_tokens: int = 0,
        vram_allocated_gb: float = 0.0,
    ) -> Tuple[bool, Optional[FacialAnalysisOutput]]:
        """Legacy compatibility wrapper."""
        parsed_output, parse_err = safe_parse_json(raw_model_output)
        from src.data_loader import map_celeba_to_taxonomy
        dummy_gt = map_celeba_to_taxonomy(ground_truth_attrs, landmarks)

        self.record_prediction(
            prediction=parsed_output,
            ground_truth=dummy_gt,
            latency_sec=latency_sec,
            ttft_sec=ttft_sec,
            generated_tokens=generated_tokens,
            peak_vram_gb=vram_allocated_gb,
            raw_prediction_text=raw_model_output,
            is_male=ground_truth_attrs.get("Male"),
            is_young=ground_truth_attrs.get("Young"),
            image_name=image_name,
        )
        return (parsed_output is not None), parsed_output

    def _extract_attributes_from_schema(self, schema: FacialAnalysisOutput) -> Dict[str, bool]:
        return {
            "Narrow_Eyes": schema.level1_anatomy.eye_dynamics == EyeDynamics.SQUINTING,
            "Smiling": (
                schema.level1_anatomy.mouth_and_jaw.movement == MouthMovement.LIP_CORNER_PULLER
                or schema.level3_affect.primary_category == PrimaryCategory.HAPPINESS_AMUSEMENT
            ),
            "Eyeglasses": schema.level2_features.eyewear in [
                Eyewear.PRESCRIPTION_GLASSES,
                Eyewear.TINTED_SUNGLASSES,
            ],
            "Arched_Eyebrows": schema.level1_anatomy.eyebrow_dynamics == EyebrowDynamics.RAISED,
            "Mouth_Slightly_Open": (
                schema.level1_anatomy.mouth_and_jaw.movement in [
                    MouthMovement.LIPS_PARTED,
                    MouthMovement.WIDE_OPEN,
                ]
                or schema.level1_anatomy.mouth_and_jaw.teeth_exposed
            ),
            "Bangs": OcclusionType.HAIR_STRANDS in schema.level2_features.occlusions,
        }

    def _check_field_completeness(self, output: FacialAnalysisOutput) -> bool:
        try:
            return (
                output.level1_anatomy is not None
                and output.level2_features is not None
                and output.level3_affect is not None
                and output.level4_causal is not None
                and len(output.level4_causal.visual_evidence) > 0
                and len(output.level4_causal.inferred_cause) > 0
            )
        except Exception:
            return False

    def _evaluate_perceptual_alignment(
        self,
        pred: FacialAnalysisOutput,
        gt_attrs: Dict[str, bool],
        is_male: Optional[bool] = None,
        is_young: Optional[bool] = None,
    ) -> None:
        pred_map = self._extract_attributes_from_schema(pred)

        for attr, pred_val in pred_map.items():
            if attr not in gt_attrs:
                continue
            gt_val = gt_attrs[attr]

            # Overall metric update
            self._update_metric(self.summary.perceptual_metrics[attr], pred_val, gt_val)

            # Subgroup metrics for Demographic Fairness Audit
            if is_male is True:
                self._update_metric(self._male_metrics[attr], pred_val, gt_val)
            elif is_male is False:
                self._update_metric(self._female_metrics[attr], pred_val, gt_val)

            if is_young is True:
                self._update_metric(self._young_metrics[attr], pred_val, gt_val)
            elif is_young is False:
                self._update_metric(self._older_metrics[attr], pred_val, gt_val)

    def _update_metric(self, metric: BinaryClassificationMetric, pred_val: bool, gt_val: bool) -> None:
        if pred_val and gt_val:
            metric.true_positives += 1
        elif pred_val and not gt_val:
            metric.false_positives += 1
        elif not pred_val and gt_val:
            metric.false_negatives += 1
        else:
            metric.true_negatives += 1

    def _evaluate_causal_faithfulness(
        self,
        image_name: str,
        pred: FacialAnalysisOutput,
        gt_attrs: Dict[str, bool],
    ) -> None:
        self.summary.causal_evaluated_samples += 1
        causal = pred.level4_causal
        evidence_text = causal.visual_evidence.lower()

        contradictions: List[str] = []

        # Contradiction: Eyeglasses hallucination
        gt_glasses = gt_attrs.get("Eyeglasses", False)
        mentions_glasses = any(k in evidence_text for k in ["glasses", "spectacle", "lens", "eyewear"])
        if mentions_glasses and not gt_glasses:
            contradictions.append("Hallucinated glasses in visual evidence when subject has none.")

        # Contradiction: Smile hallucination
        gt_smiling = gt_attrs.get("Smiling", False)
        mentions_active_smile = (
            ("smile" in evidence_text and "no smile" not in evidence_text and "absence of smile" not in evidence_text)
            or ("zygomaticus major contraction" in evidence_text or "au12" in evidence_text)
        )
        if mentions_active_smile and not gt_smiling:
            contradictions.append("Hallucinated active smile / AU12 contraction when subject is not smiling.")

        # Contradiction: Squint hallucination
        gt_narrow = gt_attrs.get("Narrow_Eyes", False)
        mentions_active_squint = (
            ("squinting" in evidence_text and "not squinting" not in evidence_text)
            or ("orbicularis oculi pars orbitalis" in evidence_text or "au6" in evidence_text)
        )
        if mentions_active_squint and not gt_narrow:
            contradictions.append("Hallucinated eye squinting when eyes are open.")

        sample_score = max(0.0, 1.0 - len(contradictions) * 0.33)
        self._faithfulness_scores.append(sample_score)

        if contradictions:
            self.summary.hallucination_instances.append({
                "image_name": image_name,
                "contradictions": contradictions,
                "visual_evidence": causal.visual_evidence,
                "inferred_cause": causal.inferred_cause,
                "confidence_score": causal.confidence_score,
            })

    def compute_summary(self) -> EvaluationSummary:
        """Alias for finalize."""
        return self.finalize()

    def finalize(self) -> EvaluationSummary:
        """Computes aggregate macro-averages, demographic disparities, and benchmark statistics."""
        s = self.summary
        total = max(1, s.total_samples)

        s.valid_json_rate = float(s.valid_json_count / total * 100.0)
        s.conformance_rate = float(s.valid_json_count / total)
        s.field_completeness_rate = float(s.complete_fields_count / total * 100.0)

        # 1. Main Perceptual Macro-Averages
        f1_list, prec_list, rec_list, bal_acc_list = [], [], [], []
        for metric in s.perceptual_metrics.values():
            metric.compute()
            f1_list.append(metric.f1_score)
            prec_list.append(metric.precision)
            rec_list.append(metric.recall)
            bal_acc_list.append(metric.balanced_accuracy)

        s.macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
        s.macro_precision = float(np.mean(prec_list)) if prec_list else 0.0
        s.macro_recall = float(np.mean(rec_list)) if rec_list else 0.0
        s.macro_balanced_accuracy = float(np.mean(bal_acc_list)) if bal_acc_list else 0.0

        # 2. Disaggregated Demographic Fairness
        for m in self._male_metrics.values(): m.compute()
        for m in self._female_metrics.values(): m.compute()
        for m in self._young_metrics.values(): m.compute()
        for m in self._older_metrics.values(): m.compute()

        male_f1s = [m.f1_score for m in self._male_metrics.values()]
        female_f1s = [m.f1_score for m in self._female_metrics.values()]
        young_f1s = [m.f1_score for m in self._young_metrics.values()]
        older_f1s = [m.f1_score for m in self._older_metrics.values()]

        s.male_macro_f1 = float(np.mean(male_f1s)) if male_f1s else s.macro_f1
        s.female_macro_f1 = float(np.mean(female_f1s)) if female_f1s else s.macro_f1
        s.gender_disparity_delta = float(abs(s.male_macro_f1 - s.female_macro_f1))

        s.young_macro_f1 = float(np.mean(young_f1s)) if young_f1s else s.macro_f1
        s.older_macro_f1 = float(np.mean(older_f1s)) if older_f1s else s.macro_f1
        s.age_disparity_delta = float(abs(s.young_macro_f1 - s.older_macro_f1))

        # 3. Geometric Consistency
        if s.pose_evaluated_samples > 0:
            s.pose_consistency_rate = float(s.pose_exact_matches / s.pose_evaluated_samples * 100.0)

        # 4. Semantic Causal Similarity
        if self._evidence_rouge_scores:
            s.rouge_l_evidence_f1 = float(np.mean(self._evidence_rouge_scores))
        if self._cause_rouge_scores:
            s.rouge_l_cause_f1 = float(np.mean(self._cause_rouge_scores))
        s.mean_causal_rouge_f1 = float((s.rouge_l_evidence_f1 + s.rouge_l_cause_f1) / 2.0)

        # 5. Causal Faithfulness & Hallucination Rate
        if s.causal_evaluated_samples > 0:
            s.faithfulness_score = float(np.mean(self._faithfulness_scores) * 100.0)
            s.causal_faithfulness_ratio = float(s.faithfulness_score / 100.0)
            hallucinated_count = len(s.hallucination_instances)
            s.hallucination_rate = float(hallucinated_count / s.causal_evaluated_samples * 100.0)

        # 6. Latency & System Profiling
        s.total_inference_time_sec = float(np.sum(self._latencies))
        s.mean_latency_sec = float(np.mean(self._latencies)) if self._latencies else 0.0
        if self._latencies:
            latencies_ms = [lat * 1000.0 for lat in self._latencies]
            s.p50_latency_ms = float(np.percentile(latencies_ms, 50))
            s.p95_latency_ms = float(np.percentile(latencies_ms, 95))
        s.mean_ttft_sec = float(np.mean(self._ttfts)) if self._ttfts else 0.0

        total_tokens = np.sum(self._token_counts)
        if s.total_inference_time_sec > 0:
            s.tokens_per_second = float(total_tokens / s.total_inference_time_sec)

        return s
