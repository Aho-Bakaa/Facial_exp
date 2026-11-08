# Hierarchical Facial Attribute & Causal Analysis with Multimodal VLMs

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ZeroGPU%20Space-blue)](https://huggingface.co/spaces/Anmolbakaa/facial-vlm-evaluator)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20Release-orange)](https://huggingface.co/datasets/Anmolbakaa/celeba-vlm-hierarchical-causal)
[![Kaggle Kernel](https://img.shields.io/badge/Kaggle-Live%20GPU%20Evaluator-20BEFF?logo=kaggle)](https://www.kaggle.com/code/goofybhatoora/celeb-vlm-live-evaluator)
[![Adapter Weights](https://img.shields.io/badge/Weights-Kaggle%20Adapter%20Dataset-success?logo=kaggle)](https://www.kaggle.com/datasets/goofybhatoora/celeb-vlm-trained-adapter)

An end-to-end multimodal Vision-Language Model framework that transforms legacy binary facial attribute classification into **anatomically grounded, Action Unit-driven hierarchical causal reasoning**.

Powered by fine-tuned **`Qwen2.5-VL-7B-Instruct`** using **4-bit NF4 QLoRA**, synchronized **Multi-Crop Region of Interest (ROI) conditioning**, and an **Evidence-First Autoregressive Visual Chain-of-Thought**.

---

## 🚀 Live Applications & Resources

| Resource | Description | Link |
| :--- | :--- | :--- |
| **Live Evaluator (HF Space)** | On-demand ZeroGPU evaluator running baseline vs fine-tuned comparison | [facial-vlm-evaluator](https://huggingface.co/spaces/Anmolbakaa/facial-vlm-evaluator) |
| **Curated Benchmark Dataset** | 400 train and 150 held-out evaluation samples with continuous biometrics | [celeba-vlm-hierarchical-causal](https://huggingface.co/datasets/Anmolbakaa/celeba-vlm-hierarchical-causal) |
| **Kaggle Live GPU Kernel** | Dual Tesla T4 interactive Gradio evaluator notebook with public URL | [celeb-vlm-live-evaluator](https://www.kaggle.com/code/goofybhatoora/celeb-vlm-live-evaluator) |
| **Pre-Trained QLoRA Adapter** | Serialized PEFT LoRA adapter weights (190.4 MB) | [celeb-vlm-trained-adapter](https://www.kaggle.com/datasets/goofybhatoora/celeb-vlm-trained-adapter) |

---

## 🧠 System Architecture & Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 1: Objective Anatomical Observations (FACS & Kinematics)         │
│ • Eye Dynamics (AU6/7), Gaze Vector, Eyebrow Dynamics (AU1+2/AU4)      │
│ • Mouth Dynamics (AU12/AU25/Neutral), Teeth Exposed (boolean)          │
│ • 3D Head Orientation (Yaw/Pitch/Roll symmetry)                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 2: Secondary Features & Physical Occlusions                     │
│ • Eyewear: None, Prescription Glasses, Sunglasses                      │
│ • Occlusions: Hands, Hair Strands, Ambient Shadows, Masks, None        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 3: Affective Expressions (Composite Semantics)                   │
│ • Primary Category: Neutral, Happiness, Sadness, Surprise, Anger...    │
│ • Intensity: Micro, Mild, Moderate, Pronounced, Extreme                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ LEVEL 4: Grounded Causal Reasoning (Visual Chain-of-Thought)           │
│ • visual_evidence: Strict physical Action Unit observations            │
│ • inferred_cause: Situational / environmental deduction                │
│ • confidence_score: Calibrated float bounded between 0.0 and 1.0       │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Technical Innovations

1. **4-Level Hierarchical Schema (`src/schema.py`)**:
   Replaces isolated binary flags with an anatomically grounded, composite semantic hierarchy validated via strict Pydantic models.
2. **Multi-Crop Region of Interest (ROI) Conditioning (`src/data_loader.py`)**:
   Instead of a single downscaled image, the vision transformer receives **three synchronized visual inputs**:
   - **Global Portrait** ($224 \times 224$): Captures head pose, global lighting, and hairstyle.
   - **Periorbital Crop** ($112 \times 112$): Captures ocular tension, eye squinting (AU6/7), and eyeglasses frames.
   - **Oral Crop** ($112 \times 112$): Captures commissure elevation (AU12), dental exposure, and lip parting (AU25).
3. **Evidence-First Anti-Hallucination Design (`src/prompt_builder.py`)**:
   Enforces emitting observable `visual_evidence` (Action Units) strictly before generating `inferred_cause`. Autoregressive attention forces the transformer to condition situational hypotheses on physical tokens.
4. **Resilient 4-Bit NF4 QLoRA Fine-Tuning (`src/train_qlora.py`)**:
   Adapts all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with paged 8-bit AdamW, double quantization, and gradient checkpointing, executing within 3.04 GB VRAM.
5. **Constrained Decoding (`src/inference.py`)**:
   The `JSONStructuralStopper` logits processor dynamically terminates generation as soon as the root JSON object balances, preventing infinite postambles without intrusive token-level logit masking.

---

## 📊 Experimental Results & Baseline Comparison

Benchmarked on **150 held-out Partition 2 test samples** drawn from the CelebA dataset. Partition 2 subjects feature **disjoint celebrity identities** from the training set (0% identity leakage).

| Metric Category | Metric | Zero-Shot Baseline (`Qwen2.5-VL-7B`) | Fine-Tuned + Multi-Crop (`Phase 4`) | Delta ($\Delta$) | Significance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Structural Validity** | **Schema Conformance** | 94.00% (141 / 150) | **100.00% (150 / 150)** | **+6.00%** | **Perfect RFC 8259 valid JSON (0 errors)** |
| **Attribute Detection** | **CelebA Macro-F1** | 0.3049 | **0.4772** | **+0.1723** | **+56.5% relative boost** |
| | • *Smiling F1* | 0.6486 (R=0.493) | **0.9079 (R=0.896)** | +0.2593 | **+39.9% gain in expression recall** |
| | • *Mouth Slightly Open F1* | 0.2474 (R=0.141) | **0.9024 (R=0.831)** | +0.6550 | **+264.7% surge; resolves lip parting failure** |
| | • *Eyeglasses F1* | 0.9333 (R=0.875) | **1.0000 (R=1.000)** | +0.0667 | **100% Precision and Recall** |
| **Causal Attribution** | **Mean Causal ROUGE-L** | 0.0616 | **0.1944** | **+0.1328** | **+215.6% relative semantic gain** |
| | • *Evidence ROUGE-L* | 0.0691 | **0.1877** | +0.1186 | Physical Action Unit grounding |
| | • *Cause ROUGE-L* | 0.0540 | **0.2011** | +0.1471 | Situational deduction |
| **Hallucination Control**| **Hallucination Rate** | 2.13% (3 / 141) | **0.00% (0 / 150)** | **-2.13%** | **Complete elimination of hallucinations** |
| | **Causal Faithfulness** | 99.30% | **100.00%** | +0.70% | Zero physical contradictions |
| **Spatial Kinematics** | **3D Pose Consistency** | 24.11% | **34.00%** | +9.89% | **+41.0% relative improvement** |
| **Demographic Fairness**| **Gender Disparity ($|M - F|$)**| 0.1104 | **0.0142** | **-0.0962** | **-87.2% disparity reduction (parity achieved)** |
| | **Age Disparity ($|Y - O|$)** | 0.0437 | **0.0279** | **-0.0158** | **-36.2% reduction in age gap** |

*For complete metric tables, confusion matrices, and subgroup breakdowns, see [BENCHMARKS.md](BENCHMARKS.md).*

---

## 📁 Repository Structure

```
Facial_exp/
├── src/                                  # Core modular system library
│   ├── schema.py                         # Pydantic 4-level taxonomy & JSON balanced parser
│   ├── data_loader.py                    # Multi-crop ROI extraction & landmark biometrics
│   ├── prompt_builder.py                 # Evidence-first multimodal chat formatting
│   ├── train_qlora.py                    # 4-bit NF4 QLoRA training engine
│   ├── eval_harness.py                   # Multi-dimensional benchmark evaluation engine
│   ├── inference.py                      # JSONStructuralStopper constrained decoding
│   ├── baseline.py                       # Zero-shot baseline evaluation runner
│   └── config.py                         # Environment detection & hardware paths
├── dataset_release/                      # Curated benchmark dataset
│   ├── train_400_curated.jsonl           # 400 curated training samples
│   ├── test_150_evaluation.jsonl         # 150 held-out evaluation samples
│   ├── dataset_manifest.csv              # Unified 550-sample metadata manifest
│   └── README.md                         # Dataset card and documentation
├── benchmark_results/                    # Audited serialized benchmark outputs
│   ├── benchmark_results.json            # Final cross-phase delta synthesis
│   ├── phase1_dataset_summary.json       # Dataset priors and landmark distributions
│   ├── phase2_baseline_results.json      # Full 150-sample baseline metrics
│   └── phase4_improved_results.json      # Full 150-sample fine-tuned test metrics
├── hf_space/                             # Hugging Face ZeroGPU deployment files
│   ├── app.py                            # Gradio 5.16 live interface with @spaces.GPU
│   ├── requirements.txt                  # Deployment dependencies
│   └── README.md                         # Space configuration metadata
├── kaggle_deployment/                    # Cloud execution notebooks
│   ├── celeb_vlm_benchmark.ipynb         # Full 5-phase benchmark pipeline notebook
│   └── run_live_evaluator_kaggle.ipynb   # Live evaluator Gradio notebook with share=True
├── streamlit_app/                        # Local interactive web evaluator
│   ├── app.py                            # Multi-tab Streamlit dashboard
│   ├── biometrics.py                     # OpenCV landmark & bounding box multi-crop module
│   ├── model_engine.py                   # Dual-mode inference execution engine
│   └── run_app.ps1                       # One-click Windows PowerShell launcher
├── run_benchmark.py                      # Master CLI orchestration pipeline
├── requirements.txt                      # Project dependencies
├── BENCHMARKS.md                         # Detailed benchmark and evaluation report
└── README.md                             # Project documentation
```

---

## 🛠️ Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/Aho-Bakaa/Facial_exp.git
cd Facial_exp
pip install -r requirements.txt
```

### 2. Local Interactive Dashboard

Run the Streamlit web dashboard to inspect images, multi-crop ROIs, and continuous biometrics:

```bash
streamlit run streamlit_app/app.py
```

Or via PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File .\streamlit_app\run_app.ps1
```

### 3. Run the Full Benchmark CLI

To run the complete 5-phase benchmark autonomously:

```bash
python run_benchmark.py --sample-size 150 --train-size 400 --val-size 100 --epochs 1
```

---

## 🔬 Experimentation & The LIMA Alignment Principle

Why fine-tuning on **400 curated samples** achieved strong generalization rather than overfitting???

1. **Foundation Model Priors**: Pre-trained 7B VLMs (`Qwen2.5-VL-7B-Instruct`) already have learned representations of facial anatomy from web-scale pre-training.
2. **Instruction Alignment vs Concept Learning**: The model does not need to learn what an eye or mouth is from scratch. It only needs to learn the **structured output format** and **Action Unit alignment**.
3. **Out-of-Distribution Validation**: All evaluations were conducted on **Partition 2** subjects (identities never seen during training). The +264.7% surge in mouth open detection and complete elimination of hallucinations confirm genuine generalization.

---

## 📜 Citation & License

This project is licensed under the Apache 2.0 License. The CelebA dataset is provided for non-commercial research purposes by the Chinese University of Hong Kong.
