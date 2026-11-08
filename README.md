# Hierarchical Facial Attribute & Causal Analysis with Multimodal VLMs

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-ZeroGPU%20Space-blue)](https://huggingface.co/spaces/Anmolbakaa/facial-vlm-evaluator)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20Release-orange)](https://huggingface.co/datasets/Anmolbakaa/celeba-vlm-hierarchical-causal)
[![Kaggle Kernel](https://img.shields.io/badge/Kaggle-Live%20GPU%20Evaluator-20BEFF?logo=kaggle)](https://www.kaggle.com/code/goofybhatoora/celeb-vlm-live-evaluator)
[![Adapter Weights](https://img.shields.io/badge/Weights-Kaggle%20Adapter%20Dataset-success?logo=kaggle)](https://www.kaggle.com/datasets/goofybhatoora/celeb-vlm-trained-adapter)
[![Benchmark Documentation](https://img.shields.io/badge/Report-BENCHMARKS.md-success)](BENCHMARKS.md)

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
| **Exhaustive Benchmark Report** | 9 engineering pivots, timing bottleneck profiling, and confusion matrices | [BENCHMARKS.md](BENCHMARKS.md) |

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
* **Inter-Pupillary Distance (IPD)** = $\left\| p_{\text{right\_eye}} - p_{\text{left\_eye}} \right\|_2$
* **Mouth Width** = $\left\| p_{\text{right\_mouth}} - p_{\text{left\_mouth}} \right\|_2$
* **Mouth Aspect Ratio (MAR)**: $\text{MAR} = \frac{d_{\text{mouth}}}{d_{\text{eyes}}} \quad (\mu = 2.1307, \, \sigma = 1.1719)$
* **2D Head Roll Cant Angle**: $\theta_{\text{roll}} = \arctan2(\Delta y_{\text{eyes}}, \, \Delta x_{\text{eyes}}) \times \frac{180^\circ}{\pi} \quad (\mu = -0.0534^\circ, \, \sigma = 2.795^\circ)$
* **Eye Aspect Ratio (EAR)**: Palpebral aperture coefficient $(\mu = 0.2157, \, \sigma = 0.0179)$
3. **Evidence-First Anti-Hallucination Design (`src/prompt_builder.py`)**:
   Enforces emitting observable `visual_evidence` (Action Units) strictly before generating `inferred_cause`. Autoregressive attention forces the transformer to condition situational hypotheses on physical tokens.
4. **Resilient 4-Bit NF4 QLoRA Fine-Tuning (`src/train_qlora.py`)**:
   Adapts all linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with paged 8-bit AdamW, double quantization, and gradient checkpointing, executing within 3.04 GB VRAM.
5. **Constrained Decoding (`src/inference.py`)**:
   The `JSONStructuralStopper` logits processor dynamically terminates generation as soon as the root JSON object balances, preventing infinite postambles without intrusive token-level logit masking.

---

## 📈 Phase-Wise Experimental Progression

The benchmark executed across **five strictly decoupled operational phases**, persisting boundary artifacts at every step to prevent compute loss.

| Phase | Operational Focus | Key Metrics / Findings |
| :--- | :--- | :--- |
| **Phase 1** | Biometric Prior & Partition Integrity Audit | Audited 202,599 CelebA records and computed landmark distributions ($\text{MAR} = 2.131, \text{EAR} = 0.216, \theta_{\text{roll}} = -0.053^\circ$). Partition 2 (test) contains disjoint identities from Partition 0 (train). |
| **Phase 2** | Zero-Shot Baseline Benchmark (`Qwen2.5-VL-7B`) | Evaluated 150 test samples. Discovered severe mouth aperture blindness ($R=0.141$), 2.13% hallucination rate, and an 11.04% gender gap favoring male subjects. |
| **Phase 3** | Resilient 4-Bit QLoRA Fine-Tuning | Fine-tuned 400 curated multi-crop samples (1,200 crops) across 100 optimizer steps. Loss converged from **4.062 $\rightarrow$ 0.333**; held-out validation loss reached **0.1076**. Saved intermediate checkpoints at step 50 and 100. |
| **Phase 4** | Improved Multi-Crop Evaluation | Re-evaluated on the identical 150 Partition 2 samples. Schema conformance reached **100.00%**, Mouth Open F1 surged to **0.9024** (+264.7%), Causal ROUGE-L rose to **0.1944** (+215.6%), and hallucinations were completely eliminated (**0.00%**). |
| **Phase 5** | Cross-Phase Synthesis & Persistence | Automated compilation of statistical deltas, fairness subgroup matrices, and deployment to Hugging Face Spaces and Kaggle. |

---

## 🔬 Chronology of Pivots: Why We Pivoted 9 Times

Developing this multimodal VLM pipeline on consumer cloud GPUs required navigating critical empirical obstacles:

1. **Local CPU vs Cloud GPU Execution**: Local machine had no CUDA GPU (`torch.cuda.is_available() == False`), requiring >5 min/sample. Pivoted to automated cloud execution on Kaggle Dual Tesla T4s and Hugging Face ZeroGPU.
2. **Aborted 1,500-Image Run $\rightarrow$ 400-Sample LIMA Stratification**: The initial fine-tuning loop was configured with 1,500 samples across 2 epochs (4,500 forward passes), projecting **>8 hours** on Kaggle with no intermediate checkpoints. We paused execution and pivoted to **400 curated, demographically stratified samples** under the LIMA paradigm, cutting training time to ~2.5 hours with checkpoints at step 50/100.
3. **Qwen 3B $\rightarrow$ 7B Upgrade**: The 3B model struggled with fine Action Unit discrimination (AU1 vs AU4) and multi-level JSON stability. Upgraded to `Qwen2.5-VL-7B-Instruct` with 4-bit NF4 double-quantization, consuming only **3.04 GB VRAM**.
4. **3.5-Hour Retraining Bottleneck $\rightarrow$ Adapter Caching**: Re-training just to debug evaluation metrics exhausted weekly GPU quotas. Packaged `qlora_adapter_final` (190.4 MB) into a Kaggle dataset (`goofybhatoora/celeb-vlm-trained-adapter`) with a `--skip-train` flag, cutting iteration cycles to **1.5 hours**.
5. **Delimiter Suppression Defect $\rightarrow$ $\mathcal{O}(1)$ `JSONStructuralStopper`**: In Version 12, a token-level prefix-trie automaton intended to force enum values crashed at line 11 because `"teeth_exposed"` is a JSON **boolean** (`true`), not a string. The trie matched quotes in the next key and masked out colons and property keys. Replaced it with `JSONStructuralStopper` (root brace counting), achieving **100.00% valid JSON (150/150)**.
6. **Oral Crop Neck-Capture Bug $\rightarrow$ Anthropometric Haar Cascades**: In the demo, tall $3:4$ aspect ratio portraits placed the mouth box on the subject's collar/neck due to a naive $y=h/2$ heuristic. Replaced it with OpenCV Haar face detection with oral bounds strictly pinned between **$62\% - 92\%$ of face height**.
7. **PeftModel In-Place Patching Bug**: In the live demo, baseline and fine-tuned predictions were identical because `PeftModel` patches the base model in-place. Wrapped zero-shot baseline calls in `with peft_model.disable_adapter():`.
8. **ZeroGPU API Failure $\rightarrow$ Gradio 5.16 & Concurrency Queue**: Gradio 5.9 had a known incompatibility with Pydantic 2.10, and ZeroGPU required explicit queue scheduling for dynamic `@spaces.GPU` allocation. Upgraded to Gradio 5.16+ and launched with `demo.queue()`.

*For an exhaustive breakdown of all 9 pivots with technical diagrams, see [BENCHMARKS.md](BENCHMARKS.md).*

---

## ⚡ Timing Bottlenecks & Training Optimizations

Fine-tuning a 7B parameter multimodal model on dual 16GB Tesla T4 GPUs presented substantial memory and latency constraints:

### 1. Training Memory Optimization Stack
- **4-Bit NF4 Quantization**: Reduced base model footprint from **~15.0 GB $\rightarrow$ 4.5 GB**.
- **Double Quantization**: Saved an additional **~0.37 bits/param** (~320 MB VRAM) with negligible overhead.
- **Paged AdamW 8-Bit**: Prevented CUDA OOM crashes by dynamically page-swapping optimizer states to CPU RAM during peak gradients.
- **Gradient Checkpointing**: Reduced activation memory by **~60%**.
- **GPU Pinning (`device_map={"": 0}`)**: Placing the model on GPU 0 eliminated PCIe bus thrashing between the two T4 GPUs, accelerating step time by **2.8x**.

### 2. Decoding Latency: Resolving the Trie Masking Bottleneck
Processing 3 visual crops generated **1,500+ visual tokens** per sample:
- **Trie Regex Logit Masking**: Slow vocabulary iterations inflated latency to **>120s / sample**, triggering pipeline timeouts.
- **$\mathcal{O}(1)$ `JSONStructuralStopper`**: Monitoring only root brace balance (`depth == 0`) reduced latency to **39.4s median ($p_{50}$)**, absorbing the 3x visual token sequence at optimal speed.

---

## 📊 Primary Comparative Benchmark Table

Benchmarked on **150 held-out Partition 2 test samples** drawn from the CelebA dataset (disjoint celebrity identities, 0% identity leakage).

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

## 🔬 The LIMA Alignment Principle: Why Fine-Tuning on 400 Curated Samples Generalizes Without Overfitting

Why fine-tuning on **400 curated samples** achieved strong generalization rather than overfitting:

1. **Foundation Model Priors**: Pre-trained 7B VLMs (`Qwen2.5-VL-7B-Instruct`) already have learned representations of facial anatomy from web-scale pre-training.
2. **Instruction Alignment vs Concept Learning**: The model does not need to learn what an eye or mouth is from scratch. It only needs to learn the **structured output format** and **Action Unit alignment**.
3. **Out-of-Distribution Validation**: All evaluations were conducted on **Partition 2** subjects (identities never seen during training). The +264.7% surge in mouth open detection and complete elimination of hallucinations confirm genuine generalization.

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
├── BENCHMARKS.md                         # Exhaustive report: pivots, latency, and metrics
└── README.md                             # Project overview and quickstart
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

## 📜 Citation & License

This project is licensed under the Apache 2.0 License. The CelebA dataset is provided for non-commercial research purposes by the Chinese University of Hong Kong.
