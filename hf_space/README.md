---
title: Facial VLM Hierarchical Causal Analysis
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Multi-crop VLM for facial Action Units and causal reasoning
---

# Facial VLM: Hierarchical Facial Attribute & Causal Analysis

This Hugging Face Space runs **`Qwen/Qwen2.5-VL-7B-Instruct`** fine-tuned with 4-bit QLoRA on CelebA portraits.
It utilizes **ZeroGPU** to provide free on-demand GPU compute only when an image analysis is submitted.

### Key Capabilities:
- **Multi-Crop Region of Interest (ROI) Conditioning**: Processes full portrait (224x224), periorbital eye region (112x112), and oral mouth region (112x112).
- **Objective Anatomy Parsing**: Action Units (AU6, AU12, AU25), Gaze, Teeth Exposed, Head Pose.
- **Evidence-First Causal Attribution**: Enforces physical Action Unit visual evidence before inferring situational causes to eliminate ungrounded hallucinations.
- **Side-by-Side Model Comparison**: Compare the Baseline Zero-Shot model against the Fine-Tuned QLoRA model.
