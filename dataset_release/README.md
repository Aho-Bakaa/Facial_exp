# CelebA-VLM: Hierarchical Facial Attribute & Causal Grounding Dataset

This dataset release contains the exact training and held-out evaluation splits used to fine-tune and benchmark **`Qwen2.5-VL-7B-Instruct`** for **Hierarchical Facial Attribute & Causal Analysis**.

---

## 1. Dataset Overview

- **Source Dataset**: Real CelebFaces Attributes Dataset (CelebA, 202,599 images across 10,177 identities) fused with official 5-point facial landmarks (`list_landmarks_align_celeba.txt`) and CelebA Dialog causal captions.
- **Strict Identity Disjoint Partitioning**: Evaluated strictly on **Partition 2** (held-out test split of 19,962 images). The test subjects share **0% overlapping identities** with Partition 0 (training), guaranteeing zero identity leakage.
- **Dataset Files**:
  - `train_400_curated.jsonl`: 400 training samples with continuous biometrics, Action Units, and 4-level ground-truth targets.
  - `test_150_evaluation.jsonl`: 150 held-out test samples with ground-truth targets.
  - `dataset_manifest.csv`: Unified manifest detailing all 550 images with metadata flags and biometric measurements.

---

## 2. Demographic & Pose Stratification

### 2.1 Training Split (400 Curated Samples)
- **Gender Balance**: 61.25% Female (245), 38.75% Male (155) — mirrors CelebA's natural $\approx 58\% / 42\%$ distribution.
- **Age Representation**: 79.0% Young (316), 21.0% Older (84) — matches CelebA's $77.4\% / 22.6\%$ prior.
- **Expression Equilibrium**: 48.0% Smiling (192) vs 52.0% Non-Smiling (208) — prevents majority-class bias.
- **Pose Diversity**: Bowed Down (28%), Profile (20%), Three-Quarter (25%), Frontal (24%), Tilted Up (3%).

### 2.2 Test Split (150 Held-Out Samples)
- **Identity Split**: 100% Partition 2 (Out-of-distribution celebrity identities).
- **Smiling Distribution**: 51.3% Smiling (77), 48.7% Non-Smiling (73).
- **Gender Distribution**: 44.0% Male (66), 56.0% Female (84).
- **Mouth Aperture**: 59.3% Mouth Slightly Open (89), 40.7% Closed (61).
- **Eyewear**: 7.3% Glasses (11), 92.7% No Glasses (139).

---

## 3. Schema & Feature Definition

Each JSONL record adheres strictly to the RFC 8259-compliant 4-level taxonomy:

```json
{
  "image_name": "023482.jpg",
  "split": "train",
  "partition_id": 0,
  "raw_attributes": {
    "Smiling": true,
    "Male": false,
    "Young": true,
    "Eyeglasses": false,
    "Mouth_Slightly_Open": true
  },
  "landmarks": {
    "lefteye_x": 69.0,
    "lefteye_y": 109.0,
    "righteye_x": 107.0,
    "righteye_y": 107.0,
    "nose_x": 89.0,
    "nose_y": 129.0,
    "leftmouth_x": 73.0,
    "leftmouth_y": 152.0,
    "rightmouth_x": 106.0,
    "rightmouth_y": 151.0
  },
  "continuous_geometry": {
    "mouth_aspect_ratio": 2.14,
    "eye_aspect_ratio": 0.22,
    "head_roll_deg": -2.95,
    "head_orientation": "Frontal"
  },
  "structured_target": {
    "level1_anatomy": {
      "eye_dynamics": "Open",
      "gaze_vector": "Direct",
      "eyebrow_dynamics": "Neutral",
      "mouth_dynamics": "AU12 (Lip Corner Puller) Contraction",
      "teeth_exposed": true,
      "head_orientation": "Frontal"
    },
    "level2_features": {
      "eyewear": "None",
      "physical_occlusions": "None"
    },
    "level3_affect": {
      "primary_expression": "Happiness / Amusement",
      "intensity": "Moderate",
      "secondary_affect": "Engagement"
    },
    "level4_causal": {
      "visual_evidence": "Bilateral contraction of zygomaticus major elevating oral commissures...",
      "inferred_cause": "Positive interpersonal reaction or photographic posing demeanor...",
      "confidence_score": 0.94
    }
  }
}
```

---

## 4. Why This Exact Fragment Was Selected (The LIMA Principle)

1. **Alignment vs. Scratch Training**: `Qwen2.5-VL-7B` is already pre-trained on billions of multimodal tokens and understands general human faces.
2. **The LIMA Paradigm** (*Less Is More for Alignment*, Zhou et al., NeurIPS 2023): Large multimodal models require only a few hundred curated, noise-free instruction demonstrations to master complex structured schemas without suffering **catastrophic forgetting**.
3. **Multi-Scale Visual Tokens**: Each sample consists of 3 distinct high-resolution image crops (Global $224\times 224$ + Periorbital $112\times 112$ + Oral $112\times 112$). Training across 400 curated samples corresponds to 1,200 distinct vision token embeddings, achieving high convergence (Validation Loss: **0.1076**) on dual Tesla T4 GPUs.
