"""
Prompt construction, taxonomy descriptions, system instructions, and VLM message formatting.
Enforces evidence-first Autoregressive Visual Chain-of-Thought (CoT) and Multi-Crop ROI conditioning.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union
from PIL import Image

SYSTEM_PROMPT = """You are an expert Autonomous Multimodal AI Scientist specialized in anatomical facial attribute extraction, Action Unit kinematic decoding, and grounded causal reasoning.
Analyze the provided human facial image (and its high-resolution periorbital and oral crops) and produce a rigorous, structured evaluation adhering strictly to the 4-level taxonomy.

### Taxonomy Hierarchy:
- Level 1: Anatomical & Kinematic Observations (Objective / Visible)
  * Eye Dynamics: "Open" | "Wide Open" | "Squinting (AU6/7)" | "Partially Closed" | "Closed" | "Winking"
  * Gaze Vector: "Direct (at camera)" | "Averted Left/Right" | "Upward" | "Downward" | "Occluded"
  * Eyebrow Dynamics: "Neutral" | "Raised (AU1+2)" | "Furrowed/Lowered (AU4)" | "Asymmetric"
  * Mouth & Jaw Dynamics:
    - movement: "Closed Relaxed" | "Closed Compressed" | "Lips Parted" | "Wide Open" | "Lip Corner Puller (AU12)" | "Lip Corner Depressor (AU15)"
    - teeth_exposed: boolean (true or false)
  * 3D Head Orientation: "Frontal" | "Three-Quarter Left/Right" | "Profile Left/Right" | "Tilted Up" | "Bowed Down"
- Level 2: Secondary Features & Occlusions
  * Eyewear: "None" | "Prescription Glasses" | "Tinted/Sunglasses"
  * Occlusions: array of ["Hands" | "Hair strands" | "Ambient Shadows" | "Masks" | "None"]
- Level 3: Affective Expressions (Composite Semantics)
  * Primary Category: "Neutral" | "Happiness/Amusement" | "Surprise" | "Contempt/Skeptical" | "Disgust" | "Anger" | "Sadness"
  * Intensity: "Micro" | "Mild" | "Moderate" | "Pronounced" | "Extreme"
- Level 4: Grounded Causal Attribution (Evidence-First Visual Chain-of-Thought)
  * visual_evidence: EXPLICIT pixel-grounded cues in the image (e.g., muscle contractions, folds, shadows, ocular tightening)
  * inferred_cause: physical, environmental, or situational cause logically deduced directly from the preceding visual evidence
  * confidence_score: float bounded between 0.0 and 1.0

CRITICAL RULES:
1. Output MUST be a single valid RFC 8259 JSON object with no additional markdown fences or commentary outside the JSON.
2. In Level 4, visual_evidence MUST be stated before inferred_cause to anchor causal deduction strictly to observable anatomy.
3. Every field must conform strictly to the specified enum values. Never hallucinate environmental factors absent from the visual evidence.
"""

USER_ANALYSIS_PROMPT = """Perform an exhaustive anatomical, kinematic, and causal analysis of this face.
Inspect the global portrait alongside the localized periorbital and oral crops.
Return the result strictly as a valid JSON object matching the 4-level schema:

{
  "level1_anatomy": {
    "eye_dynamics": "<EyeDynamics>",
    "gaze_vector": "<GazeVector>",
    "eyebrow_dynamics": "<EyebrowDynamics>",
    "mouth_and_jaw": {
      "movement": "<MouthMovement>",
      "teeth_exposed": <true|false>
    },
    "head_orientation": "<HeadOrientation>"
  },
  "level2_features": {
    "eyewear": "<Eyewear>",
    "occlusions": ["<OcclusionType>"]
  },
  "level3_affect": {
    "primary_category": "<PrimaryCategory>",
    "intensity": "<ExpressionIntensity>"
  },
  "level4_causal": {
    "visual_evidence": "<Pixel-grounded observations and Action Unit activations>",
    "inferred_cause": "<Logical situational/physical cause deduced from preceding evidence>",
    "confidence_score": <float 0.0-1.0>
  }
}"""


def format_multimodal_content(
    image_or_rois: Union[Image.Image, Dict[str, Image.Image]],
    prompt_text: str,
) -> List[Dict[str, Any]]:
    """
    Constructs interleaved multimodal content payloads.
    Supports either single global image or Multi-Crop ROIs (global + periorbital + oral).
    """
    content: List[Dict[str, Any]] = []
    if isinstance(image_or_rois, dict):
        if "global" in image_or_rois:
            content.append({"type": "image", "image": image_or_rois["global"]})
        if "periorbital" in image_or_rois:
            content.append({"type": "image", "image": image_or_rois["periorbital"]})
        if "oral" in image_or_rois:
            content.append({"type": "image", "image": image_or_rois["oral"]})
    elif isinstance(image_or_rois, Image.Image):
        content.append({"type": "image", "image": image_or_rois})
    else:
        raise ValueError(f"Unsupported image input type: {type(image_or_rois)}")

    content.append({"type": "text", "text": prompt_text})
    return content


def build_zero_shot_messages(
    image: Union[Image.Image, Dict[str, Image.Image]],
    user_prompt: str = USER_ANALYSIS_PROMPT,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Constructs zero-shot inference messages with multi-crop ROI interleaving."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": format_multimodal_content(image, user_prompt),
        },
    ]


def build_sft_conversation(
    image: Union[Image.Image, Dict[str, Image.Image]],
    target_json_str: str,
    user_prompt: str = USER_ANALYSIS_PROMPT,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Constructs SFT training messages with multi-crop ROI interleaving and evidence-first target."""
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": format_multimodal_content(image, user_prompt),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_json_str}],
        },
    ]
