"""
Strict Pydantic models and enumeration definitions for the Hierarchical Facial Attribute
and Grounded Causal Attribution taxonomy.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# Level 1: Anatomical & Kinematic Observations (Objective / Visible)
# ==============================================================================

class EyeDynamics(str, Enum):
    OPEN = "Open"
    WIDE_OPEN = "Wide Open"
    SQUINTING = "Squinting (AU6/7)"
    PARTIALLY_CLOSED = "Partially Closed"
    CLOSED = "Closed"
    WINKING = "Winking"


class GazeVector(str, Enum):
    DIRECT = "Direct (at camera)"
    AVERTED_LEFT_RIGHT = "Averted Left/Right"
    UPWARD = "Upward"
    DOWNWARD = "Downward"
    OCCLUDED = "Occluded"


class EyebrowDynamics(str, Enum):
    NEUTRAL = "Neutral"
    RAISED = "Raised (AU1+2)"
    FURROWED_LOWERED = "Furrowed/Lowered (AU4)"
    ASYMMETRIC = "Asymmetric"


class MouthMovement(str, Enum):
    CLOSED_RELAXED = "Closed Relaxed"
    CLOSED_COMPRESSED = "Closed Compressed"
    LIPS_PARTED = "Lips Parted"
    WIDE_OPEN = "Wide Open"
    LIP_CORNER_PULLER = "Lip Corner Puller (AU12)"
    LIP_CORNER_DEPRESSOR = "Lip Corner Depressor (AU15)"


class HeadOrientation(str, Enum):
    FRONTAL = "Frontal"
    THREE_QUARTER_LEFT_RIGHT = "Three-Quarter Left/Right"
    PROFILE_LEFT_RIGHT = "Profile Left/Right"
    TILTED_UP = "Tilted Up"
    BOWED_DOWN = "Bowed Down"


class MouthAndJawDynamics(BaseModel):
    movement: MouthMovement = Field(
        ...,
        description="Dynamic movement or posture of the mouth and lips."
    )
    teeth_exposed: bool = Field(
        ...,
        description="Whether upper or lower dentition is visually observable."
    )


class Level1Anatomical(BaseModel):
    eye_dynamics: EyeDynamics = Field(
        ...,
        description="Eyelid aperture and orbicularis oculi contraction."
    )
    gaze_vector: GazeVector = Field(
        ...,
        description="Pupillary trajectory and gaze fixation direction."
    )
    eyebrow_dynamics: EyebrowDynamics = Field(
        ...,
        description="Corrugator supercilii and frontalis muscle dynamics."
    )
    mouth_and_jaw: MouthAndJawDynamics = Field(
        ...,
        description="Lip posture and dental visibility."
    )
    head_orientation: HeadOrientation = Field(
        ...,
        description="Estimated 3D head pose orientation relative to camera."
    )
    mouth_aspect_ratio: Optional[float] = Field(
        default=None,
        description="Empirical mouth-aspect ratio (MAR) computed from commissure and subnasale landmarks."
    )
    head_roll_deg: Optional[float] = Field(
        default=None,
        description="Empirical 2D facial roll tilt in degrees derived from ocular cant."
    )


# ==============================================================================
# Level 2: Secondary Features & Occlusions
# ==============================================================================

class Eyewear(str, Enum):
    NONE = "None"
    PRESCRIPTION_GLASSES = "Prescription Glasses"
    TINTED_SUNGLASSES = "Tinted/Sunglasses"


class OcclusionType(str, Enum):
    HANDS = "Hands"
    HAIR_STRANDS = "Hair strands"
    AMBIENT_SHADOWS = "Ambient Shadows"
    MASKS = "Masks"
    NONE = "None"


class Level2Secondary(BaseModel):
    eyewear: Eyewear = Field(
        ...,
        description="Ocular accessories or corrective lenses present."
    )
    occlusions: List[OcclusionType] = Field(
        default_factory=lambda: [OcclusionType.NONE],
        description="Items, hair, or physical occluders partially obscuring the face."
    )

    @field_validator("occlusions")
    @classmethod
    def validate_occlusions(cls, v: List[OcclusionType]) -> List[OcclusionType]:
        if not v:
            return [OcclusionType.NONE]
        # If 'None' is combined with actual occlusions, eliminate 'None'
        if len(v) > 1 and OcclusionType.NONE in v:
            return [item for item in v if item != OcclusionType.NONE]
        return list(dict.fromkeys(v))  # deduplicate preserving order


# ==============================================================================
# Level 3: Affective Expressions (Composite Semantics)
# ==============================================================================

class PrimaryCategory(str, Enum):
    NEUTRAL = "Neutral"
    HAPPINESS_AMUSEMENT = "Happiness/Amusement"
    SURPRISE = "Surprise"
    CONTEMPT_SKEPTICAL = "Contempt/Skeptical"
    DISGUST = "Disgust"
    ANGER = "Anger"
    SADNESS = "Sadness"


class ExpressionIntensity(str, Enum):
    MICRO = "Micro"
    MILD = "Mild"
    MODERATE = "Moderate"
    PRONOUNCED = "Pronounced"
    EXTREME = "Extreme"


class Level3Affective(BaseModel):
    primary_category: PrimaryCategory = Field(
        ...,
        description="Dominant semantic emotional category synthesized from anatomical cues."
    )
    intensity: ExpressionIntensity = Field(
        ...,
        description="Subjective or objective magnitude of affective signaling."
    )


# ==============================================================================
# Level 4: Grounded Causal Attribution (Contextual Explanations - Evidence First)
# ==============================================================================

class Level4Causal(BaseModel):
    visual_evidence: str = Field(
        ...,
        min_length=3,
        description="Explicit pixel-grounded cues (e.g., shadows, wrinkles, crinkles, illumination, furrowing)."
    )
    inferred_cause: str = Field(
        ...,
        min_length=3,
        description="Physical, environmental, or situational cause logically deduced directly from the preceding visual evidence."
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score in the causal deduction bounded between 0.0 and 1.0."
    )

    @field_validator("confidence_score")
    @classmethod
    def check_bounded_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Confidence score {v} must be bounded within [0.0, 1.0]")
        return round(float(v), 4)

    @field_validator("visual_evidence", "inferred_cause")
    @classmethod
    def check_non_empty_strings(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 3:
            raise ValueError("Explanation fields must contain at least 3 characters of grounding.")
        return s


# ==============================================================================
# Root Analysis Schema
# ==============================================================================

class FacialAnalysisOutput(BaseModel):
    """
    Complete Hierarchical Facial Attribute and Grounded Causal Reasoning output schema.
    Conforms strictly to the 4-level taxonomy.
    """
    level1_anatomy: Level1Anatomical = Field(..., description="Level 1: Objective Anatomical Observations")
    level2_features: Level2Secondary = Field(..., description="Level 2: Secondary Accessories & Occlusions")
    level3_affect: Level3Affective = Field(..., description="Level 3: Composite Affective Expressions")
    level4_causal: Level4Causal = Field(..., description="Level 4: Grounded Causal Attribution")

    def to_compact_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def to_json_str(self, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)


# ==============================================================================
# Resilient JSON Parsing and Extraction Utilities
# ==============================================================================

def clean_json_string(raw: str) -> str:
    """Strip markdown code block indicators and extract the primary balanced JSON object."""
    text = raw.strip()
    
    # Strip markdown ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if match:
        text = match.group(1).strip()
    
    # Locate first opening curly brace
    first_brace = text.find("{")
    if first_brace != -1:
        # Balanced brace extraction to handle postamble text or multiple concatenations
        depth = 0
        in_string = False
        escape = False
        matched_end = -1
        for i in range(first_brace, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        matched_end = i
                        break
        if matched_end != -1:
            text = text[first_brace : matched_end + 1].strip()
        else:
            # If string was cut off before closing root object, close any open string and append missing braces
            truncated = text[first_brace:].strip()
            # If ends with an unclosed string
            if in_string:
                truncated += '"'
            # Strip trailing comma if any
            truncated = re.sub(r",\s*$", "", truncated)
            # Append missing closing braces
            truncated += "}" * max(1, depth)
            text = truncated

    # Clean trailing commas before closing braces/brackets
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


def safe_parse_json(raw_text: str) -> Tuple[Optional[FacialAnalysisOutput], Optional[str]]:
    """
    Attempts strict Pydantic validation of raw model generation.
    Returns:
        (parsed_schema_instance, None) if successful.
        (None, error_message) if schema validation or JSON decoding fails.
    """
    try:
        cleaned = clean_json_string(raw_text)
        data = json.loads(cleaned)
        validated = FacialAnalysisOutput.model_validate(data)
        return validated, None
    except json.JSONDecodeError as jde:
        return None, f"JSONDecodeError: {jde}"
    except Exception as exc:
        return None, f"PydanticValidationError: {exc}"
