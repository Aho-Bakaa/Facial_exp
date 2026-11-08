"""
Automated data ingestion, CelebA parsing, continuous landmark geometry,
multi-crop ROI extraction, stochastic biological AU bank, and resilient dataset implementations.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from src.config import SystemConfig, DEFAULT_CONFIG
from src.schema import (
    EyeDynamics,
    GazeVector,
    EyebrowDynamics,
    MouthMovement,
    MouthAndJawDynamics,
    HeadOrientation,
    Eyewear,
    OcclusionType,
    PrimaryCategory,
    ExpressionIntensity,
    Level1Anatomical,
    Level2Secondary,
    Level3Affective,
    Level4Causal,
    FacialAnalysisOutput,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==============================================================================
# Continuous Biometric Landmark Geometry
# ==============================================================================

def compute_continuous_landmarks(
    coords: Dict[str, float],
    img_width: float = 178.0,
    img_height: float = 218.0,
) -> Dict[str, float]:
    """
    Computes continuous biometric ratios from 5 canonical CelebA facial landmarks:
      - Mouth Aspect Ratio (MAR): ||p_leftmouth - p_rightmouth|| / ||p_nose - 0.5*(p_leftmouth + p_rightmouth)||
      - Eye Aspect Ratio / Normalized Interocular Distance (EAR)
      - Head Roll Angle (in degrees)
      - Commissure Vertical Asymmetry (in pixels)
      - Yaw Symmetry Ratio
      - Pitch Ratio
    """
    if not coords or "lefteye_x" not in coords:
        return {
            "mar": 1.15,
            "mouth_aspect_ratio": 1.15,
            "ear": 0.45,
            "eye_aspect_ratio": 0.45,
            "roll_deg": 0.0,
            "head_roll_deg": 0.0,
            "commissure_asym": 0.0,
            "yaw_ratio": 0.50,
            "pitch_ratio": 0.50,
        }

    p_le = np.array([coords.get("lefteye_x", 68.0), coords.get("lefteye_y", 111.0)], dtype=np.float32)
    p_re = np.array([coords.get("righteye_x", 108.0), coords.get("righteye_y", 112.0)], dtype=np.float32)
    p_nose = np.array([coords.get("nose_x", 88.0), coords.get("nose_y", 135.0)], dtype=np.float32)
    p_lm = np.array([coords.get("leftmouth_x", 72.0), coords.get("leftmouth_y", 152.0)], dtype=np.float32)
    p_rm = np.array([coords.get("rightmouth_x", 104.0), coords.get("rightmouth_y", 153.0)], dtype=np.float32)

    # 1. Mouth Aspect Ratio (MAR)
    mouth_width = float(np.linalg.norm(p_lm - p_rm))
    mouth_midpoint = 0.5 * (p_lm + p_rm)
    subnasale_to_mouth_dist = float(np.linalg.norm(p_nose - mouth_midpoint))
    mar = mouth_width / max(1.0, subnasale_to_mouth_dist)

    # 2. Interocular Distance / EAR
    interocular_dist = float(np.linalg.norm(p_re - p_le))
    ear = interocular_dist / max(1.0, img_width)

    # 3. Facial Roll Angle
    delta_x = p_re[0] - p_le[0]
    delta_y = p_re[1] - p_le[1]
    roll_deg = float(math.degrees(math.atan2(delta_y, max(1e-4, delta_x))))

    # 4. Commissure Asymmetry
    commissure_asym = float(abs(p_lm[1] - p_rm[1]))

    # 5. Yaw & Pitch
    eye_span = p_re[0] - p_le[0]
    yaw_ratio = float((p_nose[0] - p_le[0]) / max(1e-4, eye_span)) if eye_span != 0 else 0.50
    yaw_ratio = float(np.clip(yaw_ratio, -0.5, 1.5))

    eye_y_mid = (p_le[1] + p_re[1]) / 2.0
    mouth_y_mid = (p_lm[1] + p_rm[1]) / 2.0
    vertical_span = mouth_y_mid - eye_y_mid
    pitch_ratio = float((p_nose[1] - eye_y_mid) / max(1e-4, vertical_span)) if vertical_span != 0 else 0.50
    pitch_ratio = float(np.clip(pitch_ratio, -0.5, 1.5))

    return {
        "mar": round(mar, 3),
        "mouth_aspect_ratio": round(mar, 3),
        "ear": round(ear, 3),
        "eye_aspect_ratio": round(ear, 3),
        "roll_deg": round(roll_deg, 2),
        "head_roll_deg": round(roll_deg, 2),
        "commissure_asym": round(commissure_asym, 2),
        "yaw_ratio": round(yaw_ratio, 3),
        "pitch_ratio": round(pitch_ratio, 3),
    }


def compute_yaw_symmetry_ratio(coords: Dict[str, float]) -> float:
    return compute_continuous_landmarks(coords)["yaw_ratio"]


def classify_geometric_head_orientation(coords: Dict[str, float]) -> HeadOrientation:
    geom = compute_continuous_landmarks(coords)
    pitch = geom["pitch_ratio"]
    yaw = geom["yaw_ratio"]

    if pitch < 0.40:
        return HeadOrientation.TILTED_UP
    elif pitch > 0.65:
        return HeadOrientation.BOWED_DOWN

    if 0.44 <= yaw <= 0.56:
        return HeadOrientation.FRONTAL
    elif (0.30 <= yaw < 0.44) or (0.56 < yaw <= 0.70):
        return HeadOrientation.THREE_QUARTER_LEFT_RIGHT
    else:
        return HeadOrientation.PROFILE_LEFT_RIGHT


# ==============================================================================
# Multi-Crop Region-of-Interest (ROI) Extraction
# ==============================================================================

def extract_facial_rois(
    image: Image.Image,
    landmarks: Optional[Dict[str, float]] = None,
    global_size: Tuple[int, int] = (224, 224),
    crop_size: Tuple[int, int] = (112, 112),
) -> Dict[str, Image.Image]:
    """
    Extracts landmark-guided multi-scale Region-of-Interest (ROI) crops:
      1. Global Portrait: Full face resized to (224, 224).
      2. Periorbital Band: Ocular and supraorbital region resized to (112, 112).
      3. Oral Region: Subnasale to mental crease resized to (112, 112).
    """
    w, h = image.size

    # 1. Global Portrait
    global_crop = image.resize(global_size, Image.Resampling.BILINEAR)

    # 2. Periorbital Band
    if landmarks and "lefteye_x" in landmarks:
        x_le = landmarks.get("lefteye_x", 0.38 * w)
        y_le = landmarks.get("lefteye_y", 0.51 * h)
        x_re = landmarks.get("righteye_x", 0.61 * w)
        y_re = landmarks.get("righteye_y", 0.51 * h)

        center_x = (x_le + x_re) / 2.0
        center_y = (y_le + y_re) / 2.0
        eye_dist = max(30.0, abs(x_re - x_le))

        box_w = eye_dist * 1.85
        box_h = box_w * 0.75

        p_left = max(0, int(center_x - box_w / 2.0))
        p_top = max(0, int(center_y - box_h * 0.50))
        p_right = min(w, int(center_x + box_w / 2.0))
        p_bottom = min(h, int(center_y + box_h * 0.50))
        ocular_crop = image.crop((p_left, p_top, p_right, p_bottom)).resize(crop_size, Image.Resampling.BILINEAR)
    else:
        ocular_crop = image.crop((int(0.12 * w), int(0.30 * h), int(0.88 * w), int(0.60 * h))).resize(
            crop_size, Image.Resampling.BILINEAR
        )

    # 3. Oral Region
    if landmarks and "leftmouth_x" in landmarks:
        x_lm = landmarks.get("leftmouth_x", 0.40 * w)
        y_lm = landmarks.get("leftmouth_y", 0.70 * h)
        x_rm = landmarks.get("rightmouth_x", 0.58 * w)
        y_rm = landmarks.get("rightmouth_y", 0.70 * h)
        y_nose = landmarks.get("nose_y", 0.62 * h)

        m_center_x = (x_lm + x_rm) / 2.0
        m_center_y = (y_lm + y_rm) / 2.0
        mouth_dist = max(25.0, abs(x_rm - x_lm))

        box_w = mouth_dist * 1.75
        box_h = box_w * 0.85

        o_left = max(0, int(m_center_x - box_w / 2.0))
        o_top = max(0, int(min(y_nose, m_center_y - box_h * 0.45)))
        o_right = min(w, int(m_center_x + box_w / 2.0))
        o_bottom = min(h, int(m_center_y + box_h * 0.55))
        oral_crop = image.crop((o_left, o_top, o_right, o_bottom)).resize(crop_size, Image.Resampling.BILINEAR)
    else:
        oral_crop = image.crop((int(0.20 * w), int(0.60 * h), int(0.80 * w), int(0.92 * h))).resize(
            crop_size, Image.Resampling.BILINEAR
        )

    return {
        "global": global_crop,
        "periorbital": ocular_crop,
        "oral": oral_crop,
    }


# ==============================================================================
# Stochastic Biological Action Unit (AU) Bank
# ==============================================================================

AU_BANK: Dict[str, List[str]] = {
    # AU12: Zygomaticus Major (Lip Corner Puller)
    "AU12": [
        "bilateral zygomaticus major muscle contraction exerting superior-lateral traction on oral commissures (AU12)",
        "superolateral elevation of the cheilion with prominent accentuation of the nasolabial sulcus",
        "activation of zygomaticus major pulling lip corners upward with visible malar fat pad displacement",
        "dynamic tension along the modiolus raising buccal angles obliquely toward the malar eminence",
        "pronounced upward commissure displacement creating bilateral perioral skin furrowing and dental show",
        "bilateral oblique traction of the mouth corners with characteristic canine and premolar visibility",
        "zygomaticus major activation with lateral expansion of the rima oris and elevation of the vermilion border",
        "mechanical pulling of oral commissures superolaterally with soft-tissue bunching at the infraorbital margin",
        "contraction of the main zygomatic muscle bands generating an arched upper lip contour and deepened nasolabial folds",
        "elevation of the labial angles with tension propagated across the buccinator and risorius complex",
        "upward commissure vector producing bilateral cheek fullness and slight elevation of the lower eyelid contour",
        "bilateral zygomaticus major shortening causing widening of the trans-commissural distance and smile arch",
    ],
    # AU6: Orbicularis Oculi pars orbitalis (Cheek Raiser / Squint)
    "AU6": [
        "orbicularis oculi pars orbitalis contraction tightening the lower eyelid aperture (AU6)",
        "concentric orbital sphincter activation producing dynamic cutaneous furrows along the lateral canthus (crow's feet)",
        "elevation of the infraorbital cheek tissue impinging on the inferior palpebral fissure",
        "orbicularis oculi shortening with slight depression of the eyebrow tail and skin gathering at the temples",
        "pars orbitalis contraction narrowing the ocular aperture with deepening of the sub-ocular groove",
        "infraorbital skin bunching creating pronounced crescent-shaped wrinkles below both orbital margins",
        "bilateral lower eyelid elevation with concentric skin bunching around the external orbital rim",
        "tightening of the palpebral aperture with lateral canthal radiating rhytids (Duchenne marker)",
        "orbital sphincter tension elevating the malar soft tissue and reducing horizontal eye aperture",
        "active contraction of the lateral fibers of orbicularis oculi with soft-tissue displacement toward the nasion",
        "squinting dynamics characterized by lower eyelid flattening and prominent lateral orbit furrowing",
        "inferior and lateral orbital ring tension compressing the periorbital soft tissue matrix",
    ],
    # AU4: Corrugator Supercilii (Brow Lowerer / Furrower)
    "AU4": [
        "corrugator supercilii contraction pulling the medial brow heads inferomedially (AU4)",
        "medial brow depression producing vertical glabellar furrowing between supraorbital arches",
        "shortening of the corrugator muscle belly creating vertical rhytids above the radix",
        "inferomedial convergence of the eyebrow heads with lowering of the supraorbital skin fold",
        "active contraction of the corrugator and depressor supercilii forming prominent glabellar folds",
        "medial eyebrow depression with oblique tissue gathering in the inter-eyebrow zone",
        "corrugator activation narrowing the glabella with downward displacement of the medial brow margins",
        "depressed medial supraorbital margins accompanied by deep vertical cutaneous furrows",
        "bilateral corrugator tension drawing eyebrows downward into a furrowed posture",
        "adduction and depression of the eyebrow heads causing shadowing in the superior orbital groove",
        "prominent glabellar furrowing reflecting tonic contraction of the medial brow depressors",
        "downward and inward displacement of the eyebrows with visible crease formation across the nasal bridge",
    ],
    # AU1+2: Frontalis (Brow Raiser)
    "AU1_2": [
        "bilateral frontalis muscle contraction raising eyebrows superiorly (AU1+2)",
        "superior elevation of the supraorbital arches creating horizontal frontal forehead striations",
        "frontalis pars medialis and lateralis activation elevating both eyebrows across their full span",
        "uninhibited frontalis contraction raising the ciliary margins well clear of the orbital rim",
        "upward tension of the epicranius frontalis resulting in widened supra-eyelid exposure",
        "high arched eyebrow posture accompanied by fine horizontal dermic furrows across the forehead",
        "frontalis recruitment pulling eyebrows upward with accentuated palpebral aperture widening",
        "symmetrical brow elevation highlighting superior supraorbital bony contours",
        "pronounced upward excursion of the eyebrow arches with characteristic forehead skin folding",
        "contraction of the forehead musculature lifting both medial and lateral eyebrow extremities",
    ],
    # AU25/26: Lips Parted / Jaw Drop
    "AU25_26": [
        "relaxation of the orbicularis oris allowing separation of upper and lower vermilion borders (AU25)",
        "mandibular depression and parting of the lips exposing the interdental acoustic space (AU26)",
        "open oral aperture with separation of labial margins and partial incisal reveal",
        "depressor labii inferioris and pterygoid relaxation causing vertical divergence of the lips",
        "parted lip posture with visible dark acoustic buccal cavity between the dental arches",
        "vertical lip opening with exposure of maxillary central and lateral incisors",
        "moderate mouth opening accompanied by relaxation of perioral tone",
        "labial border separation highlighting the vermilion border and tongue shadow",
        "patent oral aperture without active muscular compression along the lip seal",
        "open mouth architecture with unconstrained jaw posture and clear inter-labial spacing",
    ],
    # Neutral Mouth
    "NEUTRAL_MOUTH": [
        "quiescent orbicularis oris tonicity maintaining a sealed, uncompressed labial fissure",
        "relaxed oral commissures in horizontal equilibrium without superior or inferior displacement",
        "resting perioral tone with preserved philtral ridges and closed lip margins",
        "neutral mouth posture displaying no dental exposure or commissure traction",
        "bilateral horizontal commissure alignment with quiescent mentalis and depressor anguli oris",
        "unstrained lip seal with natural vermilion volume and absence of lateral tension",
        "horizontal oral cleft at rest with smooth perioral cutaneous contours",
        "physiologic resting posture of the lips with relaxed modiolus bilaterally",
        "symmetrical closed lip contour exhibiting basal mucosal contact without strain",
        "relaxed oral anatomy without observable zygomatic or depressor muscular recruitment",
    ],
    # Neutral Eyes
    "NEUTRAL_EYES": [
        "resting palpebral fissure with normal corneal exposure and uncontracted orbicularis oculi",
        "unconstricted ocular aperture without lateral canthal skin bunching or ptosis",
        "equilateral ocular opening with relaxed levator palpebrae superioris tone",
        "neutral eyelid dynamics showing unoccluded pupil and iris contours",
        "smooth periorbital cutaneous surface without dynamic squinting or furrow rhytids",
        "symmetric palpebral aperture height with quiescent lower eyelid margin",
        "unwrinkled peri-canthal tissue and unhindered binocular visual axis",
        "physiologic resting aperture of both orbits with stable forward pupillary orientation",
        "relaxed orbital margins with clean demarcation of the supra-tarsal sulcus",
        "natural open eye posture showing clear scleral contrast without squint lines",
    ],
    # Eyewear
    "GLASSES": [
        "ocular spectacle frame resting across the nasal dorsum with temples extending to the helix",
        "corrective ophthalmic lenses with subtle rim reflections positioned anterior to both orbits",
        "rigid optical frame traversing the nasion with translucent lenses covering the eye sockets",
        "spectacle bridge seated above the nasal radix with slender metallic/acetate rim contours",
        "prescription eyewear introducing localized specular lens highlights and rim boundaries",
        "clear corrective lenses framed symmetrically over the orbital apertures",
    ],
    # Causal Explanations - Smiling
    "CAUSES_SMILE": [
        "positive social engagement, humorous conversational stimulus, or pleasant interpersonal interaction",
        "photographic studio compliance, polite portrait smiling response, or expressive greeting",
        "spontaneous affective amusement triggered by lighthearted external context",
        "approachable social demeanor and genuine Duchenne mirth in an interactive setting",
        "sociable portrait pose displaying deliberate warmth and affective signaling",
        "pleasurable cognitive appraisal evoking spontaneous bilateral facial muscle activation",
        "affable social display intended to project friendliness and mutual rapport",
        "affective response to amusing visual or acoustic stimuli during capture",
    ],
    # Causal Explanations - Neutral
    "CAUSES_NEUTRAL": [
        "focused attentive baseline demeanor during standard studio photographic capture",
        "neutral cognitive baseline without overt affective stimulation or emotional arousal",
        "sedate formal portrait posture exhibiting controlled and calm facial expression",
        "objective observation mode during still documentary or archival portrait recording",
        "passive resting state with absence of social or comedic interactive stimuli",
        "formal non-emotive portrait compliance under controlled studio lighting",
        "deliberate composed facial equilibrium devoid of transient emotional signaling",
    ],
    # Causal Explanations - Squinting
    "CAUSES_SQUINT": [
        "intense ambient studio illumination, bright sunlight glare, or ocular reflex to flash",
        "effortful visual acuity accommodation or squinting toward distant focal targets",
        "concomitant Duchenne contraction accompanying robust laughter and zygomatic pulling",
        "photophobia reflex elicited by high directional luminance upon the subject's face",
        "protective orbital contraction against wind, airborne particles, or elevated brightness",
    ],
}


# ==============================================================================
# Deterministic & Stochastic Taxonomy Rule-Mapping
# ==============================================================================

def map_celeba_to_taxonomy(
    attrs: Dict[str, bool],
    landmarks: Optional[Dict[str, float]] = None,
    seed_key: Optional[str] = None,
) -> FacialAnalysisOutput:
    """
    Maps CelebA binary attributes and continuous landmarks into the 4-level taxonomy.
    Features:
      - Continuous Biometrics: MAR, EAR, Roll angle, Commissure asymmetry injected dynamically.
      - Stochastic Lexical AU Bank: 12 biologically diverse descriptions per Action Unit.
      - Inverted Autoregressive Dependency: visual_evidence generated FIRST, then inferred_cause!
      - Seeded deterministically per image to prevent static template memorization across samples.
    """
    # Deterministic RNG derived from image identifier
    seed = int(hashlib.md5((seed_key or "celeba_sample").encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    # 1. Compute continuous geometry
    geom = compute_continuous_landmarks(landmarks or {})
    mar = geom["mar"]
    roll_deg = geom["roll_deg"]
    asym = geom["commissure_asym"]
    yaw_ratio = geom["yaw_ratio"]

    # 2. Extract binary attribute flags
    smiling = attrs.get("Smiling", False)
    mouth_open = attrs.get("Mouth_Slightly_Open", False)
    narrow_eyes = attrs.get("Narrow_Eyes", False)
    arched_brows = attrs.get("Arched_Eyebrows", False)
    bushy_brows = attrs.get("Bushy_Eyebrows", False)
    has_glasses = attrs.get("Eyeglasses", False)
    has_bangs = attrs.get("Bangs", False)
    has_hat = attrs.get("Wearing_Hat", False)

    # 3. Level 1: Anatomical & Kinematic Observations
    eye_dynamics = EyeDynamics.SQUINTING if narrow_eyes else EyeDynamics.OPEN
    eyebrow_dynamics = EyebrowDynamics.RAISED if arched_brows else (
        EyebrowDynamics.FURROWED_LOWERED if bushy_brows and not smiling else EyebrowDynamics.NEUTRAL
    )

    if smiling:
        movement = MouthMovement.LIP_CORNER_PULLER
        teeth = True
    elif mouth_open:
        movement = MouthMovement.LIPS_PARTED
        teeth = True
    else:
        movement = MouthMovement.CLOSED_RELAXED
        teeth = False

    mouth_and_jaw = MouthAndJawDynamics(movement=movement, teeth_exposed=teeth)
    orientation = classify_geometric_head_orientation(landmarks or {})
    gaze = GazeVector.DIRECT if orientation == HeadOrientation.FRONTAL else GazeVector.AVERTED_LEFT_RIGHT

    level1 = Level1Anatomical(
        eye_dynamics=eye_dynamics,
        gaze_vector=gaze,
        eyebrow_dynamics=eyebrow_dynamics,
        mouth_and_jaw=mouth_and_jaw,
        head_orientation=orientation,
        mouth_aspect_ratio=mar,
        head_roll_deg=roll_deg,
    )

    # 4. Level 2: Secondary Features & Occlusions
    eyewear = Eyewear.PRESCRIPTION_GLASSES if has_glasses else Eyewear.NONE
    occlusions: List[OcclusionType] = []
    if has_bangs:
        occlusions.append(OcclusionType.HAIR_STRANDS)
    if has_hat:
        occlusions.append(OcclusionType.AMBIENT_SHADOWS)
    if not occlusions:
        occlusions.append(OcclusionType.NONE)

    level2 = Level2Secondary(eyewear=eyewear, occlusions=occlusions)

    # 5. Level 3: Affective Expressions
    if smiling:
        primary_affect = PrimaryCategory.HAPPINESS_AMUSEMENT
        intensity = ExpressionIntensity.PRONOUNCED if (mouth_open or mar > 1.4) else ExpressionIntensity.MODERATE
    else:
        primary_affect = PrimaryCategory.NEUTRAL
        intensity = ExpressionIntensity.MODERATE

    level3 = Level3Affective(primary_category=primary_affect, intensity=intensity)

    # 6. Level 4: Grounded Causal Attribution (EVIDENCE FIRST -> THEN INFERRED CAUSE)
    evidence_tokens: List[str] = []
    cause_tokens: List[str] = []

    # Sample stochastic biological descriptions
    if smiling:
        au12_desc = rng.choice(AU_BANK["AU12"])
        evidence_tokens.append(f"{au12_desc} (measured MAR={mar:.2f}, roll={roll_deg:.1f}°)")
        cause_tokens.append(rng.choice(AU_BANK["CAUSES_SMILE"]))
    else:
        mouth_desc = rng.choice(AU_BANK["NEUTRAL_MOUTH"])
        evidence_tokens.append(f"{mouth_desc} (quiescent MAR={mar:.2f}, roll={roll_deg:.1f}°)")
        cause_tokens.append(rng.choice(AU_BANK["CAUSES_NEUTRAL"]))

    if narrow_eyes:
        au6_desc = rng.choice(AU_BANK["AU6"])
        evidence_tokens.append(au6_desc)
        cause_tokens.append(rng.choice(AU_BANK["CAUSES_SQUINT"]))
    else:
        eye_desc = rng.choice(AU_BANK["NEUTRAL_EYES"])
        evidence_tokens.append(eye_desc)

    if arched_brows:
        au12_brow = rng.choice(AU_BANK["AU1_2"])
        evidence_tokens.append(au12_brow)

    if has_glasses:
        glasses_desc = rng.choice(AU_BANK["GLASSES"])
        evidence_tokens.append(glasses_desc)
        cause_tokens.append("refractive correction and ocular visual acuity preservation")

    if asym > 3.0:
        evidence_tokens.append(f"unilateral commissure vertical delta of {asym:.1f}px")

    visual_evidence = "; ".join(evidence_tokens)
    inferred_cause = "; ".join(cause_tokens)

    level4 = Level4Causal(
        visual_evidence=visual_evidence,
        inferred_cause=inferred_cause,
        confidence_score=0.92,
    )

    return FacialAnalysisOutput(
        level1_anatomy=level1,
        level2_features=level2,
        level3_affect=level3,
        level4_causal=level4,
    )


# ==============================================================================
# CelebA Parser Functions
# ==============================================================================

def parse_celeba_attributes(filepath: Path) -> Tuple[List[str], Dict[str, Dict[str, bool]]]:
    """Parses list_attr_celeba.csv or list_attr_celeba.txt."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Attribute file not found at: {filepath}")

    attributes: Dict[str, Dict[str, bool]] = {}
    attr_names: List[str] = []

    if filepath.suffix.lower() == ".csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            img_col_idx = 0
            for idx, col in enumerate(header):
                if col.lower() in ["image_id", "image_name", "img_id"]:
                    img_col_idx = idx
                    break
            attr_names = [col.strip() for idx, col in enumerate(header) if idx != img_col_idx]
            for row in reader:
                if not row or len(row) < len(header):
                    continue
                img_name = row[img_col_idx].strip()
                flags = {
                    attr_names[i]: (row[idx].strip() == "1")
                    for i, idx in enumerate([j for j in range(len(row)) if j != img_col_idx])
                }
                attributes[img_name] = flags
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            _ = f.readline()  # count line
            attr_names = f.readline().strip().split()
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_name = parts[0]
                flags = {attr: (val == "1") for attr, val in zip(attr_names, parts[1:])}
                attributes[img_name] = flags

    logger.info("Parsed %d attribute records from %s", len(attributes), filepath.name)
    return attr_names, attributes


def parse_celeba_landmarks(filepath: Path) -> Dict[str, Dict[str, float]]:
    """Parses list_landmarks_align_celeba.csv or list_landmarks_align_celeba.txt."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Landmark file not found at: {filepath}")

    landmarks: Dict[str, Dict[str, float]] = {}

    if filepath.suffix.lower() == ".csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            img_col_idx = 0
            for idx, col in enumerate(header):
                if col.lower() in ["image_id", "image_name", "img_id"]:
                    img_col_idx = idx
                    break
            coord_names = [col.strip().lower() for idx, col in enumerate(header) if idx != img_col_idx]
            for row in reader:
                if not row or len(row) < len(header):
                    continue
                img_name = row[img_col_idx].strip()
                row_coords = [float(v.strip()) for idx, v in enumerate(row) if idx != img_col_idx]
                landmarks[img_name] = {k: v for k, v in zip(coord_names, row_coords)}
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            _ = f.readline()
            header = [c.strip().lower() for c in f.readline().strip().split()]
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                img_name = parts[0]
                coords = {col: float(val) for col, val in zip(header, parts[1:])}
                landmarks[img_name] = coords

    logger.info("Parsed %d landmark records from %s", len(landmarks), filepath.name)
    return landmarks


def parse_celeba_partitions(filepath: Path) -> Dict[str, int]:
    """Parses list_eval_partition.csv or list_eval_partition.txt."""
    filepath = Path(filepath)
    partitions: Dict[str, int] = {}
    if not filepath.exists():
        return partitions

    if filepath.suffix.lower() == ".csv":
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            img_col_idx = 0
            part_col_idx = 1
            for idx, col in enumerate(header):
                if col.lower() in ["image_id", "image_name", "img_id"]:
                    img_col_idx = idx
                elif col.lower() in ["partition", "eval_partition", "partition_id"]:
                    part_col_idx = idx
            for row in reader:
                if len(row) >= 2:
                    partitions[row[img_col_idx].strip()] = int(row[part_col_idx].strip())
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    partitions[parts[0]] = int(parts[1])

    logger.info("Parsed %d partition entries from %s", len(partitions), filepath.name)
    return partitions


def create_deterministic_splits(
    image_names: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Dict[str, int]:
    partitions: Dict[str, int] = {}
    for name in image_names:
        hash_val = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % 100000 / 100000.0
        if hash_val < train_ratio:
            partitions[name] = 0
        elif hash_val < (train_ratio + val_ratio):
            partitions[name] = 1
        else:
            partitions[name] = 2
    return partitions


# ==============================================================================
# PyTorch CelebA VLM Dataset Implementation
# ==============================================================================

class CelebAVLMDataset(Dataset):
    """
    PyTorch Dataset providing:
      - Multi-crop Region-of-Interest (ROI) images: global, periorbital, oral
      - Ground-truth binary attributes & demographic metadata (Male, Young, etc.)
      - Continuous biometric geometry
      - Stochastic biological AU-grounded schema targets
    """

    def __init__(
        self,
        image_dir: Path,
        attr_file: Path,
        landmarks_file: Optional[Path] = None,
        partition_file: Optional[Path] = None,
        split: str = "test",  # "train", "val", "test"
        sample_limit: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.split = split.lower()
        self.sample_limit = sample_limit
        self.seed = seed

        self.attr_names, self.all_attributes = parse_celeba_attributes(attr_file)

        self.all_landmarks: Dict[str, Dict[str, float]] = {}
        if landmarks_file and landmarks_file.exists():
            self.all_landmarks = parse_celeba_landmarks(landmarks_file)

        split_map = {"train": 0, "val": 1, "test": 2}
        target_partition = split_map.get(self.split, 2)

        partitions: Dict[str, int] = {}
        if partition_file and partition_file.exists():
            partitions = parse_celeba_partitions(partition_file)
        if not partitions:
            partitions = create_deterministic_splits(list(self.all_attributes.keys()))

        self.sample_names: List[str] = [
            name
            for name, p_id in partitions.items()
            if p_id == target_partition and name in self.all_attributes
        ]

        rng = np.random.default_rng(self.seed)
        rng.shuffle(self.sample_names)

        if self.sample_limit and len(self.sample_names) > self.sample_limit:
            self.sample_names = self.sample_names[: self.sample_limit]

        logger.info(
            "Initialized CelebAVLMDataset: split=%s, samples=%d",
            self.split,
            len(self.sample_names),
        )

    def __len__(self) -> int:
        return len(self.sample_names)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name = self.sample_names[idx]
        img_path = self.image_dir / img_name
        if not img_path.exists():
            for cand in [
                self.image_dir / "img_align_celeba" / img_name,
                self.image_dir.parent / "img_align_celeba" / img_name,
                self.image_dir.parent / "img_align_celeba" / "img_align_celeba" / img_name,
            ]:
                if cand.exists():
                    img_path = cand
                    break

        attrs = self.all_attributes.get(img_name, {})
        landmarks = self.all_landmarks.get(img_name, {})

        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            raise FileNotFoundError(
                f"CelebA image {img_name} not found on disk at {self.image_dir}."
            )

        # 1. Multi-crop ROI generation
        rois = extract_facial_rois(image, landmarks if landmarks else None)

        # 2. Continuous Biometrics
        geom = compute_continuous_landmarks(landmarks if landmarks else {})

        # 3. Grounded target schema with stochastic lexical perturbation & evidence-first ordering
        structured_target = map_celeba_to_taxonomy(
            attrs,
            landmarks if landmarks else None,
            seed_key=img_name,
        )

        return {
            "image_name": img_name,
            "image_path": str(img_path),
            "image": rois["global"],
            "rois": rois,
            "raw_attributes": attrs,
            "landmarks": landmarks,
            "continuous_geometry": geom,
            "structured_target": structured_target,
            "is_male": attrs.get("Male", False),
            "is_young": attrs.get("Young", True),
        }


# ==============================================================================
# Curated Dataset Export Utility
# ==============================================================================

def export_curated_training_dataset(
    dataset: CelebAVLMDataset,
    output_path: Path,
    max_samples: Optional[int] = None,
) -> Path:
    """Exports curated dataset with stochastic biological targets and multi-crop paths."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limit = min(len(dataset), max_samples or len(dataset))
    logger.info("Exporting %d curated training examples to %s...", limit, output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for idx in range(limit):
            sample = dataset[idx]
            record = {
                "image_name": sample["image_name"],
                "image_path": sample["image_path"],
                "raw_attributes": sample["raw_attributes"],
                "landmarks": sample["landmarks"],
                "continuous_geometry": sample["continuous_geometry"],
                "structured_target": sample["structured_target"].to_compact_dict(),
            }
            f.write(json.dumps(record) + "\n")

    logger.info("Curated dataset export completed: %s", output_path)
    return output_path
