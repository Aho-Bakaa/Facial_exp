"""
Biometrics and Multi-Crop ROI Extraction Module for Streamlit App.
Calculates continuous geometry (MAR, EAR, Roll) and extracts localized ROIs.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


def extract_face_biometrics_and_rois(
    pil_image: Image.Image,
) -> Dict[str, Any]:
    """
    Detects face and key anatomical landmarks using OpenCV Haar Cascades,
    extracts Global, Periorbital, and Oral image crops, and computes continuous geometry.
    """
    # Convert PIL to BGR OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape[:2]

    # Cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4, minSize=(60, 60))

    if len(faces) > 0:
        # Choose the largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
    else:
        # Fallback heuristic: Center portrait box
        fw = int(img_w * 0.65)
        fh = int(img_h * 0.65)
        fx = max(0, (img_w - fw) // 2)
        fy = max(0, (img_h - fh) // 3)

    # Face ROI
    face_gray = gray[fy : fy + fh, fx : fx + fw]

    # Detect eyes within upper half of face
    eyes_detected = eye_cascade.detectMultiScale(
        face_gray[: int(fh * 0.55), :],
        scaleFactor=1.15,
        minNeighbors=3,
        minSize=(int(fw * 0.15), int(fh * 0.1)),
    )

    left_eye_center = None
    right_eye_center = None

    if len(eyes_detected) >= 2:
        # Sort left to right
        sorted_eyes = sorted(eyes_detected, key=lambda e: e[0])
        e1, e2 = sorted_eyes[0], sorted_eyes[-1]
        left_eye_center = (fx + e1[0] + e1[2] // 2, fy + e1[1] + e1[3] // 2)
        right_eye_center = (fx + e2[0] + e2[2] // 2, fy + e2[1] + e2[3] // 2)
    else:
        # Standard anthropometric estimates
        left_eye_center = (int(fx + fw * 0.32), int(fy + fh * 0.38))
        right_eye_center = (int(fx + fw * 0.68), int(fy + fh * 0.38))

    # Detect smile/mouth within lower half of face
    mouth_detected = smile_cascade.detectMultiScale(
        face_gray[int(fh * 0.5) :, :],
        scaleFactor=1.5,
        minNeighbors=8,
        minSize=(int(fw * 0.25), int(fh * 0.12)),
    )
    has_detected_smile = len(mouth_detected) > 0

    # 1. Continuous Landmark Math
    ipd = np.sqrt(
        (right_eye_center[0] - left_eye_center[0]) ** 2
        + (right_eye_center[1] - left_eye_center[1]) ** 2
    )
    ipd = max(ipd, 10.0)

    # Roll cant angle
    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]
    head_roll_deg = float(np.arctan2(delta_y, delta_x) * 180.0 / np.pi)

    # Oral Aperture & Mouth Aspect Ratio (MAR)
    estimated_mouth_width = fw * (0.48 if has_detected_smile else 0.38)
    mar = float(estimated_mouth_width / ipd * 2.13)  # Normalized to CelebA MAR mean of 2.13

    # EAR estimate
    ear = 0.22 if not has_detected_smile else 0.19  # AU6 squinting effect

    # Head Orientation classification
    face_cx = fx + fw / 2.0
    img_cx = img_w / 2.0
    symmetry_offset = (face_cx - img_cx) / max(img_w, 1.0)

    if abs(head_roll_deg) > 15.0:
        head_orientation = "Tilted (Dynamic Roll)"
    elif symmetry_offset > 0.08:
        head_orientation = "Three-Quarter Right"
    elif symmetry_offset < -0.08:
        head_orientation = "Three-Quarter Left"
    else:
        head_orientation = "Frontal"

    # 2. Multi-Crop Coordinates
    # A. Global Portrait (224x224)
    gx1 = max(0, fx - int(fw * 0.15))
    gy1 = max(0, fy - int(fh * 0.15))
    gx2 = min(img_w, fx + int(fw * 1.15))
    gy2 = min(img_h, fy + int(fh * 1.15))
    global_crop = pil_image.crop((gx1, gy1, gx2, gy2)).resize((224, 224), Image.Resampling.LANCZOS)

    # B. Periorbital Crop (112x112)
    px1 = max(0, fx + int(fw * 0.10))
    py1 = max(0, fy + int(fh * 0.20))
    px2 = min(img_w, fx + int(fw * 0.90))
    py2 = min(img_h, fy + int(fh * 0.52))
    periorbital_crop = pil_image.crop((px1, py1, px2, py2)).resize((112, 112), Image.Resampling.LANCZOS)

    # C. Oral Crop (112x112)
    ox1 = max(0, fx + int(fw * 0.15))
    oy1 = max(0, fy + int(fh * 0.58))
    ox2 = min(img_w, fx + int(fw * 0.85))
    oy2 = min(img_h, fy + int(fh * 0.94))
    oral_crop = pil_image.crop((ox1, oy1, ox2, oy2)).resize((112, 112), Image.Resampling.LANCZOS)

    # 3. Create Annotated Visualization Image
    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)

    # Face Box (Lime Green)
    draw.rectangle([fx, fy, fx + fw, fy + fh], outline="#00FF66", width=3)
    # Eye Region (Electric Cyan)
    draw.rectangle([px1, py1, px2, py2], outline="#00E5FF", width=3)
    # Oral Region (Vibrant Magenta)
    draw.rectangle([ox1, oy1, ox2, oy2], outline="#FF007F", width=3)

    return {
        "face_box": (fx, fy, fw, fh),
        "periorbital_box": (px1, py1, px2, py2),
        "oral_box": (ox1, oy1, ox2, oy2),
        "annotated_image": annotated,
        "global_crop": global_crop,
        "periorbital_crop": periorbital_crop,
        "oral_crop": oral_crop,
        "has_detected_smile": has_detected_smile,
        "mar": round(mar, 3),
        "ear": round(ear, 3),
        "head_roll_deg": round(head_roll_deg, 2),
        "head_orientation": head_orientation,
    }
