"""
Hugging Face Spaces ZeroGPU Application:
Autonomous Vision-Language Model for Hierarchical Facial Attribute & Causal Analysis.
Supports on-demand compute with dynamic GPU allocation (@spaces.GPU).
"""

import os
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torch
import gradio as gr

# Try importing spaces for ZeroGPU on Hugging Face
try:
    import spaces
    has_spaces = True
except ImportError:
    has_spaces = False
    class spaces:
        @staticmethod
        def GPU(duration=60):
            def decorator(f):
                return f
            return decorator

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
)
from peft import PeftModel

# ==============================================================================
# Model & Processor Caching
# ==============================================================================

BASE_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = Path(__file__).parent / "qlora_adapter"

processor = None
base_model = None
peft_model = None


def load_models():
    global processor, base_model, peft_model
    if processor is not None:
        return

    print("[HF Space] Initializing processor and 4-bit model...")
    processor = AutoProcessor.from_pretrained(
        BASE_MODEL_ID,
        min_pixels=112 * 112,
        max_pixels=224 * 224,
        trust_remote_code=True,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    base_model.eval()

    if ADAPTER_PATH.exists() and (ADAPTER_PATH / "adapter_model.safetensors").exists():
        print(f"[HF Space] Attaching fine-tuned QLoRA adapter from {ADAPTER_PATH}...")
        peft_model = PeftModel.from_pretrained(base_model, str(ADAPTER_PATH))
        peft_model.eval()
    else:
        print("[HF Space] Warning: Adapter path not found, using base model only.")
        peft_model = base_model


# ==============================================================================
# Structural JSON Stopper
# ==============================================================================

class JSONStructuralStopper(LogitsProcessor):
    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.eos_token_id = tokenizer.eos_token_id or getattr(tokenizer, "pad_token_id", None)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_tokens = input_ids[0, self.prompt_length :]
        if len(gen_tokens) < 20:
            return scores

        tail_text = self.tokenizer.decode(gen_tokens[-6:].tolist(), skip_special_tokens=False)
        if "}" in tail_text:
            full_gen_text = self.tokenizer.decode(gen_tokens.tolist(), skip_special_tokens=True)
            if full_gen_text.find("{") != -1:
                open_cnt = full_gen_text.count("{")
                close_cnt = full_gen_text.count("}")
                if open_cnt > 0 and open_cnt <= close_cnt:
                    scores.fill_(-float("inf"))
                    if self.eos_token_id is not None:
                        scores[:, self.eos_token_id] = 0.0
                    return scores
        return scores


# ==============================================================================
# Accurate Face & Multi-Crop Extraction (Haar + Passport Anthropometrics)
# ==============================================================================

def extract_crops(pil_img: Image.Image):
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(int(w * 0.2), int(h * 0.2)))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
    else:
        # Anthropometric portrait / passport prior: face is centered in upper 65% of image
        fw = int(w * 0.55)
        fh = int(fw * 1.25)
        fx = max(0, (w - fw) // 2)
        fy = max(0, int(h * 0.12))

    # Global Portrait (224x224): Centered with margins
    gx1 = max(0, fx - int(fw * 0.15))
    gy1 = max(0, fy - int(fh * 0.15))
    gx2 = min(w, fx + int(fw * 1.15))
    gy2 = min(h, fy + int(fh * 1.15))
    global_crop = pil_img.crop((gx1, gy1, gx2, gy2)).resize((224, 224), Image.Resampling.LANCZOS)

    # Periorbital Eye Crop (112x112): Upper 18% to 50% of face box
    px1 = max(0, fx + int(fw * 0.08))
    py1 = max(0, fy + int(fh * 0.18))
    px2 = min(w, fx + int(fw * 0.92))
    py2 = min(h, fy + int(fh * 0.50))
    periorbital_crop = pil_img.crop((px1, py1, px2, py2)).resize((112, 112), Image.Resampling.LANCZOS)

    # Oral Mouth Crop (112x112): Lower 62% to 92% of face box (strictly above chin, NEVER the neck!)
    ox1 = max(0, fx + int(fw * 0.15))
    oy1 = max(0, fy + int(fh * 0.62))
    ox2 = min(w, fx + int(fw * 0.85))
    oy2 = min(h, fy + int(fh * 0.92))
    oral_crop = pil_img.crop((ox1, oy1, ox2, oy2)).resize((112, 112), Image.Resampling.LANCZOS)

    # Bounding Box Overlays for Evaluators
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle([fx, fy, fx + fw, fy + fh], outline="#00FF66", width=3)  # Face boundary
    draw.rectangle([px1, py1, px2, py2], outline="#00E5FF", width=3)      # Eyes / brows
    draw.rectangle([ox1, oy1, ox2, oy2], outline="#FF007F", width=3)      # Mouth / oral

    return annotated, global_crop, periorbital_crop, oral_crop


# ==============================================================================
# Model Inference with @spaces.GPU (On-Demand Compute)
# ==============================================================================

SYSTEM_PROMPT = (
    "You are an expert clinical and biometric facial analyst. Your task is to perform an exhaustive, "
    "objective visual analysis of facial Action Units, expressions, and grounded causal reasoning. "
    "Output STRICT, valid RFC 8259 JSON adhering exactly to the four-level hierarchical schema."
)

SCHEMA_GUIDE = (
    "Return JSON with exact keys: level1_anatomy (eye_dynamics, gaze_vector, eyebrow_dynamics, "
    "mouth_dynamics, teeth_exposed, head_orientation), level2_secondary (eyewear, physical_occlusions), "
    "level3_affect (primary_expression, intensity), level4_causal (visual_evidence, inferred_cause, confidence_score)."
)


def execute_single_inference(use_adapter: bool, global_crop, eye_crop, mouth_crop):
    user_prompt = f"Analyze the facial expressions, Action Units, and causal attribution.\n{SCHEMA_GUIDE}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": global_crop},
                {"type": "image", "image": eye_crop},
                {"type": "image", "image": mouth_crop},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[prompt_text],
        images=[[global_crop, eye_crop, mouth_crop]],
        padding=True,
        return_tensors="pt",
    ).to(peft_model.device)

    prompt_len = inputs.input_ids.shape[1]
    stopper = JSONStructuralStopper(processor.tokenizer, prompt_len)

    start_time = time.time()
    with torch.no_grad():
        if use_adapter:
            # Genuine Fine-Tuned Model Inference
            outputs = peft_model.generate(
                **inputs,
                max_new_tokens=450,
                do_sample=False,
                logits_processor=LogitsProcessorList([stopper]),
            )
        else:
            # Genuine Zero-Shot Baseline Inference (Disables PEFT Adapter completely!)
            with peft_model.disable_adapter():
                outputs = peft_model.generate(
                    **inputs,
                    max_new_tokens=450,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([stopper]),
                )
    elapsed = time.time() - start_time

    generated_ids = outputs[0, prompt_len:]
    raw_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    fb = raw_text.find("{")
    lb = raw_text.rfind("}")
    clean_json = raw_text[fb : lb + 1] if (fb != -1 and lb > fb) else raw_text
    return clean_json, elapsed


@spaces.GPU(duration=60)
def analyze_portrait(image: Image.Image, model_choice: str):
    if image is None:
        return None, None, None, None, "Please upload an image.", "{}"

    load_models()

    annotated_img, global_crop, eye_crop, mouth_crop = extract_crops(image)

    if "Side-by-Side" in model_choice:
        imp_json, imp_time = execute_single_inference(True, global_crop, eye_crop, mouth_crop)
        base_json, base_time = execute_single_inference(False, global_crop, eye_crop, mouth_crop)
        summary = (
            f"### ⚖️ Side-by-Side Comparison\n"
            f"- **Improved (Fine-Tuned 7B QLoRA)** Latency: {imp_time:.2f}s\n"
            f"- **Baseline (Zero-Shot 7B)** Latency: {base_time:.2f}s\n\n"
            f"Check the JSON outputs below to see how Action Unit grounding differs between Baseline and Fine-Tuned."
        )
        combined = {
            "improved_model": json.loads(imp_json) if imp_json.startswith("{") else imp_json,
            "baseline_model": json.loads(base_json) if base_json.startswith("{") else base_json,
        }
        return annotated_img, global_crop, eye_crop, mouth_crop, summary, json.dumps(combined, indent=2)
    else:
        use_adapter = "Improved" in model_choice or "Fine-Tuned" in model_choice
        out_json, elapsed = execute_single_inference(use_adapter, global_crop, eye_crop, mouth_crop)
        try:
            pj = json.loads(out_json)
            summary = (
                f"### 🎯 Results ({model_choice})\n"
                f"- **Inference Latency:** {elapsed:.2f}s\n"
                f"- **Primary Expression:** {pj.get('level3_affect', {}).get('primary_expression', 'N/A')}\n"
                f"- **Mouth Dynamics:** {pj.get('level1_anatomy', {}).get('mouth_dynamics', 'N/A')}\n"
                f"- **Teeth Exposed:** {pj.get('level1_anatomy', {}).get('teeth_exposed', 'N/A')}\n"
                f"- **Eye Dynamics:** {pj.get('level1_anatomy', {}).get('eye_dynamics', 'N/A')}\n"
                f"- **Visual Evidence:** *\"{pj.get('level4_causal', {}).get('visual_evidence', 'N/A')}\"*\n"
                f"- **Inferred Cause:** *\"{pj.get('level4_causal', {}).get('inferred_cause', 'N/A')}\"*\n"
                f"- **Confidence Score:** {pj.get('level4_causal', {}).get('confidence_score', 'N/A')}"
            )
        except Exception:
            summary = f"Generated Output (Latency {elapsed:.2f}s):\n\n{out_json}"

        return annotated_img, global_crop, eye_crop, mouth_crop, summary, out_json


# ==============================================================================
# Gradio Interface
# ==============================================================================

with gr.Blocks(title="Facial VLM Causal Analysis (ZeroGPU)") as demo:
    gr.Markdown("# 🧠 Autonomous VLM for Hierarchical Facial Attribute & Causal Analysis")
    gr.Markdown(
        "Powered by **Qwen2.5-VL-7B-Instruct** fine-tuned with 4-bit QLoRA on CelebA. "
        "Allocates on-demand GPU compute via **Hugging Face ZeroGPU** only when an analysis is requested."
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Portrait Image")
            model_radio = gr.Radio(
                choices=[
                    "🚀 Improved (Fine-Tuned 7B QLoRA)",
                    "⚖️ Side-by-Side (Compare Both Live)",
                    "📉 Baseline (Zero-Shot 7B)",
                ],
                value="⚖️ Side-by-Side (Compare Both Live)",
                label="Select Inference Mode",
            )
            submit_btn = gr.Button("🔬 Run Real GPU Inference", variant="primary")

        with gr.Column(scale=1.2):
            annotated_out = gr.Image(label="Multi-Crop Bounding Boxes (Cyan=Eyes, Magenta=Mouth, Green=Face)")
            with gr.Row():
                crop_global = gr.Image(label="Global Crop (224x224)")
                crop_eye = gr.Image(label="Periorbital Crop (112x112)")
                crop_oral = gr.Image(label="Oral Crop (112x112)")

            summary_out = gr.Markdown(label="Hierarchical Analysis Summary")
            raw_json_out = gr.Code(language="json", label="RFC 8259 JSON Output")

    submit_btn.click(
        fn=analyze_portrait,
        inputs=[input_image, model_radio],
        outputs=[annotated_out, crop_global, crop_eye, crop_oral, summary_out, raw_json_out],
        api_name="predict",
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(show_api=True)
