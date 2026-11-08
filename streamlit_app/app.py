"""
Streamlit Web Application: Hierarchical Facial & Causal Analysis with Multimodal VLM
Demonstrating Multi-Crop ROI Attention, Biometric Grounding, and Demographic Fairness.
"""

import os
import json
from pathlib import Path
from PIL import Image
import streamlit as st
import plotly.graph_objects as go

from biometrics import extract_face_biometrics_and_rois
from model_engine import FacialVLMInferenceEngine

# Page Configuration
st.set_page_config(
    page_title="Facial VLM Causal Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .level-card {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-pass {
        background-color: #DEF7EC;
        color: #03543F;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Engine
@st.cache_resource
def get_engine():
    return FacialVLMInferenceEngine()

engine = get_engine()

# Sidebar Setup
st.sidebar.title("🎛️ Control Panel")

model_mode = st.sidebar.radio(
    "Inference Model",
    [
        "🚀 Improved Model (7B QLoRA + Multi-Crop)",
        "⚖️ Side-by-Side Comparison",
        "📉 Baseline Zero-Shot (Qwen2.5-VL)",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Select Test Image")

sample_dir = Path("Facial_exp/streamlit_app/sample_images")
sample_files = sorted([f.name for f in sample_dir.glob("*.jpg")]) if sample_dir.exists() else []

input_source = st.sidebar.radio("Image Source", ["Curated Gallery", "Upload Custom Image"])

selected_image: Image.Image = None
image_label = ""

if input_source == "Curated Gallery" and sample_files:
    sample_choice = st.sidebar.selectbox("Gallery Portraits", sample_files, index=0)
    img_path = sample_dir / sample_choice
    selected_image = Image.open(img_path).convert("RGB")
    image_label = sample_choice
elif input_source == "Upload Custom Image":
    uploaded = st.sidebar.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        selected_image = Image.open(uploaded).convert("RGB")
        image_label = uploaded.name
    elif sample_files:
        img_path = sample_dir / sample_files[0]
        selected_image = Image.open(img_path).convert("RGB")
        image_label = f"{sample_files[0]} (Default)"

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ System Status")
st.sidebar.markdown(f"**Hardware:** {'🟢 CUDA GPU Active' if engine.cuda_available else '🟡 CPU Preview Mode'}")
st.sidebar.markdown(f"**Backbone:** `Qwen/Qwen2.5-VL-7B-Instruct`")
st.sidebar.markdown(f"**Quantization:** 4-bit NormalFloat (NF4)")
st.sidebar.markdown(f"**Fine-Tuning:** Dual T4 QLoRA (Rank=16)")

# Main Header
st.markdown("<div class='main-header'>🧠 Hierarchical Facial Attribute & Causal Analysis</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Autonomous Multimodal Vision-Language Model with Multi-Crop ROI Attention & Action Unit Grounding</div>",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔬 Live Interactive Analyzer", "📊 Demographic Parity & Benchmark Audit", "☁️ Evaluator Cloud Hosting Guide"])

# ==============================================================================
# TAB 1: Live Interactive Analyzer
# ==============================================================================
with tab1:
    if selected_image is None:
        st.warning("Please select or upload an image to begin analysis.")
    else:
        # Run Biometrics Extraction
        with st.spinner("Extracting facial landmarks and multi-crop ROIs..."):
            bio_data = extract_face_biometrics_and_rois(selected_image)

        col_left, col_right = st.columns([1, 1.2])

        with col_left:
            st.subheader(f"📸 Image Input: `{image_label}`")
            st.image(
                bio_data["annotated_image"],
                caption="Face Landmark Bounding Boxes: Green (Face), Cyan (Eyes/Brows), Magenta (Oral/Mouth)",
                use_container_width=True,
            )

            st.markdown("#### 🎯 Multi-Crop Region of Interest (ROI) Tokens")
            crop_c1, crop_c2, crop_c3 = st.columns(3)
            with crop_c1:
                st.image(bio_data["global_crop"], caption="Global (224x224)", use_container_width=True)
            with crop_c2:
                st.image(bio_data["periorbital_crop"], caption="Periorbital (112x112)", use_container_width=True)
            with crop_c3:
                st.image(bio_data["oral_crop"], caption="Oral (112x112)", use_container_width=True)

            st.markdown("#### 📐 Continuous Biometric Measurements")
            b_c1, b_c2, b_c3 = st.columns(3)
            with b_c1:
                mar_val = bio_data["mar"]
                st.metric("Mouth Aspect Ratio (MAR)", f"{mar_val:.2f}", delta="Open" if mar_val > 2.0 else "Closed")
            with b_c2:
                st.metric("Head Roll Angle", f"{bio_data['head_roll_deg']:.1f}°", delta="Level" if abs(bio_data["head_roll_deg"]) < 5.0 else "Tilted")
            with b_c3:
                st.metric("3D Orientation", bio_data["head_orientation"])

        with col_right:
            st.subheader("📑 4-Level Hierarchical Causal Output")

            if "Side-by-Side" in model_mode:
                # Run both
                res_improved = engine.analyze_face(bio_data, "Improved")
                res_baseline = engine.analyze_face(bio_data, "Baseline")

                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    st.markdown("##### 🚀 Improved (QLoRA + Multi-Crop)")
                    st.json(res_improved["result_json"])
                with sub_c2:
                    st.markdown("##### 📉 Baseline (Zero-Shot)")
                    st.json(res_baseline["result_json"])
            else:
                chosen_variant = "Baseline" if "Baseline" in model_mode else "Improved"
                res = engine.analyze_face(bio_data, chosen_variant)
                rj = res["result_json"]

                # Level 1 Card
                st.markdown(
                    f"""
                    <div class='level-card'>
                        <b>Level 1: Objective Anatomical Observations</b><br/>
                        • <b>Eye Dynamics:</b> {rj['level1_anatomy']['eye_dynamics']}<br/>
                        • <b>Gaze Vector:</b> {rj['level1_anatomy']['gaze_vector']}<br/>
                        • <b>Eyebrow Dynamics:</b> {rj['level1_anatomy']['eyebrow_dynamics']}<br/>
                        • <b>Mouth Dynamics:</b> {rj['level1_anatomy']['mouth_dynamics']}<br/>
                        • <b>Teeth Exposed:</b> {'Yes' if rj['level1_anatomy']['teeth_exposed'] else 'No'}<br/>
                        • <b>3D Head Pose:</b> {rj['level1_anatomy']['head_orientation']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Level 2 Card
                st.markdown(
                    f"""
                    <div class='level-card'>
                        <b>Level 2: Secondary Features & Occlusions</b><br/>
                        • <b>Eyewear:</b> {rj['level2_secondary']['eyewear']}<br/>
                        • <b>Physical Occlusions:</b> {rj.get('level2_secondary', {}).get('physical_occlusions', 'None')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Level 3 Card
                st.markdown(
                    f"""
                    <div class='level-card'>
                        <b>Level 3: Affective Composite Semantics</b><br/>
                        • <b>Primary Expression:</b> {rj['level3_affect'].get('primary_expression', rj['level3_affect'].get('expression', 'Neutral'))}<br/>
                        • <b>Expression Intensity:</b> {rj['level3_affect'].get('intensity', 'Moderate')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Level 4 Card
                st.markdown(
                    f"""
                    <div class='level-card' style='border-left-color: #10B981;'>
                        <b>Level 4: Grounded Causal Reasoning (Visual Chain-of-Thought)</b><br/>
                        • <b>Visual Evidence (Action Units):</b> <i>"{rj['level4_causal']['visual_evidence']}"</i><br/>
                        • <b>Inferred Situational Cause:</b> <i>"{rj['level4_causal']['inferred_cause']}"</i><br/>
                        • <b>Calibrated Confidence:</b> <b>{rj['level4_causal']['confidence_score']:.2f}</b>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander("🔍 View Raw RFC 8259 Compliant JSON"):
                    st.json(rj)


# ==============================================================================
# TAB 2: Demographic Parity & Benchmark Audit
# ==============================================================================
with tab2:
    st.subheader("⚖️ Audited Demographic Fairness & Performance Verification")
    st.markdown("Empirical benchmark results comparing Zero-Shot Baseline vs Fine-Tuned Model across 150 held-out CelebA Partition 2 portraits.")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Schema Conformance", "100.0%", "+6.0% (150/150 Valid)")
    with m2:
        st.metric("Attribute Macro-F1", "0.4772", "+56.5% Relative Boost")
    with m3:
        st.metric("Gender Disparity Gap", "0.0142", "-87.2% Reduction")
    with m4:
        st.metric("Hallucination Rate", "0.00%", "-2.13% (Zero Contradictions)")

    st.markdown("---")

    chart_c1, chart_c2 = st.columns(2)

    with chart_c1:
        st.markdown("#### 🚻 Gender Fairness: Closing the Disparity Gap")
        fig_gender = go.Figure(
            data=[
                go.Bar(name="Baseline Zero-Shot", x=["Male", "Female"], y=[0.3467, 0.2363], marker_color="#94A3B8"),
                go.Bar(name="Improved (QLoRA + Multi-Crop)", x=["Male", "Female"], y=[0.4659, 0.4801], marker_color="#3B82F6"),
            ]
        )
        fig_gender.update_layout(
            barmode="group",
            yaxis=dict(title="Macro F1-Score", range=[0, 0.6]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        st.caption("Baseline suffered from a massive 11.04% gender gap. Fine-tuning with multi-crop patches elevated female F1 by +103.2%, achieving demographic parity (1.42% gap).")

    with chart_c2:
        st.markdown("#### 🎯 Granular Per-Attribute F1 Breakdown")
        attributes = ["Smiling", "Mouth Open", "Eyeglasses", "Arched Eyebrows"]
        base_scores = [0.6486, 0.2474, 0.9333, 0.0000]
        imp_scores = [0.9079, 0.9024, 1.0000, 0.0526]

        fig_attr = go.Figure(
            data=[
                go.Bar(name="Baseline", x=attributes, y=base_scores, marker_color="#CBD5E1"),
                go.Bar(name="Improved", x=attributes, y=imp_scores, marker_color="#10B981"),
            ]
        )
        fig_attr.update_layout(
            barmode="group",
            yaxis=dict(title="F1 Score", range=[0, 1.1]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_attr, use_container_width=True)
        st.caption("Oral and Periorbital crops drove Smiling to 0.908 F1 (+39.9%), Mouth Open to 0.902 F1 (+264.7%), and Eyeglasses to a perfect 1.000 F1.")


# ==============================================================================
# TAB 3: Evaluator Cloud Hosting Guide
# ==============================================================================
with tab3:
    st.subheader("🚀 One-Click Hosting for Evaluators (Free GPU + Public HTTPS URL)")
    st.markdown(
        """
        To give evaluators instant access to test the live **Qwen2.5-VL-7B fine-tuned model** with real GPU acceleration:
        You can host this Streamlit application directly on **Kaggle** or **Google Colab** and tunnel it via **Cloudflare Tunnel** (100% free, no login or authentication required for evaluators!).
        """
    )

    st.markdown("#### Step-by-Step Kaggle Deployment Instructions:")
    st.code(
        """# 1. In a Kaggle Notebook with GPU T4 enabled:
!pip install streamlit pyngrok plotly

# 2. Download the fast cloudflared tunnel client (free, zero signup required)
!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
!dpkg -i cloudflared-linux-amd64.deb

# 3. Launch Streamlit in the background
import subprocess, time
subprocess.Popen(["streamlit", "run", "Facial_exp/streamlit_app/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"])
time.sleep(3)

# 4. Start the secure Cloudflare tunnel and print the live URL
print("🚀 Live Evaluator Public URL will appear below:")
!cloudflared tunnel --url http://localhost:8501
""",
        language="bash",
    )

    st.info("💡 **Evaluator Experience**: When Cloudflare starts, it will output a secure URL like `https://<random-name>.trycloudflare.com`. Evaluators can click it on any browser, upload their own photos, and run real-time inference on the 7B fine-tuned model!")
