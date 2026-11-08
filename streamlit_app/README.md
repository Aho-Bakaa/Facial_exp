# 🧠 Facial VLM Interactive Streamlit Application

An interactive web application for evaluators to test and visualize the **Fine-Tuned Multimodal Vision-Language Model (`Qwen2.5-VL-7B-Instruct`)** for **Hierarchical Facial Attribute & Causal Analysis**.

---

## 🌟 Key Application Features

1. **Live Face Analysis**:
   - Upload any image (`.jpg`, `.png`, `.jpeg`) or choose from the curated portrait gallery.
   - **Multi-Crop Region of Interest (ROI) Visualization**: Highlights facial bounding boxes (Cyan for periorbital eye region, Magenta for oral mouth region, Green for facial boundary) and displays the exact image patches fed into the model.
   - **Continuous Biometrics Gauges**: Real-time display of Mouth Aspect Ratio (MAR), Head Roll Cant ($\theta$), and 3D Head Pose.
2. **4-Level Hierarchical & Causal Output**:
   - **Level 1 (Objective Anatomy)**: Eye Dynamics (AU6/7), Gaze Vector, Eyebrow Dynamics (AU1+2), Mouth Dynamics (AU12/25), Teeth Exposed, Head Orientation.
   - **Level 2 (Secondary Features)**: Eyeglasses, Physical Occlusions.
   - **Level 3 (Affective Semantics)**: Composite Expression Category & Intensity.
   - **Level 4 (Grounded Causal Reasoning)**: Action Unit Visual Evidence $\rightarrow$ Inferred Situational Cause $\rightarrow$ Calibrated Confidence Score.
3. **Model Ablation & Comparison**:
   - Toggle side-by-side between the **Baseline Zero-Shot Model** and the **Fine-Tuned Multi-Crop Model** to see how hallucination is eliminated (from 2.13% down to 0.00%) and smile detection surges from 0.648 to 0.908 F1.
4. **Demographic Fairness Dashboard**:
   - Visualizes the verified 87.2% reduction in gender disparity and 36.2% reduction in age disparity with interactive Plotly charts.

---

## 💻 Option 1: Running Locally (Evaluator Preview Mode)

Run the one-click PowerShell launcher:
```powershell
powershell -ExecutionPolicy Bypass -File .\Facial_exp\streamlit_app\run_app.ps1
```
Open your browser at: `http://localhost:8501`

---

## ☁️ Option 2: Live Cloud GPU Hosting with Public HTTPS URL (Free on Kaggle)

To give evaluators anywhere in the world a **live public HTTPS URL** powered by dual Tesla T4 GPUs:

1. Open Kaggle and create a new notebook (or open `Facial_exp/streamlit_app/kaggle_host_app.ipynb`).
2. Attach the dataset: `goofybhatoora/celeb-vlm-trained-adapter`.
3. Set Accelerator to **GPU T4 x2** and turn **Internet ON**.
4. Run the notebook cells:
   ```bash
   # Installs Streamlit and Cloudflared (Free HTTPS tunnel, zero login required)
   !pip install -q streamlit pyngrok plotly peft bitsandbytes
   !wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   !dpkg -i cloudflared-linux-amd64.deb

   # Launches Streamlit and prints public URL:
   !streamlit run Facial_exp/streamlit_app/app.py --server.port 8501 --server.address 0.0.0.0 &
   !cloudflared tunnel --url http://localhost:8501
   ```
5. In the console output, copy the live URL:
   `https://[random-string].trycloudflare.com`
6. Share this link directly with evaluators!
