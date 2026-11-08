"""
Generates the standalone Kaggle Notebook for hosting the Streamlit app on Kaggle Dual T4 with Cloudflare Tunnel.
"""

import json
from pathlib import Path

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Facial VLM Streamlit Web Application: Live Cloud GPU Hosting\n",
                "This notebook launches the interactive Streamlit app powered by the fine-tuned `Qwen2.5-VL-7B-Instruct` model and generates a **free public HTTPS URL** via Cloudflare Tunnel so evaluators can test the model live from anywhere in the world.\n",
                "\n",
                "### Prerequisites:\n",
                "- **Accelerator**: GPU T4 x2 (or GPU P100)\n",
                "- **Internet**: ON\n",
                "- **Mounted Input**: `goofybhatoora/celeb-vlm-trained-adapter`"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Install Streamlit, Peft, and Cloudflared\n",
                "!pip install -q streamlit pyngrok plotly peft bitsandbytes\n",
                "\n",
                "# Download and install Cloudflared (free HTTPS tunneling, zero login required)\n",
                "!wget -q -nc https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb\n",
                "!dpkg -i cloudflared-linux-amd64.deb\n",
                "print('Cloudflared installed successfully.')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Clone or locate Streamlit app code\n",
                "import os, shutil\n",
                "from pathlib import Path\n",
                "\n",
                "app_dir = Path('/kaggle/working/streamlit_app')\n",
                "app_dir.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Locate adapter directory\n",
                "adapter_path = '/kaggle/input/celeb-vlm-trained-adapter/qlora_adapter_final'\n",
                "if not os.path.exists(adapter_path):\n",
                "    for root, dirs, files in os.walk('/kaggle/input'):\n",
                "        if 'adapter_model.safetensors' in files:\n",
                "            adapter_path = root\n",
                "            break\n",
                "print(f'Using adapter from: {adapter_path}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Launch Streamlit and Cloudflare Tunnel\n",
                "import subprocess, time\n",
                "\n",
                "# Launch Streamlit in background\n",
                "st_process = subprocess.Popen([\n",
                "    'streamlit', 'run', '/kaggle/working/Facial_exp/streamlit_app/app.py',\n",
                "    '--server.port', '8501',\n",
                "    '--server.address', '0.0.0.0',\n",
                "    '--server.headless', 'true'\n",
                "])\n",
                "time.sleep(5)\n",
                "print('Streamlit launched on port 8501.')\n",
                "print('\\n' + '='*70)\n",
                "print('🌐 LIVE EVALUATOR HTTPS LINK WILL APPEAR BELOW (look for .trycloudflare.com):')\n",
                "print('='*70 + '\\n')\n",
                "\n",
                "# Launch cloudflared tunnel\n",
                "!cloudflared tunnel --url http://localhost:8501"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_path = Path("Facial_exp/streamlit_app/kaggle_host_app.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Generated {out_path} successfully.")
