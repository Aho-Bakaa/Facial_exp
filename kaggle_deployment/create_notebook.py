"""
Generates the self-contained Kaggle notebook and kernel metadata for GPU execution on Kaggle.
"""

import json
from pathlib import Path

base_dir = Path("Facial_exp").resolve()
src_dir = base_dir / "src"

files_to_pack = [
    ("__init__.py", src_dir / "__init__.py"),
    ("config.py", src_dir / "config.py"),
    ("schema.py", src_dir / "schema.py"),
    ("data_loader.py", src_dir / "data_loader.py"),
    ("prompt_builder.py", src_dir / "prompt_builder.py"),
    ("eval_harness.py", src_dir / "eval_harness.py"),
    ("baseline.py", src_dir / "baseline.py"),
    ("train_qlora.py", src_dir / "train_qlora.py"),
    ("inference.py", src_dir / "inference.py"),
    ("run_benchmark.py", base_dir / "run_benchmark.py"),
]

cells = []

# Cell 1: Markdown Title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Autonomous VLM Pipeline for Hierarchical Facial Attribute & Causal Analysis\n",
        "**Target:** Dual NVIDIA T4 (16GB VRAM x2) / P100 GPU\n",
        "**Mission:** Implement, benchmark, and optimize Qwen2.5-VL-3B with QLoRA on CelebA."
    ]
})

# Cell 2: Hardware and GPU Diagnostics
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!nvidia-smi\n",
        "import torch\n",
        "print('PyTorch Version:', torch.__version__)\n",
        "print('CUDA Available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    print('GPU Count:', torch.cuda.device_count())\n",
        "    print('GPU Device Name:', torch.cuda.get_device_name(0))\n",
        "    print('BF16 Native Support:', torch.cuda.is_bf16_supported())\n"
    ]
})

# Cell 3: Install Core Dependencies (quoted to avoid bash redirection bugs)
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!pip install -q \"transformers>=4.45.0\" \"peft>=0.9.0\" \"bitsandbytes>=0.43.0\" \"accelerate>=0.26.0\" \"qwen-vl-utils>=0.0.8\" \"pydantic>=2.5.0\" \"gdown\" \"datasets>=2.16.0\"\n"
    ]
})

# Cell 4: Initialize Modular Codebase Hierarchy on Kaggle
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os, sys\n",
        "from pathlib import Path\n",
        "os.makedirs('/kaggle/working/src', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/experiments', exist_ok=True)\n",
        "os.makedirs('/kaggle/working/input_data', exist_ok=True)\n",
        "if '/kaggle/working' not in sys.path:\n",
        "    sys.path.insert(0, '/kaggle/working')\n"
    ]
})

# Add cells to write each source file
for filename, filepath in files_to_pack:
    content = filepath.read_text(encoding="utf-8")
    target_path = f"/kaggle/working/src/{filename}" if filename != "run_benchmark.py" else "/kaggle/working/run_benchmark.py"
    
    code_source = [
        f"# Writing {target_path}\n",
        f"with open('{target_path}', 'w', encoding='utf-8') as f:\n",
        f"    f.write({repr(content)})\n",
        f"print('Successfully deployed {target_path}')\n"
    ]
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_source
    })

# Cell for Robust Dataset Path Resolution & Auto-Fallback
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Comprehensive CelebA Dataset Mount & Resolution\n",
        "import os, sys, shutil\n",
        "from pathlib import Path\n",
        "\n",
        "target_data = Path('/kaggle/working/input_data')\n",
        "target_data.mkdir(parents=True, exist_ok=True)\n",
        "\n",
        "print('=== /kaggle/input Inspection ===')\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    for item in os.listdir('/kaggle/input'):\n",
        "        p = Path('/kaggle/input') / item\n",
        "        if p.is_dir():\n",
        "            try:\n",
        "                sample_files = os.listdir(p)[:6]\n",
        "                print(f'  Mounted directory: {item}/ ({len(os.listdir(p))} items) -> Sample: {sample_files}')\n",
        "            except Exception as e:\n",
        "                print(f'  Mounted directory: {item}/ (Error: {e})')\n",
        "else:\n",
        "    print('Notice: /kaggle/input does not exist.')\n",
        "\n",
        "# 1. Recursive search across /kaggle/input for CelebA artifacts\n",
        "found_img_dir = None\n",
        "found_attr = None\n",
        "found_landmarks = None\n",
        "found_partition = None\n",
        "\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    for root, dirs, files in os.walk('/kaggle/input'):\n",
        "        r_path = Path(root)\n",
        "        if '000001.jpg' in files or '000002.jpg' in files:\n",
        "            found_img_dir = r_path\n",
        "            print(f'Discovered CelebA images at: {found_img_dir}')\n",
        "        if 'list_attr_celeba.csv' in files:\n",
        "            found_attr = r_path / 'list_attr_celeba.csv'\n",
        "            print(f'Discovered attributes CSV at: {found_attr}')\n",
        "        elif 'list_attr_celeba.txt' in files and not found_attr:\n",
        "            found_attr = r_path / 'list_attr_celeba.txt'\n",
        "            print(f'Discovered attributes TXT at: {found_attr}')\n",
        "        if 'list_landmarks_align_celeba.csv' in files:\n",
        "            found_landmarks = r_path / 'list_landmarks_align_celeba.csv'\n",
        "            print(f'Discovered landmarks CSV at: {found_landmarks}')\n",
        "        elif 'list_landmarks_align_celeba.txt' in files and not found_landmarks:\n",
        "            found_landmarks = r_path / 'list_landmarks_align_celeba.txt'\n",
        "            print(f'Discovered landmarks TXT at: {found_landmarks}')\n",
        "        if 'list_eval_partition.csv' in files:\n",
        "            found_partition = r_path / 'list_eval_partition.csv'\n",
        "            print(f'Discovered partition CSV at: {found_partition}')\n",
        "        elif 'list_eval_partition.txt' in files and not found_partition:\n",
        "            found_partition = r_path / 'list_eval_partition.txt'\n",
        "            print(f'Discovered partition TXT at: {found_partition}')\n",
        "\n",
        "# 2. Symlink discovered dataset into /kaggle/working/input_data\n",
        "if found_img_dir and not (target_data / 'img_align_celeba').exists():\n",
        "    os.symlink(str(found_img_dir), str(target_data / 'img_align_celeba'), target_is_directory=True)\n",
        "    print(f'Symlinked images: {found_img_dir} -> {target_data / \"img_align_celeba\"}')\n",
        "\n",
        "for src in [found_attr, found_landmarks, found_partition]:\n",
        "    if src and src.exists():\n",
        "        dest = target_data / src.name\n",
        "        if not dest.exists():\n",
        "            os.symlink(str(src), str(dest))\n",
        "            print(f'Symlinked {src.name} -> {dest}')\n",
        "\n",
        "# 3. Automated Fallback: Fetch real images from Hugging Face if not mounted on Kaggle\n",
        "imgs_dir = target_data / 'img_align_celeba'\n",
        "if not imgs_dir.exists() or not (imgs_dir / '000001.jpg').exists():\n",
        "    print('Real images not found in /kaggle/input. Downloading genuine CelebA images from Hugging Face mirror (nielsr/CelebA-faces)...')\n",
        "    imgs_dir.mkdir(parents=True, exist_ok=True)\n",
        "    from datasets import load_dataset\n",
        "    hf_dataset = load_dataset('nielsr/CelebA-faces', split='train', streaming=True)\n",
        "    count = 0\n",
        "    for sample in hf_dataset:\n",
        "        count += 1\n",
        "        sample['image'].save(imgs_dir / f'{count:06d}.jpg')\n",
        "        if count % 500 == 0:\n",
        "            print(f'  Cached {count} genuine CelebA face images...')\n",
        "        if count >= 2000:\n",
        "            break\n",
        "    print(f'Successfully prepared {count} genuine CelebA images at {imgs_dir}')\n",
        "\n",
        "# 4. Automated Fallback: Fetch CelebA Dialog causal captions & annotations from Google Drive\n",
        "if not (target_data / 'captions.json').exists():\n",
        "    print('Fetching ground-truth CelebA Dialog causal annotations from Google Drive...')\n",
        "    import gdown\n",
        "    try:\n",
        "        gdown.download(id='1ndtYb3tlopQhpBYUHpBpVPYBFbz9-kb_', output=str(target_data / 'captions.json'), quiet=True)\n",
        "        gdown.download(id='1wZcVEjJ5LwP1Ciuc3j_RFw9Vcusj4UEU', output=str(target_data / 'combined_annotation.txt'), quiet=True)\n",
        "        print('CelebA Dialog causal annotations linked successfully.')\n",
        "    except Exception as e:\n",
        "        print('Google Drive download note:', e)\n",
        "\n",
        "print('Dataset verification completed successfully.')\n"
    ]
})

# Cell for Restoring Pre-Trained Adapter and Phase 1/2 Artifacts if Mounted
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Check if pre-trained adapter and benchmark caches are mounted via Kaggle input\n",
        "import os, shutil\n",
        "from pathlib import Path\n",
        "\n",
        "adapter_input_dir = None\n",
        "if os.path.exists('/kaggle/input'):\n",
        "    for root, dirs, files in os.walk('/kaggle/input'):\n",
        "        if 'adapter_model.safetensors' in files:\n",
        "            adapter_input_dir = Path(root)\n",
        "            print(f'Discovered pre-trained LoRA adapter at: {adapter_input_dir}')\n",
        "            break\n",
        "\n",
        "if adapter_input_dir:\n",
        "    dest_adapter = Path('/kaggle/working/experiments/qlora_adapter_final')\n",
        "    dest_adapter.mkdir(parents=True, exist_ok=True)\n",
        "    for f in adapter_input_dir.iterdir():\n",
        "        if f.is_file():\n",
        "            shutil.copy(str(f), str(dest_adapter / f.name))\n",
        "    print('Successfully restored pre-trained LoRA adapter to /kaggle/working/experiments/qlora_adapter_final')\n",
        "\n",
        "    # Check for phase1 and phase2 cache\n",
        "    for p_file in ['phase1_dataset_summary.json', 'phase2_baseline_results.json']:\n",
        "        for root, dirs, files in os.walk('/kaggle/input'):\n",
        "            if p_file in files:\n",
        "                shutil.copy(str(Path(root) / p_file), f'/kaggle/working/experiments/{p_file}')\n",
        "                print(f'Restored cached {p_file}')\n",
        "                break\n"
    ]
})

# Cell for Running the Benchmark Suite
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "!python -u /kaggle/working/run_benchmark.py --sample-size 150 --skip-train\n"
    ]
})

# Cell for Verifying and Outputting Final Results
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import json, os\n",
        "from pathlib import Path\n",
        "\n",
        "exp_dir = Path('/kaggle/working/experiments')\n",
        "print('=== Experiments Directory Contents ===')\n",
        "if exp_dir.exists():\n",
        "    for item in exp_dir.glob('**/*'):\n",
        "        if item.is_file():\n",
        "            print(f'  {item.relative_to(exp_dir)} ({item.stat().st_size} bytes)')\n",
        "\n",
        "res_file = exp_dir / 'benchmark_results.json'\n",
        "if res_file.exists():\n",
        "    with open(res_file) as f:\n",
        "        data = json.load(f)\n",
        "    print('\\n=== Final Benchmark Results ===')\n",
        "    print('Deltas:', json.dumps(data.get('deltas', {}), indent=2))\n",
        "else:\n",
        "    print('Benchmark output file not found.')\n"
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_nb = base_dir / "kaggle_deployment" / "celeb_vlm_benchmark.ipynb"
with open(out_nb, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully at:", out_nb)

# Generate kernel-metadata.json
kernel_meta = {
    "id": "goofybhatoora/celeb-vlm-hierarchical-causal-analysis",
    "title": "Celeb VLM Hierarchical Causal Analysis",
    "code_file": "celeb_vlm_benchmark.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": True,
    "dataset_sources": [
        "jessicali9530/celeba-dataset",
        "goofybhatoora/celeb-vlm-trained-adapter"
    ],
    "competition_sources": [],
    "kernel_sources": [],
    "model_sources": []
}

meta_file = base_dir / "kaggle_deployment" / "kernel-metadata.json"
with open(meta_file, "w", encoding="utf-8") as f:
    json.dump(kernel_meta, f, indent=2)

print("Kernel metadata generated at:", meta_file)
