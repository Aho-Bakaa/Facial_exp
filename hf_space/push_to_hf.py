"""
Uploads the local hf_space folder directly to the Hugging Face Space.
Usage:
    python push_to_hf.py --token <HF_WRITE_TOKEN>
or
    set HF_TOKEN=<HF_WRITE_TOKEN>
    python push_to_hf.py
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN", None), help="Hugging Face Write Token")
    parser.add_argument("--space_id", type=str, default="Anmolbakaa/facial-vlm-evaluator", help="Hugging Face Space ID")
    args = parser.parse_args()

    api = HfApi(token=args.token)
    repo_id = args.space_id
    space_dir = Path(__file__).parent

    print(f"Uploading files from {space_dir} to Space {repo_id}...")
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=repo_id,
        repo_type="space",
        ignore_patterns=["push_to_hf.py", "__pycache__/*"],
    )
    print(f"Successfully uploaded! View your Space at: https://huggingface.co/spaces/{repo_id}")

if __name__ == "__main__":
    main()
