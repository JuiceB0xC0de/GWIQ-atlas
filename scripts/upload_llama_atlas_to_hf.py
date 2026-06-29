#!/usr/bin/env python3
"""
Upload the full Llama-3.1-8B atlas tree from Modal volume to HuggingFace.

Run this locally after the Modal workflow completes:

    python scripts/upload_llama_atlas_to_hf.py

Requires:
    - modal CLI configured
    - HF_TOKEN env var or huggingface login
    - The atlas has been built on the gwiq-atlas-output volume

The script downloads /gwiq-output/atlas from Modal to a local temp dir,
then pushes it to juiceb0xc0de/llama-3.1-8b-atlas.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ID = "juiceb0xc0de/llama-3.1-8b-atlas"
VOLUME_NAME = "gwiq-atlas-output"
REMOTE_ATLAS_PATH = "atlas"


def modal_volume_get(remote_path: str, local_dir: Path):
    """Download a path from the Modal volume to a local directory."""
    cmd = ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_dir)]
    print(f"[download] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"modal volume get failed: {result.returncode}")


def upload_to_hf(local_atlas_dir: Path, repo_id: str):
    """Upload the atlas tree to HuggingFace Hub as a dataset."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Installing huggingface_hub...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
        from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=token)

    # Ensure repo exists (private by default; change to public=True if desired)
    try:
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True, private=False)
        print(f"[hf] repo ready: {repo_id}")
    except Exception as e:
        print(f"[hf] repo create/exists check: {e}")

    # Upload all files under local_atlas_dir, preserving tree structure
    print(f"[hf] uploading {local_atlas_dir} to {repo_id} ...")
    api.upload_folder(
        folder_path=str(local_atlas_dir),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        multi_commits=True,
    )
    print(f"[hf] upload complete: https://huggingface.co/datasets/{repo_id}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local_atlas = tmp_path / "atlas"
        local_atlas.mkdir(parents=True, exist_ok=True)

        print("=== downloading atlas from Modal volume ===")
        modal_volume_get(REMOTE_ATLAS_PATH, local_atlas)

        # Verify key files
        sqlite = local_atlas / "atlas.sqlite"
        manifest = local_atlas / "manifest.json"
        if not sqlite.exists():
            raise SystemExit(f"atlas.sqlite not found after download: {sqlite}")
        if not manifest.exists():
            raise SystemExit(f"manifest.json not found after download: {manifest}")

        print(f"[verify] atlas.sqlite: {sqlite.stat().st_size / 1e6:.1f} MB")
        print(f"[verify] manifest.json: {manifest.stat().st_size} bytes")

        print("\n=== uploading to HuggingFace ===")
        upload_to_hf(local_atlas, REPO_ID)

    print("\n=== done ===")


if __name__ == "__main__":
    main()
