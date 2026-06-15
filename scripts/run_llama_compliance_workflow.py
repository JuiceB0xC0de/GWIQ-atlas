#!/usr/bin/env python3
"""
Run this inside a Modal GPU shell that has qwip-atlas installed.
The gwiq-atlas-output volume must be mounted at /gwiq-output.

Example launch from local:
    cd ~/modal/mapping-corpus
    modal shell atlas11.py::gpu_shell

Inside the shell:
    python /gwiq-output/run_llama_compliance_workflow.py

Or copy this file to /gwiq-output first:
    modal volume put gwiq-atlas-output \
        ~/GWIQ-atlas/scripts/run_llama_compliance_workflow.py \
        run_llama_compliance_workflow.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.1-8B"
CENSUS_DIR = Path("/gwiq-output/llama-3-8b/census")
ATLAS_DIR = Path("/gwiq-output/atlas")
CORPORA_DIR = Path("/gwiq-output/corpora")
COMPLIANCE_OUT = Path("/gwiq-output/llama-3-8b/compliance_behaviour_scores.json")


def run(cmd: list[str], **kwargs):
    print(f"\n[run] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise SystemExit(f"command failed with {result.returncode}: {' '.join(cmd)}")
    return result


def main():
    # 1. Verify census is present
    print("=== census files ===")
    npz_files = sorted(CENSUS_DIR.glob("l*_census_raw.npz"))
    print(f"found {len(npz_files)} layer census files")
    if len(npz_files) != 32:
        print("WARNING: expected 32 layers for Llama-3.1-8B", file=sys.stderr)

    # 2. Extract compliance-behaviour scores (corp vs authentic)
    print("\n=== compliance behaviour extraction ===")
    COMPLIANCE_OUT.parent.mkdir(parents=True, exist_ok=True)
    run([
        "qwip-atlas", "compliance-behaviour-local",
        "--model", MODEL_ID,
        "--positive", str(CORPORA_DIR / "corporate_stems.jsonl"),
        "--negative", str(CORPORA_DIR / "authentic_bella_samples.jsonl"),
        "--layers", "0-31",
        "--output", str(COMPLIANCE_OUT),
        "--batch-size", "16",
        "--device-map", "",
        "--positive-prompt-key", "text",
        "--negative-prompt-key", "text",
        "--positive-label", "corporate",
        "--negative-label", "authentic",
    ])

    # 3. Merge compliance scores into the atlas
    print("\n=== merge compliance behaviour ===")
    run([
        "qwip-build-atlas", "--atlas", str(ATLAS_DIR),
        "merge-compliance-behaviour", "--report", str(COMPLIANCE_OUT),
    ])

    # 4. Rebuild SQLite mirror + cross_layer summaries
    print("\n=== index ===")
    run(["qwip-build-atlas", "--atlas", str(ATLAS_DIR), "index"])

    # 5. Status check
    print("\n=== status ===")
    run(["qwip-build-atlas", "--atlas", str(ATLAS_DIR), "status"])

    # 6. Copy raw .npz census into atlas tree for HF upload
    #    (qwip-build-atlas defaults to no-copy for .npz to save space)
    print("\n=== stage raw census into atlas tree ===")
    for npz in npz_files:
        layer = int(npz.stem.split("_")[0][1:])
        census_dir = ATLAS_DIR / "layers" / str(layer) / "census"
        census_dir.mkdir(parents=True, exist_ok=True)
        dst = census_dir / "raw.npz"
        shutil.copy2(npz, dst)
        print(f"  copied L{layer:02d}: {npz.name} -> {dst}")

    print("\n=== done ===")
    print("Next: exit shell and run scripts/upload_llama_atlas_to_hf.py locally")


if __name__ == "__main__":
    main()
