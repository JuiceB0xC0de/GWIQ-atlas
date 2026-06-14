#!/usr/bin/env python3
"""
Run this inside a Modal GPU shell that has qwip-atlas + sae-lens installed.

Uses OpenMOSS Llama-Scope SAEs (SAELens-compatible) for full MLP coverage on
Llama-3.1-8B-Base. Variants "32x" / "64x" are kept as output labels but map to
Llama-Scope's 8x (32K features) and 32x (128K features) MLP releases. Writes
sae_l<N>_<variant>.npz to /gwiq-output/llama-3-8b/sae/.

After this finishes:
    qwip-build-atlas --atlas /gwiq-output/atlas index
    python -m qwip_atlas.merge_sae --variant 32x --atlas /gwiq-output/atlas --sae-dir /gwiq-output/llama-3-8b/sae
    python -m qwip_atlas.merge_sae --variant 64x --atlas /gwiq-output/atlas --sae-dir /gwiq-output/llama-3-8b/sae
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.1-8B"
CORPUS_REPO = "juiceb0xc0de/mapping-prompts"
SAE_VARIANTS = {
    # output_label: (SAELens release, sae_id suffix for layer L)
    "32x": ("llama_scope_lxm_8x", "m_8x"),
    "64x": ("llama_scope_lxm_32x", "m_32x"),
}
N_LAYERS = 32
OUTPUT_DIR = Path("/gwiq-output/llama-3-8b/sae")


def _load_jsonl_from_hf(repo_id: str, filename: str, prompt_key: str, token: str | None) -> list[dict]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=token)
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get(prompt_key, "")
            if text:
                rows.append({"prompt": text, **rec})
    return rows


def _resolve_layers(model):
    import torch.nn as nn
    from collections import deque
    queue = deque([model])
    while queue:
        m = queue.popleft()
        layers = getattr(m, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return list(layers)
        for _, child in m.named_children():
            queue.append(child)
    raise RuntimeError("Cannot find layers ModuleList")


def _load_saes(variant: str, device: str, dtype):
    from sae_lens import SAE

    release, suffix = SAE_VARIANTS[variant]
    saes = {}
    for layer in range(N_LAYERS):
        sae_id = f"l{layer}{suffix}"
        print(f"[sae] loading {variant} {sae_id} (release={release}) ...")
        sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        saes[layer] = sae.to(dtype=dtype)
    return saes


def _encode_pass(model, tokenizer, saes, corpus, group_fn, n_groups, batch_size, max_length, device):
    import torch
    from tqdm import tqdm

    target_layers = sorted(saes.keys())
    first_sae = next(iter(saes.values()))
    d_sae = first_sae.cfg.d_sae

    acc = {
        L: {
            "sum": torch.zeros(n_groups, d_sae, dtype=torch.float64, device=device),
            "sumsq": torch.zeros(n_groups, d_sae, dtype=torch.float64, device=device),
            "active": torch.zeros(d_sae, dtype=torch.float64, device=device),
        }
        for L in target_layers
    }
    group_n = torch.zeros(n_groups, dtype=torch.float64, device=device)

    n_batches = (len(corpus) + batch_size - 1) // batch_size
    for bstart in tqdm(range(0, len(corpus), batch_size), total=n_batches, desc="batches"):
        batch = corpus[bstart : bstart + batch_size]
        prompts = [r["prompt"] for r in batch]
        groups = [group_fn(r) for r in batch]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        mask = enc["attention_mask"].bool()
        enc = {k: v.to(device) for k, v in enc.items()}

        captured: dict[int, torch.Tensor] = {}

        def make_hook(layer: int):
            def _hook(mod, inp, out):
                captured[layer] = (out[0] if isinstance(out, tuple) else out).detach()
            return _hook

        layers = _resolve_layers(model)
        handles = [layers[L].mlp.register_forward_hook(make_hook(L)) for L in target_layers]

        with torch.no_grad():
            model(**enc, use_cache=False)

        for h in handles:
            h.remove()

        for L in target_layers:
            mlp_out = captured[L]
            sae = saes[L]
            for gid in set(groups):
                idx = [i for i, g in enumerate(groups) if g == gid]
                if not idx:
                    continue
                sub = mlp_out[idx]
                sub_mask = mask[idx]
                tokens = sub[sub_mask]
                if tokens.numel() == 0:
                    continue

                acts = sae.encode(tokens)
                k = getattr(sae.cfg, "k", None) or acts.shape[-1]
                vals, inds = acts.topk(k, dim=-1)
                vals = vals.clamp_min(0).double()

                flat_i = inds.reshape(-1)
                flat_v = vals.reshape(-1)
                acc[L]["sum"][gid].index_add_(0, flat_i, flat_v)
                acc[L]["sumsq"][gid].index_add_(0, flat_i, flat_v * flat_v)
                acc[L]["active"].index_add_(0, flat_i, torch.ones_like(flat_v))
                if L == target_layers[0]:
                    group_n[gid] += tokens.shape[0]

        captured.clear()

    return acc, group_n


def _fstat_from_sums(acc_L, group_n):
    import torch

    s, ss = acc_L["sum"], acc_L["sumsq"]
    n = group_n.unsqueeze(1).clamp_min(1)
    G = s.shape[0]
    N = group_n.sum().clamp_min(1)
    grand = s.sum(0) / N
    mean_g = s / n
    ssb = (group_n.unsqueeze(1) * (mean_g - grand) ** 2).sum(0)
    ssw = (ss - s**2 / n).sum(0).clamp_min(0)
    df_b, df_w = max(G - 1, 1), torch.clamp(N - G, min=1)
    F = (ssb / df_b) / (ssw / df_w + 1e-12)
    return F.float().cpu().numpy()


def run_variant(variant: str, batch_size: int = 16, max_length: int = 256):
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n=== SAE variant: {variant} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=dtype, device_map=device, token=token)
    model.eval()

    saes = _load_saes(variant, device, dtype)
    first_sae = next(iter(saes.values()))
    d_sae = first_sae.cfg.d_sae

    # Topic pass
    topic = _load_jsonl_from_hf(CORPUS_REPO, "prompts.jsonl", "prompt", token)
    cats = sorted({r["category"] for r in topic})
    cat_id = {c: i for i, c in enumerate(cats)}
    print(f"[sae] topic: {len(topic)} prompts, {len(cats)} categories")

    t_acc, t_n = _encode_pass(
        model, tokenizer, saes, topic,
        lambda r: cat_id[r["category"]], len(cats),
        batch_size, max_length, device,
    )

    # Compliance pass
    corp = _load_jsonl_from_hf(CORPUS_REPO, "corporate_stems.jsonl", "text", token)
    auth = _load_jsonl_from_hf(CORPUS_REPO, "authentic_bella_samples.jsonl", "text", token)
    bcorpus = [{"prompt": r["prompt"], "_g": 0} for r in corp] + [{"prompt": r["prompt"], "_g": 1} for r in auth]
    print(f"[sae] compliance: {len(corp)} corp + {len(auth)} auth")

    b_acc, b_n = _encode_pass(
        model, tokenizer, saes, bcorpus,
        lambda r: r["_g"], 2,
        batch_size, max_length, device,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    N_topic = float(t_n.sum().item())

    for L in sorted(saes.keys()):
        topic_fstat = _fstat_from_sums(t_acc[L], t_n)
        activation_rate = (t_acc[L]["active"] / max(N_topic, 1)).float().cpu().numpy()
        bouncer_fstat = _fstat_from_sums(b_acc[L], b_n)

        s, n = b_acc[L]["sum"], b_n.clamp_min(1).unsqueeze(1)
        mean = (s / n).float().cpu().numpy()
        mean_corp, mean_auth = mean[0], mean[1]
        bouncer_delta = mean_corp - mean_auth

        out_path = OUTPUT_DIR / f"sae_l{L}_{variant}.npz"
        np.savez_compressed(
            out_path,
            categories=np.array(cats),
            topic_fstat=topic_fstat.astype(np.float32),
            activation_rate=activation_rate.astype(np.float32),
            bouncer_fstat=bouncer_fstat.astype(np.float32),
            bouncer_delta=bouncer_delta.astype(np.float32),
            mean_corp=mean_corp.astype(np.float32),
            mean_auth=mean_auth.astype(np.float32),
        )
        print(f"[sae] wrote {out_path}")

    return {"variant": variant, "layers": N_LAYERS, "features": d_sae}


def main(batch_size: int = 16, max_length: int = 256):
    for variant in ["32x", "64x"]:
        run_variant(variant, batch_size, max_length)
    print("\n=== done ===")
    print("Next:")
    print("  qwip-build-atlas --atlas /gwiq-output/atlas index")
    print("  python -m qwip_atlas.merge_sae --variant 32x --atlas /gwiq-output/atlas --sae-dir /gwiq-output/llama-3-8b/sae")
    print("  python -m qwip_atlas.merge_sae --variant 64x --atlas /gwiq-output/atlas --sae-dir /gwiq-output/llama-3-8b/sae")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()
    main(args.batch_size, args.max_length)
