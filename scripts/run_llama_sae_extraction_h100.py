#!/usr/bin/env python3
"""
H100/H200 chunked Llama-3.1-8B SAE extraction.

Optimized for 80-140 GB VRAM. Loads the model once, then processes MLP layers in
CHUNK_SIZE chunks. SAEs for each chunk are loaded once per chunk, kept on GPU,
and activations never leave GPU. This avoids OOM from loading all 32 SAEs at
once while still being much faster than CPU-round-trip strategies.

Uses OpenMOSS Llama-Scope SAEs:
  "32x" -> llama_scope_lxm_8x  (32K features)
  "64x" -> llama_scope_lxm_32x (128K features)

Writes sae_l<N>_<variant>.npz to /gwiq-output/llama-3-8b/sae/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.1-8B"
CORPUS_REPO = "juiceb0xc0de/mapping-prompts"
SAE_VARIANTS = {
    "32x": ("llama_scope_lxm_8x", "m_8x"),
    "64x": ("llama_scope_lxm_32x", "m_32x"),
}
N_LAYERS = 32
CHUNK_SIZE = 8
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


def _load_sae(variant: str, layer: int, device: str, dtype):
    from sae_lens import SAE

    release, suffix = SAE_VARIANTS[variant]
    sae_id = f"l{layer}{suffix}"
    print(f"[sae] loading {variant} {sae_id} ...")
    result = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae = result[0] if isinstance(result, tuple) else result
    return sae.to(dtype=dtype)


def _encode_chunk_pass(model, tokenizer, saes: dict, chunk_layers: list, corpus, group_fn, n_groups,
                       batch_size, max_length, device, desc: str):
    import torch
    from tqdm import tqdm

    acc = {
        L: {
            "sum": torch.zeros(n_groups, saes[L].cfg.d_sae, dtype=torch.float64, device=device),
            "sumsq": torch.zeros(n_groups, saes[L].cfg.d_sae, dtype=torch.float64, device=device),
            "active": torch.zeros(saes[L].cfg.d_sae, dtype=torch.float64, device=device),
        }
        for L in chunk_layers
    }
    group_n = torch.zeros(n_groups, dtype=torch.float64, device=device)

    layers = _resolve_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(target_layer: int):
        def _hook(mod, inp, out):
            captured[target_layer] = (out[0] if isinstance(out, tuple) else out).detach()
        return _hook

    n_batches = (len(corpus) + batch_size - 1) // batch_size
    for bstart in tqdm(range(0, len(corpus), batch_size), total=n_batches, desc=desc):
        batch = corpus[bstart : bstart + batch_size]
        prompts = [r["prompt"] for r in batch]
        groups = [group_fn(r) for r in batch]

        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        mask = enc["attention_mask"].bool()
        enc = {k: v.to(device) for k, v in enc.items()}

        captured.clear()
        handles = [layers[L].mlp.register_forward_hook(make_hook(L)) for L in chunk_layers]
        with torch.no_grad():
            model(**enc, use_cache=False)
        for h in handles:
            h.remove()

        for L in chunk_layers:
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
                if L == chunk_layers[0]:
                    group_n[gid] += tokens.shape[0]

        captured.clear()

    return acc, group_n


def _fstat_from_sums(acc, group_n):
    import torch

    s, ss = acc["sum"], acc["sumsq"]
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


def _write_layer_npz(layer: int, variant: str, t_acc, t_n, b_acc, b_n, cats, N_topic: float):
    import numpy as np

    topic_fstat = _fstat_from_sums(t_acc, t_n)
    activation_rate = (t_acc["active"] / max(N_topic, 1)).float().cpu().numpy()
    bouncer_fstat = _fstat_from_sums(b_acc, b_n)

    s, n = b_acc["sum"], b_n.clamp_min(1).unsqueeze(1)
    mean = (s / n).float().cpu().numpy()
    mean_corp, mean_auth = mean[0], mean[1]
    bouncer_delta = mean_corp - mean_auth

    out_path = OUTPUT_DIR / f"sae_l{layer}_{variant}.npz"
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


def run_variant(variant: str, batch_size: int = 128, max_length: int = 256):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    device = "cuda"
    dtype = torch.bfloat16

    print(f"\n=== SAE variant: {variant} ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=dtype, device_map=device, token=token)
    model.eval()

    topic = _load_jsonl_from_hf(CORPUS_REPO, "prompts.jsonl", "prompt", token)
    cats = sorted({r["category"] for r in topic})
    cat_id = {c: i for i, c in enumerate(cats)}
    print(f"[sae] topic: {len(topic)} prompts, {len(cats)} categories")

    corp = _load_jsonl_from_hf(CORPUS_REPO, "corporate_stems.jsonl", "text", token)
    auth = _load_jsonl_from_hf(CORPUS_REPO, "authentic_bella_samples.jsonl", "text", token)
    bcorpus = [{"prompt": r["prompt"], "_g": 0} for r in corp] + [{"prompt": r["prompt"], "_g": 1} for r in auth]
    print(f"[sae] compliance: {len(corp)} corp + {len(auth)} auth")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    d_sae = None

    for chunk_start in range(0, N_LAYERS, CHUNK_SIZE):
        chunk_layers = list(range(chunk_start, min(chunk_start + CHUNK_SIZE, N_LAYERS)))
        print(f"\n[sae] === chunk layers {chunk_layers[0]}-{chunk_layers[-1]} ===")

        saes = {L: _load_sae(variant, L, device, dtype) for L in chunk_layers}
        if d_sae is None:
            d_sae = next(iter(saes.values())).cfg.d_sae

        t_acc_chunk, t_n = _encode_chunk_pass(
            model, tokenizer, saes, chunk_layers, topic,
            lambda r: cat_id[r["category"]], len(cats),
            batch_size, max_length, device,
            desc=f"topic L{chunk_layers[0]}-{chunk_layers[-1]}",
        )

        b_acc_chunk, b_n = _encode_chunk_pass(
            model, tokenizer, saes, chunk_layers, bcorpus,
            lambda r: r["_g"], 2,
            batch_size, max_length, device,
            desc=f"comp L{chunk_layers[0]}-{chunk_layers[-1]}",
        )

        N_topic = float(t_n.sum().item())
        for L in chunk_layers:
            _write_layer_npz(L, variant, t_acc_chunk[L], t_n, b_acc_chunk[L], b_n, cats, N_topic)

        del saes, t_acc_chunk, b_acc_chunk
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    return {"variant": variant, "layers": N_LAYERS, "features": d_sae}


def main(batch_size: int = 128, max_length: int = 256):
    for variant in ["32x", "64x"]:
        run_variant(variant, batch_size, max_length)
    print("\n=== done ===")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()
    main(args.batch_size, args.max_length)
