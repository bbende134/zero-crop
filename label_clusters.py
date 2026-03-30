#!/usr/bin/env python3
# Ensure CUDA 13 libs (libnvJitLink.so.13 etc.) are on LD_LIBRARY_PATH before
# any native extension (bitsandbytes, triton) is loaded.
import os, sys as _sys
_cu13 = os.path.join(os.path.dirname(__file__),
    ".venv/lib/python3.12/site-packages/nvidia/cu13/lib")
if os.path.isdir(_cu13) and _cu13 not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _cu13 + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(_sys.executable, [_sys.executable] + _sys.argv)
del _cu13

"""
Hybrid cluster labeling: centroid-to-CORINE matching + Qwen generative summary.

For each cluster (from all 3 methods), produces:
  - Top-3 nearest CORINE class matches (cosine similarity)
  - Free-text summary from Qwen3.5-4B generation (with thinking)

Generation is parallelised across cuda:0 and cuda:1 — each GPU handles half
the clusters for each method simultaneously.

Saves results to data_corine/clusters/cluster_labels_finetuned_aug_v2.json
"""

import json
import numpy as np
from pathlib import Path

# ── 1. Load data ──────────────────────────────────────────────────────

OUT_DIR = Path("data_corine/clusters")
embeddings = np.load(OUT_DIR / "embeddings_aug.npy")

meta = []
with open(OUT_DIR / "meta_aug.jsonl") as f:
    for line in f:
        meta.append(json.loads(line))

with open("data_corine/corine_classes_includes.json") as f:
    corine_classes = json.load(f)

labels_all = {
    "kmeans":   np.load(OUT_DIR / "labels_kmeans_aug.npy"),
    "hdbscan":  np.load(OUT_DIR / "labels_hdbscan_aug.npy"),
    "optics":   np.load(OUT_DIR / "labels_optics_aug.npy"),
}

print(f"Loaded {len(embeddings)} embeddings, {len(corine_classes)} CORINE classes")

# ── 2. Embed CORINE class descriptions (encoder on cuda:0) ───────────

import torch
from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter

FINETUNED_LORA_PATH = Path("training_data/qwen_lora_finetuned.pt")
enc_device = "cuda:0" if torch.cuda.is_available() else "cpu"

encoder = Qwen3EmbeddingAdapter(freeze_encoder=True, lora=True).to(enc_device)
if FINETUNED_LORA_PATH.exists():
    print(f"Loading fine-tuned LoRA weights from {FINETUNED_LORA_PATH}")
    ckpt = torch.load(FINETUNED_LORA_PATH, map_location=enc_device, weights_only=False)
    missing, unexpected = encoder._model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    print(f"  Loaded LoRA weights (missing={len(missing)}, unexpected={len(unexpected)})")
else:
    print(f"WARNING: {FINETUNED_LORA_PATH} not found, using base model")

encoder.eval()
corine_codes = list(corine_classes.keys())
corine_descs = list(corine_classes.values())

print("Encoding CORINE class descriptions...")
with torch.no_grad():
    class_embs = encoder.encode_batch(corine_descs, chunk_size=16, normalize=True)
class_embs = class_embs.cpu().float().numpy()  # (44, 2560)

# Free encoder VRAM before spawning generation workers
del encoder
torch.cuda.empty_cache()

# ── 3. Precompute cluster jobs (CPU, main process) ────────────────────

def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_n @ a_n


def build_cluster_jobs(labels):
    """Return list of dicts with everything needed for generation."""
    jobs = []
    for cid in sorted(set(labels)):
        mask = labels == cid
        count = int(mask.sum())
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)

        sims = cosine_sim(centroid, class_embs)
        top3_idx = np.argsort(sims)[-3:][::-1]
        matches = [
            {"code": corine_codes[i], "similarity": round(float(sims[i]), 4)}
            for i in top3_idx
        ]

        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        n_rep = min(20, len(dists))
        nearest_local = np.argsort(dists)[:n_rep]
        rng = np.random.default_rng(seed=int(cid) % (2**31))
        pool = np.setdiff1d(np.arange(len(dists)), nearest_local)
        n_rand = min(20, len(pool))
        rand_local = (rng.choice(pool, size=n_rand, replace=False)
                      if n_rand > 0 else np.array([], dtype=int))
        combined_local = np.concatenate([nearest_local, rand_local]).astype(int)
        global_idx = np.where(mask)[0][combined_local]
        rep_sentences = [meta[i]["sentence"] for i in global_idx]

        jobs.append({
            "cid": int(cid),
            "count": count,
            "top_corine_matches": matches,
            "rep_sentences": rep_sentences,
        })
    return jobs


# ── 4. Generation worker ──────────────────────────────────────────────

def generation_worker(jobs_chunk: list[dict], device: str, result_path: str):
    """Load gen model on `device`, process jobs, write partial JSON."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch, json, re
    from tqdm import tqdm

    MODEL_ID = "Qwen/Qwen3.5-4B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    gen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True, device_map=device,
    ).eval()

    SYSTEM = (
        "You are a geospatial land cover expert analysing CORINE land cover taxonomy. "
        "You are given a representative sample of sentences from a single semantic cluster "
        "extracted from Wikipedia articles about CORINE land cover classes.\n\n"
        "Your task:\n"
        "1. Identify the dominant land cover theme or ecological concept that unifies these sentences.\n"
        "2. Note any secondary themes or outlier topics present.\n"
        "3. Output a concise cluster label (max 10 words) followed by a 1-2 sentence description "
        "of the cluster's semantic content and any notable heterogeneity.\n\n"
        "Format strictly as:\n"
        "LABEL: <short label>\n"
        "DESC: <description>"
    )

    partial = {}
    for job in tqdm(jobs_chunk, desc=device, position=0 if device == "cuda:0" else 1):
        sentences = job["rep_sentences"]
        joined = "\n".join(f"[{i+1}] {s}" for i, s in enumerate(sentences[:40]))
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": joined},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=4096).to(device)
        THINK_END_ID = 248069
        THINKING_BUDGET = 1024  # tokens allocated for reasoning
        RESPONSE_TOKENS = 120   # tokens for LABEL + DESC

        with torch.no_grad():
            # Pass 1: let the model think
            thinking_out = gen_model.generate(
                **inputs,
                max_new_tokens=THINKING_BUDGET,
                do_sample=False,
                temperature=1.0,
            )
            thinking_ids = thinking_out.sequences if hasattr(thinking_out, "sequences") else thinking_out
            # Force-close the thinking block then generate the response
            think_end = torch.tensor([[THINK_END_ID]], device=device)
            combined = torch.cat([thinking_ids, think_end], dim=1)
            combined_mask = torch.ones(combined.shape, dtype=torch.long, device=device)
            # Pass 2: generate actual response after </think>
            full_out = gen_model.generate(
                input_ids=combined,
                attention_mask=combined_mask,
                max_new_tokens=RESPONSE_TOKENS,
                do_sample=False,
                temperature=1.0,
            )

        full_ids = full_out.sequences if hasattr(full_out, "sequences") else full_out
        response_ids = full_ids[0][combined.shape[1]:]
        decoded = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        partial[str(job["cid"])] = {
            "count": job["count"],
            "top_corine_matches": job["top_corine_matches"],
            "representative_sentences": sentences[:10],
            "qwen_summary": decoded,
        }

    with open(result_path, "w") as f:
        json.dump(partial, f, ensure_ascii=False)
    print(f"\n[{device}] done — wrote {len(partial)} clusters to {result_path}")


# ── 5. Run each method with dual-GPU parallel generation ─────────────

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    results = {}

    for method, labels in labels_all.items():
        print(f"\n=== {method.upper()} ===")
        jobs = build_cluster_jobs(labels)
        print(f"  {len(jobs)} clusters to label")

        mid = len(jobs) // 2
        chunk0, chunk1 = jobs[:mid], jobs[mid:]

        tmp0 = str(OUT_DIR / f"_tmp_{method}_gpu0.json")
        tmp1 = str(OUT_DIR / f"_tmp_{method}_gpu1.json")

        p0 = mp.Process(target=generation_worker, args=(chunk0, "cuda:0", tmp0))
        p1 = mp.Process(target=generation_worker, args=(chunk1, "cuda:1", tmp1))
        p0.start(); p1.start()
        p0.join();  p1.join()

        # Merge partial results preserving sorted cluster order
        merged = {}
        for tmp in (tmp0, tmp1):
            with open(tmp) as f:
                merged.update(json.load(f))
            Path(tmp).unlink()

        results[method] = dict(sorted(merged.items(), key=lambda x: int(x[0])))
        print(f"  {method}: {len(merged)} clusters labelled")

    # ── 6. Save ───────────────────────────────────────────────────────────

    out_path = OUT_DIR / "cluster_labels_finetuned_aug_v2.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
