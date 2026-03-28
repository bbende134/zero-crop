#!/usr/bin/env python3
"""
Hybrid cluster labeling: centroid-to-CORINE matching + Qwen generative summary.

For each cluster (from all 3 methods), produces:
  - Top-3 nearest CORINE class matches (cosine similarity)
  - Free-text summary from Qwen3.5-4B generation

Saves results to data_corine/clusters/cluster_labels.json
"""

import json
import numpy as np
from pathlib import Path

# ── 1. Load data ─────────────────────────────────────────────────────

OUT_DIR = Path("data_corine/clusters")
embeddings = np.load(OUT_DIR / "embeddings.npy")

meta = []
with open(OUT_DIR / "meta.jsonl") as f:
    for line in f:
        meta.append(json.loads(line))

with open("data_corine/corine_classes_includes.json") as f:
    corine_classes = json.load(f)

labels_all = {
    "kmeans": np.load(OUT_DIR / "labels_kmeans.npy"),
    "hdbscan": np.load(OUT_DIR / "labels_hdbscan.npy"),
    "optics": np.load(OUT_DIR / "labels_optics.npy"),
}

print(f"Loaded {len(embeddings)} embeddings, {len(corine_classes)} CORINE classes")

# ── 2. Embed CORINE class descriptions ───────────────────────────────

import torch
from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = Qwen3EmbeddingAdapter(freeze_encoder=True, lora=False).to(device).eval()

corine_codes = list(corine_classes.keys())
corine_descs = list(corine_classes.values())

print("Encoding CORINE class descriptions...")
with torch.no_grad():
    class_embs = encoder.encode_batch(corine_descs, chunk_size=16, normalize=True)
class_embs = class_embs.cpu().float().numpy()  # (44, 2560)

# ── 3. Load Qwen for generation ──────────────────────────────────────

from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-4B"
print(f"Loading {MODEL_ID} for generation...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
gen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, trust_remote_code=True
).to(device).eval()


def generate_summary(sentences: list[str]) -> str:
    joined = "\n".join(f"- {s[:200]}" for s in sentences[:8])
    messages = [
        {"role": "system", "content": (
            "You label semantic clusters. Given sentences from CORINE land cover "
            "Wikipedia articles, output ONLY a short label (max 15 words) for the "
            "land cover type or topic that unifies them. No explanation."
        )},
        {"role": "user", "content": joined},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = gen_model.generate(
            **inputs, max_new_tokens=30,
            do_sample=False, temperature=1.0,
        )
    full = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return full.strip()


# ── 4. Label each cluster for each method ─────────────────────────────

def cosine_sim(a, b):
    a_n = a / (np.linalg.norm(a) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_n @ a_n


def label_clusters(labels, method_name):
    unique = sorted(set(labels))
    cluster_info = {}

    for cid in unique:
        mask = labels == cid
        count = int(mask.sum())
        cluster_embs = embeddings[mask]
        centroid = cluster_embs.mean(axis=0)

        # Top-3 CORINE class matches
        sims = cosine_sim(centroid, class_embs)
        top3_idx = np.argsort(sims)[-3:][::-1]
        matches = [
            {"code": corine_codes[i], "similarity": round(float(sims[i]), 4)}
            for i in top3_idx
        ]

        # Representative sentences (closest to centroid)
        dists = np.linalg.norm(cluster_embs - centroid, axis=1)
        nearest_idx = np.argsort(dists)[:8]
        global_idx = np.where(mask)[0][nearest_idx]
        rep_sentences = [meta[i]["sentence"] for i in global_idx]

        # Generative summary
        label_str = cid if cid != -1 else "noise"
        print(f"  [{method_name}] cluster {label_str} ({count} pts)...", end=" ", flush=True)
        summary = generate_summary(rep_sentences)
        print("done")

        cluster_info[str(cid)] = {
            "count": count,
            "top_corine_matches": matches,
            "representative_sentences": rep_sentences[:5],
            "qwen_summary": summary,
        }

    return cluster_info


results = {}
for method, labels in labels_all.items():
    print(f"\n=== {method.upper()} ===")
    results[method] = label_clusters(labels, method)

# ── 5. Save ───────────────────────────────────────────────────────────

out_path = OUT_DIR / "cluster_labels.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {out_path}")
