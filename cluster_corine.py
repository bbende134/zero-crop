#!/usr/bin/env python3
"""
Cluster CORINE wiki sentences using Qwen3.5-4B embeddings.

Reads data_corine/corine_wiki_char_count.jsonl, embeds each sentence
via last-token pooling from Qwen3.5-4B, then clusters with KMeans.
Saves results + a 2D UMAP visualization.
"""

import json
import re
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── 1. Load & sentence-split ──────────────────────────────────────────

DATA_PATH = Path("data_corine/corine_wiki_char_count.jsonl")

records = []
with open(DATA_PATH) as f:
    for line in f:
        records.append(json.loads(line))

# Sentence-split each wiki text, keeping provenance
SENT_RE = re.compile(r'(?<=[.!?])\s+')

sentences = []   # list of str
meta = []        # list of dict(code, term, sent_idx)

for rec in records:
    code = rec["code"]
    for term, text in rec["wiki_texts"].items():
        sents = [s.strip() for s in SENT_RE.split(text) if len(s.strip()) > 30]
        for i, s in enumerate(sents):
            # Truncate very long "sentences" (probably parsing artefacts)
            sentences.append(s[:512])
            meta.append({"code": code, "term": term, "sent_idx": i})

print(f"Total sentences: {len(sentences)}")

# ── 2. Embed with Qwen3.5-4B (or load cached) ────────────────────────

OUT_DIR = Path("data_corine/clusters")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EMB_CACHE = OUT_DIR / "embeddings.npy"
META_CACHE = OUT_DIR / "meta.jsonl"

if EMB_CACHE.exists():
    print(f"Loading cached embeddings from {EMB_CACHE}")
    embeddings = np.load(EMB_CACHE)
    print(f"Embeddings shape: {embeddings.shape}")
else:
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Qwen3EmbeddingAdapter(freeze_encoder=True, lora=False).to(device).eval()

    BATCH = 32
    emb_list = []

    print("Encoding sentences...")
    for i in tqdm(range(0, len(sentences), BATCH)):
        batch = sentences[i : i + BATCH]
        with torch.no_grad():
            emb = encoder.encode_batch(batch, chunk_size=BATCH, normalize=True)
        emb_list.append(emb.cpu().float())

    embeddings = torch.cat(emb_list, dim=0).numpy()  # (N, 2560)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(EMB_CACHE, embeddings)
    with open(META_CACHE, "w") as f:
        for i, m in enumerate(meta):
            f.write(json.dumps({**m, "sentence": sentences[i]}, ensure_ascii=False) + "\n")
    print(f"Saved embeddings to {EMB_CACHE}")
    print(f"Saved metadata to {META_CACHE}")

# ── 3. Cluster with KMeans ────────────────────────────────────────────

from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score
import hdbscan

# ── 3a. KMeans ────────────────────────────────────────────────────────

N_CLUSTERS = 44  # same as number of CORINE classes

print(f"\n--- KMeans (k={N_CLUSTERS}) ---")
km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42, verbose=0)
labels_km = km.fit_predict(embeddings)
sil_km = silhouette_score(embeddings, labels_km, sample_size=min(5000, len(labels_km)))
print(f"Silhouette score: {sil_km:.4f}")

# ── 3b. HDBSCAN ──────────────────────────────────────────────────────

print("\n--- HDBSCAN ---")
hdb = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean")
labels_hdb = hdb.fit_predict(embeddings)
n_hdb = len(set(labels_hdb) - {-1})
n_noise_hdb = (labels_hdb == -1).sum()
print(f"Clusters found: {n_hdb}, noise points: {n_noise_hdb}")
if n_hdb > 1:
    mask = labels_hdb != -1
    sil_hdb = silhouette_score(embeddings[mask], labels_hdb[mask],
                               sample_size=min(5000, mask.sum()))
    print(f"Silhouette score (excl. noise): {sil_hdb:.4f}")
else:
    sil_hdb = -1.0

# ── 3c. OPTICS ────────────────────────────────────────────────────────

print("\n--- OPTICS ---")
optics = OPTICS(min_samples=10, metric="euclidean", n_jobs=-1)
labels_opt = optics.fit_predict(embeddings)
n_opt = len(set(labels_opt) - {-1})
n_noise_opt = (labels_opt == -1).sum()
print(f"Clusters found: {n_opt}, noise points: {n_noise_opt}")
if n_opt > 1:
    mask = labels_opt != -1
    sil_opt = silhouette_score(embeddings[mask], labels_opt[mask],
                               sample_size=min(5000, mask.sum()))
    print(f"Silhouette score (excl. noise): {sil_opt:.4f}")
else:
    sil_opt = -1.0

# ── 4. Save results ──────────────────────────────────────────────────

def save_clustering(labels, name, sil):
    results = []
    for i, (sent, m) in enumerate(zip(sentences, meta)):
        results.append({
            "cluster": int(labels[i]),
            "code": m["code"],
            "term": m["term"],
            "sent_idx": m["sent_idx"],
            "sentence": sent,
        })
    with open(OUT_DIR / f"clustered_{name}.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    cluster_summary = {}
    for r in results:
        cid = r["cluster"]
        if cid not in cluster_summary:
            cluster_summary[cid] = {"codes": set(), "terms": set(), "count": 0}
        cluster_summary[cid]["codes"].add(r["code"])
        cluster_summary[cid]["terms"].add(r["term"])
        cluster_summary[cid]["count"] += 1
    with open(OUT_DIR / f"summary_{name}.json", "w") as f:
        summary_out = {
            str(k): {
                "count": v["count"],
                "codes": sorted(v["codes"]),
                "sample_terms": sorted(v["terms"])[:5],
            }
            for k, v in sorted(cluster_summary.items())
        }
        json.dump(summary_out, f, indent=2, ensure_ascii=False)
    print(f"Saved {name}: {len(results)} sentences, silhouette={sil:.4f}")

save_clustering(labels_km, "kmeans", sil_km)
save_clustering(labels_hdb, "hdbscan", sil_hdb)
save_clustering(labels_opt, "optics", sil_opt)

# Save all label arrays as .npy for easy reloading
np.save(OUT_DIR / "labels_kmeans.npy", labels_km)
np.save(OUT_DIR / "labels_hdbscan.npy", labels_hdb)
np.save(OUT_DIR / "labels_optics.npy", labels_opt)

# Save fitted cluster objects via pickle for later inspection
import pickle
with open(OUT_DIR / "kmeans_model.pkl", "wb") as f:
    pickle.dump(km, f)
with open(OUT_DIR / "hdbscan_model.pkl", "wb") as f:
    pickle.dump(hdb, f)
with open(OUT_DIR / "optics_model.pkl", "wb") as f:
    pickle.dump(optics, f)
print(f"Saved all label arrays and fitted models to {OUT_DIR}/")

# ── 5. PCA visualization (all 3 methods) ─────────────────────────────

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("\nComputing PCA for 2D projection...")
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(embeddings)
np.save(OUT_DIR / "pca_coords.npy", coords)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (labels, name, sil) in zip(axes, [
    (labels_km, "KMeans", sil_km),
    (labels_hdb, "HDBSCAN", sil_hdb),
    (labels_opt, "OPTICS", sil_opt),
]):
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=labels, cmap="tab20", s=4, alpha=0.5,
    )
    ax.set_title(f"{name} (sil={sil:.3f})")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, shrink=0.8)

plt.suptitle("CORINE wiki sentences — Qwen3.5-4B embeddings", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "clusters_comparison.png", dpi=150)
print(f"Saved comparison plot to {OUT_DIR / 'clusters_comparison.png'}")

print("\nDone!")
