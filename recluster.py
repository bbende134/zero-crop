#!/usr/bin/env python3
"""
Re-cluster: for each method, remove noise (-1) and re-cluster the largest cluster.
Uses KMeans, HDBSCAN, and OPTICS on the sub-embeddings of the biggest cluster.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, OPTICS
from sklearn.metrics import silhouette_score
import hdbscan

OUT_DIR = Path("data_corine/clusters")
REOUT = OUT_DIR / "reclustered"
REOUT.mkdir(parents=True, exist_ok=True)

embeddings = np.load(OUT_DIR / "embeddings.npy")
meta = []
with open(OUT_DIR / "meta.jsonl") as f:
    for line in f:
        meta.append(json.loads(line))

methods = {
    "kmeans": np.load(OUT_DIR / "labels_kmeans.npy"),
    "hdbscan": np.load(OUT_DIR / "labels_hdbscan.npy"),
    "optics": np.load(OUT_DIR / "labels_optics.npy"),
}


def find_biggest_cluster(labels):
    """Return the label of the largest non-noise cluster."""
    unique, counts = np.unique(labels, return_counts=True)
    # exclude noise
    valid = [(c, u) for c, u in zip(counts, unique) if u != -1]
    valid.sort(reverse=True)
    return valid[0][1]  # label of biggest


def run_subclustering(sub_embs, name_prefix):
    """Run all 3 clustering methods on a subset. Returns dict of results."""
    n = len(sub_embs)
    results = {}

    # KMeans — pick k as sqrt(n) capped at 44
    k = min(44, max(2, int(np.sqrt(n))))
    print(f"    KMeans k={k}...")
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    lbl_km = km.fit_predict(sub_embs)
    sil_km = silhouette_score(sub_embs, lbl_km, sample_size=min(5000, n))
    results["kmeans"] = {"labels": lbl_km, "k": k, "silhouette": sil_km}
    print(f"      silhouette={sil_km:.4f}")

    # HDBSCAN
    print(f"    HDBSCAN...")
    hdb = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean")
    lbl_hdb = hdb.fit_predict(sub_embs)
    n_cl = len(set(lbl_hdb) - {-1})
    n_noise = (lbl_hdb == -1).sum()
    sil_hdb = -1.0
    if n_cl > 1:
        mask = lbl_hdb != -1
        sil_hdb = silhouette_score(sub_embs[mask], lbl_hdb[mask], sample_size=min(5000, mask.sum()))
    results["hdbscan"] = {"labels": lbl_hdb, "n_clusters": n_cl, "noise": int(n_noise), "silhouette": sil_hdb}
    print(f"      clusters={n_cl}, noise={n_noise}, silhouette={sil_hdb:.4f}")

    # OPTICS
    print(f"    OPTICS...")
    opt = OPTICS(min_samples=10, metric="euclidean", n_jobs=-1)
    lbl_opt = opt.fit_predict(sub_embs)
    n_cl = len(set(lbl_opt) - {-1})
    n_noise = (lbl_opt == -1).sum()
    sil_opt = -1.0
    if n_cl > 1:
        mask = lbl_opt != -1
        sil_opt = silhouette_score(sub_embs[mask], lbl_opt[mask], sample_size=min(5000, mask.sum()))
    results["optics"] = {"labels": lbl_opt, "n_clusters": n_cl, "noise": int(n_noise), "silhouette": sil_opt}
    print(f"      clusters={n_cl}, noise={n_noise}, silhouette={sil_opt:.4f}")

    return results


for method_name, labels in methods.items():
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"{'='*60}")

    # Remove noise
    non_noise_mask = labels != -1
    n_removed = (~non_noise_mask).sum()
    print(f"  Removing {n_removed} noise points")

    # Find biggest cluster
    biggest = find_biggest_cluster(labels)
    big_mask = labels == biggest
    big_count = big_mask.sum()
    print(f"  Biggest cluster: {biggest} ({big_count} points)")

    # Extract sub-embeddings
    sub_idx = np.where(big_mask)[0]
    sub_embs = embeddings[sub_idx]
    sub_meta = [meta[i] for i in sub_idx]

    print(f"  Re-clustering {len(sub_embs)} points...")
    sub_results = run_subclustering(sub_embs, method_name)

    # Save everything
    prefix = REOUT / method_name
    np.save(f"{prefix}_sub_indices.npy", sub_idx)
    np.save(f"{prefix}_sub_embeddings.npy", sub_embs)

    summary = {"source_method": method_name, "biggest_cluster_id": int(biggest),
               "biggest_cluster_size": int(big_count), "noise_removed": int(n_removed),
               "sub_methods": {}}

    for sub_method, res in sub_results.items():
        lbl = res.pop("labels")
        np.save(f"{prefix}_sub_labels_{sub_method}.npy", lbl)
        res_clean = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in res.items()}
        summary["sub_methods"][sub_method] = res_clean

        # Save per-sentence results
        rows = []
        for i, idx in enumerate(sub_idx):
            rows.append({
                "cluster": int(lbl[i]),
                "code": sub_meta[i]["code"],
                "term": sub_meta[i]["term"],
                "sentence": sub_meta[i]["sentence"],
            })
        with open(f"{prefix}_sub_clustered_{sub_method}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(f"{prefix}_sub_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved to {prefix}_sub_*")

# ── PCA visualization ────────────────────────────────────────────────

from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

for method_name in methods:
    prefix = REOUT / method_name
    sub_embs = np.load(f"{prefix}_sub_embeddings.npy")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(sub_embs)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, sub_method in zip(axes, ["kmeans", "hdbscan", "optics"]):
        lbl = np.load(f"{prefix}_sub_labels_{sub_method}.npy")
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=lbl, cmap="tab20", s=3, alpha=0.4)
        n_cl = len(set(lbl) - {-1})
        ax.set_title(f"{sub_method} ({n_cl} clusters)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.colorbar(scatter, ax=ax, shrink=0.8)

    plt.suptitle(f"Re-clustered biggest cluster from {method_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{prefix}_recluster_pca.png", dpi=150)
    print(f"Saved {prefix}_recluster_pca.png")

print("\nDone!")
