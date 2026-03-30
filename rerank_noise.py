#!/usr/bin/env python3
"""
Reassign HDBSCAN noise points to their nearest cluster centroid.

Noise points (label == -1) are kept by HDBSCAN because they sit below the
density threshold, not because they are semantically unrelated.  Reranking
by cosine similarity to cluster centroids recovers most of them.

Outputs (all _reranked suffix):
  labels_hdbscan_aug_reranked.npy  — full label array, noise -> nearest cluster
  clustered_hdbscan_aug_reranked.jsonl
  summary_hdbscan_aug_reranked.json
  rerank_stats.json                — per-cluster growth + confidence stats
"""

import json
import numpy as np
from pathlib import Path

SIM_THRESHOLD = 0.5   # min cosine similarity to accept reassignment

OUT_DIR = Path("data_corine/clusters")

# ── 1. Load ───────────────────────────────────────────────────────────

embeddings  = np.load(OUT_DIR / "embeddings_aug.npy")
labels_hdb  = np.load(OUT_DIR / "labels_hdbscan_aug.npy")

meta = []
with open(OUT_DIR / "meta_aug.jsonl") as f:
    for line in f:
        meta.append(json.loads(line))

sentences = [m["sentence"] for m in meta]

real_ids = sorted(set(labels_hdb) - {-1})
print(f"Real clusters: {len(real_ids)}")
print(f"Noise points:  {(labels_hdb == -1).sum()}")

# ── 2. Compute centroids ──────────────────────────────────────────────

centroids = np.stack([
    embeddings[labels_hdb == cid].mean(axis=0) for cid in real_ids
])  # (K, D)

# ── 3. Rerank noise by cosine similarity to centroids ─────────────────

noise_mask  = labels_hdb == -1
noise_embs  = embeddings[noise_mask]
noise_idx   = np.where(noise_mask)[0]

cents_n = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9)
noise_n = noise_embs / (np.linalg.norm(noise_embs, axis=1, keepdims=True) + 1e-9)

sims     = noise_n @ cents_n.T          # (N_noise, K)
best_pos = sims.argmax(axis=1)          # index into real_ids
best_sim = sims.max(axis=1)

assigned = np.array([real_ids[p] for p in best_pos])
accepted  = best_sim >= SIM_THRESHOLD   # bool mask over noise points

n_accepted = accepted.sum()
n_rejected = (~accepted).sum()
print(f"\nThreshold={SIM_THRESHOLD}")
print(f"  Reassigned: {n_accepted} ({100*n_accepted/len(noise_idx):.1f}%)")
print(f"  Still noise: {n_rejected} ({100*n_rejected/len(noise_idx):.1f}%)")

# ── 4. Build new label array ──────────────────────────────────────────

new_labels = labels_hdb.copy()
for j, global_i in enumerate(noise_idx):
    if accepted[j]:
        new_labels[global_i] = assigned[j]
    # else stays -1

np.save(OUT_DIR / "labels_hdbscan_aug_reranked.npy", new_labels)
print(f"\nSaved reranked labels → {OUT_DIR}/labels_hdbscan_aug_reranked.npy")

# ── 5. Save per-sentence JSONL ────────────────────────────────────────

# Build a confidence map: original cluster points get sim=1.0,
# reassigned noise points carry their actual cosine similarity.
confidence = np.ones(len(labels_hdb), dtype=np.float32)
for j, global_i in enumerate(noise_idx):
    if accepted[j]:
        confidence[global_i] = float(best_sim[j])
    else:
        confidence[global_i] = float(best_sim[j])  # still record it even if below thresh

results = []
for i, (sent, m) in enumerate(zip(sentences, meta)):
    results.append({
        "cluster": int(new_labels[i]),
        "code": m["code"],
        "term": m["term"],
        "sent_idx": m["sent_idx"],
        "sentence": sent,
        "was_noise": bool(noise_mask[i]),
        "centroid_sim": round(float(confidence[i]), 4),
    })

with open(OUT_DIR / "clustered_hdbscan_aug_reranked.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ── 6. Summary per cluster ────────────────────────────────────────────

cluster_summary = {}
for r in results:
    cid = r["cluster"]
    if cid == -1:
        continue
    if cid not in cluster_summary:
        cluster_summary[cid] = {
            "codes": set(), "terms": set(),
            "count_original": 0, "count_reranked": 0,
            "sims": [],
        }
    cluster_summary[cid]["codes"].add(r["code"])
    cluster_summary[cid]["terms"].add(r["term"])
    cluster_summary[cid]["count_reranked"] += 1
    if not r["was_noise"]:
        cluster_summary[cid]["count_original"] += 1
    if r["was_noise"]:
        cluster_summary[cid]["sims"].append(r["centroid_sim"])

# Load original hdb labels summary for comparison
with open(OUT_DIR / "summary_hdbscan_aug.json") as f:
    orig_summary = json.load(f)

summary_out = {}
for cid in sorted(cluster_summary.keys()):
    v = cluster_summary[cid]
    orig_count = int(orig_summary.get(str(cid), {}).get("count", v["count_original"]))
    added = v["count_reranked"] - orig_count
    mean_sim = float(np.mean(v["sims"])) if v["sims"] else None
    summary_out[str(cid)] = {
        "count_original": orig_count,
        "count_reranked": v["count_reranked"],
        "noise_added": added,
        "noise_mean_sim": round(mean_sim, 4) if mean_sim is not None else None,
        "codes": sorted(v["codes"]),
        "sample_terms": sorted(v["terms"])[:5],
    }

with open(OUT_DIR / "summary_hdbscan_aug_reranked.json", "w") as f:
    json.dump(summary_out, f, indent=2, ensure_ascii=False)

# ── 7. Global rerank stats ────────────────────────────────────────────

stats = {
    "threshold": SIM_THRESHOLD,
    "total_sentences": len(labels_hdb),
    "original_noise": int(noise_mask.sum()),
    "reassigned": int(n_accepted),
    "still_noise": int(n_rejected),
    "reassigned_pct": round(100 * n_accepted / len(noise_idx), 2),
    "sim_stats": {
        "mean": round(float(best_sim.mean()), 4),
        "median": round(float(np.median(best_sim)), 4),
        "p10": round(float(np.percentile(best_sim, 10)), 4),
        "p90": round(float(np.percentile(best_sim, 90)), 4),
    },
    "top10_growth": sorted(
        [{"cluster": cid,
          "original": v["count_original"],
          "reranked": v["count_reranked"],
          "added": v["noise_added"]}
         for cid, v in summary_out.items()],
        key=lambda x: -x["added"]
    )[:10],
}

with open(OUT_DIR / "rerank_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nTop-5 clusters by noise absorbed:")
for e in stats["top10_growth"][:5]:
    print(f"  cluster {e['cluster']:3s}: {e['original']} -> {e['reranked']} (+{e['added']})")

print(f"\nSaved summary  → {OUT_DIR}/summary_hdbscan_aug_reranked.json")
print(f"Saved JSONL    → {OUT_DIR}/clustered_hdbscan_aug_reranked.jsonl")
print(f"Saved stats    → {OUT_DIR}/rerank_stats.json")
print("\nDone!")
