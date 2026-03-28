#!/usr/bin/env python3
"""
Rebuild: take HDBSCAN results, discard noise and the mega-cluster,
then reconstruct a cleaned CORINE wiki dataset from the remaining
well-defined clusters.
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("data_corine/clusters")

# ── Load ──────────────────────────────────────────────────────────────

labels_hdb = np.load(OUT_DIR / "labels_hdbscan.npy")
meta = []
with open(OUT_DIR / "meta.jsonl") as f:
    for line in f:
        meta.append(json.loads(line))

# Find biggest non-noise cluster
unique, counts = np.unique(labels_hdb, return_counts=True)
valid = [(c, u) for c, u in zip(counts, unique) if u != -1]
valid.sort(reverse=True)
biggest = valid[0][1]

# ── Filter: keep noise + mega-cluster, drop the small defined clusters ─

keep_mask = (labels_hdb == -1) | (labels_hdb == biggest)
discard_mask = ~keep_mask

n_noise = int((labels_hdb == -1).sum())
n_mega = int((labels_hdb == biggest).sum())
n_keep = int(keep_mask.sum())
n_discard = int(discard_mask.sum())
n_small_clusters = len(set(labels_hdb[discard_mask]))

print(f"Keeping: noise={n_noise} + mega-cluster {biggest}={n_mega} = {n_keep}")
print(f"Discarding: {n_discard} sentences in {n_small_clusters} small clusters")

# ── Rebuild per CORINE code ───────────────────────────────────────────

rebuilt = defaultdict(lambda: defaultdict(list))
for i in np.where(keep_mask)[0]:
    m = meta[i]
    rebuilt[m["code"]][m["term"]].append(m["sentence"])

# Load original to preserve structure
original = {}
with open("data_corine/corine_wiki_char_count.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        original[rec["code"]] = rec

SENT_RE = re.compile(r'(?<=[.!?])\s+')

cleaned = []
for code in sorted(original.keys()):
    orig = original[code]
    clean_rec = {
        "code": code,
        "description": orig["description"],
        "qwen_generated_terms": orig["qwen_generated_terms"],
        "wiki_texts_cleaned": {},
    }
    if code in rebuilt:
        for term, sents in rebuilt[code].items():
            clean_rec["wiki_texts_cleaned"][term] = " ".join(sents)

    orig_count = sum(
        len([s for s in SENT_RE.split(txt) if len(s.strip()) > 30])
        for txt in orig["wiki_texts"].values()
    )
    clean_count = sum(len(ss) for ss in rebuilt.get(code, {}).values())
    clean_rec["original_sentence_count"] = orig_count
    clean_rec["cleaned_sentence_count"] = clean_count
    cleaned.append(clean_rec)

# ── Save ──────────────────────────────────────────────────────────────

out_path = Path("data_corine/corine_wiki_cleaned.jsonl")
with open(out_path, "w") as f:
    for rec in cleaned:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

total_orig = sum(r["original_sentence_count"] for r in cleaned)
total_clean = sum(r["cleaned_sentence_count"] for r in cleaned)
codes_with_data = sum(1 for r in cleaned if r["cleaned_sentence_count"] > 0)

print(f"\nCleaned dataset: {out_path}")
print(f"Classes with data: {codes_with_data}/{len(cleaned)}")
print(f"Sentences: {total_orig} -> {total_clean} ({100*total_clean/total_orig:.1f}% retained)")

summary_path = Path("data_corine/cleaning_summary.json")
with open(summary_path, "w") as f:
    json.dump({
        "method": "hdbscan",
        "kept_noise": n_noise,
        "kept_mega_cluster": n_mega,
        "mega_cluster_id": int(biggest),
        "discarded_small_clusters": n_discard,
        "discarded_cluster_count": n_small_clusters,
        "kept_sentences": n_keep,
        "total_original_sentences": total_orig,
        "total_cleaned_sentences": total_clean,
        "retention_pct": round(100 * total_clean / total_orig, 2),
        "classes_with_data": codes_with_data,
        "classes_total": len(cleaned),
    }, f, indent=2)
print(f"Summary: {summary_path}")
