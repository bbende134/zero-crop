"""
flow_4 — Satellite-Grounded Contrastive Text Encoder
=====================================================

Text → Qwen3 (LoRA) → ProjectionHead → 64-d L2-norm
                                            ↕ InfoNCE
                satellite grid: (256, 609, 64) AlphaEarth embeddings

Inference: cosine_sim(projected_text, sat_grid) → density map.

No classifier, no lookup table. Novel queries work because the model
learns the language → spectral mapping via contrastive training against
satellite embeddings.

Training data: 40 land cover classes (HRL crops + CORINE), each with
Wikipedia-derived text descriptions and a ground-truth density map.
"""

import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow_3 import load_raw_texts, _KEEP_PATTERNS, _DROP_PATTERNS


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Flow4Config:
    # Data
    pairs_cache: str = "pipeline_cache/pairs_7c381c15589c.pt"
    sat_grid_path: str = "pipeline_cache/sat_grid_27317cfe3740.npy"
    text_descriptions_path: str = "data_corine/corine_wiki_char_count.jsonl"
    hrl_descriptions_path: str = "data_corine/hrl_wiki_char_count.jsonl"
    output_dir: str = "training_data_flow4_v4_extended"

    # Satellite
    sat_emb_dim: int = 64

    # Projection head
    proj_hidden_1: int = 512
    proj_hidden_2: int = 128

    # Contrastive
    init_temperature: float = 0.07
    pixel_loss_weight: float = 1.0
    density_loss_weight: float = 1.0
    distractor_loss_weight: float = 0.3
    n_pos_pixels: int = 128
    n_neg_pixels: int = 128

    # Training
    n_epochs: int = 20
    lr: float = 1e-4
    batch_size: int = 256 # = all classes per step
    grad_accum_steps: int = 2
    val_fraction: float = 0.15
    plot_every: int = 5

    # Geo
    lat_min: float = 45.737
    lat_max: float = 48.585
    lon_min: float = 16.113
    lon_max: float = 22.897

    # Qwen
    qwen_model_id: str = "Qwen/Qwen3.5-9B"
    qwen_emb_dim: int = 0  # auto-detect

    # LLM query augmentation
    # Generates K extra descriptions per class via a local llama-server to
    # close the train/inference gap (training sees Wikipedia sentences;
    # inference queries tend to be short phrases).
    llm_expand_url: str = "http://192.168.242.180:8080/v1"
    llm_expand_k: int = 5       # descriptions to generate per class
    llm_expand_cache: str = "pipeline_cache/llm_augmented_texts_{key}.json"  # {key} filled at runtime
    llm_expand_enabled: bool = False

    # Named geographic features (Option C)
    # Fetched from OSM via Nominatim, rasterized onto the sat grid, and injected
    # as extra training classes so the model learns named-place → land-cover mapping.
    geo_features_enabled: bool = True
    geo_features_cache: str = "pipeline_cache/geo_pairs_{H}x{W}.pt"  # {H},{W} filled at runtime
    geo_river_buffer_deg: float = 0.008  # ~900 m corridor buffer for linear features


# ============================================================
# MODELS
# ============================================================

class SatelliteProjectionHead(nn.Module):
    """Projects Qwen text embeddings into 64-d satellite embedding space."""

    def __init__(self, text_dim: int, sat_dim: int = 64,
                 hidden_1: int = 512, hidden_2: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_1),
            nn.SiLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.SiLU(),
            nn.Linear(hidden_2, sat_dim),
        )

    def forward(self, text_emb):
        return F.normalize(self.proj(text_emb), dim=-1)


class LearnableTemperature(nn.Module):
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def forward(self):
        return self.log_temp.exp().clamp(min=1.0, max=20.0)


# ============================================================
# CORINE code → human-readable name (standard CLC nomenclature)
# ============================================================

_CORINE_NAMES = {
    "111": "Continuous urban fabric",
    "112": "Discontinuous urban fabric",
    "121": "Industrial or commercial units",
    "122": "Road and rail networks",
    "123": "Port areas",
    "124": "Airports",
    "131": "Mineral extraction sites",
    "132": "Dump sites",
    "133": "Construction sites",
    "141": "Green urban areas",
    "142": "Sport and leisure facilities",
    "211": "Non-irrigated arable land",
    "212": "Permanently irrigated land",
    "213": "Rice fields",
    "221": "Vineyards",
    "222": "Fruit trees and berry plantations",
    "223": "Olive groves",
    "231": "Pastures",
    "241": "Annual crops with permanent crops",
    "242": "Complex cultivation patterns",
    "243": "Agriculture with natural vegetation",
    "244": "Agro-forestry areas",
    "311": "Broad-leaved forest",
    "312": "Coniferous forest",
    "313": "Mixed forest",
    "321": "Natural grasslands",
    "322": "Moors and heathland",
    "323": "Sclerophyllous vegetation",
    "324": "Transitional woodland-shrub",
    "331": "Beaches dunes sands",
    "332": "Bare rocks",
    "333": "Sparsely vegetated areas",
    "334": "Burnt areas",
    "335": "Glaciers and perpetual snow",
    "411": "Inland marshes",
    "412": "Peat bogs",
    "421": "Salt marshes",
    "422": "Salines",
    "423": "Intertidal flats",
    "511": "Water courses",
    "512": "Water bodies",
    "521": "Coastal lagoons",
    "522": "Estuaries",
    "523": "Sea and ocean",
}


# ============================================================
# CLASS GROUPS — classes that share the same satellite spectral
# cluster and must NOT be treated as negatives in InfoNCE.
# ============================================================

_CLASS_GROUPS = {
    # Water: all still/flowing water + named features (same sat cluster)
    "511": "water",
    "512": "water",
    "Lake Balaton": "water",
    "Lake Velence": "water",
    "Lake Fertő": "water",
    "Kis-Balaton wetland": "water",
    "Danube river": "water",
    "Tisza river": "water",

    # Wetland: marshes/floodplains (spectral overlap with water edges)
    "411": "wetland",
    "Gemenc floodplain": "wetland",

    # Forest: all tree canopy types (IoU 0.45-0.91)
    "311": "forest",
    "312": "forest",
    "313": "forest",
    "324": "forest",

    # Grassland: grass/low vegetation
    "321": "grassland",
    "231": "grassland",
    "permanent_grassland": "grassland",
    "Hortobágy steppe": "grassland",

    # Arable: broad-acre crops + temporal phases (IoU 0.80-0.99)
    "wheat": "arable",
    "barley": "arable",
    "maize": "arable",
    "other_cereals": "arable",
    "rapeseed": "arable",
    "sunflower": "arable",
    "211": "arable",
    "unclassified_arable": "arable",
    "main_crop_harvest_date": "arable",
    "bare_soil_before_sowing": "arable",
    "bare_soil_after_harvest": "arable",

    # Mixed agriculture (IoU 0.86)
    "242": "mixed_agri",
    "243": "mixed_agri",

    # Vineyards/orchards only — genuine spatial overlap (IoU 0.32-0.35)
    # olives, nuts, unclassified_permanent are too tiny/disjoint to group
    "grapes": "perm_crop",
    "221": "perm_crop",
    "222": "perm_crop",
    "fruits": "perm_crop",
}
# Classes not listed → singleton group (no masking applied)


def build_same_group_mask(class_names):
    """Build (N, N) bool mask: True where i≠j but same spectral group."""
    N = len(class_names)
    group_ids = []
    for i, name in enumerate(class_names):
        group_ids.append(_CLASS_GROUPS.get(name, f"_singleton_{i}"))
    mask = torch.zeros(N, N, dtype=torch.bool)
    for i in range(N):
        for j in range(N):
            if i != j and group_ids[i] == group_ids[j]:
                mask[i, j] = True
    return mask


# ============================================================
# LOSSES
# ============================================================

def centroid_infonce_loss(text_embs, sat_centroids, temperature):
    """Symmetric InfoNCE (CLIP-style) over (text, centroid) pairs.

    text_embs:     (N, 64) L2-normalized
    sat_centroids: (N, 64) L2-normalized
    temperature:   scalar

    Returns scalar loss.
    """
    logits = (text_embs @ sat_centroids.T) * temperature  # (N, N)
    labels = torch.arange(len(text_embs), device=text_embs.device)
    loss_t2s = F.cross_entropy(logits, labels)
    loss_s2t = F.cross_entropy(logits.T, labels)
    return (loss_t2s + loss_s2t) / 2


def density_weighted_infonce_loss(text_embs, sat_grid_flat, density_maps_flat,
                                  temperature, same_group_mask=None):
    """Distribution-aligned InfoNCE with optional same-group masking.

    Score between text_i and class_j = expected cosine similarity under class j's
    spatial distribution:
        score(i,j) = Σ_k [ density_j[k] * sim(text_i, sat[k]) ] / Σ_k density_j[k]

    Preserves spatial spread — bimodal distributions stay bimodal instead of
    collapsing to a geographically meaningless centroid.

    same_group_mask: (N, N) bool — True where i≠j but same spectral group.
    Masked positions are set to -inf so they don't act as negatives.
    """
    N = text_embs.shape[0]
    sim = text_embs @ sat_grid_flat.T

    dm_norm = density_maps_flat / (density_maps_flat.sum(dim=1, keepdim=True) + 1e-8)

    scores = (sim @ dm_norm.T) * temperature  # (N, N)

    if same_group_mask is not None:
        scores = scores.masked_fill(same_group_mask.to(scores.device), float('-inf'))

    labels = torch.arange(N, device=text_embs.device)
    loss_t2s = F.cross_entropy(scores, labels)
    loss_s2t = F.cross_entropy(scores.T, labels)
    return (loss_t2s + loss_s2t) / 2


def pixel_contrastive_loss(text_embs, sat_grid_flat, density_maps_flat,
                           class_indices, n_pos, n_neg, temperature,
                           sat_centroids=None, same_group_mask=None):
    """Pixel-level contrastive with hard negative mining (vectorized).

    text_embs:        (B, 64) L2-normalized
    sat_grid_flat:    (H*W, 64) L2-normalized
    density_maps_flat: (n_classes, H*W)
    class_indices:    (B,) class index per text
    sat_centroids:    (n_classes, 64) for hard negative mining (optional)
    same_group_mask:  (n_classes, n_classes) bool — skip same-group for hard negs
    """
    B = text_embs.shape[0]
    n_classes = density_maps_flat.shape[0]
    device = text_embs.device
    losses = []

    # Precompute class similarities for hard negative mining
    hard_neg_classes = None
    if sat_centroids is not None and n_classes > 3:
        with torch.no_grad():
            cls_sim = sat_centroids @ sat_centroids.T  # (C, C)
            # Zero out self-similarity and same-group pairs
            cls_sim.fill_diagonal_(-1.0)
            if same_group_mask is not None:
                cls_sim[same_group_mask.to(cls_sim.device)] = -1.0
            # Top-3 most similar classes per class (excluding self + same group)
            hard_neg_classes = cls_sim.topk(min(3, n_classes - 1)).indices  # (C, 3)

    n_hard = n_neg // 2  # half from confusable classes, half random

    for i in range(B):
        cls = class_indices[i]
        dm = density_maps_flat[cls]

        # Positive: cells where this class has density
        pos_mask = dm > 1e-4
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:
            continue
        if len(pos_indices) <= n_pos:
            pos_sample = pos_indices
        else:
            weights = dm[pos_indices]
            weights = weights / weights.sum()
            pos_sample = pos_indices[torch.multinomial(weights, n_pos, replacement=False)]

        # Negative sampling: hard + random
        neg_mask = dm < 1e-6
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        if len(neg_indices) == 0:
            continue

        neg_parts = []

        # Hard negatives: sample from high-density cells of confusable classes
        if hard_neg_classes is not None:
            hard_pool = []
            for conf_cls in hard_neg_classes[cls]:
                conf_dm = density_maps_flat[conf_cls]
                # Cells that are high-density for the confusable class but
                # NOT high-density for our class (the discriminative boundary)
                hard_mask = (conf_dm > 1e-4) & neg_mask
                hard_idx = hard_mask.nonzero(as_tuple=True)[0]
                if len(hard_idx) > 0:
                    hard_pool.append(hard_idx)
            if hard_pool:
                hard_pool = torch.cat(hard_pool)
                n_sample = min(n_hard, len(hard_pool))
                hard_sample = hard_pool[torch.randint(len(hard_pool), (n_sample,), device=device)]
                neg_parts.append(hard_sample)

        # Random negatives: fill remaining budget
        n_random = n_neg - (len(neg_parts[0]) if neg_parts else 0)
        if n_random > 0 and len(neg_indices) > 0:
            n_sample = min(n_random, len(neg_indices))
            rand_sample = neg_indices[torch.randint(len(neg_indices), (n_sample,), device=device)]
            neg_parts.append(rand_sample)

        if not neg_parts:
            continue
        neg_sample = torch.cat(neg_parts)

        pos_embs = sat_grid_flat[pos_sample]  # (n_p, 64)
        neg_embs = sat_grid_flat[neg_sample]  # (n_n, 64)

        # Vectorized: (n_p, 1) positive sims + (n_p, n_n) negative sims
        sim_pos = (text_embs[i] * pos_embs).sum(dim=-1, keepdim=True) * temperature  # (n_p, 1)
        sim_neg = (text_embs[i:i+1] @ neg_embs.T) * temperature  # (1, n_n)
        sim_neg = sim_neg.expand(len(pos_sample), -1)  # (n_p, n_n)

        logits = torch.cat([sim_pos, sim_neg], dim=1)  # (n_p, 1+n_n)
        labels = torch.zeros(len(pos_sample), dtype=torch.long, device=device)
        losses.append(F.cross_entropy(logits, labels))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


def _is_distractor(text: str) -> bool:
    """True only if the sentence matches drop patterns WITHOUT any keep pattern.

    Sentences matching both (e.g. 'wheat exports from Hungary') are treated as
    relevant — the keep signal wins.
    """
    t = text.lower()
    if not any(p in t for p in _DROP_PATTERNS):
        return False
    return not any(p in t for p in _KEEP_PATTERNS)


def distractor_uniformity_loss(projected_distractors, sat_grid_flat, temperature):
    """Penalize distractor embeddings for producing peaked geographic distributions.

    Pushes cosine similarity against all satellite cells toward zero —
    i.e., the embedding should carry no geographic signal.

    projected_distractors: (D, 64) L2-normalized
    sat_grid_flat:         (H*W, 64) L2-normalized
    """
    sim = (projected_distractors @ sat_grid_flat.T) * temperature  # (D, H*W)
    return sim.pow(2).mean()


# ============================================================
# DATA
# ============================================================

def compute_sat_centroids(pairs, sat_grid):
    """Density-weighted average of sat grid per class, L2-normalized.

    Returns (n_classes, 64) tensor.
    """
    H, W, D = sat_grid.shape
    sat_flat = sat_grid.reshape(H * W, D).astype(np.float64)
    centroids = []
    for p in pairs:
        dm_raw = p["density_map"]
        dm = (dm_raw.numpy() if hasattr(dm_raw, "numpy") else np.asarray(dm_raw)).flatten().astype(np.float64)
        w_sum = dm.sum()
        if w_sum < 1e-8:
            c = sat_flat.mean(axis=0)
        else:
            c = (dm[:, np.newaxis] / w_sum * sat_flat).sum(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids.append(c.astype(np.float32))
    return torch.from_numpy(np.stack(centroids))


def _augment_neither_sentences(sents):
    """For each 'neither' sentence, append the closest keep/both sentence from the
    same class pool (token-overlap Jaccard). Returns the augmented list in-place.

    keep/both sentences are kept unchanged. Pure distractors are kept as-is
    (the uniformity loss handles them during training).
    """
    def _tokens(s):
        return set(s.lower().split())

    def _classify(s):
        sl = s.lower()
        has_keep = any(p in sl for p in _KEEP_PATTERNS)
        has_drop = any(p in sl for p in _DROP_PATTERNS)
        if has_drop and not has_keep:
            return "drop"
        if not has_keep and not has_drop:
            return "neither"
        return "keep"

    keep_sents = [s for s in sents if _classify(s) == "keep"]
    if not keep_sents:
        return sents  # nothing to anchor to

    keep_tokens = [_tokens(s) for s in keep_sents]

    augmented = []
    n_augmented = 0
    for s in sents:
        if _classify(s) != "neither":
            augmented.append(s)
            continue
        # Jaccard similarity to each keep sentence
        s_tok = _tokens(s)
        best_idx, best_score = 0, -1.0
        for i, kt in enumerate(keep_tokens):
            union = s_tok | kt
            if union:
                score = len(s_tok & kt) / len(union)
                if score > best_score:
                    best_score, best_idx = score, i
        augmented.append(s + ". " + keep_sents[best_idx])
        n_augmented += 1
    return augmented, n_augmented


# Named Hungarian geographic features to inject as training classes.
# (display_name, nominatim_query, is_linear)
# is_linear=True → geometry is buffered by geo_river_buffer_deg before rasterizing.
_GEO_FEATURES = [
    ("Lake Balaton",         "Balaton, Hungary",         False),
    ("Lake Velence",         "Velencei-tó, Hungary",     False),
    ("Lake Fertő",           "Fertő, Hungary",           False),
    ("Kis-Balaton wetland",  "Kis-Balaton, Hungary",     False),
    ("Danube river",         "Duna, Magyarország",       True),
    ("Tisza river",          "Tisza, Hungary",           True),
    ("Hortobágy steppe",     "Hortobágy, Hungary",       False),
    ("Gemenc floodplain",    "Gemenc, Hungary",          False),
]


def _fetch_nominatim_geometry(query: str):
    """Fetch the boundary polygon for a named place via Nominatim GeoJSON API."""
    import requests
    from shapely.geometry import shape
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "geojson", "polygon_geojson": 1, "limit": 1},
            headers={"User-Agent": "zero-crop-training/1.0"},
            timeout=30,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            return None
        return shape(features[0]["geometry"])
    except Exception as e:
        print(f"    [geo] Nominatim failed for '{query}': {e}")
        return None


def _rasterize_geom(geom, H: int, W: int, cfg) -> np.ndarray:
    """Rasterize a shapely geometry onto the (H, W) grid, clipped to the Hungary bbox."""
    from shapely.geometry import box
    from shapely import contains_xy
    bbox = box(cfg.lon_min, cfg.lat_min, cfg.lon_max, cfg.lat_max)
    clipped = geom.intersection(bbox)
    if clipped.is_empty:
        return None
    lons = np.linspace(cfg.lon_min, cfg.lon_max, W)
    lats = np.linspace(cfg.lat_max, cfg.lat_min, H)   # row 0 = north
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    mask = contains_xy(clipped, lon_grid.ravel(), lat_grid.ravel()).reshape(H, W)
    return mask.astype(np.float32)


def build_or_load_geo_pairs(H: int, W: int, cfg, cache_dir: Path) -> list:
    """Build density maps for named Hungarian geographic features from OSM.

    Returns list of pair dicts {class_name, desc_key, density_map} — same format
    as CORINE pairs so they can be appended directly to the pairs list.

    Cache key includes H×W so changing resolution auto-invalidates.
    Delete the cache file to force a re-fetch from Nominatim.
    """
    cache_path = cache_dir / f"geo_pairs_{H}x{W}.pt"
    if cache_path.exists():
        pairs = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"  [geo] cache HIT → {cache_path.name}  ({len(pairs)} features)")
        return pairs

    print(f"  [geo] Fetching {len(_GEO_FEATURES)} named features from Nominatim ...")
    pairs = []
    for name, query, is_linear in _GEO_FEATURES:
        print(f"    {name!r} ...", end=" ", flush=True)
        geom = _fetch_nominatim_geometry(query)
        if geom is None:
            print("SKIP (no geometry)")
            continue
        if is_linear:
            geom = geom.buffer(cfg.geo_river_buffer_deg)
        dm = _rasterize_geom(geom, H, W, cfg)
        if dm is None or dm.sum() == 0:
            print("SKIP (empty after clip to bbox)")
            continue
        pairs.append({"class_name": name, "desc_key": name, "density_map": dm})
        print(f"OK  ({int(dm.sum()):,} cells, {dm.sum()/dm.size*100:.1f}% of grid)")

    torch.save(pairs, cache_path)
    print(f"  [geo] {len(pairs)} features cached → {cache_path.name}")
    return pairs


_LLM_EXPAND_SYSTEM = (
    "You are a geographic land-cover encyclopedia specializing in satellite-observable features. "
    "Given a land cover class name or a named geographic feature, write exactly 2–3 Wikipedia-style "
    "sentences describing it in terms of what a satellite would observe: "
    "the land cover type, surface characteristics, spectral signature, ecology, and spatial extent "
    "in Central Europe or Hungary. "
    "If the input is a named place (lake, river, mountain, city, national park), translate it "
    "into its land cover description — describe what the feature IS and looks like from above, "
    "NOT just where it is. Do NOT repeat the proper name in the output. "
    "Do NOT include headings, bullet points, or meta-commentary. "
    "Output only the description sentences."
)


def build_llm_augmented_texts(class_names: list, cfg) -> dict:
    """Generate K LLM descriptions per class, cached to disk.

    Returns {class_name: [sentence, ...]} or {} if LLM is unavailable.
    The cache is keyed by (class_names, k, url) so changing any of those
    invalidates it automatically.
    """
    import json, hashlib

    cache_key = hashlib.md5(
        f"{sorted(class_names)}|{cfg.llm_expand_k}|{cfg.llm_expand_url}".encode()
    ).hexdigest()[:12]
    # Key is embedded in the filename — different class sets → different files, never overwrite
    cache_path = Path(cfg.llm_expand_cache.replace("{key}", cache_key))

    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        total = sum(len(v) for k, v in cached.items() if not k.startswith("_"))
        print(f"  [LLM aug] cache HIT → {cache_path.name}  ({total} sentences for {len(class_names)} classes)")
        return {k: v for k, v in cached.items() if not k.startswith("_")}

    try:
        from openai import OpenAI
        client = OpenAI(base_url=cfg.llm_expand_url, api_key="none")
        # Quick health check
        client.models.list()
    except Exception as e:
        print(f"  [LLM aug] server unreachable ({e}) — skipping augmentation")
        return {}

    augmented = {}
    print(f"  [LLM aug] generating {cfg.llm_expand_k} descriptions × {len(class_names)} classes "
          f"from {cfg.llm_expand_url} ...")
    for name in class_names:
        sents = []
        for _ in range(cfg.llm_expand_k):
            try:
                resp = client.chat.completions.create(
                    model="qwen",
                    messages=[
                        {"role": "system", "content": _LLM_EXPAND_SYSTEM},
                        {"role": "user", "content": name},
                    ],
                    max_tokens=-1,
                    temperature=0.7,   # diversity across K samples
                )
                text = resp.choices[0].message.content.strip()
                if text:
                    sents.append(text)
            except Exception as e:
                print(f"    [LLM aug] failed for '{name}': {e}")
                break
        augmented[name] = sents
        print(f"    {name}: {len(sents)} sentences")

    # Save cache — filename encodes the key, so old caches are never overwritten
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(augmented, f, indent=2)
    total = sum(len(v) for v in augmented.values())
    print(f"  [LLM aug] {total} sentences saved → {cache_path.name}")
    return augmented


def build_datasets(pairs, raw_texts, cfg):
    """Build class names, centroids, density maps, and text pools."""
    class_names = []
    class_to_idx = {}
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        if desc_key not in class_to_idx:
            class_to_idx[desc_key] = len(class_names)
            class_names.append(p["class_name"])
    n_classes = len(class_names)

    density_maps = torch.stack([
        torch.as_tensor(np.asarray(p["density_map"]), dtype=torch.float32) for p in pairs
    ])

    # Build per-class text pools
    texts_by_class = {i: [] for i in range(n_classes)}
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        cls_idx = class_to_idx[desc_key]
        sents = raw_texts.get(desc_key, [p["class_name"]])
        texts_by_class[cls_idx].extend(sents)

    # Inject CORINE human-readable names as extra training text.
    # The raw data keys classes by numeric code ("512") but the model
    # needs to understand queries like "water bodies" or "broad-leaved forest".
    n_corine_injected = 0
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        cls_idx = class_to_idx[desc_key]
        corine_name = _CORINE_NAMES.get(desc_key)
        if corine_name:
            texts_by_class[cls_idx].append(corine_name)
            texts_by_class[cls_idx].append(
                f"{corine_name} as observed from satellite imagery")
            n_corine_injected += 2
    print(f"  CORINE name sentences injected: {n_corine_injected}")

    # LLM-generated descriptions — closes the train/inference gap:
    # training sees Wikipedia sentences; inference queries are short phrases that
    # get expanded by the same LLM prompt, so including LLM text in training
    # ensures the model sees both registers.
    if cfg.llm_expand_enabled:
        llm_texts = build_llm_augmented_texts(class_names, cfg)
        n_llm = 0
        for cls_idx, name in enumerate(class_names):
            extra = llm_texts.get(name, [])
            texts_by_class[cls_idx].extend(extra)
            n_llm += len(extra)
        print(f"  LLM-augmented sentences added: {n_llm}")

    # Neither-sentence augmentation disabled
    print(f"  Neither-sentence augmentation: disabled")

    # Train/val split
    rng = random.Random(42)
    train_texts = {i: [] for i in range(n_classes)}
    val_texts = {i: [] for i in range(n_classes)}
    for cls_idx in range(n_classes):
        sents = list(texts_by_class[cls_idx])
        rng.shuffle(sents)
        n_val = max(1, int(len(sents) * cfg.val_fraction))
        val_texts[cls_idx] = sents[:n_val]
        train_texts[cls_idx] = sents[n_val:]

    n_train = sum(len(v) for v in train_texts.values())
    n_val = sum(len(v) for v in val_texts.values())
    print(f"  Classes: {n_classes}")
    print(f"  Train sentences: {n_train}, Val sentences: {n_val}")
    print(f"  Density maps: {density_maps.shape}")

    return class_names, density_maps, train_texts, val_texts


def sample_contrastive_batch(texts_by_class, n_classes):
    """Sample one sentence per class for full NxN contrastive batch."""
    texts = []
    for cls_idx in range(n_classes):
        pool = texts_by_class[cls_idx]
        texts.append(random.choice(pool) if pool else f"class {cls_idx}")
    return texts, torch.arange(n_classes)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_predictions(proj_head, text_encoder, sat_grid_flat_norm,
                          density_maps, class_names, queries, cfg,
                          epoch, viz_dir, device, hungary_mask=None):
    """Render cosine similarity maps for test queries."""
    proj_head.eval()
    H, W = density_maps.shape[1], density_maps.shape[2]
    n = len(queries)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    extent = [cfg.lon_min, cfg.lon_max, cfg.lat_max, cfg.lat_min]

    with torch.no_grad():
        for row, query in enumerate(queries):
            emb = text_encoder.encode_raw(query)  # (1, qwen_dim)
            projected = proj_head(emb.to(device))  # (1, 64)
            sim = (projected @ sat_grid_flat_norm.T)[0]  # (H*W,)
            sim_map = sim.reshape(H, W).cpu().numpy()
            if hungary_mask is not None:
                sim_map[~hungary_mask] = np.nan
            density = np.where(hungary_mask if hungary_mask is not None else True,
                               sim_map.clip(0, 1), np.nan)

            # Heatmap
            ax = axes[row, 0]
            cmap = plt.cm.YlOrRd.copy()
            cmap.set_bad(color="lightgrey")   # outside Hungary = grey
            im = ax.imshow(density, cmap=cmap, origin="upper",
                           extent=extent, vmin=0, vmax=0.5)
            ax.set_title(f'"{query[:50]}"', fontsize=9)
            ax.axis("off")

            # Histogram: only inside-Hungary cells
            sim_np = sim.cpu().numpy()
            if hungary_mask is not None:
                sim_inside = sim_np[hungary_mask.ravel()]
            else:
                sim_inside = sim_np
            ax2 = axes[row, 1]
            ax2.hist(sim_inside, bins=50, color="steelblue", alpha=0.7)
            ax2.axvline(0, color="red", linestyle="--", alpha=0.5)
            ax2.set_title(f"sim distribution (mean={sim_inside.mean():.3f})", fontsize=9)
            ax2.set_xlim(-0.3, 0.5)

    plt.suptitle(f"Epoch {epoch}", fontsize=11)
    plt.tight_layout()
    plt.savefig(viz_dir / f"epoch_{epoch:03d}.png", dpi=120)
    plt.close(fig)
    proj_head.train()


# ============================================================
# TRAINING
# ============================================================

def validate(proj_head, text_encoder, sat_centroids, val_texts,
             n_classes, device, max_per_class: int = 3, batch_size: int = 40):
    """Compute retrieval R@1 and R@5 on validation texts (batched)."""
    proj_head.eval()
    tokenizer = text_encoder._tokenizer
    qwen_device = text_encoder.input_device

    # Collect (sentence, class_idx) pairs, capped per class
    all_sents, all_labels = [], []
    for cls_idx in range(n_classes):
        for sent in val_texts[cls_idx][:max_per_class]:
            all_sents.append(sent)
            all_labels.append(cls_idx)

    correct_1, correct_5 = 0, 0

    with torch.no_grad():
        for start in range(0, len(all_sents), batch_size):
            batch_sents = all_sents[start:start + batch_size]
            batch_labels = all_labels[start:start + batch_size]

            inputs = tokenizer(
                batch_sents, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            ids = inputs["input_ids"].to(qwen_device)
            mask = inputs["attention_mask"].to(qwen_device)

            with torch.autocast(qwen_device.type, dtype=torch.bfloat16):
                embs = text_encoder(ids, mask)
                projected = proj_head(embs.to(device))  # (B, 64)

            sims = projected @ sat_centroids.T  # (B, n_classes)
            top5s = sims.topk(5).indices  # (B, 5)

            for i, cls_idx in enumerate(batch_labels):
                top5 = top5s[i].tolist()
                if cls_idx == top5[0]:
                    correct_1 += 1
                if cls_idx in top5:
                    correct_5 += 1

    proj_head.train()
    total = len(all_sents)
    r1 = correct_1 / max(1, total) * 100
    r5 = correct_5 / max(1, total) * 100
    return r1, r5


def train(proj_head, temperature, text_encoder, sat_centroids, sat_grid_flat_norm,
          density_maps, class_names, train_texts, val_texts, cfg, device,
          hungary_mask=None):
    """Main training loop."""

    n_classes = len(class_names)
    sat_centroids_dev = sat_centroids.to(device)
    sat_grid_flat_dev = sat_grid_flat_norm.to(device)
    H, W = density_maps.shape[1], density_maps.shape[2]
    dm_flat = density_maps.reshape(n_classes, -1).to(device)

    # Build same-group mask for InfoNCE masking
    same_group_mask = build_same_group_mask(class_names)
    n_masked = same_group_mask.sum().item() // 2  # undirected pairs
    groups_used = set(_CLASS_GROUPS[n] for n in class_names if n in _CLASS_GROUPS)
    print(f"  Same-group mask: {n_masked} class pairs masked across {len(groups_used)} groups")

    # Optimizer
    qwen_params = [p for p in text_encoder.parameters() if p.requires_grad]
    n_qwen = sum(p.numel() for p in qwen_params)
    n_proj = sum(p.numel() for p in proj_head.parameters())
    print(f"\n  Qwen LoRA params: {n_qwen:,}")
    print(f"  Projection head params: {n_proj:,}")

    optimizer = torch.optim.AdamW([
        {"params": qwen_params, "lr": cfg.lr},
        {"params": proj_head.parameters(), "lr": cfg.lr * 10},
        {"params": temperature.parameters(), "lr": cfg.lr},
    ])

    # One "batch" = one sentence per class = 40 texts
    # n_batches = enough to cycle through most training sentences per epoch
    min_pool = min(len(v) for v in train_texts.values())
    n_batches = max(min_pool * 2, 40)
    total_steps = n_batches * cfg.n_epochs
    warmup_steps = int(total_steps * 0.1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps])

    tokenizer = text_encoder._tokenizer
    qwen_device = text_encoder.input_device

    viz_dir = Path(cfg.output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    test_queries = [
        "wheat", "maize", "sunflower", "water bodies",
        "deciduous forest", "urban residential areas",
        "grassland near rivers", "quantum physics textbook",
    ]

    train_losses, val_r1s, val_r5s, temps = [], [], [], []

    # Resume
    ckpt_path = Path(cfg.output_dir) / "checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        print(f"[resume] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        proj_head.load_state_dict(ckpt["proj_head"])
        temperature.load_state_dict(ckpt["temperature"])
        lora_state = ckpt.get("qwen_lora", {})
        if lora_state:
            text_encoder.load_state_dict(lora_state, strict=False)
            print(f"  Restored {len(lora_state)} LoRA weight tensors")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        train_losses = ckpt.get("train_losses", [])
        val_r1s = ckpt.get("val_r1s", [])
        val_r5s = ckpt.get("val_r5s", [])
        temps = ckpt.get("temps", [])
        print(f"  Resuming from epoch {start_epoch}")

    print(f"\nTraining: {n_classes} classes, {n_batches} batches/epoch")
    print(f"  Grad accum: {cfg.grad_accum_steps}, effective batch: {n_classes * cfg.grad_accum_steps}")

    proj_head.train()
    text_encoder.train()

    for epoch in range(start_epoch, cfg.n_epochs):
        ep_loss = 0.0
        ep_centroid_loss = 0.0
        ep_density_loss = 0.0
        ep_pixel_loss = 0.0
        ep_distractor_loss = 0.0

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1:3d}",
                    leave=False, ncols=100)

        for batch_i in pbar:
            texts, class_indices = sample_contrastive_batch(train_texts, n_classes)
            distractor_mask = torch.tensor([_is_distractor(t) for t in texts])
            class_indices = class_indices.to(device)

            inputs = tokenizer(
                texts, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            ids = inputs["input_ids"].to(qwen_device)
            mask = inputs["attention_mask"].to(qwen_device)

            with torch.autocast(qwen_device.type, dtype=torch.bfloat16):
                embs = text_encoder(ids, mask)  # (40, qwen_dim)
                projected = proj_head(embs.to(device))  # (40, 64)
                temp = temperature()

                # Centroid loss: logged for monitoring only, not in backward graph
                with torch.no_grad():
                    loss_c = centroid_infonce_loss(projected, sat_centroids_dev, temp)

                loss_dw = density_weighted_infonce_loss(
                    projected, sat_grid_flat_dev, dm_flat, temp,
                    same_group_mask=same_group_mask)
                loss_p = pixel_contrastive_loss(
                    projected, sat_grid_flat_dev, dm_flat,
                    class_indices, cfg.n_pos_pixels, cfg.n_neg_pixels, temp,
                    sat_centroids=sat_centroids_dev,
                    same_group_mask=same_group_mask)

                distractor_idx = distractor_mask.nonzero(as_tuple=True)[0].to(device)
                if len(distractor_idx) > 0:
                    loss_d = distractor_uniformity_loss(
                        projected[distractor_idx], sat_grid_flat_dev, temp)
                else:
                    loss_d = torch.tensor(0.0, device=device)

                loss = (cfg.density_loss_weight * loss_dw
                        + cfg.pixel_loss_weight * loss_p
                        + cfg.distractor_loss_weight * loss_d) / cfg.grad_accum_steps

            loss.backward()

            if (batch_i + 1) % cfg.grad_accum_steps == 0 or batch_i == n_batches - 1:
                all_params = qwen_params + list(proj_head.parameters()) + list(temperature.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            ep_loss += loss.item() * cfg.grad_accum_steps
            ep_centroid_loss += loss_c.item()
            ep_density_loss += loss_dw.item()
            ep_pixel_loss += loss_p.item()
            ep_distractor_loss += loss_d.item()

            pbar.set_postfix({
                "loss": f"{ep_loss/(batch_i+1):.4f}",
                "dw": f"{ep_density_loss/(batch_i+1):.4f}",
                "temp": f"{temp.item():.2f}",
            })

        avg_loss = ep_loss / n_batches
        avg_c = ep_centroid_loss / n_batches
        avg_dw = ep_density_loss / n_batches
        avg_p = ep_pixel_loss / n_batches
        avg_d = ep_distractor_loss / n_batches
        cur_temp = temperature().item()
        train_losses.append(avg_loss)
        temps.append(cur_temp)

        # Validation
        r1, r5 = validate(proj_head, text_encoder, sat_centroids_dev,
                          val_texts, n_classes, device)
        val_r1s.append(r1)
        val_r5s.append(r5)

        print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f} (centroid={avg_c:.4f}, "
              f"density_w={avg_dw:.4f}, pixel={avg_p:.4f}, distractor={avg_d:.4f}), "
              f"R@1={r1:.1f}%, R@5={r5:.1f}%, temp={cur_temp:.2f}")

        # --- Progress plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs_x = list(range(1, len(train_losses) + 1))

        axes[0].plot(epochs_x, train_losses, color="coral", marker="o", markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Total Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_x, val_r1s, label="R@1", color="steelblue", marker="o", markersize=3)
        axes[1].plot(epochs_x, val_r5s, label="R@5", color="coral", marker="o", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Retrieval %")
        axes[1].set_title("Validation Retrieval")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs_x, temps, color="green", marker="o", markersize=3)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Temperature")
        axes[2].set_title("Learned Temperature")
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("Flow 4 — Contrastive Training")
        plt.tight_layout()
        plt.savefig(viz_dir / "progress.png", dpi=120)
        plt.close(fig)

        # --- Visualization ---
        if cfg.plot_every > 0 and ((epoch + 1) % cfg.plot_every == 0 or epoch == 0):
            visualize_predictions(
                proj_head, text_encoder, sat_grid_flat_dev,
                density_maps, class_names, test_queries,
                cfg, epoch + 1, viz_dir, device,
                hungary_mask=hungary_mask)

        # --- Checkpoint ---
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        lora_state = {k: v.cpu() for k, v in text_encoder.state_dict().items()
                      if "lora" in k.lower()}
        torch.save({
            "epoch": epoch,
            "proj_head": proj_head.state_dict(),
            "temperature": temperature.state_dict(),
            "qwen_lora": lora_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sat_centroids": sat_centroids.cpu(),
            "class_names": class_names,
            "config": vars(cfg),
            "train_losses": train_losses,
            "val_r1s": val_r1s,
            "val_r5s": val_r5s,
            "temps": temps,
        }, ckpt_path)

    # Final save
    lora_state = {k: v.cpu() for k, v in text_encoder.state_dict().items()
                  if "lora" in k.lower()}
    torch.save({
        "proj_head": proj_head.state_dict(),
        "temperature": temperature.state_dict(),
        "qwen_lora": lora_state,
        "sat_centroids": sat_centroids.cpu(),
        "class_names": class_names,
        "config": vars(cfg),
    }, Path(cfg.output_dir) / "flow4_model.pt")
    print(f"\n  Saved to {Path(cfg.output_dir) / 'flow4_model.pt'}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=None,
                        help="Qwen model ID, e.g. Qwen/Qwen3-14B")
    parser.add_argument("--multi-gpu", action="store_true")
    args = parser.parse_args()

    cfg = Flow4Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.model:
        cfg.qwen_model_id = args.model

    # Never overwrite a completed training run — bump version suffix if needed
    out = Path(cfg.output_dir)
    if (out / "flow4_model.pt").exists():
        import re
        base = re.sub(r"_v\d+$", "", str(out))
        v = 1
        while True:
            candidate = Path(f"{base}_v{v}")
            if not (candidate / "flow4_model.pt").exists():
                cfg.output_dir = str(candidate)
                print(f"  [safe] '{out}' has a completed model — using '{candidate}' instead")
                break
            v += 1

    multi_gpu = args.multi_gpu
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, multi_gpu: {multi_gpu}")

    import data_pipeline as dp
    from data_pipeline import (
        build_or_load_distributions,
        build_or_load_pairs,
        build_or_load_sat_grid,
    )
    cache_dir = Path("pipeline_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    expected_H = dp.TARGET_RES
    expected_W = dp.target_width(dp.TARGET_RES)

    # --- Step 1: Pairs (density maps + class metadata) ---
    print("\n" + "=" * 60)
    print(f"STEP 1: Pairs cache  [expected density map: {expected_H}×{expected_W}]")
    print("=" * 60)

    def _pairs_resolution_ok(p):
        dm = np.asarray(p[0]["density_map"])
        return dm.shape == (expected_H, expected_W)

    pairs = None
    pairs_path = Path(cfg.pairs_cache)
    if pairs_path.exists():
        p = torch.load(pairs_path, map_location="cpu", weights_only=False)
        if _pairs_resolution_ok(p):
            pairs = p
            print(f"  [OK]     config path: {pairs_path.name}  "
                  f"({len(pairs)} pairs, {expected_H}×{expected_W})")
        else:
            dm_shape = np.asarray(p[0]["density_map"]).shape
            print(f"  [SKIP]   config path: {pairs_path.name}  "
                  f"wrong resolution {dm_shape} ≠ {(expected_H, expected_W)}")

    if pairs is None:
        for candidate in sorted(cache_dir.glob("pairs_*.pt"),
                                key=lambda p: p.stat().st_mtime, reverse=True):
            p = torch.load(candidate, map_location="cpu", weights_only=False)
            if _pairs_resolution_ok(p):
                pairs = p
                print(f"  [OK]     found cache: {candidate.name}  "
                      f"({len(pairs)} pairs, {expected_H}×{expected_W})")
                break
            else:
                dm_shape = np.asarray(p[0]["density_map"]).shape
                print(f"  [SKIP]   {candidate.name}  resolution {dm_shape}")

    if pairs is None:
        print(f"  [BUILD]  No matching pairs cache — building at {expected_H}×{expected_W}...")
        distributions = build_or_load_distributions(cache_dir, dp.TARGET_RES)
        dist_hash = dp._hash(
            dp.R3_DIR, dp.R4_DIR, dp.CORINE_GEOJSON,
            dp.LAT_MIN, dp.LAT_MAX, dp.LON_MIN, dp.LON_MAX,
            dp.TARGET_RES, dp.MIN_PIXELS,
        )
        pairs = build_or_load_pairs(distributions, cache_dir, dist_hash, device)
        print(f"  [DONE]   {len(pairs)} pairs built")

    # --- Step 2: Satellite embedding grid ---
    print("\n" + "=" * 60)
    print(f"STEP 2: Satellite grid  [expected: {expected_H}×{expected_W}×{dp.SAT_EMB_DIM}]")
    print("=" * 60)

    sat_grid = None
    sat_grid_path = Path(cfg.sat_grid_path)
    if sat_grid_path.exists():
        g = np.load(sat_grid_path, mmap_mode="r")
        if g.shape[:2] == (expected_H, expected_W):
            sat_grid = np.array(g)
            print(f"  [OK]     config path: {sat_grid_path.name}  shape={sat_grid.shape}")
        else:
            print(f"  [SKIP]   config path: {sat_grid_path.name}  "
                  f"wrong shape {g.shape} ≠ ({expected_H},{expected_W},...)")

    if sat_grid is None:
        for candidate in sorted(cache_dir.glob("sat_grid_*.npy"),
                                key=lambda p: p.stat().st_mtime, reverse=True):
            g = np.load(candidate, mmap_mode="r")
            if g.shape[:2] == (expected_H, expected_W):
                sat_grid = np.array(g)
                print(f"  [OK]     found cache: {candidate.name}  shape={sat_grid.shape}")
                break
            else:
                print(f"  [SKIP]   {candidate.name}  shape={g.shape}")

    if sat_grid is None:
        print(f"  [BUILD]  No matching sat_grid — building at {expected_H}×{expected_W}...")
        sat_grid = build_or_load_sat_grid(cache_dir, dp.TARGET_RES, dp.SAT_EMB_PATH)
        print(f"  [DONE]   sat_grid shape={sat_grid.shape}")
    H, W, D = sat_grid.shape

    # L2-normalize per cell (once, for cosine similarity)
    sat_grid_flat = sat_grid.reshape(-1, D)
    norms = np.linalg.norm(sat_grid_flat, axis=1, keepdims=True) + 1e-8
    sat_grid_flat_norm = torch.from_numpy((sat_grid_flat / norms).astype(np.float32))
    print(f"  Normalized: {sat_grid_flat_norm.shape}, to {device}")

    # --- Hungary border mask ---
    hungary_mask = dp.build_or_load_hungary_mask(cache_dir, H)  # (H, W) bool
    hungary_mask_flat = torch.from_numpy(hungary_mask.reshape(-1))  # (H*W,)
    # Zero out outside-Hungary cells so they never contribute to similarity
    sat_grid_flat_norm[~hungary_mask_flat] = 0.0
    print(f"  Hungary mask applied: {hungary_mask_flat.sum():,} valid cells "
          f"({hungary_mask_flat.float().mean()*100:.1f}%)")

    # --- Step 2b: Named geographic features ---
    if cfg.geo_features_enabled:
        print("\n" + "=" * 60)
        print("STEP 2b: Named geographic features (OSM → density maps)")
        print("=" * 60)
        geo_pairs = build_or_load_geo_pairs(H, W, cfg, cache_dir)
        if geo_pairs:
            pairs = list(pairs) + geo_pairs
            print(f"  Total pairs after geo injection: {len(pairs)}")

    # --- Step 3: Compute centroids ---
    print("\n" + "=" * 60)
    print("STEP 3: Compute satellite centroids")
    print("=" * 60)
    sat_centroids = compute_sat_centroids(pairs, sat_grid)
    print(f"  Centroids: {sat_centroids.shape}")

    # --- Step 4: Load texts & build datasets ---
    print("\n" + "=" * 60)
    print("STEP 4: Load text descriptions & build datasets")
    print("=" * 60)
    raw_texts = load_raw_texts(
        cfg.text_descriptions_path,
        extra_paths=[cfg.hrl_descriptions_path],
    )
    class_names, density_maps, train_texts, val_texts = \
        build_datasets(pairs, raw_texts, cfg)

    # --- Step 5: Load Qwen ---
    print("\n" + "=" * 60)
    print("STEP 5: Load Qwen text encoder")
    print("=" * 60)
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
    text_encoder = Qwen3EmbeddingAdapter(
        model_id=cfg.qwen_model_id,
        freeze_encoder=True,
        lora=True,
        lora_r=32,
        lora_alpha=64,
        multi_gpu=multi_gpu,
    )
    text_encoder = text_encoder.to(device)
    cfg.qwen_emb_dim = text_encoder.target_dim
    n_lora = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    print(f"  Qwen loaded, emb_dim={cfg.qwen_emb_dim}")
    print(f"  LoRA trainable params: {n_lora:,}")

    # --- Step 6: Build projection head ---
    print("\n" + "=" * 60)
    print("STEP 6: Build projection head + temperature")
    print("=" * 60)
    proj_head = SatelliteProjectionHead(
        text_dim=cfg.qwen_emb_dim,
        sat_dim=cfg.sat_emb_dim,
        hidden_1=cfg.proj_hidden_1,
        hidden_2=cfg.proj_hidden_2,
    ).to(device)
    temperature = LearnableTemperature(cfg.init_temperature).to(device)
    print(f"  ProjectionHead: {cfg.qwen_emb_dim} → {cfg.proj_hidden_1} → {cfg.proj_hidden_2} → {cfg.sat_emb_dim}")

    # --- Step 7: Train ---
    print("\n" + "=" * 60)
    print("STEP 7: Contrastive training")
    print("=" * 60)
    train(proj_head, temperature, text_encoder, sat_centroids,
          sat_grid_flat_norm, density_maps, class_names,
          train_texts, val_texts, cfg, device=device,
          hungary_mask=hungary_mask)

    print("\nDone.")


if __name__ == "__main__":
    main()
