"""
data_pipeline.py — Preprocessing for flow_4 contrastive training
=================================================================

Builds and caches:
  1. Density maps   — per-class spatial distributions (from CORINE/HRL rasters)
  2. Pairs cache    — density maps + class metadata (.pt)
  3. Sat grid       — AlphaEarth embeddings binned into (H, W, 64) grid
                      using per-cell median aggregation, empty cells filled
                      via nearest-neighbour (.npy)

Run this before flow_4.py whenever you change resolution or the source data.

Usage:
    .venv/bin/python data_pipeline.py
    .venv/bin/python data_pipeline.py --resolution 512
    .venv/bin/python data_pipeline.py --sat-emb-path data_corine/embedding_inspection/embeddings_with_corine.parquet
"""

import argparse
import hashlib
import pickle
import numpy as np
import torch
from pathlib import Path


# ============================================================
# CONFIG
# ============================================================

SAT_EMB_PATH   = "data_corine/embedding_inspection/embeddings_with_corine.parquet"
SAT_EMB_DIM    = 64
TARGET_RES     = 512                          # grid height in pixels
LAT_MIN, LAT_MAX = 45.737, 48.585
LON_MIN, LON_MAX = 16.113, 22.897
CACHE_DIR      = Path("pipeline_cache")

CORINE_DESCRIPTIONS = "data_corine/corine_wiki_char_count.jsonl"
HRL_DESCRIPTIONS    = "data_corine/hrl_wiki_char_count.jsonl"
R3_DIR              = "data_corine/Results-3"
R4_DIR              = "data_corine/Results-4"
CORINE_GEOJSON      = "data_corine/Results/U2018_CLC2018_V2020_20u1.json"
MIN_PIXELS          = 500
QWEN_EMB_DIM        = 2560


def _hash(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def target_width(res):
    aspect = (LON_MAX - LON_MIN) / (LAT_MAX - LAT_MIN)
    return int(res * aspect)


# ============================================================
# STEP 1 — Density maps (distributions)
# ============================================================

def build_or_load_distributions(cache_dir: Path, res: int) -> dict:
    print("\n" + "=" * 60)
    print("STEP 1: Density maps (CORINE / HRL rasters)")
    print("=" * 60)

    dist_hash = _hash(R3_DIR, R4_DIR, CORINE_GEOJSON,
                      LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                      res, MIN_PIXELS)
    dist_cache = cache_dir / f"distributions_{dist_hash}.pkl"

    if dist_cache.exists():
        print(f"[cache] HIT → {dist_cache.name}")
        with open(dist_cache, "rb") as f:
            distributions = pickle.load(f)
        print(f"  {len(distributions)} classes")
        return distributions

    from flow_2 import PipelineConfig
    from full_flow import process_all_rasters
    cfg = PipelineConfig()
    cfg.target_resolution = res
    cfg.r3_dir = R3_DIR
    cfg.r4_dir = R4_DIR
    cfg.corine_geojson_path = CORINE_GEOJSON
    cfg.min_pixels_for_class = MIN_PIXELS

    distributions = process_all_rasters(cfg)
    with open(dist_cache, "wb") as f:
        pickle.dump(distributions, f)
    print(f"  Built {len(distributions)} classes → {dist_cache.name}")
    return distributions


# ============================================================
# STEP 2 — Pairs cache
# ============================================================

def build_or_load_pairs(distributions: dict, cache_dir: Path,
                        dist_hash_str: str, device: str) -> list:
    print("\n" + "=" * 60)
    print("STEP 2: Pairs cache (density maps + class metadata)")
    print("=" * 60)

    pairs_hash = _hash(dist_hash_str, CORINE_DESCRIPTIONS,
                       HRL_DESCRIPTIONS, QWEN_EMB_DIM)
    pairs_cache = cache_dir / f"pairs_{pairs_hash}.pt"

    if pairs_cache.exists():
        print(f"[cache] HIT (hash) → {pairs_cache.name}")
        pairs = torch.load(pairs_cache, map_location="cpu", weights_only=False)
        print(f"  {len(pairs)} pairs loaded")
        return pairs

    # Shape-based fallback: accept any pairs cache whose density maps match resolution
    expected_H = TARGET_RES
    expected_W = target_width(TARGET_RES)
    for existing in sorted(cache_dir.glob("pairs_*.pt"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
        p = torch.load(existing, map_location="cpu", weights_only=False)
        dm_shape = np.asarray(p[0]["density_map"]).shape
        if dm_shape == (expected_H, expected_W):
            print(f"[cache] HIT (shape {expected_H}×{expected_W}) → {existing.name}")
            torch.save(p, pairs_cache)
            return p
        print(f"  [skip] {existing.name}  density_map shape={dm_shape}")

    # Upsample fallback: resize density maps from any existing pairs cache
    # — no Qwen re-encoding needed, just bilinear interpolation of spatial maps
    import torch.nn.functional as F_nn
    for existing in sorted(cache_dir.glob("pairs_*.pt"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
        p = torch.load(existing, map_location="cpu", weights_only=False)
        dm_shape = np.asarray(p[0]["density_map"]).shape
        print(f"  [upsample] {existing.name}  {dm_shape} → ({expected_H},{expected_W})")
        upsampled = []
        for pair in p:
            pair = dict(pair)
            dm = torch.as_tensor(np.asarray(pair["density_map"]), dtype=torch.float32)
            dm = F_nn.interpolate(
                dm.unsqueeze(0).unsqueeze(0),
                size=(expected_H, expected_W),
                mode="bilinear",
                align_corners=False,
            ).squeeze()
            pair["density_map"] = dm.numpy()
            upsampled.append(pair)
        torch.save(upsampled, pairs_cache)
        print(f"  [upsample] saved → {pairs_cache.name}")
        return upsampled

    from full_flow import build_training_pairs
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter

    print("  Building pairs from scratch (requires Qwen encoding)...")
    enc = Qwen3EmbeddingAdapter(target_dim=QWEN_EMB_DIM, freeze_encoder=True)
    enc = enc.to(device).eval()
    pairs = build_training_pairs(
        distributions, CORINE_DESCRIPTIONS, enc,
        extra_descriptions_paths=[HRL_DESCRIPTIONS],
    )
    del enc
    torch.cuda.empty_cache()
    torch.save(pairs, pairs_cache)
    print(f"  {len(pairs)} pairs → {pairs_cache.name}")
    return pairs


# ============================================================
# STEP 3 — Satellite embedding grid (median per cell)
# ============================================================

def build_or_load_sat_grid(cache_dir: Path, res: int,
                           sat_emb_path: str) -> np.ndarray:
    print("\n" + "=" * 60)
    print("STEP 3: AlphaEarth satellite grid (median aggregation)")
    print("=" * 60)

    H = res
    W = target_width(res)
    cache_key = _hash(f"sat_grid_median|{H}x{W}|{sat_emb_path}|{SAT_EMB_DIM}")
    cache_path = cache_dir / f"sat_grid_{cache_key}.npy"

    if cache_path.exists():
        print(f"[cache] HIT (hash) → {cache_path.name}")
        grid = np.load(cache_path)
        print(f"  Grid shape: {grid.shape}")
        return grid

    # Shape-based fallback: accept any existing sat_grid with matching dimensions
    # (handles files built by flow_2.py or previous runs with a different hash prefix)
    for existing in sorted(cache_dir.glob("sat_grid_*.npy"),
                           key=lambda p: p.stat().st_mtime, reverse=True):
        g = np.load(existing, mmap_mode="r")
        if g.shape[:2] == (H, W):
            grid = np.array(g)
            print(f"[cache] HIT (shape {H}×{W}) → {existing.name}")
            # re-save under the canonical hash name so future runs find it instantly
            np.save(cache_path, grid)
            return grid
        print(f"  [skip] {existing.name}  shape={g.shape}")

    import pandas as pd
    from scipy.ndimage import distance_transform_edt

    print(f"  Loading embeddings from {sat_emb_path} ...")
    emb_cols = [f"v{i}" for i in range(SAT_EMB_DIM)]
    df = pd.read_parquet(sat_emb_path)
    lats = df["lat"].values.astype(np.float64)
    lons = df["lon"].values.astype(np.float64)
    embs = df[emb_cols].values.astype(np.float32)
    print(f"  {len(df):,} embeddings loaded")

    y_idx = ((LAT_MAX - lats) / (LAT_MAX - LAT_MIN) * H).clip(0, H - 1).astype(np.int32)
    x_idx = ((lons - LON_MIN) / (LON_MAX - LON_MIN) * W).clip(0, W - 1).astype(np.int32)
    flat_idx = (y_idx * W + x_idx).astype(np.int64)

    print(f"  Binning into {H}×{W} grid using per-cell median ...")
    sort_order = np.argsort(flat_idx, kind="stable")
    sorted_idx = flat_idx[sort_order]
    sorted_embs = embs[sort_order]

    unique_cells, first_occ, _ = np.unique(sorted_idx,
                                           return_index=True,
                                           return_counts=True)
    grid_flat = np.zeros((H * W, SAT_EMB_DIM), dtype=np.float32)
    splits = np.split(sorted_embs, first_occ[1:])
    for cell, group in zip(unique_cells, splits):
        grid_flat[cell] = np.median(group, axis=0)

    filled = np.zeros(H * W, dtype=bool)
    filled[unique_cells] = True
    n_empty = (~filled).sum()
    print(f"  Filled: {filled.sum():,}/{H*W:,} cells  —  {n_empty:,} empty → NN fill")

    # Fill empty cells with nearest filled neighbour
    grid = grid_flat.reshape(H, W, SAT_EMB_DIM)
    filled_grid = filled.reshape(H, W)
    _, nearest = distance_transform_edt(~filled_grid, return_indices=True)
    grid[~filled_grid] = grid[nearest[0][~filled_grid], nearest[1][~filled_grid]]

    np.save(cache_path, grid)
    print(f"  Saved → {cache_path.name}  shape={grid.shape}")
    return grid


# ============================================================
# STEP 4 — Hungary border mask
# ============================================================

CORINE_GEOJSON = "data_corine/Results/U2018_CLC2018_V2020_20u1.json"


def build_or_load_hungary_mask(cache_dir: Path, res: int) -> np.ndarray:
    """Boolean mask (H, W) — True inside Hungary's actual borders.

    Derived by dissolving all CORINE land-cover polygons into a single
    Hungary outline, then rasterizing onto the grid.
    Cached as hungary_mask_{H}x{W}.npy.
    """
    H = res
    W = target_width(res)
    cache_path = cache_dir / f"hungary_mask_{H}x{W}.npy"

    if cache_path.exists():
        mask = np.load(cache_path)
        print(f"[cache] Hungary mask HIT → {cache_path.name}  "
              f"({mask.sum():,}/{H*W:,} cells inside)")
        return mask

    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union

    print(f"  Building Hungary border mask at {H}×{W} from CORINE features...")
    with open(CORINE_GEOJSON, encoding="utf-8") as f:
        fc = json.load(f)
    geoms = [shape(feat["geometry"]) for feat in fc["features"] if feat.get("geometry")]
    hungary = unary_union(geoms)

    lons = np.linspace(LON_MIN, LON_MAX, W)
    lats = np.linspace(LAT_MAX, LAT_MIN, H)   # row 0 = north
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    from shapely import contains_xy
    mask = contains_xy(hungary, lon_grid.ravel(), lat_grid.ravel()).reshape(H, W)

    np.save(cache_path, mask)
    print(f"  Mask saved → {cache_path.name}  "
          f"({mask.sum():,}/{H*W:,} = {mask.mean()*100:.1f}% inside Hungary)")
    return mask


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=TARGET_RES)
    parser.add_argument("--sat-emb-path", default=SAT_EMB_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    args = parser.parse_args()

    res = args.resolution
    sat_path = args.sat_emb_path
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Resolution : {res}×{target_width(res)}")
    print(f"Sat emb    : {sat_path}")
    print(f"Device     : {device}")
    print(f"Cache dir  : {cache_dir}")

    # Step 1
    dist_hash = _hash(R3_DIR, R4_DIR, CORINE_GEOJSON,
                      LAT_MIN, LAT_MAX, LON_MIN, LON_MAX,
                      res, MIN_PIXELS)
    distributions = build_or_load_distributions(cache_dir, res)

    # Step 2
    pairs = build_or_load_pairs(distributions, cache_dir, dist_hash, device)

    # Step 3
    sat_grid = build_or_load_sat_grid(cache_dir, res, sat_path)

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  Pairs  : {len(pairs)} classes")
    print(f"  Sat grid: {sat_grid.shape}")
    print(f"  Cache  : {cache_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
