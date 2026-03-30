#!/usr/bin/env python3
import os, sys as _sys
_cu13 = os.path.join(os.path.dirname(__file__),
    ".venv/lib/python3.12/site-packages/nvidia/cu13/lib")
if os.path.isdir(_cu13) and _cu13 not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = _cu13 + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    os.execv(_sys.executable, [_sys.executable] + _sys.argv)
del _cu13

"""
CORINE x AlphaEarth Embedding Inspection
=========================================
Bulk-extract satellite embeddings from Milvus, spatial-join with CORINE
land cover polygons, and produce six research-grade analyses measuring
semantic cohesion of embeddings within CORINE classes.

Usage:
    uv run python inspect_corine_embeddings.py
    uv run python inspect_corine_embeddings.py --skip-milvus --skip-join
    uv run python inspect_corine_embeddings.py --tile-limit 100  # quick test
"""

import argparse
import json
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
plt.style.use(["science", "ieee", "no-latex"])
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from pymilvus import Collection, connections
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Point, box
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Config ────────────────────────────────────────────────────────────────

@dataclass
class Config:
    corine_geojson: str = "data_corine/Results/U2018_CLC2018_V2020_20u1.json"
    corine_classes: str = "data_corine/corine_classes_includes.json"
    milvus_host: str = "192.168.242.182"
    milvus_port: str = "19530"
    collection: str = "high_res_hun_2018"
    embedding_field: str = "vector"
    embedding_dim: int = 64
    tile_size: float = 0.05
    tile_limit: int = 16384
    output_dir: str = "data_corine/embedding_inspection"
    lat_min: float = 45.737
    lat_max: float = 48.585
    lon_min: float = 16.113
    lon_max: float = 22.897
    seed: int = 42
    dpi: int = 300
    device: str = "cuda:0"
    max_workers: int = 8
    umap_subsample: int = 20000
    sim_subsample_per_class: int = 500
    spatial_pca_classes: List[str] = field(default_factory=lambda: ["211", "311", "512", "112"])


HIERARCHY_LABELS = {
    "1": "Artificial surfaces",
    "2": "Agricultural areas",
    "3": "Forest & semi-natural",
    "4": "Wetlands",
    "5": "Water bodies",
}

HIERARCHY_COLORS = {
    "1": "#e41a1c",
    "2": "#ff7f00",
    "3": "#4daf4a",
    "4": "#377eb8",
    "5": "#984ea3",
}


def get_l1(code: str) -> str:
    return code[0]


def get_l1_label(code: str) -> str:
    return HIERARCHY_LABELS.get(code[0], "Unknown")


# ── Section 2: Bulk Milvus Extraction ─────────────────────────────────────

def connect_milvus(cfg: Config) -> Collection:
    connections.connect(alias="default", host=cfg.milvus_host, port=cfg.milvus_port)
    coll = Collection(cfg.collection)
    coll.load()
    return coll


def query_tile(coll_name: str, host: str, port: str, lat0: float, lat1: float,
               lon0: float, lon1: float, emb_field: str, page_size: int = 16384) -> List[dict]:
    """Query a single tile from Milvus with iterative offset pagination."""
    try:
        conn_alias = f"tile_{lat0:.3f}_{lon0:.3f}"
        connections.connect(alias=conn_alias, host=host, port=port)
        coll = Collection(coll_name, using=conn_alias)
        coll.load()
        expr = (f"lat >= {lat0} && lat < {lat1} && "
                f"lon >= {lon0} && lon < {lon1}")
        all_results = []
        offset = 0
        while True:
            batch = coll.query(
                expr=expr, output_fields=[emb_field, "lat", "lon"],
                limit=page_size, offset=offset,
            )
            all_results.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        connections.disconnect(conn_alias)
        return all_results
    except Exception as e:
        print(f"  Tile ({lat0:.2f},{lon0:.2f}) failed: {e}")
        return []


def collect_embeddings_tiled(cfg: Config) -> pd.DataFrame:
    """Bulk-extract embeddings by tiling Hungary."""
    cache = Path(cfg.output_dir) / "embeddings_raw.parquet"
    if cache.exists():
        print(f"Loading cached embeddings from {cache}")
        return pd.read_parquet(cache)

    lat_edges = np.arange(cfg.lat_min, cfg.lat_max + cfg.tile_size, cfg.tile_size)
    lon_edges = np.arange(cfg.lon_min, cfg.lon_max + cfg.tile_size, cfg.tile_size)
    tiles = [(lat_edges[i], lat_edges[i + 1], lon_edges[j], lon_edges[j + 1])
             for i in range(len(lat_edges) - 1) for j in range(len(lon_edges) - 1)]
    print(f"Querying {len(tiles)} tiles from Milvus ({cfg.collection})...")

    all_rows = []
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
        futures = {
            pool.submit(
                query_tile, cfg.collection, cfg.milvus_host, cfg.milvus_port,
                t[0], t[1], t[2], t[3], cfg.embedding_field,
            ): t for t in tiles
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Milvus tiles"):
            results = fut.result()
            for r in results:
                vec = r[cfg.embedding_field]
                row = {"lat": r["lat"], "lon": r["lon"]}
                for d in range(len(vec)):
                    row[f"v{d}"] = vec[d]
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"Collected {len(df)} embeddings")

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    print(f"Cached to {cache}")
    return df


# ── Section 3: CORINE Spatial Join ────────────────────────────────────────

def assign_corine_classes(emb_df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Spatial join embeddings with CORINE polygons."""
    cache = Path(cfg.output_dir) / "embeddings_with_corine.parquet"
    if cache.exists():
        print(f"Loading cached spatial join from {cache}")
        return pd.read_parquet(cache)

    print(f"Loading CORINE GeoJSON ({cfg.corine_geojson})...")
    gdf = gpd.read_file(cfg.corine_geojson)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # Clip to Hungary bounds
    hungary = box(cfg.lon_min, cfg.lat_min, cfg.lon_max, cfg.lat_max)
    gdf = gdf[gdf.geometry.intersects(hungary)].copy()
    gdf = gdf[["Code_18", "geometry"]].copy()
    gdf["Code_18"] = gdf["Code_18"].astype(str)
    print(f"  {len(gdf)} CORINE features, {gdf['Code_18'].nunique()} unique classes")

    print("Building point geometries...")
    geometry = gpd.points_from_xy(emb_df["lon"], emb_df["lat"])
    points_gdf = gpd.GeoDataFrame(emb_df, geometry=geometry, crs="EPSG:4326")

    print("Spatial join (this may take a few minutes)...")
    joined = gpd.sjoin(points_gdf, gdf, how="inner", predicate="within")
    joined = joined.drop(columns=["geometry", "index_right"])
    result = pd.DataFrame(joined)

    print(f"  Joined: {len(result)} embeddings with CORINE class")
    counts = result["Code_18"].value_counts()
    print(f"  Classes present: {len(counts)}")
    print(f"  Top 5: {dict(counts.head())}")

    result.to_parquet(cache, index=False)
    print(f"Cached to {cache}")
    return result


# ── Helpers ───────────────────────────────────────────────────────────────

def get_embedding_matrix(df: pd.DataFrame, dim: int = 64) -> np.ndarray:
    """Extract embedding columns as numpy array."""
    cols = [f"v{d}" for d in range(dim)]
    return df[cols].values.astype(np.float32)


def load_class_names(cfg: Config) -> Dict[str, str]:
    with open(cfg.corine_classes, "r") as f:
        raw = json.load(f)
    return {k: v[:80] for k, v in raw.items()}


def get_class_colors(codes: List[str]) -> Dict[str, np.ndarray]:
    """Assign colors based on L1 hierarchy."""
    colors = {}
    for c in codes:
        colors[c] = HIERARCHY_COLORS.get(c[0], "#999999")
    return colors


def subsample_per_class(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    parts = []
    for code, grp in df.groupby("Code_18"):
        if len(grp) > n:
            parts.append(grp.sample(n, random_state=rng))
        else:
            parts.append(grp)
    return pd.concat(parts, ignore_index=True)


# ── Analysis 1: Intra/Inter-class Similarity Matrix ──────────────────────

def analysis_similarity_matrix(df: pd.DataFrame, cfg: Config,
                               class_names: Dict[str, str]) -> Tuple[plt.Figure, dict]:
    print("\n=== Analysis 1: Intra/Inter-class Similarity Matrix ===")
    sub = subsample_per_class(df, cfg.sim_subsample_per_class, cfg.seed)
    codes = sorted(sub["Code_18"].unique())
    n_classes = len(codes)
    code_to_idx = {c: i for i, c in enumerate(codes)}

    sim_matrix = np.zeros((n_classes, n_classes))
    stats = {}

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Precompute normalized embeddings per class on GPU
    class_embs = {}
    for c in codes:
        mask = sub["Code_18"] == c
        emb = get_embedding_matrix(sub[mask], cfg.embedding_dim)
        t = torch.tensor(emb, device=device)
        class_embs[c] = F.normalize(t, dim=1)

    for i, ci in enumerate(tqdm(codes, desc="Similarity")):
        ei = class_embs[ci]
        for j, cj in enumerate(codes):
            if j < i:
                sim_matrix[i, j] = sim_matrix[j, i]
                continue
            ej = class_embs[cj]
            # Compute mean cosine similarity
            cos = torch.mm(ei, ej.T)
            if i == j:
                # Exclude self-pairs (diagonal of cos matrix)
                mask = ~torch.eye(cos.shape[0], dtype=torch.bool, device=device)
                vals = cos[mask]
            else:
                vals = cos.flatten()
            sim_matrix[i, j] = vals.mean().item()

    # Collect stats
    for i, c in enumerate(codes):
        intra = sim_matrix[i, i]
        inter_vals = [sim_matrix[i, j] for j in range(n_classes) if j != i]
        stats[c] = {
            "intra_similarity": float(intra),
            "inter_similarity_mean": float(np.mean(inter_vals)),
            "cohesion_gap": float(intra - np.mean(inter_vals)),
            "n_samples": int((sub["Code_18"] == c).sum()),
        }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8.5))
    short_labels = [f"{c}" for c in codes]
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.2, vmax=1.0, aspect="equal")

    # Draw L1 group boundaries
    boundaries = []
    prev_l1 = None
    for i, c in enumerate(codes):
        l1 = get_l1(c)
        if l1 != prev_l1 and prev_l1 is not None:
            boundaries.append(i - 0.5)
        prev_l1 = l1
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=1.2)
        ax.axvline(b, color="black", linewidth=1.2)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(short_labels, rotation=90, fontsize=5)
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels(short_labels, fontsize=5)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Cosine Similarity")
    ax.set_title("Intra/Inter-class Embedding Similarity (CORINE $\\times$ AlphaEarth)")
    fig.tight_layout()
    return fig, stats


# ── Analysis 2: UMAP ─────────────────────────────────────────────────────

def analysis_umap(df: pd.DataFrame, cfg: Config,
                  class_names: Dict[str, str]) -> Tuple[plt.Figure, np.ndarray]:
    print("\n=== Analysis 2: UMAP Visualization ===")

    # Subsample for visualization clarity
    if len(df) > cfg.umap_subsample:
        sub = subsample_per_class(
            df, max(50, cfg.umap_subsample // df["Code_18"].nunique()), cfg.seed
        )
        if len(sub) > cfg.umap_subsample:
            sub = sub.sample(cfg.umap_subsample, random_state=cfg.seed)
    else:
        sub = df

    emb = get_embedding_matrix(sub, cfg.embedding_dim)
    codes = sub["Code_18"].values

    try:
        import umap
        print("  Using UMAP (n_neighbors=30, min_dist=0.3, cosine)")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.3, metric="cosine",
                            n_components=2, random_state=cfg.seed)
        coords = reducer.fit_transform(emb)
    except ImportError:
        print("  umap-learn not available, falling back to t-SNE")
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, metric="cosine", random_state=cfg.seed,
                       perplexity=30).fit_transform(emb)

    unique_codes = sorted(set(codes))
    l1_groups = sorted(set(get_l1(c) for c in unique_codes))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel A: by CORINE class
    ax = axes[0]
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_codes))
    for i, c in enumerate(unique_codes):
        mask = codes == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=2, alpha=0.4,
                   color=cmap(i), label=c, rasterized=True)
    ax.set_title("By CORINE Class (44)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    # Panel B: by L1 hierarchy
    ax = axes[1]
    for l1 in l1_groups:
        mask = np.array([get_l1(c) == l1 for c in codes])
        label = HIERARCHY_LABELS.get(l1, l1)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=2, alpha=0.4,
                   color=HIERARCHY_COLORS.get(l1, "#999"), label=label, rasterized=True)
    ax.legend(fontsize=6, markerscale=4, loc="best", framealpha=0.9)
    ax.set_title("By L1 Hierarchy (5)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    fig.suptitle("Embedding Space Structure", y=1.02)
    fig.tight_layout()
    return fig, coords


# ── Analysis 3: Silhouette ────────────────────────────────────────────────

def analysis_silhouette(df: pd.DataFrame, cfg: Config,
                        class_names: Dict[str, str]) -> Tuple[plt.Figure, dict]:
    print("\n=== Analysis 3: Per-class Silhouette Scores ===")

    # Subsample for speed (silhouette is O(n^2))
    max_total = 15000
    sub = subsample_per_class(df, max(30, max_total // df["Code_18"].nunique()), cfg.seed)
    if len(sub) > max_total:
        sub = sub.sample(max_total, random_state=cfg.seed)

    emb = get_embedding_matrix(sub, cfg.embedding_dim)
    labels = sub["Code_18"].values

    # Need numeric labels for silhouette
    unique_codes = sorted(set(labels))
    code_to_int = {c: i for i, c in enumerate(unique_codes)}
    int_labels = np.array([code_to_int[c] for c in labels])

    print(f"  Computing silhouette on {len(sub)} samples, {len(unique_codes)} classes...")
    sil_samples = silhouette_samples(emb, int_labels, metric="cosine")

    stats = {}
    code_means = []
    for c in unique_codes:
        mask = labels == c
        vals = sil_samples[mask]
        mean_s = float(np.mean(vals))
        stats[c] = {
            "mean_silhouette": mean_s,
            "std_silhouette": float(np.std(vals)),
            "n_samples": int(mask.sum()),
        }
        code_means.append((c, mean_s, int(mask.sum())))

    # Sort by silhouette score
    code_means.sort(key=lambda x: x[1], reverse=True)
    overall = float(np.mean(sil_samples))
    stats["_overall"] = overall
    print(f"  Overall silhouette: {overall:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(6, 8))
    codes_sorted = [x[0] for x in code_means]
    means_sorted = [x[1] for x in code_means]
    counts_sorted = [x[2] for x in code_means]
    colors = [HIERARCHY_COLORS.get(c[0], "#999") for c in codes_sorted]

    bars = ax.barh(range(len(codes_sorted)), means_sorted, color=colors, edgecolor="none")
    ax.set_yticks(range(len(codes_sorted)))
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(codes_sorted, counts_sorted)], fontsize=5)
    ax.axvline(overall, color="black", linestyle="--", linewidth=0.8, label=f"Overall: {overall:.3f}")
    ax.set_xlabel("Mean Silhouette Score")
    ax.set_title("Per-class Silhouette (cosine metric)")
    ax.legend(fontsize=7)
    ax.invert_yaxis()

    # Add L1 legend
    for l1, label in HIERARCHY_LABELS.items():
        ax.barh([], [], color=HIERARCHY_COLORS[l1], label=f"{l1}xx: {label}")
    ax.legend(fontsize=5, loc="lower right")

    fig.tight_layout()
    return fig, stats


# ── Analysis 4: Dendrogram vs CORINE Hierarchy ───────────────────────────

def analysis_dendrogram(df: pd.DataFrame, cfg: Config,
                        class_names: Dict[str, str]) -> Tuple[plt.Figure, dict]:
    print("\n=== Analysis 4: Dendrogram vs CORINE Hierarchy ===")

    codes = sorted(df["Code_18"].unique())
    centroids = []
    for c in codes:
        mask = df["Code_18"] == c
        emb = get_embedding_matrix(df[mask], cfg.embedding_dim)
        centroids.append(emb.mean(axis=0))
    centroids = np.array(centroids)

    # Cosine distance matrix
    dist_condensed = pdist(centroids, metric="cosine")

    # Linkage (average for cosine distances)
    Z = linkage(dist_condensed, method="average")

    # Cophenetic correlation with CORINE hierarchy distances
    corine_dist = np.zeros((len(codes), len(codes)))
    for i, ci in enumerate(codes):
        for j, cj in enumerate(codes):
            if ci[0] != cj[0]:
                corine_dist[i, j] = 2.0  # different L1
            elif ci[:2] != cj[:2]:
                corine_dist[i, j] = 1.0  # same L1, different L2
            elif ci != cj:
                corine_dist[i, j] = 0.5  # same L2, different L3
    corine_condensed = squareform(corine_dist)
    coph_corr, _ = cophenet(Z, corine_condensed)
    print(f"  Cophenetic correlation with CORINE hierarchy: {coph_corr:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    leaf_colors = {i: HIERARCHY_COLORS.get(codes[i][0], "#999") for i in range(len(codes))}

    # Color function for dendrogram
    from scipy.cluster.hierarchy import leaves_list
    dendrogram(
        Z, labels=codes, ax=ax, leaf_rotation=90, leaf_font_size=6,
        above_threshold_color="#888",
    )

    # Color the leaf labels
    xlabels = ax.get_xticklabels()
    for lbl in xlabels:
        code = lbl.get_text()
        lbl.set_color(HIERARCHY_COLORS.get(code[0], "#999"))
        lbl.set_fontweight("bold")

    ax.set_ylabel("Cosine Distance")
    ax.set_title(f"Embedding-based Dendrogram (cophenetic corr. with CORINE: {coph_corr:.3f})")
    fig.tight_layout()

    return fig, {"cophenetic_correlation": float(coph_corr)}


# ── Analysis 5: Centroid Similarity Heatmap ───────────────────────────────

def analysis_centroid_heatmap(df: pd.DataFrame, cfg: Config,
                              class_names: Dict[str, str]) -> Tuple[plt.Figure, np.ndarray]:
    print("\n=== Analysis 5: Centroid Cosine Similarity Heatmap ===")

    codes = sorted(df["Code_18"].unique())
    centroids = []
    for c in codes:
        mask = df["Code_18"] == c
        emb = get_embedding_matrix(df[mask], cfg.embedding_dim)
        centroids.append(emb.mean(axis=0))

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    t = F.normalize(torch.tensor(np.array(centroids), device=device), dim=1)
    sim_matrix = torch.mm(t, t.T).cpu().numpy()

    # Find L1 boundaries for annotation
    boundaries = []
    prev_l1 = None
    for i, c in enumerate(codes):
        l1 = get_l1(c)
        if l1 != prev_l1 and prev_l1 is not None:
            boundaries.append(i - 0.5)
        prev_l1 = l1

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-0.3, vmax=1.0, aspect="equal")

    for b in boundaries:
        ax.axhline(b, color="black", linewidth=1.2)
        ax.axvline(b, color="black", linewidth=1.2)

    ax.set_xticks(range(len(codes)))
    ax.set_xticklabels(codes, rotation=90, fontsize=5)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=5)

    # Color tick labels by L1
    for lbl in ax.get_xticklabels():
        lbl.set_color(HIERARCHY_COLORS.get(lbl.get_text()[0], "#999"))
    for lbl in ax.get_yticklabels():
        lbl.set_color(HIERARCHY_COLORS.get(lbl.get_text()[0], "#999"))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cosine Similarity")
    ax.set_title("Class Centroid Similarity Matrix")
    fig.tight_layout()

    return fig, sim_matrix


# ── Analysis 6: Spatial PCA ──────────────────────────────────────────────

def _rasterize_pc(lats: np.ndarray, lons: np.ndarray, values: np.ndarray,
                   lat_bounds: Tuple[float, float], lon_bounds: Tuple[float, float],
                   resolution: float = 0.005) -> Tuple[np.ndarray, tuple]:
    """Bin PC values onto a regular lat/lon grid, returning a 2D raster and extent."""
    lat_edges = np.arange(lat_bounds[0], lat_bounds[1] + resolution, resolution)
    lon_edges = np.arange(lon_bounds[0], lon_bounds[1] + resolution, resolution)

    sum_grid = np.zeros((len(lat_edges) - 1, len(lon_edges) - 1), dtype=np.float64)
    cnt_grid = np.zeros_like(sum_grid)

    lat_idx = np.searchsorted(lat_edges, lats) - 1
    lon_idx = np.searchsorted(lon_edges, lons) - 1

    valid = ((lat_idx >= 0) & (lat_idx < sum_grid.shape[0]) &
             (lon_idx >= 0) & (lon_idx < sum_grid.shape[1]))
    lat_idx, lon_idx, values = lat_idx[valid], lon_idx[valid], values[valid]

    np.add.at(sum_grid, (lat_idx, lon_idx), values)
    np.add.at(cnt_grid, (lat_idx, lon_idx), 1)

    with np.errstate(invalid="ignore"):
        mean_grid = np.where(cnt_grid > 0, sum_grid / cnt_grid, np.nan)

    extent = (lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1])
    return mean_grid, extent


def analysis_spatial_pca(df: pd.DataFrame, cfg: Config,
                         class_names: Dict[str, str]) -> List[plt.Figure]:
    print("\n=== Analysis 6: Spatial PCA for Selected Classes ===")

    selected = [c for c in cfg.spatial_pca_classes if c in df["Code_18"].unique()]
    if not selected:
        print("  No selected classes found in data, skipping.")
        return [plt.figure()]

    lat_bounds = (cfg.lat_min, cfg.lat_max)
    lon_bounds = (cfg.lon_min, cfg.lon_max)
    figs = []

    for code in selected:
        mask = df["Code_18"] == code
        sub = df[mask]
        emb = get_embedding_matrix(sub, cfg.embedding_dim)
        lats = sub["lat"].values
        lons = sub["lon"].values

        pca = PCA(n_components=3)
        pcs = pca.fit_transform(emb)
        var_explained = pca.explained_variance_ratio_

        name_short = class_names.get(code, "")[:60]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle(f"CORINE {code}: {name_short}", fontsize=9, y=1.02)

        for col in range(3):
            ax = axes[col]
            raster, extent = _rasterize_pc(
                lats, lons, pcs[:, col], lat_bounds, lon_bounds, resolution=0.005
            )
            vmax = np.nanpercentile(np.abs(raster), 98)
            im = ax.imshow(raster, origin="lower", extent=extent, aspect=1.5,
                           cmap="coolwarm", vmin=-vmax, vmax=vmax,
                           interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(f"PC{col+1} ({var_explained[col]:.1%})", fontsize=8)
            ax.set_xlabel("Longitude", fontsize=7)
            if col == 0:
                ax.set_ylabel("Latitude", fontsize=7)
            ax.tick_params(labelsize=6)

        fig.tight_layout()
        figs.append(fig)
        print(f"  {code}: {len(sub)} points, var explained "
              f"{var_explained[0]:.1%}/{var_explained[1]:.1%}/{var_explained[2]:.1%}")

    return figs


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CORINE x AlphaEarth Embedding Inspection")
    parser.add_argument("--skip-milvus", action="store_true", help="Use cached embeddings")
    parser.add_argument("--skip-join", action="store_true", help="Use cached spatial join")
    parser.add_argument("--tile-limit", type=int, default=3000, help="Max results per tile")
    parser.add_argument("--tile-size", type=float, default=0.1, help="Tile size in degrees")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument("--device", default="cuda:0", help="Torch device")
    parser.add_argument("--output-dir", default="data_corine/embedding_inspection")
    parser.add_argument("--spatial-pca-classes", nargs="+", default=["211", "311", "512", "112"])
    args = parser.parse_args()

    cfg = Config(
        tile_limit=args.tile_limit,
        tile_size=args.tile_size,
        dpi=args.dpi,
        device=args.device,
        output_dir=args.output_dir,
        spatial_pca_classes=args.spatial_pca_classes,
    )
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Collect embeddings
    if not args.skip_milvus:
        emb_df = collect_embeddings_tiled(cfg)
    else:
        cache = out / "embeddings_raw.parquet"
        if not cache.exists():
            print(f"ERROR: --skip-milvus but {cache} not found. Run without --skip-milvus first.")
            return
        emb_df = pd.read_parquet(cache)
        print(f"Loaded {len(emb_df)} cached embeddings")

    # Step 2: Spatial join
    if not args.skip_join:
        joined_df = assign_corine_classes(emb_df, cfg)
    else:
        cache = out / "embeddings_with_corine.parquet"
        if not cache.exists():
            print(f"ERROR: --skip-join but {cache} not found. Run without --skip-join first.")
            return
        joined_df = pd.read_parquet(cache)
        print(f"Loaded {len(joined_df)} cached joined embeddings")

    # Filter classes with too few samples
    counts = joined_df["Code_18"].value_counts()
    min_samples = 20
    valid_classes = counts[counts >= min_samples].index.tolist()
    dropped = counts[counts < min_samples]
    if len(dropped) > 0:
        print(f"Dropping {len(dropped)} classes with <{min_samples} samples: {list(dropped.index)}")
    joined_df = joined_df[joined_df["Code_18"].isin(valid_classes)].copy()

    print(f"\nDataset summary:")
    print(f"  Total embeddings: {len(joined_df)}")
    print(f"  Classes: {joined_df['Code_18'].nunique()}")
    print(f"  Per-class range: {counts[valid_classes].min()} - {counts[valid_classes].max()}")

    class_names = load_class_names(cfg)

    # Run all 6 analyses
    all_stats = {}

    # 1. Similarity matrix
    fig1, stats1 = analysis_similarity_matrix(joined_df, cfg, class_names)
    fig1.savefig(out / "fig1_similarity_matrix.png", dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig1)
    all_stats["similarity"] = stats1

    # 2. UMAP
    fig2, umap_coords = analysis_umap(joined_df, cfg, class_names)
    fig2.savefig(out / "fig2_umap.png", dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig2)

    # 3. Silhouette
    fig3, stats3 = analysis_silhouette(joined_df, cfg, class_names)
    fig3.savefig(out / "fig3_silhouette.png", dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig3)
    all_stats["silhouette"] = stats3

    # 4. Dendrogram
    fig4, stats4 = analysis_dendrogram(joined_df, cfg, class_names)
    fig4.savefig(out / "fig4_dendrogram.png", dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig4)
    all_stats["dendrogram"] = stats4

    # 5. Centroid heatmap
    fig5, sim_mat = analysis_centroid_heatmap(joined_df, cfg, class_names)
    fig5.savefig(out / "fig5_centroid_heatmap.png", dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig5)

    # 6. Spatial PCA (one figure per class)
    pca_figs = analysis_spatial_pca(joined_df, cfg, class_names)
    for i, (fig6, code) in enumerate(zip(pca_figs, cfg.spatial_pca_classes)):
        fig6.savefig(out / f"fig6_spatial_pca_{code}.png", dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig6)

    # Save summary stats
    stats_path = out / "summary_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nSaved summary stats to {stats_path}")

    print(f"\nAll figures saved to {out}/")
    print("Done.")


if __name__ == "__main__":
    main()
