"""
SpatialBasisField v2 — Text-conditioned spatial distribution model
===================================================================

Key changes from v1:
  1. FiLM conditioning: text modulates spatial features at every layer
  2. Sigmoid factors: independent gating instead of softmax winner-take-all
  3. Reduced coordinate capacity: forces reliance on text signal
  4. Discrimination loss: penalizes identical outputs for different texts
  5. Class-level val split: measures actual generalization
  6. Text dropout: random zeroing of text embedding during training (like CFG)
"""

import hashlib
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class PipelineConfig:
    # --- Data paths ---
    r3_dir: str = "data_corine/Results-3"
    r4_dir: str = "data_corine/Results-4"
    corine_geojson_path: str = "data_corine/Results/U2018_CLC2018_V2020_20u1.json"
    text_descriptions_path: str = "data_corine/corine_wiki_char_count.jsonl"
    hrl_descriptions_path: str = "data_corine/hrl_wiki_char_count.jsonl"
    output_dir: str = "training_data"
    cache_dir: str = "pipeline_cache"

    # --- Geographic bounds (Hungary) ---
    lat_min: float = 45.737
    lat_max: float = 48.585
    lon_min: float = 16.113
    lon_max: float = 22.897

    # --- Processing ---
    target_resolution: int = 256
    min_pixels_for_class: int = 500

    # --- Satellite embeddings (AlphaEarth) ---
    sat_emb_path: str = "data_corine/embedding_inspection/embeddings_with_corine_old.parquet"
    sat_emb_dim: int = 64
    n_sat_augments: int = 16        # sub-centroid augmentations per class

    # --- Model ---
    n_bases: int = 24
    n_fourier_freqs: int = 64
    hidden_dim: int = 128
    coord_hidden: int = 128
    text_emb_dim: int = 64          # satellite embedding dim (was 2560 Qwen)
    text_proj_dim: int = 64         # project sat emb before FiLM

    # --- TextToSatBridge (text → satellite space projection) ---
    bridge_hidden_dim: int = 256
    bridge_weight: float = 1.0      # weight on alignment loss
    bridge_start_epoch: int = 50    # epoch to start mixing text path
    text_mix_ratio: float = 0.3     # fraction of batches using bridge conditioning
    bridge_lr: float = 1e-4

    # --- Qwen (frozen, used only for bridge training) ---
    qwen_emb_dim: int = 2560        # Qwen3 hidden dim — bridge input

    # --- Training ---
    n_epochs: int = 400
    lr: float = 3e-4
    batch_size: int = 65_536
    samples_per_epoch: int = 2_000_000
    val_samples: int = 100_000
    weight_decay: float = 1e-5
    text_dropout: float = 0.0        # no dropout — always condition on sat embedding
    discrimination_weight: float = 0.5
    disc_warmup_epochs: int = 50
    disc_ramp_epochs: int = 50
    plot_every: int = 20
    val_class_fraction: float = 0.2
    disc_max_pairs: int = 40

    @property
    def target_width(self):
        aspect = (self.lon_max - self.lon_min) / (self.lat_max - self.lat_min)
        return int(self.target_resolution * aspect)


# ============================================================
# MODEL
# ============================================================

class FourierFeatures(nn.Module):
    """Random Fourier features for coordinate encoding."""

    def __init__(self, n_input: int = 2, n_freqs: int = 32, sigma: float = 10.0):
        super().__init__()
        self.n_freqs = n_freqs
        B = torch.randn(n_input, n_freqs) * sigma
        self.register_buffer('B', B)

    @property
    def output_dim(self):
        return 2 + 2 * self.n_freqs

    def forward(self, coords):
        proj = coords @ self.B
        return torch.cat([coords, torch.sin(2 * math.pi * proj),
                          torch.cos(2 * math.pi * proj)], dim=-1)


class FiLMLayer(nn.Module):
    """Single linear layer with Feature-wise Linear Modulation from text.

    h = SiLU(gamma * Linear(x) + beta)
    where (gamma, beta) = Linear(text_emb)
    """

    def __init__(self, in_dim: int, out_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.film = nn.Linear(cond_dim, out_dim * 2)
        # Initialize film to identity modulation: gamma=1, beta=0
        nn.init.zeros_(self.film.weight)
        nn.init.constant_(self.film.bias[:out_dim], 1.0)   # gamma = 1
        nn.init.zeros_(self.film.bias[out_dim:])             # beta = 0

    def forward(self, x, cond):
        h = self.linear(x)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        return F.silu(gamma * h + beta)


class TextToSatBridge(nn.Module):
    """Projects Qwen text embedding → satellite embedding space (L2-normalized).

    Trained alongside the spatial model via a cosine alignment loss against
    per-class AlphaEarth satellite centroids.  At inference any free-text
    query is routed through Qwen (frozen) → this bridge → spatial model.
    """

    def __init__(self, text_dim: int = 2560, sat_dim: int = 64,
                 hidden_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, sat_dim),
        )

    def forward(self, text_emb):
        return F.normalize(self.proj(text_emb), dim=-1)


class SpatialBasisFieldV2(nn.Module):
    """Text-conditioned spatial distribution model with FiLM conditioning.

    Architecture:
        Text path:
            text_emb (2560) → LayerNorm → project (128) → conditions every layer

        Coord path (FiLM-conditioned by text at every layer):
            (lat, lon) → Fourier(32 freqs) → FiLM-MLP(128) → 128-dim features

        Basis heads:
            128 → 64 → 1 (sigmoid) per basis  ×  K bases

        Factor head:
            text_proj (128) → MLP → K factors (sigmoid, independent gating)

        Output:
            Σ_k  factor_k × basis_k(lat, lon)

    Key design decisions:
        - FiLM makes spatial features text-dependent at every layer
        - Sigmoid factors: each basis independently gated (no winner-take-all)
        - Reduced Fourier freqs (32 not 64): spatial trunk can't memorize alone
        - Text dropout: zeroes text randomly → model must gracefully degrade
        - No basis_scale/bias params: sigmoid heads already output [0,1]
    """

    def __init__(self, text_dim=2560, text_proj_dim=128, n_bases=24,
                 n_freqs=32, hidden_dim=128, coord_hidden=128):
        super().__init__()
        self.n_bases = n_bases
        self.text_proj_dim = text_proj_dim

        # --- Text projection ---
        self.text_norm = nn.LayerNorm(text_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, text_proj_dim),
            nn.SiLU(),
        )

        # --- Text → factor weights (sigmoid, not softmax) ---
        self.text_to_factors = nn.Sequential(
            nn.Linear(text_proj_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_bases),
            # sigmoid applied in forward()
        )

        # --- Coordinate encoding ---
        self.fourier = FourierFeatures(n_input=2, n_freqs=n_freqs)

        # --- FiLM-conditioned coordinate trunk ---
        coord_input_dim = self.fourier.output_dim
        self.trunk_layer1 = FiLMLayer(coord_input_dim, coord_hidden, text_proj_dim)
        self.trunk_layer2 = FiLMLayer(coord_hidden, coord_hidden, text_proj_dim)
        self.trunk_layer3 = FiLMLayer(coord_hidden, coord_hidden, text_proj_dim)

        # --- Vectorized basis heads (single batched op, not 24 sequential) ---
        self.basis_hidden_dim = hidden_dim // 2
        self.basis_layer1 = nn.Linear(coord_hidden, n_bases * self.basis_hidden_dim)
        self.basis_layer2_w = nn.Parameter(torch.randn(n_bases, self.basis_hidden_dim, 1) * 0.02)
        # Init bias to -4: sigmoid(-4)≈0.018 per basis.
        # Initial density = n_bases * sigmoid(0) * 0.018 ≈ 0.22 — well below 1.0 so
        # gradients flow freely through the clamp from the first batch.
        self.basis_layer2_b = nn.Parameter(torch.full((n_bases, 1), -4.0))

    def _get_text_cond(self, text_emb):
        """Project text to conditioning vector."""
        return self.text_proj(self.text_norm(text_emb))

    def forward(self, coords, text_emb):
        """
        coords:   (B, 2) normalized lat/lon in [-1, 1]
        text_emb: (B, text_dim) text embedding

        Returns:  (B,) predicted density in [0, 1]
        """
        # Text conditioning
        text_cond = self._get_text_cond(text_emb)  # (B, text_proj_dim)

        # Factor weights — sigmoid for independent gating
        factors = torch.sigmoid(self.text_to_factors(text_cond))  # (B, n_bases)

        # Coordinate features — FiLM conditioned by text
        coord_feat = self.fourier(coords)                            # (B, fourier_dim)
        h = self.trunk_layer1(coord_feat, text_cond)                 # (B, coord_hidden)
        h = self.trunk_layer2(h, text_cond)                          # (B, coord_hidden)
        h = self.trunk_layer3(h, text_cond)                          # (B, coord_hidden)

        # Evaluate all basis heads in one vectorized op
        h_bases = self.basis_layer1(h)                                         # (B, K*D)
        h_bases = F.silu(h_bases.view(-1, self.n_bases, self.basis_hidden_dim)) # (B, K, D)
        basis_out = torch.einsum('bkd,kdo->bk', h_bases, self.basis_layer2_w) + self.basis_layer2_b.squeeze(-1)
        basis_out = torch.sigmoid(basis_out)                                   # (B, K)

        # Weighted combination
        density = (basis_out * factors).sum(dim=-1)  # (B,)
        return density.clamp(0, 1)

    @torch.no_grad()
    def render_map(self, text_emb, H=256, W=480, device="cuda"):
        """Render a full density map for a given text embedding."""
        self.eval()
        lat_grid = torch.linspace(-1, 1, H, device=device)
        lon_grid = torch.linspace(-1, 1, W, device=device)
        grid_lat, grid_lon = torch.meshgrid(lat_grid, lon_grid, indexing='ij')
        coords = torch.stack([grid_lat.flatten(), grid_lon.flatten()], dim=-1)

        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        text_emb = text_emb.to(device)
        text_expanded = text_emb.expand(coords.shape[0], -1)

        chunk = 100_000
        parts = []
        for i in range(0, len(coords), chunk):
            parts.append(self.forward(coords[i:i+chunk], text_expanded[i:i+chunk]))
        return torch.cat(parts).view(H, W).cpu().numpy()

    @torch.no_grad()
    def get_basis_maps(self, H=256, W=480, device="cuda"):
        """Render all basis maps using a neutral (zero) text conditioning."""
        self.eval()
        lat_grid = torch.linspace(-1, 1, H, device=device)
        lon_grid = torch.linspace(-1, 1, W, device=device)
        grid_lat, grid_lon = torch.meshgrid(lat_grid, lon_grid, indexing='ij')
        coords = torch.stack([grid_lat.flatten(), grid_lon.flatten()], dim=-1)

        # Zero text → FiLM gives identity-ish modulation (due to init)
        zero_text = torch.zeros(1, self.text_proj_dim, device=device)
        zero_text_expanded = zero_text.expand(coords.shape[0], -1)

        coord_feat = self.fourier(coords)
        h = self.trunk_layer1(coord_feat, zero_text_expanded)
        h = self.trunk_layer2(h, zero_text_expanded)
        h = self.trunk_layer3(h, zero_text_expanded)

        h_bases = self.basis_layer1(h)
        h_bases = F.silu(h_bases.view(-1, self.n_bases, self.basis_hidden_dim))
        basis_logits = torch.einsum('bkd,kdo->bk', h_bases, self.basis_layer2_w) + self.basis_layer2_b.squeeze(-1)
        basis_all = torch.sigmoid(basis_logits)  # (H*W, K)
        maps = [basis_all[:, k].view(H, W).cpu().numpy() for k in range(self.n_bases)]
        return maps


_KEEP_PATTERNS = [
    "grow", "cultivat", "soil", "climate", "vegetation", "habitat",
    "landscape", "region", "area", "found in", "distribut", "elevation",
    "temperate", "continental", "plain", "lowland", "forest", "field",
    "crop", "leaf", "canopy", "flower", "root", "seed", "harvest",
    "irrigat", "rainfall", "drought", "fertile", "arid", "humid",
    "grassland", "meadow", "pasture", "woodland", "shrub", "wetland",
    "river", "lake", "marsh", "peat", "sand", "clay", "loam",
    "plant", "tree", "herb", "annual", "perennial", "deciduous",
    "satellite", "reflectance", "spectral", "surface", "cover",
    "land use", "agricultural", "urban", "industrial", "residential",
    "water", "pond", "reservoir", "stream", "floodplain",
    "hungary", "pannonian", "carpathian", "danube", "tisza",
]
_DROP_PATTERNS = [
    "born", "died", "century", "recipe", "cuisine", "cooking",
    "export", "import", "million tonnes", "gdp", "economy",
    "kingdom", "phylum", "genus", "family poaceae",
    "isbn", "doi.org", "issn", "archived from",
    "football", "stadium", "championship", "olympic",
    "album", "song", "film", "movie", "novel", "author",
]


def filter_relevant_sentences(sentences: List[str]) -> List[str]:
    """Keep sentences about physical appearance, geography, agriculture, ecology."""
    filtered = []
    for s in sentences:
        s_lower = s.lower()
        if any(drop in s_lower for drop in _DROP_PATTERNS):
            continue
        if any(keep in s_lower for keep in _KEEP_PATTERNS):
            filtered.append(s)
    return filtered


def load_raw_texts(descriptions_path, extra_paths=None, filter_relevance=True):
    """Load raw text sentences from JSONL files. Returns {desc_key: [sentences]}."""
    descriptions = {}

    def _load(path):
        p = Path(path)
        if not p.exists():
            return
        if p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    key = rec.get("code", "")
                    texts = rec.get("wiki_texts", {})
                    chunks = []
                    for article in texts.values():
                        sents = [s.strip() for s in article.replace("\n", " ").split(". ") if len(s.strip()) > 30]
                        if filter_relevance:
                            relevant = filter_relevant_sentences(sents)
                            chunks.extend(relevant if relevant else sents)
                        else:
                            chunks.extend(sents)
                    if chunks:
                        descriptions[key] = chunks

    _load(descriptions_path)
    for extra in (extra_paths or []):
        _load(extra)

    # Minimum enrichment: ensure every class has at least 3 sentences
    for key, sents in list(descriptions.items()):
        if len(sents) < 3:
            class_label = key.replace("_", " ")
            descriptions[key].extend([
                f"Land cover characterized by {class_label}",
                f"Areas of {class_label} as observed from satellite imagery",
                f"Spatial distribution of {class_label} in Hungary",
            ])

    if filter_relevance:
        total = sum(len(v) for v in descriptions.values())
        print(f"  Text filtering: {len(descriptions)} classes, {total} total sentences")
        for key, sents in sorted(descriptions.items(), key=lambda x: len(x[1])):
            print(f"    {key:30s}: {len(sents):4d} sentences")

    return descriptions


# ============================================================
# SATELLITE EMBEDDING UTILITIES
# ============================================================

# Maps HRL crop class desc_keys to CORINE Code_18 for satellite centroid lookup.
# All arable crop types → 211 (non-irrigated arable land);
# grapes → 221; permanent crops → 222; grassland → 231.
_HRL_TO_CORINE: Dict[str, str] = {
    "wheat": "211", "barley": "211", "maize": "211", "rice": "213",
    "other_cereals": "211", "fresh_vegetables": "211", "dry_pulses": "211",
    "potatoes": "211", "sugar_beet": "211", "sunflower": "211",
    "soybeans": "211", "rapeseed": "211", "flax_cotton_hemp": "211",
    "grapes": "221", "olives": "222", "fruits": "222", "nuts": "222",
    "unclassified_arable": "211", "unclassified_permanent": "231",
    "main_crop_harvest_date": "211", "permanent_grassland": "231",
    "bare_soil_before_sowing": "211", "bare_soil_after_harvest": "211",
}


def _fetch_milvus_tile(lat0: float, lat1: float, lon0: float, lon1: float,
                       host: str = "192.168.242.182", port: str = "19530",
                       collection: str = "high_res_hun_2018",
                       emb_field: str = "vector",
                       page_size: int = 16384) -> List[dict]:
    """Fetch all embeddings in a lat/lon tile via iterative offset pagination."""
    from pymilvus import Collection, connections
    alias = f"satcent_{lat0:.3f}_{lon0:.3f}_{id(object())}"
    try:
        connections.connect(alias=alias, host=host, port=port)
        coll = Collection(collection, using=alias)
        coll.load()
        expr = (f"lat >= {lat0} && lat < {lat1} && "
                f"lon >= {lon0} && lon < {lon1}")
        rows = []
        offset = 0
        while True:
            batch = coll.query(
                expr=expr, output_fields=[emb_field, "lat", "lon"],
                limit=page_size, offset=offset,
            )
            rows.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size
        connections.disconnect(alias)
        return rows
    except Exception as e:
        print(f"  Tile ({lat0:.2f},{lon0:.2f}) failed: {e}")
        return []


def _fetch_milvus_with_corine_join(cfg: "PipelineConfig") -> "pd.DataFrame":
    """Fetch all embeddings from Milvus and spatial-join with CORINE polygons.

    Tiles Hungary into 0.05° cells, fetches with offset pagination per tile,
    then assigns Code_18 via a geopandas point-in-polygon join.
    Caches the result to data_corine/embedding_inspection/embeddings_with_corine.parquet.
    """
    import geopandas as gpd
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from shapely.geometry import box

    cache = Path("data_corine/embedding_inspection/embeddings_with_corine.parquet")
    if cache.exists():
        print(f"  [cache] Spatial-join parquet found → {cache}")
        return pd.read_parquet(cache)

    # Load CORINE polygons clipped to Hungary
    print(f"  Loading CORINE polygons for spatial join...")
    gdf = gpd.read_file(cfg.corine_geojson_path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    hungary = box(cfg.lon_min, cfg.lat_min, cfg.lon_max, cfg.lat_max)
    gdf = gdf[gdf.geometry.intersects(hungary)][["Code_18", "geometry"]].copy()
    gdf["Code_18"] = gdf["Code_18"].astype(str)
    print(f"  {len(gdf)} CORINE features, {gdf['Code_18'].nunique()} classes")

    # Tile Hungary
    tile_size = 0.05
    lat_edges = np.arange(cfg.lat_min, cfg.lat_max + tile_size, tile_size)
    lon_edges = np.arange(cfg.lon_min, cfg.lon_max + tile_size, tile_size)
    tiles = [
        (lat_edges[i], lat_edges[i + 1], lon_edges[j], lon_edges[j + 1])
        for i in range(len(lat_edges) - 1)
        for j in range(len(lon_edges) - 1)
    ]
    print(f"  Fetching {len(tiles)} tiles from Milvus high_res_hun_2018 (~25M rows)...")

    all_rows: List[dict] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch_milvus_tile, *t): t for t in tiles}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Milvus tiles"):
            for r in fut.result():
                vec = r["vector"]
                row: dict = {"lat": r["lat"], "lon": r["lon"]}
                for d, v in enumerate(vec):
                    row[f"v{d}"] = float(v)
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    print(f"  Fetched {len(df):,} embeddings — running spatial join...")

    emb_gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326"
    )
    joined = gpd.sjoin(emb_gdf, gdf, how="left", predicate="within")
    joined = joined.dropna(subset=["Code_18"])
    result = joined.drop(columns=["geometry", "index_right"], errors="ignore")

    cache.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache, index=False)
    print(f"  Cached {len(result):,} joined embeddings → {cache}")
    return result


def build_class_satellite_embeddings(cfg: "PipelineConfig",
                                     cache_dir: Path) -> dict:
    """Build per-class L2-normalized satellite centroids from high_res_hun_2018.

    Returns a dict with two keys:
      "centroids":  {code18: np.ndarray(64,)}
      "augmented":  {code18: np.ndarray(K, 64)}  — K=cfg.n_sat_augments sub-centroids
    """
    import hashlib
    import pandas as pd

    cache_key = hashlib.md5(
        f"sat_centroids|{cfg.n_sat_augments}".encode()
    ).hexdigest()[:12]
    cache_path = cache_dir / f"sat_centroids_{cache_key}.pt"

    if cache_path.exists():
        print(f"[cache] HIT → {cache_path}")
        return torch.load(cache_path, weights_only=False)

    # Load embedding + CORINE data
    old_parquet = Path(cfg.sat_emb_path)
    if old_parquet.exists():
        print(f"  Loading existing parquet: {old_parquet}")
        df = pd.read_parquet(old_parquet)
    else:
        df = _fetch_milvus_with_corine_join(cfg)

    emb_cols = [f"v{i}" for i in range(cfg.sat_emb_dim)]
    missing = [c for c in emb_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing embedding columns: {missing[:3]}...")

    print(f"  Computing centroids for {df['Code_18'].nunique()} classes "
          f"from {len(df):,} embeddings...")

    rng = np.random.RandomState(42)
    centroids: Dict[str, np.ndarray] = {}
    augmented: Dict[str, np.ndarray] = {}

    for code, group in df.groupby("Code_18"):
        vecs = group[emb_cols].values.astype(np.float32)
        c = vecs.mean(axis=0)
        c /= np.linalg.norm(c) + 1e-8
        centroids[str(code)] = c

        K = cfg.n_sat_augments
        n = len(vecs)
        sub_size = max(1, n // 2)
        augs = []
        for _ in range(K):
            idx = rng.choice(n, size=min(n, sub_size), replace=False)
            sub = vecs[idx].mean(axis=0)
            sub /= np.linalg.norm(sub) + 1e-8
            augs.append(sub)
        augmented[str(code)] = np.stack(augs)

    result = {"centroids": centroids, "augmented": augmented}
    torch.save(result, cache_path)
    print(f"  {len(centroids)} centroids built → {cache_path}")
    return result


def build_satellite_pairs(pairs: List[dict], sat_data: dict) -> List[dict]:
    """Enrich training pairs with satellite conditioning embeddings.

    Replaces avg_embedding / sentence_embeddings with AlphaEarth satellite
    centroids, and saves the original Qwen embeddings as qwen_avg_embedding /
    qwen_sentence_embeddings for use by the TextToSatBridge during training.
    """
    centroids = sat_data["centroids"]
    augmented = sat_data["augmented"]

    # Global fallback centroid
    all_c = np.stack(list(centroids.values()))
    global_mean = all_c.mean(axis=0)
    global_mean /= np.linalg.norm(global_mean) + 1e-8

    enriched = []
    n_direct, n_mapped, n_fallback = 0, 0, 0

    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])

        # Resolve desc_key → Code_18
        if desc_key in centroids:
            code = desc_key
            n_direct += 1
        elif _HRL_TO_CORINE.get(desc_key) in centroids:
            code = _HRL_TO_CORINE[desc_key]
            n_mapped += 1
        elif desc_key.startswith("corine_") and desc_key[7:] in centroids:
            code = desc_key[7:]
            n_direct += 1
        else:
            code = None
            n_fallback += 1
            print(f"  [warn] No centroid for {desc_key!r} — using global mean")

        if code:
            sat_avg = torch.from_numpy(centroids[code]).float()
            sat_augs = torch.from_numpy(augmented[code]).float()
        else:
            sat_avg = torch.from_numpy(global_mean).float()
            sat_augs = sat_avg.unsqueeze(0).expand(16, -1).clone()

        new_p = dict(p)
        new_p["qwen_avg_embedding"] = p["avg_embedding"].clone()
        new_p["qwen_sentence_embeddings"] = p["sentence_embeddings"].clone()
        new_p["avg_embedding"] = sat_avg
        new_p["sentence_embeddings"] = sat_augs
        enriched.append(new_p)

    print(f"  Satellite pairs: {n_direct} direct, {n_mapped} HRL-mapped, "
          f"{n_fallback} fallback (global mean)")
    return enriched


def build_satellite_embedding_grid(cfg: "PipelineConfig",
                                   cache_dir: Path) -> np.ndarray:
    """Bin satellite embeddings into a (H, W, sat_emb_dim) spatial grid.

    Each cell holds the mean embedding of all satellite observations that
    fall in that pixel.  Empty cells are filled with the global mean.
    Cached as sat_grid_{hash}.npy.
    """
    import hashlib
    import pandas as pd

    H = cfg.target_resolution
    W = cfg.target_width

    cache_key = hashlib.md5(
        f"sat_grid|{H}x{W}|{cfg.sat_emb_path}|{cfg.sat_emb_dim}".encode()
    ).hexdigest()[:12]
    cache_path = cache_dir / f"sat_grid_{cache_key}.npy"

    if cache_path.exists():
        print(f"[cache] HIT → {cache_path}")
        return np.load(cache_path)

    old_parquet = Path(cfg.sat_emb_path)
    if old_parquet.exists():
        print(f"  Loading parquet for grid: {old_parquet}")
        df = pd.read_parquet(old_parquet)
    else:
        df = _fetch_milvus_with_corine_join(cfg)

    emb_cols = [f"v{i}" for i in range(cfg.sat_emb_dim)]
    lats = df["lat"].values.astype(np.float64)
    lons = df["lon"].values.astype(np.float64)
    embs = df[emb_cols].values.astype(np.float32)

    y_idx = ((cfg.lat_max - lats) / (cfg.lat_max - cfg.lat_min) * H).clip(0, H - 1).astype(np.int32)
    x_idx = ((lons - cfg.lon_min) / (cfg.lon_max - cfg.lon_min) * W).clip(0, W - 1).astype(np.int32)
    flat_idx = (y_idx * W + x_idx).astype(np.int64)

    print(f"  Binning {len(df):,} embeddings into {H}×{W} grid ({cfg.sat_emb_dim} dims)...")
    cnt = np.bincount(flat_idx, minlength=H * W).astype(np.float32)  # (H*W,)
    grid_flat = np.stack([
        np.bincount(flat_idx, weights=embs[:, d].astype(np.float64), minlength=H * W)
        for d in range(cfg.sat_emb_dim)
    ], axis=1).astype(np.float32)  # (H*W, sat_emb_dim)

    filled = cnt > 0
    grid_flat[filled] /= cnt[filled, np.newaxis]
    grid_flat[~filled] = embs.mean(axis=0)  # fill empty cells with global mean

    grid = grid_flat.reshape(H, W, cfg.sat_emb_dim)
    np.save(cache_path, grid)
    print(f"  Grid built: {int(filled.sum()):,}/{H*W:,} cells filled → {cache_path}")
    return grid


def build_density_weighted_sat_pairs(pairs: List[dict],
                                     sat_grid: np.ndarray,
                                     cfg: "PipelineConfig") -> List[dict]:
    """Compute per-class density-weighted satellite centroids.

    Each class centroid = density_map-weighted average of sat_grid embeddings
    at that class's spatial footprint.  Gives distinct centroids for classes
    that share a CORINE Code_18 (e.g. wheat vs sunflower, both Code_18=211)
    because they occupy different geographic regions with different spectral
    signatures.
    """
    H, W, D = sat_grid.shape
    sat_flat = sat_grid.reshape(H * W, D)  # (H*W, 64)
    rng = np.random.RandomState(42)
    K = cfg.n_sat_augments

    enriched = []
    for p in pairs:
        dm = p["density_map"].flatten().astype(np.float64)  # (H*W,)
        w_sum = dm.sum()

        if w_sum < 1e-8:
            centroid = sat_flat.mean(axis=0)
        else:
            centroid = (dm[:, np.newaxis] / w_sum * sat_flat).sum(axis=0)  # (D,)

        norm = np.linalg.norm(centroid) + 1e-8
        centroid = (centroid / norm).astype(np.float32)

        # Sub-centroid augmentations: resample 50% of positive-density pixels
        pos = np.where(dm > 1e-4)[0]
        augs = []
        for _ in range(K):
            if len(pos) < 4:
                augs.append(centroid.copy())
            else:
                sub = rng.choice(pos, size=max(1, len(pos) // 2), replace=False)
                w_sub = dm[sub]
                w_sub = w_sub / (w_sub.sum() + 1e-8)
                c = (w_sub[:, np.newaxis] * sat_flat[sub]).sum(axis=0)
                c = (c / (np.linalg.norm(c) + 1e-8)).astype(np.float32)
                augs.append(c)

        new_p = dict(p)
        # Preserve Qwen embeddings for bridge training
        new_p["qwen_avg_embedding"] = p["avg_embedding"].clone()
        new_p["qwen_sentence_embeddings"] = p["sentence_embeddings"].clone()
        # Replace conditioning with density-weighted centroid
        new_p["avg_embedding"] = torch.from_numpy(centroid).float()
        new_p["sentence_embeddings"] = torch.from_numpy(np.stack(augs)).float()
        enriched.append(new_p)

    print(f"  Density-weighted sat pairs: {len(enriched)} classes")
    return enriched


# ============================================================
# GPU-NATIVE DATA SAMPLER
# ============================================================

class GPUSampler:
    """GPU-native data sampler — all sampling happens on GPU, no DataLoader.

    Preloads density maps and text embeddings as GPU tensors.
    Stores raw text sentences per map for on-the-fly Qwen encoding.
    50% uniform coordinates, 50% importance-sampled from signal regions.
    """

    def __init__(self, pairs, cfg, device="cuda",
                 n_text_tokens=8, text_dropout=0.0):
        self.device = device
        self.n_maps = len(pairs)
        self.n_text_tokens = n_text_tokens
        self.text_dropout = text_dropout
        self.batch_size = cfg.batch_size
        self.samples_per_epoch = cfg.samples_per_epoch

        # All maps are same shape (target_resolution, target_width)
        H, W = pairs[0]["density_map"].shape
        self.H, self.W = H, W

        # Stack all density maps: (n_maps, H*W)
        density_flat = np.stack([p["density_map"].flatten() for p in pairs])
        self.density_maps = torch.from_numpy(
            np.stack([p["density_map"] for p in pairs])
        ).float().to(device)
        self.density_flat = self.density_maps.view(self.n_maps, -1)

        # Importance sampling weights per map: (n_maps, H*W)
        probs = torch.from_numpy(density_flat).float() + 1e-6
        probs = probs / probs.sum(dim=1, keepdim=True)
        self.sample_weights = probs.to(device)

        # Satellite embeddings for spatial model conditioning
        self.avg_embeddings = torch.stack(
            [p["avg_embedding"] for p in pairs]
        ).float().to(device)

        max_sents = max(len(p["sentence_embeddings"]) for p in pairs)
        sat_dim = pairs[0]["sentence_embeddings"].shape[-1]
        self.sent_embs = torch.zeros(self.n_maps, max_sents, sat_dim, device=device)
        self.n_sents = torch.zeros(self.n_maps, dtype=torch.long, device=device)
        for i, p in enumerate(pairs):
            n = len(p["sentence_embeddings"])
            self.sent_embs[i, :n] = p["sentence_embeddings"].to(device)
            self.n_sents[i] = n

        # Qwen embeddings for bridge training (pre-computed, frozen)
        has_qwen = "qwen_avg_embedding" in pairs[0]
        if has_qwen:
            self.qwen_avg_embs = torch.stack(
                [p["qwen_avg_embedding"] for p in pairs]
            ).float().to(device)
            q_max = max(len(p["qwen_sentence_embeddings"]) for p in pairs)
            q_dim = pairs[0]["qwen_sentence_embeddings"].shape[-1]
            self.qwen_sent_embs = torch.zeros(
                self.n_maps, q_max, q_dim, device=device
            )
            self.qwen_n_sents = torch.zeros(
                self.n_maps, dtype=torch.long, device=device
            )
            for i, p in enumerate(pairs):
                n = len(p["qwen_sentence_embeddings"])
                self.qwen_sent_embs[i, :n] = p["qwen_sentence_embeddings"].to(device)
                self.qwen_n_sents[i] = n
        else:
            self.qwen_avg_embs = None
            self.qwen_sent_embs = None
            self.qwen_n_sents = None

        # Per-class inverse-frequency weights for balanced loss
        pixel_masses = self.density_maps.sum(dim=(1, 2))  # (n_maps,)
        inv_freq = 1.0 / torch.sqrt(pixel_masses + 1e-6)
        self.class_weights = (inv_freq / inv_freq.mean()).to(device)  # normalized, mean=1

        print(f"  GPUSampler: {self.n_maps} maps, {H}x{W}, "
              f"{max_sents} sat-sents, "
              f"qwen={'yes' if has_qwen else 'no'}")
        print(f"  Class weights (min={self.class_weights.min():.3f}, "
              f"max={self.class_weights.max():.3f}, mean={self.class_weights.mean():.3f})")

    def __len__(self):
        return self.samples_per_epoch

    @property
    def n_batches(self):
        return self.samples_per_epoch // self.batch_size

    def _sample_coords_and_targets(self, B):
        """Sample coordinates and density targets on GPU."""
        device = self.device
        map_indices = torch.randint(0, self.n_maps, (B,), device=device)
        half = B // 2

        y_uniform = torch.randint(0, self.H, (half,), device=device)
        x_uniform = torch.randint(0, self.W, (half,), device=device)

        imp_maps = map_indices[half:]
        chunk_size = 2048
        flat_idx_parts = []
        for i in range(0, len(imp_maps), chunk_size):
            chunk_w = self.sample_weights[imp_maps[i:i+chunk_size]]
            flat_idx_parts.append(torch.multinomial(chunk_w, 1).squeeze(-1))
        flat_idx = torch.cat(flat_idx_parts)
        y_imp = flat_idx // self.W
        x_imp = flat_idx % self.W

        y_all = torch.cat([y_uniform, y_imp])
        x_all = torch.cat([x_uniform, x_imp])
        flat_coords = y_all * self.W + x_all
        targets = self.density_flat[map_indices, flat_coords]

        lat_norm = (y_all.float() / self.H) * 2 - 1
        lon_norm = (x_all.float() / self.W) * 2 - 1
        coords = torch.stack([lat_norm, lon_norm], dim=-1)

        return coords, targets, map_indices

    def _sample_precomputed_embs(self, map_indices):
        """Sample text embeddings from precomputed cache."""
        B = len(map_indices)
        n_tok = self.n_text_tokens
        map_n_sents = self.n_sents[map_indices]
        rand_idx = torch.rand(B, n_tok, device=self.device)
        sent_idx = (rand_idx * map_n_sents.unsqueeze(1).float()).long()
        sent_idx = sent_idx.clamp(max=self.sent_embs.shape[1] - 1)
        expanded_maps = map_indices.unsqueeze(1).expand(-1, n_tok)
        selected = self.sent_embs[expanded_maps, sent_idx]
        text_embs = selected.mean(dim=1)

        if self.text_dropout > 0:
            mask = torch.rand(B, device=self.device) < self.text_dropout
            text_embs[mask] = 0.0

        return text_embs

    def sample_batch(self):
        """Batch with precomputed satellite embeddings."""
        B = self.batch_size
        coords, targets, map_indices = self._sample_coords_and_targets(B)
        text_embs = self._sample_precomputed_embs(map_indices)
        return coords, text_embs, targets, map_indices

    def _sample_qwen_embs(self, map_indices):
        """Sample Qwen text embeddings for bridge alignment training."""
        if self.qwen_sent_embs is None:
            raise RuntimeError("GPUSampler has no Qwen embeddings (pairs not satellite-enriched)")
        B = len(map_indices)
        n_tok = self.n_text_tokens
        map_n = self.qwen_n_sents[map_indices]
        rand_idx = torch.rand(B, n_tok, device=self.device)
        sent_idx = (rand_idx * map_n.unsqueeze(1).float()).long()
        sent_idx = sent_idx.clamp(max=self.qwen_sent_embs.shape[1] - 1)
        expanded = map_indices.unsqueeze(1).expand(-1, n_tok)
        selected = self.qwen_sent_embs[expanded, sent_idx]
        return selected.mean(dim=1)


# ============================================================
# TRAINING
# ============================================================

def compute_discrimination_loss(model, coords, text_embs, map_indices, n_coords=32):
    """At shared coordinates, different text should produce different outputs.

    Iterates over ALL unique class pairs in the batch (not just one random pair),
    uses shared coordinates, and penalizes predictions being too similar.
    Loss: 1 / (1 + |pred_a - pred_b|)  → 1 when identical, → 0 when different.
    """
    unique_maps = torch.unique(map_indices)
    n_classes = len(unique_maps)
    if n_classes < 2:
        return torch.tensor(0.0, device=coords.device)

    # Precompute indices per class
    class_indices = {}
    for m in unique_maps:
        class_indices[m.item()] = (map_indices == m).nonzero(as_tuple=True)[0]

    total_loss = torch.tensor(0.0, device=coords.device)
    n_pairs_done = 0

    # Cap the number of class pairs to avoid blowup with many classes
    class_list = unique_maps.tolist()
    max_pairs = 40
    if n_classes * (n_classes - 1) // 2 > max_pairs:
        # Random subset of pairs
        import itertools
        all_pairs = list(itertools.combinations(class_list, 2))
        perm = torch.randperm(len(all_pairs))[:max_pairs].tolist()
        pair_list = [all_pairs[i] for i in perm]
    else:
        import itertools
        pair_list = list(itertools.combinations(class_list, 2))

    for map_a, map_b in pair_list:
        idx_a = class_indices[map_a]
        idx_b = class_indices[map_b]
        n = min(n_coords, len(idx_a), len(idx_b))
        if n < 2:
            continue

        # Skip pairs that share a satellite centroid (identical conditioning).
        # For these pairs pred_a == pred_b by construction, so disc loss = 1.0
        # always with zero useful gradient — it just fights the MSE loss.
        cos_sim = F.cosine_similarity(
            text_embs[idx_a[0]].unsqueeze(0),
            text_embs[idx_b[0]].unsqueeze(0),
        ).item()
        if cos_sim > 0.95:
            continue

        # Random subset of coordinates
        sel_a = idx_a[torch.randperm(len(idx_a))[:n]]
        sel_b = idx_b[torch.randperm(len(idx_b))[:n]]

        # Use coordinates from class A, query with both text embeddings
        shared_coords = coords[sel_a]
        pred_a = model(shared_coords, text_embs[sel_a])
        pred_b = model(shared_coords, text_embs[sel_b])

        # 1/(1+|diff|) — penalizes similarity, good gradients everywhere
        diff = (pred_a - pred_b).abs()
        total_loss = total_loss + (1.0 / (1.0 + diff)).mean()
        n_pairs_done += 1

    if n_pairs_done == 0:
        return torch.tensor(0.0, device=coords.device)
    return total_loss / n_pairs_done


def split_pairs_by_class(pairs, val_fraction=0.2, seed=42):
    """Hold out entire classes for validation.

    This tests whether the model generalizes to unseen text→spatial mappings,
    not just unseen sentences for known classes.
    """
    n_val = max(1, round(len(pairs) * val_fraction))
    indices = list(range(len(pairs)))
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    val_set = set(indices[:n_val])

    train = [p for i, p in enumerate(pairs) if i not in val_set]
    val = [p for i, p in enumerate(pairs) if i in val_set]

    print(f"  Train classes: {len(train)}, Val classes: {len(val)}")
    print(f"  Val class names: {[p['class_name'] for p in val]}")
    return train, val


def pretrain_qwen_classifier(text_encoder, raw_texts, pairs, cfg, device="cuda"):
    """Stage 1: Pretrain Qwen LoRA via sentence → class classification.

    Each sentence from the raw texts is treated as one input.
    A temporary classification head maps the Qwen embedding (2560-d) to
    N_classes logits, trained with cross-entropy.

    After pretraining the classification head is discarded — only the
    LoRA weights survive into Stage 2.
    """
    import random

    # Build class mapping: desc_key → class_index
    class_names = []
    class_to_idx = {}
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        if desc_key not in class_to_idx:
            class_to_idx[desc_key] = len(class_names)
            class_names.append(desc_key)
    n_classes = len(class_names)

    # Build flat dataset: list of (sentence, class_index)
    sentence_pool = []
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        cls_idx = class_to_idx[desc_key]
        sents = raw_texts.get(desc_key, [f"{p['class_name']} in Hungary"])
        for s in sents:
            sentence_pool.append((s, cls_idx))

    print(f"  Classification pretraining:")
    print(f"    {n_classes} classes, {len(sentence_pool)} total sentences")
    print(f"    Epochs: {cfg.pretrain_qwen_epochs}, batch_size: {cfg.pretrain_qwen_batch_size}")

    # Temporary classification head (discarded after pretraining)
    qwen_device = next(text_encoder.parameters()).device
    cls_head = nn.Linear(cfg.text_emb_dim, n_classes).to(qwen_device)

    # Optimizer: LoRA params + classification head
    qwen_params = [p for p in text_encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": qwen_params, "lr": cfg.pretrain_qwen_lr},
        {"params": cls_head.parameters(), "lr": cfg.pretrain_qwen_lr * 10},
    ])

    total_steps = cfg.pretrain_qwen_epochs * (len(sentence_pool) // cfg.pretrain_qwen_batch_size + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    tokenizer = text_encoder.tokenizer
    text_encoder.train()

    n_batches_per_epoch = len(sentence_pool) // cfg.pretrain_qwen_batch_size + 1

    for epoch in tqdm(range(cfg.pretrain_qwen_epochs), desc="Stage 1", ncols=100):
        random.shuffle(sentence_pool)
        epoch_loss = 0.0
        n_correct = 0
        n_total = 0
        n_batches = 0

        pbar = tqdm(range(0, len(sentence_pool), cfg.pretrain_qwen_batch_size),
                    desc=f"  Epoch {epoch+1:3d}", leave=False, ncols=120)
        for i in pbar:
            batch = sentence_pool[i:i + cfg.pretrain_qwen_batch_size]
            texts = [s for s, _ in batch]
            labels = torch.tensor([c for _, c in batch], dtype=torch.long, device=qwen_device)

            # Tokenize and encode
            inputs = tokenizer(
                texts, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            input_ids = inputs["input_ids"].to(qwen_device)
            attn_mask = inputs["attention_mask"].to(qwen_device)

            with torch.autocast(qwen_device.type, dtype=torch.bfloat16):
                embs = text_encoder(input_ids, attn_mask)  # (B, 2560)
                logits = cls_head(embs)                     # (B, n_classes)
                loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qwen_params, 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_correct += (logits.argmax(dim=-1) == labels).sum().item()
            n_total += len(labels)
            n_batches += 1

            pbar.set_postfix({
                "loss": f"{epoch_loss/n_batches:.4f}",
                "acc": f"{n_correct/max(1,n_total)*100:.1f}%",
            })

        acc = n_correct / max(1, n_total) * 100
        avg_loss = epoch_loss / max(1, n_batches)

    print(f"  ✅ Qwen preconditioning complete — {n_classes} classes, final acc={acc:.1f}%")

    # Discard classification head, keep LoRA weights in text_encoder
    del cls_head, optimizer, scheduler
    torch.cuda.empty_cache()


def train(model, train_sampler, cfg, device="cuda", sample_pairs=None,
          val_sampler=None, bridge=None):
    """Training loop with GPU-native sampling, AMP, discrimination loss,
    and optional TextToSatBridge for zero-shot text inference.

    Stages:
      Epochs 0 .. bridge_start_epoch-1:
        Spatial model trains on satellite embeddings only.
      Epochs bridge_start_epoch .. end:
        text_mix_ratio fraction of batches use bridge(Qwen_emb) as conditioning.
        Alignment loss penalizes bridge output diverging from satellite centroids.
    """
    import random
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.to(device)
    if bridge is not None:
        bridge.to(device)

    # Resume from checkpoint
    ckpt_path = Path(cfg.output_dir) / "checkpoint_v2.pt"
    start_epoch = 0
    epoch_losses, val_losses = [], []
    disc_losses, align_losses = [], []
    saved_opt, saved_sched = None, None

    if ckpt_path.exists():
        print(f"[resume] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model"])
            if bridge is not None and "bridge" in ckpt:
                bridge.load_state_dict(ckpt["bridge"])
                print(f"  Loaded bridge weights from checkpoint")
            saved_opt = ckpt["optimizer"]
            saved_sched = ckpt["scheduler"]
            start_epoch = ckpt["epoch"] + 1
            epoch_losses = ckpt.get("losses", [])
            val_losses = ckpt.get("val_losses", [])
            disc_losses = ckpt.get("disc_losses", [])
            align_losses = ckpt.get("align_losses", [])
            print(f"  Resuming from epoch {start_epoch}")
        except (RuntimeError, KeyError) as e:
            print(f"  [warn] Checkpoint incompatible (architecture changed) — starting fresh")
            print(f"         {e}")
            start_epoch = 0

    # Optimizer: spatial model + optional bridge
    param_groups = [
        {"params": model.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
    ]
    if bridge is not None:
        n_bridge = sum(p.numel() for p in bridge.parameters())
        print(f"  Bridge trainable params: {n_bridge:,}")
        param_groups.append(
            {"params": bridge.parameters(), "lr": cfg.bridge_lr, "weight_decay": 0.0}
        )

    optimizer = torch.optim.AdamW(param_groups, foreach=False)
    if saved_opt:
        try:
            optimizer.load_state_dict(saved_opt)
        except (ValueError, KeyError):
            print("  [warn] Could not restore optimizer state (param groups changed)")

    n_batches_per_epoch = train_sampler.n_batches
    total_steps = n_batches_per_epoch * cfg.n_epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if saved_sched:
        try:
            scheduler.load_state_dict(saved_sched)
        except (ValueError, KeyError):
            print("  [warn] Could not restore scheduler state")

    viz_dir = Path(cfg.output_dir) / "training_viz_v2"
    if cfg.plot_every > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    bridge_active = bridge is not None and train_sampler.qwen_sent_embs is not None
    print(f"\nTraining: {n_params:,} spatial params, {train_sampler.samples_per_epoch:,} samples/epoch")
    print(f"  {n_batches_per_epoch} batches/epoch × {cfg.batch_size:,} = {n_batches_per_epoch * cfg.batch_size:,} samples")
    print(f"  Discrimination weight: {cfg.discrimination_weight}")
    print(f"  Sat embedding dropout: {cfg.text_dropout}")
    print(f"  Bridge: {'ON (starts epoch ' + str(cfg.bridge_start_epoch) + ')' if bridge_active else 'OFF'}")
    print(f"  AMP: bfloat16")

    model.train()
    if bridge is not None:
        bridge.train()

    for epoch in range(start_epoch, cfg.n_epochs):
        ep_mse, ep_disc, ep_align = 0.0, 0.0, 0.0
        use_bridge_this_epoch = (
            bridge_active and epoch >= cfg.bridge_start_epoch
        )

        pbar = tqdm(range(n_batches_per_epoch), desc=f"Epoch {epoch+1:3d}",
                    leave=False, ncols=120)
        for batch_i in pbar:
            coords, targets, map_indices = train_sampler._sample_coords_and_targets(cfg.batch_size)

            # --- Choose conditioning: satellite centroid or bridge(Qwen) ---
            use_bridge_batch = (
                use_bridge_this_epoch and random.random() < cfg.text_mix_ratio
            )

            if use_bridge_batch:
                qwen_embs = train_sampler._sample_qwen_embs(map_indices)
                cond_embs = bridge(qwen_embs)   # (B, sat_dim) — L2-normalized
            else:
                cond_embs = train_sampler._sample_precomputed_embs(map_indices)

            # Satellite embedding dropout (only on sat path, not bridge path)
            if not use_bridge_batch and cfg.text_dropout > 0:
                mask = torch.rand(len(cond_embs), device=device) < cfg.text_dropout
                cond_embs = cond_embs.clone()
                cond_embs[mask] = 0.0

            # --- Forward with AMP ---
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(coords, cond_embs)
                class_w = train_sampler.class_weights[map_indices]
                signal_w = 1.0 + 4.0 * targets
                mse_loss = (class_w * signal_w * (preds - targets) ** 2).mean()

                disc_loss = compute_discrimination_loss(
                    model, coords, cond_embs, map_indices, n_coords=32
                )

                if epoch < cfg.disc_warmup_epochs:
                    disc_w = 0.0
                elif epoch < cfg.disc_warmup_epochs + cfg.disc_ramp_epochs:
                    disc_w = cfg.discrimination_weight * (
                        (epoch - cfg.disc_warmup_epochs) / cfg.disc_ramp_epochs
                    )
                else:
                    disc_w = cfg.discrimination_weight

                # Bridge alignment loss: cosine distance between bridge output
                # and per-class satellite centroids (computed on unique classes)
                align_loss = torch.tensor(0.0, device=device)
                if use_bridge_this_epoch and bridge is not None:
                    unique_maps = torch.unique(map_indices)
                    q_avg = train_sampler.qwen_avg_embs[unique_maps]
                    s_avg = train_sampler.avg_embeddings[unique_maps]
                    bridge_out = bridge(q_avg)
                    align_loss = (
                        1 - F.cosine_similarity(bridge_out, s_avg, dim=-1)
                    ).mean()

                loss = (mse_loss + disc_w * disc_loss
                        + cfg.bridge_weight * align_loss)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if bridge is not None:
                torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_mse += mse_loss.item()
            ep_disc += disc_loss.item()
            ep_align += align_loss.item()

            pbar.set_postfix({
                "mse": f"{ep_mse/(batch_i+1):.4f}",
                "disc": f"{ep_disc/(batch_i+1):.4f}",
                "align": f"{ep_align/(batch_i+1):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
            })

        avg_mse = ep_mse / n_batches_per_epoch
        avg_disc = ep_disc / n_batches_per_epoch
        avg_align = ep_align / n_batches_per_epoch
        epoch_losses.append(avg_mse)
        disc_losses.append(avg_disc)
        align_losses.append(avg_align)

        # --- Validation (always uses satellite embeddings) ---
        avg_val = float("nan")
        if val_sampler is not None:
            model.eval()
            val_sum = 0.0
            n_val_batches = val_sampler.n_batches
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(n_val_batches):
                    c, t, tgt, mi = val_sampler.sample_batch()
                    p = model(c, t)
                    cw = val_sampler.class_weights[mi]
                    sw = 1.0 + 4.0 * tgt
                    val_sum += (cw * sw * (p - tgt) ** 2).mean().item()
            avg_val = val_sum / max(1, n_val_batches)
            val_losses.append(avg_val)
            model.train()
            if bridge is not None:
                bridge.train()

        lr_now = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_str = f", val={avg_val:.5f}" if not math.isnan(avg_val) else ""
            align_str = f", align={avg_align:.4f}" if use_bridge_this_epoch else ""
            print(f"Epoch {epoch+1:4d}: mse={avg_mse:.5f}, disc={avg_disc:.5f}"
                  f"{align_str}{val_str}, lr={lr_now:.2e}")

        # --- Visualization + checkpoint ---
        if cfg.plot_every > 0 and ((epoch + 1) % cfg.plot_every == 0 or epoch == 0):
            model.eval()
            n_viz = min(4, len(sample_pairs)) if sample_pairs else 0
            n_cols = n_viz + 1
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
            if n_cols == 1:
                axes = [axes]

            axes[0].plot(epoch_losses, linewidth=1.5, label="train MSE")
            if val_losses:
                axes[0].plot(val_losses, linewidth=1.5, linestyle="--", label="val MSE")
            if disc_losses:
                ax2 = axes[0].twinx()
                ax2.plot(disc_losses, linewidth=1.2, color="coral", alpha=0.7, label="disc")
                if any(a > 0 for a in align_losses):
                    ax2.plot(align_losses, linewidth=1.2, color="steelblue",
                             alpha=0.7, label="align")
                ax2.set_ylabel("Disc / Align", fontsize=8)
                ax2.tick_params(axis='y', labelsize=7)
            axes[0].legend(fontsize=8)
            axes[0].set_title("Loss")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)

            if sample_pairs:
                with torch.no_grad():
                    for ax, pair in zip(axes[1:], sample_pairs[:n_viz]):
                        dm = model.render_map(
                            pair["avg_embedding"].to(device),
                            H=cfg.target_resolution, W=cfg.target_width,
                            device=device,
                        )
                        ax.imshow(dm, cmap="YlOrRd", origin="upper",
                                  extent=[cfg.lon_min, cfg.lon_max,
                                          cfg.lat_max, cfg.lat_min])
                        ax.set_title(pair["class_name"], fontsize=9)
                        ax.axis("off")

            plt.suptitle(f"Epoch {epoch+1}", fontsize=11)
            plt.tight_layout()
            plt.savefig(viz_dir / f"epoch_{epoch+1:04d}.png", dpi=120)
            plt.close(fig)

            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            ckpt_data = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "losses": epoch_losses,
                "val_losses": val_losses,
                "disc_losses": disc_losses,
                "align_losses": align_losses,
                "config": vars(cfg),
            }
            if bridge is not None:
                ckpt_data["bridge"] = bridge.state_dict()
            torch.save(ckpt_data, ckpt_path)
            print(f"  [ckpt] Saved → {ckpt_path}")
            model.train()
            if bridge is not None:
                bridge.train()

    return model


# ============================================================
# MAIN
# ============================================================

def _cfg_hash(*parts) -> str:
    blob = "|".join(str(p) for p in parts)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None, help="e.g. cuda:0, cuda:1, cpu")
    parser.add_argument("--output-dir", default=None, help="override cfg.output_dir")
    args = parser.parse_args()

    cfg = PipelineConfig()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Step 1: Load cached distributions ---
    # (reuse your existing v1 cache — density maps haven't changed)
    print("=" * 60)
    print("STEP 1: Load distributions (from v1 cache)")
    print("=" * 60)

    dist_hash = _cfg_hash(
        cfg.r3_dir, cfg.r4_dir, cfg.corine_geojson_path,
        cfg.lat_min, cfg.lat_max, cfg.lon_min, cfg.lon_max,
        cfg.target_resolution, cfg.min_pixels_for_class,
    )
    dist_cache = cache_dir / f"distributions_{dist_hash}.pkl"

    if dist_cache.exists():
        print(f"[cache] HIT → {dist_cache}")
        with open(dist_cache, "rb") as f:
            distributions = pickle.load(f)
        print(f"  {len(distributions)} distribution maps")
    else:
        # If no cache, import and run v1 processing
        from pipeline import process_all_rasters
        distributions = process_all_rasters(cfg)
        with open(dist_cache, "wb") as f:
            pickle.dump(distributions, f)

    # --- Step 2: Load cached pairs ---
    print("\n" + "=" * 60)
    print("STEP 2: Load text-paired training data")
    print("=" * 60)

    # Pairs cache always uses full Qwen dim (2560) — satellite enrichment
    # happens in Step 3 on top of these cached Qwen embeddings.
    pairs_hash = _cfg_hash(
        dist_hash, cfg.text_descriptions_path, cfg.hrl_descriptions_path,
        cfg.qwen_emb_dim,
    )
    pairs_cache = cache_dir / f"pairs_{pairs_hash}.pt"

    if pairs_cache.exists():
        print(f"[cache] HIT → {pairs_cache}")
        pairs = torch.load(pairs_cache, weights_only=False)
        print(f"  {len(pairs)} pairs loaded")
    else:
        from full_flow import build_training_pairs
        from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
        enc = Qwen3EmbeddingAdapter(target_dim=cfg.qwen_emb_dim, freeze_encoder=True)
        enc = enc.to(device).eval()
        pairs = build_training_pairs(
            distributions, cfg.text_descriptions_path, enc,
            extra_descriptions_paths=[cfg.hrl_descriptions_path],
        )
        del enc
        torch.cuda.empty_cache()
        torch.save(pairs, pairs_cache)

    # --- Step 3: Build satellite centroids and enrich pairs ---
    print("\n" + "=" * 60)
    print("STEP 3: Build AlphaEarth satellite conditioning")
    print("=" * 60)

    sat_grid = build_satellite_embedding_grid(cfg, cache_dir)
    pairs = build_density_weighted_sat_pairs(pairs, sat_grid, cfg)

    # --- Step 4: Build GPU samplers ---
    print("\n" + "=" * 60)
    print("STEP 4: Build GPU-native samplers (class-level split)")
    print("=" * 60)

    train_pairs, val_pairs = split_pairs_by_class(
        pairs, val_fraction=cfg.val_class_fraction
    )

    train_sampler = GPUSampler(
        train_pairs, cfg, device=device,
        text_dropout=cfg.text_dropout,
    )
    val_sampler = GPUSampler(
        val_pairs, cfg, device=device,
        text_dropout=0.0,
    )
    val_sampler.samples_per_epoch = cfg.val_samples
    print(f"  Train: {train_sampler.samples_per_epoch:,} samples, {len(train_pairs)} classes")
    print(f"  Val:   {val_sampler.samples_per_epoch:,} samples, {len(val_pairs)} classes")

    # --- Step 5: Build spatial model + TextToSatBridge ---
    print("\n" + "=" * 60)
    print("STAGE 2: Train SpatialBasisFieldV2 + TextToSatBridge")
    print("=" * 60)

    model = SpatialBasisFieldV2(
        text_dim=cfg.text_emb_dim,
        text_proj_dim=cfg.text_proj_dim,
        n_bases=cfg.n_bases,
        n_freqs=cfg.n_fourier_freqs,
        hidden_dim=cfg.hidden_dim,
        coord_hidden=cfg.coord_hidden,
    )

    # Bridge requires Qwen embeddings to be stored in the pairs
    has_qwen = train_sampler.qwen_sent_embs is not None
    bridge = None
    if has_qwen:
        bridge = TextToSatBridge(
            text_dim=cfg.qwen_emb_dim,
            sat_dim=cfg.sat_emb_dim,
            hidden_dim=cfg.bridge_hidden_dim,
        )
        print(f"  TextToSatBridge: {cfg.qwen_emb_dim}→{cfg.bridge_hidden_dim}→{cfg.sat_emb_dim}")
        print(f"  Bridge starts at epoch {cfg.bridge_start_epoch}, "
              f"mix ratio {cfg.text_mix_ratio:.0%}")
    else:
        print("  [warn] No Qwen embeddings found in pairs — bridge disabled")

    model = train(model, train_sampler, cfg, device,
                  sample_pairs=pairs, val_sampler=val_sampler,
                  bridge=bridge)

    # --- Step 6: Save ---
    print("\n" + "=" * 60)
    print("STEP 6: Save final model")
    print("=" * 60)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_data = {
        "model_state_dict": model.state_dict(),
        "config": vars(cfg),
        "class_names": [p["class_name"] for p in pairs],
    }
    if bridge is not None:
        save_data["bridge"] = bridge.state_dict()
    torch.save(save_data, out / "spatial_basis_field_v2.pt")
    print(f"✅ Saved to {out / 'spatial_basis_field_v2.pt'}")


if __name__ == "__main__":
    main()
