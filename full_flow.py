#!/usr/bin/env python3
"""
Spatial Basis Decomposition: Data Pipeline & Training
======================================================

Connects Copernicus raster data + text descriptions → trains a coordinate-based
neural field that maps (lat, lon, text_embedding) → density.

Data sources:
  - HRL Crop Types 2021 (10m, 19 classes)
  - Woody Vegetation Layer 2021 (5m)
  - Small Woody Features 2021 (5m/100m)
  - Cropping Seasons 2021 (10m)
  - WorldCereal 2021 (10m, optional)
  - Your custom text descriptions per class

Architecture: SpatialBasisField
  text_embedding → factor weights (K factors)
  (lat, lon) → Fourier features → K basis maps
  output = Σ factor_k × basis_k(lat, lon)
"""

import hashlib
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import warnings

# ============================================================
# PART 1: CONFIGURATION
# ============================================================

@dataclass
class PipelineConfig:
    """All paths and parameters in one place."""
    
    # --- Data paths (2018 HRL only) ---
    r3_dir: str = "data_corine/Results-3"   # CTY, CPMCH, HER
    r4_dir: str = "data_corine/Results-4"   # CPBSA, CPBSB
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
    target_resolution: int = 256  # Downsample rasters to this height
    # Width auto-calculated from aspect ratio: ~480 for Hungary
    kde_bandwidth: float = 0.01   # For point → density conversion (in degrees)
    min_pixels_for_class: int = 500  # Skip classes with fewer pixels than this
    
    # --- Model ---
    n_bases: int = 24          # Number of spatial basis functions
    n_fourier_freqs: int = 64  # Fourier feature frequencies
    hidden_dim: int = 256
    text_emb_dim: int = 2560   # Qwen3-Embedding-4B hidden dim
    
    # --- Training ---
    n_epochs: int = 400
    lr: float = 3e-4
    batch_size: int = 8192     # Coordinate samples per batch (not images!)
    samples_per_map: int = 50000   # Coordinates sampled per distribution map
    weight_decay: float = 1e-5
    plot_every: int = 20           # Save sample map + loss curve every N epochs (0 = off)
    
    @property
    def target_width(self):
        aspect = (self.lon_max - self.lon_min) / (self.lat_max - self.lat_min)
        return int(self.target_resolution * aspect)


# ============================================================
# PART 2: RASTER → DISTRIBUTION EXTRACTION
# ============================================================

# --- HRL Crop Types 2021 (CLMS_HRLVLCC_CTY) ---
# Codes from CLMS_HRLVLCC_CTY_R10.clr / product documentation.
# desc_key matches keys in class_descriptions.json / corine_wiki_char_count.jsonl.

HRL_CROP_CLASSES = {
    # code: (short_name, description_key)
    # Official labels from CLMS_HRLVLCC_CTY_R10.qml
    1110: ("wheat",                  "wheat"),
    1120: ("barley",                 "barley"),
    1130: ("maize",                  "maize"),           # QML: "Maize" (was incorrectly "cereals")
    1140: ("rice",                   "rice"),
    1150: ("other_cereals",          "other_cereals"),
    1210: ("fresh_vegetables",       "fresh_vegetables"),
    1220: ("dry_pulses",             "dry_pulses"),
    1310: ("potatoes",               "potatoes"),
    1320: ("sugar_beet",             "sugar_beet"),
    1410: ("sunflower",              "sunflower"),
    1420: ("soybeans",               "soybeans"),
    1430: ("rapeseed",               "rapeseed"),
    1440: ("flax_cotton_hemp",       "flax_cotton_hemp"),
    2100: ("grapes",                 "grapes"),           # QML: "Grapes" (was "vineyard")
    2200: ("olives",                 "olives"),
    2310: ("fruits",                 "fruits"),
    2320: ("nuts",                   "nuts"),
    3100: ("unclassified_arable",    "unclassified_arable"),
    3200: ("unclassified_permanent", "unclassified_permanent"),
}

# WVL (CLMS_HRLVLCC_WVL): binary layer — 1 = woody vegetation area
WOODY_VEG_CLASSES = {
    1: ("woody_vegetation", "woody_vegetation"),
}

# CPCSY (CLMS_HRLVLCC_CPCSY): cropping season pattern — labels from CLMS_HRLVLCC_CPCSY_R10.qml
# 0=No annual cropland, 1=1 growing season, 2=2 growing seasons
CROPPING_SEASON_CLASSES = {
    1: ("single_growing_season", "single_growing_season"),
    2: ("double_growing_season", "double_growing_season"),
}

# Results-3 / Results-4 (2018 HRL) — non-categorical layers.
# Each is a single spatial feature extracted as a presence/absence density map.
#
# CPMCH: Main Crop Harvest date — uint16 YYYYDDD (e.g. 17181=Jul 2017, 18365=Jan 2018).
#        QML labels: Apr 2017 … Jan 2018. Presence = value in [17090, 18365].
#        nodata sentinels: 0, 65526, 65527, 65531, 65532, 65533, 65535.
#
# HER:   Grassland layer — binary uint8.
#        QML label: 0 = non-grassland, 1 = permanent and temporary grassland.
#        NOT a herbaceous cover %; it is a binary grassland presence map.
#
# CPBSA: Bare Soil Before Sowing — uint16, values = DOY in 5-day steps (10, 15 … 150).
#        QML labels: "10 days" … "150 days" = day-of-year of detected bare soil.
#        nodata sentinels: 0, 65526-65535.
#
# CPBSB: Bare Soil After Harvest — same encoding as CPBSA.
HRL2018_SINGLE_LAYERS = [
    # (product_prefix, short_name, desc_key, valid_min, valid_max)
    ("CLMS_HRLVLCC_CPMCH", "main_crop_harvest_date", "main_crop_harvest_date", 17090, 18365),
    ("CLMS_HRLVLCC_HER",   "permanent_grassland",    "permanent_grassland",    1,     1),
    ("CLMS_HRLVLCC_CPBSA", "bare_soil_before_sowing","bare_soil_before_sowing",10,    150),
    ("CLMS_HRLVLCC_CPBSB", "bare_soil_after_harvest","bare_soil_after_harvest", 10,   150),
]


def load_raster_as_array(tif_path: str,
                         crop_bounds_wgs84: Optional[Tuple[float, float, float, float]] = None,
                         ) -> Tuple[np.ndarray, dict]:
    """Load a GeoTIFF, reproject to WGS84, and crop to Hungary bounds.

    Always returns data in EPSG:4326 so density maps align with the lat/lon
    grid used for training and plotting.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.windows import from_bounds as window_from_bounds

    wgs84 = CRS.from_epsg(4326)

    with rasterio.open(tif_path) as src:
        src_crs = src.crs or wgs84

        if crop_bounds_wgs84 is not None:
            lon_min, lat_min, lon_max, lat_max = crop_bounds_wgs84
        else:
            # Use full raster extent reprojected to WGS84
            from rasterio.warp import transform_bounds
            lon_min, lat_min, lon_max, lat_max = transform_bounds(src_crs, wgs84, *src.bounds)

        # Calculate output transform in WGS84 at native-ish resolution
        transform_out, width_out, height_out = calculate_default_transform(
            src_crs, wgs84, src.width, src.height,
            left=src.bounds.left, bottom=src.bounds.bottom,
            right=src.bounds.right, top=src.bounds.top,
        )

        # Build output array covering the WGS84 bounds
        from rasterio.transform import from_bounds as transform_from_bounds
        transform_crop = transform_from_bounds(lon_min, lat_min, lon_max, lat_max,
                                               width_out, height_out)
        dest = np.zeros((height_out, width_out), dtype=src.dtypes[0])

        try:
            reproject(
                source=rasterio.band(src, 1),
                destination=dest,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=transform_crop,
                dst_crs=wgs84,
                resampling=Resampling.nearest,
                src_nodata=src.nodata,
                dst_nodata=0,
            )
        except Exception:
            # Tile doesn't overlap — return empty
            return np.zeros((0, 0), dtype=src.dtypes[0]), {'nodata': src.nodata, 'crs': str(src_crs)}

        # Check if anything landed inside our crop area
        if dest.max() == 0 and (src.nodata is None or src.nodata != 0):
            return np.zeros((0, 0), dtype=src.dtypes[0]), {'nodata': src.nodata, 'crs': str(src_crs)}

        meta = {'crs': 'EPSG:4326', 'nodata': src.nodata}

    return dest, meta


def extract_class_mask(raster: np.ndarray, class_value: int, 
                       nodata: Optional[float] = None) -> np.ndarray:
    """Extract binary mask for a single class value."""
    mask = (raster == class_value)
    if nodata is not None:
        mask &= (raster != nodata)
    return mask.astype(np.float32)


def mask_to_density(mask: np.ndarray, target_h: int, target_w: int,
                    sigma: float = 2.0) -> np.ndarray:
    """Convert binary mask to smooth density map at target resolution.
    
    Steps:
    1. Downsample using area averaging (preserves total mass)
    2. Apply Gaussian smoothing for continuous density
    3. Normalize to [0, 1]
    
    This is much faster than KDE and works directly on the raster grid.
    """
    from scipy.ndimage import gaussian_filter, zoom
    
    # Step 1: Area-average downsample
    # This is equivalent to computing the fraction of the class in each output pixel
    h_ratio = target_h / mask.shape[0]
    w_ratio = target_w / mask.shape[1]
    
    # Use zoom with order=1 (bilinear) for smooth downsampling
    density = zoom(mask, (h_ratio, w_ratio), order=1)
    
    # Step 2: Gaussian smooth to create continuous field
    if sigma > 0:
        density = gaussian_filter(density, sigma=sigma)
    
    # Step 3: Normalize
    max_val = density.max()
    if max_val > 0:
        density = density / max_val
    
    return density


def process_all_rasters(cfg: PipelineConfig) -> Dict[str, Dict]:
    """Process all downloaded rasters into (class_key, density_map) pairs.
    
    Returns:
        {
            "wheat": {"density": np.array, "source": "crop_types", "pixel_count": N},
            "broadleaf_forest": {"density": np.array, "source": "woody_veg", ...},
            ...
        }
    """
    distributions = {}
    target_h = cfg.target_resolution
    target_w = cfg.target_width
    
    print(f"Target resolution: {target_w}x{target_h}")
    print(f"Geographic bounds: [{cfg.lon_min}, {cfg.lon_max}] x [{cfg.lat_min}, {cfg.lat_max}]")
    hungary_bounds = (cfg.lon_min, cfg.lat_min, cfg.lon_max, cfg.lat_max)
    
    # --- 1. HRL Crop Types 2018 (Results-3) ---
    # Accumulate canvas per class across ALL tiles before applying pixel threshold,
    # so sparse classes (rice, olives, nuts) aren't dropped due to low per-tile counts.
    r3_cty_tifs = list(Path(cfg.r3_dir).glob("CLMS_HRLVLCC_CTY*/*.tif"))
    if r3_cty_tifs:
        print(f"\n--- HRL Crop Types 2018: {len(r3_cty_tifs)} tiles ---")
        class_canvas: Dict[int, np.ndarray] = {}
        class_pixels: Dict[int, int] = {}
        for tif in r3_cty_tifs:
            raster, meta = load_raster_as_array(str(tif), crop_bounds_wgs84=hungary_bounds)
            if raster.size == 0:
                continue
            unique_vals = set(np.unique(raster).tolist())
            for code, (short_name, desc_key) in HRL_CROP_CLASSES.items():
                if code not in unique_vals:
                    continue
                mask = extract_class_mask(raster, code, meta.get('nodata'))
                n_pixels = int(mask.sum())
                if n_pixels == 0:
                    continue
                density = mask_to_density(mask, target_h, target_w)
                class_canvas[code] = np.maximum(class_canvas[code], density) if code in class_canvas else density
                class_pixels[code] = class_pixels.get(code, 0) + n_pixels

        for code, (short_name, desc_key) in HRL_CROP_CLASSES.items():
            if code not in class_canvas:
                continue
            total_px = class_pixels[code]
            if total_px < cfg.min_pixels_for_class:
                print(f"  Skipping {short_name}: only {total_px} total pixels")
                continue
            distributions[short_name] = {
                "density": class_canvas[code],
                "source": "crop_types_2018",
                "pixel_count": total_px,
                "desc_key": desc_key,
            }
            print(f"  {short_name}: {total_px:,} pixels")

    # --- 2. Results-3/4 single-concept layers (CPMCH, HER, CPBSA, CPBSB) ---
    _r34_dirs = {
        "CLMS_HRLVLCC_CPMCH": cfg.r3_dir,
        "CLMS_HRLVLCC_HER":   cfg.r3_dir,
        "CLMS_HRLVLCC_CPBSA": cfg.r4_dir,
        "CLMS_HRLVLCC_CPBSB": cfg.r4_dir,
    }
    for prefix, short_name, desc_key, valid_min, valid_max in HRL2018_SINGLE_LAYERS:
        tifs = list(Path(_r34_dirs[prefix]).glob(f"{prefix}*/*.tif"))
        if not tifs:
            continue
        print(f"\n--- {short_name} ({prefix}): {len(tifs)} tiles ---")
        canvas = None
        total_pixels = 0
        for tif in tifs:
            raster, meta = load_raster_as_array(str(tif), crop_bounds_wgs84=hungary_bounds)
            if raster.size == 0:
                continue
            # Build presence mask: pixel in [valid_min, valid_max] and != nodata sentinel
            mask = ((raster >= valid_min) & (raster <= valid_max)).astype(np.float32)
            # Treat 65533 (0xFFFD) as uint16 nodata sentinel even when nodata attr is None
            mask[raster == 65533] = 0
            n_pixels = int(mask.sum())
            if n_pixels == 0:
                continue
            density = mask_to_density(mask, target_h, target_w)
            canvas = np.maximum(canvas, density) if canvas is not None else density
            total_pixels += n_pixels
        if canvas is not None and total_pixels >= cfg.min_pixels_for_class:
            distributions[short_name] = {
                "density": canvas,
                "source": "hrl_2018",
                "pixel_count": total_pixels,
                "desc_key": desc_key,
            }
            print(f"  {short_name}: {total_pixels:,} pixels, "
                  f"density range [{canvas.min():.3f}, {canvas.max():.3f}]")

    # --- 7. CORINE Land Cover (vector) ---
    corine_dists = process_corine_geojson(cfg)
    distributions.update(corine_dists)

    print(f"\n{'='*60}")
    print(f"Total distribution maps: {len(distributions)}")
    for name, info in distributions.items():
        print(f"  {name:25s}: {info['pixel_count']:>10,} pixels, source={info['source']}")

    return distributions


def process_corine_geojson(cfg: PipelineConfig) -> Dict[str, Dict]:
    """Rasterize CORINE Land Cover GeoJSON by Code_18 into density maps.

    The GeoJSON is in EPSG:3035 (no explicit CRS tag — standard for CORINE).
    Each unique Code_18 (3-digit) becomes one distribution map, keyed by that
    code string so it matches corine_wiki_char_count.jsonl directly.
    """
    import geopandas as gpd
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import box

    p = Path(cfg.corine_geojson_path)
    if not p.exists():
        print(f"CORINE GeoJSON not found at {p}, skipping.")
        return {}

    print(f"\n--- CORINE Land Cover: {p.name} ---")
    gdf = gpd.read_file(str(p))
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")  # GeoJSON coords are WGS84 lon/lat

    # Bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max)
    x_min, y_min = cfg.lon_min, cfg.lat_min
    x_max, y_max = cfg.lon_max, cfg.lat_max

    # Rasterize at 2× target resolution then downsample for accuracy
    target_h = cfg.target_resolution
    target_w = cfg.target_width
    raster_h, raster_w = target_h * 2, target_w * 2
    # from_bounds expects (left, bottom, right, top)
    transform = from_bounds(x_min, y_min, x_max, y_max, raster_w, raster_h)

    # Clip to Hungary bounds first
    hungary_box = box(x_min, y_min, x_max, y_max)
    gdf = gdf[gdf.geometry.intersects(hungary_box)].copy()
    print(f"  Features within Hungary bounds: {len(gdf)}")

    distributions: Dict[str, Dict] = {}

    for code, group in gdf.groupby("Code_18"):
        code_str = str(code)
        shapes = [(geom, 1) for geom in group.geometry if geom is not None and not geom.is_empty]
        if not shapes:
            continue

        burned = rio_rasterize(
            shapes,
            out_shape=(raster_h, raster_w),
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(np.float32)

        n_pixels = int(burned.sum())
        if n_pixels < cfg.min_pixels_for_class:
            print(f"  Skipping CORINE {code_str}: only {n_pixels} pixels")
            continue

        density = mask_to_density(burned, target_h, target_w)
        distributions[f"corine_{code_str}"] = {
            "density": density,
            "source": "corine",
            "pixel_count": n_pixels,
            "desc_key": code_str,  # matches corine_wiki_char_count.jsonl keys
        }
        print(f"  CORINE {code_str}: {n_pixels:>8,} pixels, "
              f"density range [{density.min():.3f}, {density.max():.3f}]")

    print(f"  Total CORINE classes: {len(distributions)}")
    return distributions


# ============================================================
# PART 3: TEXT DESCRIPTION → EMBEDDING PAIRING
# ============================================================

# 
def build_training_pairs(
    distributions: Dict[str, Dict],
    descriptions_path: str,
    text_encoder,  # Your Qwen3EmbeddingAdapter
    extra_descriptions_paths: Optional[List[str]] = None,
    device: str = "cuda",
) -> List[Dict]:
    """Pair each distribution map with its text embeddings.
    
    Returns list of:
    {
        "class_name": str,
        "density_map": np.ndarray (H, W),  # [0, 1]
        "sentence_embeddings": torch.Tensor (N_sentences, emb_dim),
        "avg_embedding": torch.Tensor (emb_dim,),
    }
    """
    # Load descriptions — supports both JSON and JSONL formats.
    # JSONL (corine_wiki_char_count.jsonl): each line has {desc_key, wiki_texts, ...}
    # JSON (class_descriptions.json): {desc_key: {"sentences": [...]}}
    descriptions: Dict[str, List[str]] = {}  # desc_key → list of text chunks

    def _load_into(path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Description file not found: {path}")

        if p.suffix == ".jsonl":
            with open(p, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    key = rec.get("code", "")
                    texts = rec.get("wiki_texts", {})
                    chunks: List[str] = []
                    for article in texts.values():
                        sentences_raw = [s.strip() for s in article.replace("\n", " ").split(". ") if len(s.strip()) > 30]
                        chunks.extend(sentences_raw)
                    if chunks:
                        descriptions[key] = chunks
        else:
            with open(p, encoding="utf-8") as f:
                raw = json.load(f)
            for key, val in raw.items():
                if isinstance(val, dict) and "sentences" in val:
                    descriptions[key] = val["sentences"]
                elif isinstance(val, list):
                    descriptions[key] = val

    _load_into(descriptions_path)
    for extra in (extra_descriptions_paths or []):
        _load_into(extra)

    if not descriptions:
        raise RuntimeError(
            f"No descriptions loaded from '{descriptions_path}' "
            f"(and extras: {extra_descriptions_paths}). "
            "Run the wiki search scripts first."
        )

    # Validate every distribution has a description before encoding anything.
    missing = [
        (name, info["desc_key"])
        for name, info in distributions.items()
        if info["desc_key"] not in descriptions
    ]
    if missing:
        lines = "\n".join(f"  {name!r} → desc_key={key!r}" for name, key in missing)
        raise RuntimeError(
            f"{len(missing)} distribution(s) have no description entry:\n{lines}\n"
            "Add them to the JSONL files and re-run the wiki search scripts."
        )

    pairs = []

    for class_name, info in distributions.items():
        desc_key = info["desc_key"]
        sentences = descriptions[desc_key]
        
        # Encode all sentences
        with torch.no_grad():
            embs = []
            for sent in sentences:
                emb = text_encoder.encode_raw(sent, normalize=False)  # [1, dim]
                embs.append(emb.squeeze(0).cpu())
            
            sent_tensor = torch.stack(embs)      # [N_sentences, dim]
            avg_emb = sent_tensor.mean(dim=0)     # [dim]
        
        pairs.append({
            "class_name": class_name,
            "desc_key": desc_key,
            "density_map": info["density"],
            "sentence_embeddings": sent_tensor,
            "avg_embedding": avg_emb,
            "source": info["source"],
        })
        
        print(f"  {class_name:25s}: {len(sentences)} sentences, "
              f"emb_norm={avg_emb.norm():.2f}")
    
    return pairs


# ============================================================
# PART 4: COORDINATE-LEVEL DATASET
# ============================================================

class CoordinateDensityDataset(torch.utils.data.Dataset):
    """Samples (lat, lon, text_emb, density) coordinate-level training pairs.
    
    Instead of treating each map as one sample, we sample individual coordinates
    from all maps. This converts 25 maps into millions of training points.
    
    Sampling strategy:
    - 50% uniform random coordinates (learn background = 0)
    - 50% importance-sampled from high-density regions (learn signal)
    """
    
    def __init__(self, training_pairs: List[Dict], cfg: PipelineConfig,
                 n_text_tokens: int = 8, samples_per_epoch: int = 500_000):
        self.pairs = training_pairs
        self.cfg = cfg
        self.n_text_tokens = n_text_tokens
        self.samples_per_epoch = samples_per_epoch

        # Precompute CDF for importance sampling per map
        self._importance_cdfs = []
        for pair in self.pairs:
            density = pair["density_map"]
            flat = density.flatten()
            # Add small epsilon to allow sampling everywhere
            probs = flat + 1e-6
            probs /= probs.sum()
            cdf = np.cumsum(probs)
            self._importance_cdfs.append(cdf)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Pick a random distribution map
        map_idx = np.random.randint(len(self.pairs))
        pair = self.pairs[map_idx]
        density_map = pair["density_map"]  # (H, W) in [0, 1]
        H, W = density_map.shape

        # Sample coordinate: 50% uniform, 50% importance-sampled
        if np.random.random() < 0.5:
            # Uniform random
            y_pixel = np.random.randint(H)
            x_pixel = np.random.randint(W)
        else:
            # Importance sample from density
            cdf = self._importance_cdfs[map_idx]
            flat_idx = np.searchsorted(cdf, np.random.random())
            flat_idx = min(flat_idx, H * W - 1)
            y_pixel = flat_idx // W
            x_pixel = flat_idx % W

        # Convert pixel to normalized coordinates [-1, 1]
        lat_norm = (y_pixel / H) * 2 - 1  # -1 = south, +1 = north
        lon_norm = (x_pixel / W) * 2 - 1  # -1 = west, +1 = east

        # Get density at this coordinate
        density_value = density_map[y_pixel, x_pixel]

        # Sample text tokens (random subset of sentence embeddings)
        sent_embs = pair["sentence_embeddings"]  # [N_sents, dim]
        n_sents = len(sent_embs)
        if n_sents >= self.n_text_tokens:
            indices = np.random.choice(n_sents, self.n_text_tokens, replace=False)
        else:
            indices = np.array([i % n_sents for i in range(self.n_text_tokens)])
            np.random.shuffle(indices)

        text_emb = sent_embs[indices].mean(dim=0)  # Average to single vector

        return (
            torch.tensor([lat_norm, lon_norm], dtype=torch.float32),
            text_emb.float(),
            torch.tensor(density_value, dtype=torch.float32),
            torch.tensor(map_idx, dtype=torch.long),
        )


# ============================================================
# PART 5: SPATIAL BASIS FIELD MODEL
# ============================================================

class FourierFeatures(nn.Module):
    """Fourier positional encoding for continuous coordinate inputs.
    
    Maps (lat, lon) → high-dimensional feature vector that lets the MLP
    learn high-frequency spatial patterns (rivers, mountain ridges, etc.)
    """
    
    def __init__(self, n_input: int = 2, n_freqs: int = 64, 
                 sigma: float = 10.0, learnable: bool = False):
        super().__init__()
        self.n_freqs = n_freqs
        
        # Random Fourier features (Gaussian kernel approximation)
        B = torch.randn(n_input, n_freqs) * sigma
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    @property
    def output_dim(self):
        return 2 + 2 * self.n_freqs  # raw coords + sin/cos
    
    def forward(self, coords):
        """coords: (..., 2) normalized lat/lon in [-1, 1]"""
        proj = coords @ self.B  # (..., n_freqs)
        return torch.cat([
            coords,
            torch.sin(2 * np.pi * proj),
            torch.cos(2 * np.pi * proj),
        ], dim=-1)

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

class SpatialBasisField(nn.Module):
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

        # --- Per-basis heads ---
        self.basis_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(coord_hidden, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            ) for _ in range(n_bases)
        ])

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

        # Evaluate each basis head
        basis_out = torch.cat([head(h) for head in self.basis_heads], dim=-1)  # (B, n_bases)

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

        maps = []
        for head in self.basis_heads:
            b = head(h).squeeze(-1).view(H, W).cpu().numpy()
            maps.append(b)
        return maps

# ============================================================
# PART 6: TRAINING LOOP
# ============================================================

def train(model, dataset, cfg: PipelineConfig, device="cuda", sample_pairs=None, val_dataset=None):
    """Train the SpatialBasisField model."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

    # Move to device first — optimizer must be created after, so foreach AdamW
    # sees all parameters on the correct device from the start.
    model.to(device)

    ckpt_path = Path(cfg.output_dir) / "checkpoint.pt"
    start_epoch = 0
    epoch_losses: list[float] = []
    val_losses:   list[float] = []
    saved_opt_state = None
    saved_sched_state = None
    if ckpt_path.exists():
        print(f"[train] Resuming from checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        saved_opt_state = ckpt["optimizer"]
        saved_sched_state = ckpt["scheduler"]
        start_epoch = ckpt["epoch"] + 1
        epoch_losses = ckpt.get("losses", [])
        val_losses   = ckpt.get("val_losses", [])
        print(f"  Resuming from epoch {start_epoch}/{cfg.n_epochs}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    if saved_opt_state is not None:
        optimizer.load_state_dict(saved_opt_state)

    # Cosine decay with warmup
    total_steps = len(loader) * cfg.n_epochs
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    if saved_sched_state is not None:
        scheduler.load_state_dict(saved_sched_state)

    viz_dir = Path(cfg.output_dir) / "training_viz"
    if cfg.plot_every > 0:
        viz_dir.mkdir(parents=True, exist_ok=True)

    model.train()

    print(f"\nTraining: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Dataset: {len(dataset)} samples/epoch, batch_size={cfg.batch_size}")
    print(f"Epochs: {cfg.n_epochs}")

    for epoch in range(start_epoch, cfg.n_epochs):
        epoch_loss = 0
        n_batches = 0

        for coords, text_embs, targets, map_indices in loader:
            coords = coords.to(device)
            text_embs = text_embs.to(device)
            targets = targets.to(device)

            # Forward
            preds = model(coords, text_embs)

            # Loss: MSE + signal-aware weighting
            weights = 1.0 + 4.0 * targets  # Background=1x, signal up to 5x
            loss = (weights * (preds - targets) ** 2).mean()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        lr = optimizer.param_groups[0]['lr']
        epoch_losses.append(avg_loss)

        # Validation
        avg_val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for coords, text_embs, targets, _ in val_loader:
                    coords    = coords.to(device)
                    text_embs = text_embs.to(device)
                    targets   = targets.to(device)
                    preds     = model(coords, text_embs)
                    weights   = 1.0 + 4.0 * targets
                    val_loss_sum += (weights * (preds - targets) ** 2).mean().item()
                    val_batches += 1
            avg_val_loss = val_loss_sum / val_batches
            val_losses.append(avg_val_loss)
            model.train()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            val_str = f", val={avg_val_loss:.6f}" if val_loader is not None else ""
            print(f"Epoch {epoch+1:4d}/{cfg.n_epochs}: loss={avg_loss:.6f}{val_str}, lr={lr:.2e}")

        if cfg.plot_every > 0 and ((epoch + 1) % cfg.plot_every == 0 or epoch == 0):
            model.eval()
            n_viz = min(4, len(sample_pairs)) if sample_pairs else 0
            n_cols = n_viz + 1
            fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
            if n_cols == 1:
                axes = [axes]

            axes[0].plot(epoch_losses, linewidth=1.5, label="train")
            if val_losses:
                axes[0].plot(val_losses, linewidth=1.5, linestyle="--", label="val")
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
                            H=cfg.target_resolution, W=cfg.target_width, device=device,
                        )
                        ax.imshow(dm, cmap="YlOrRd", origin="upper",
                                  extent=[cfg.lon_min, cfg.lon_max, cfg.lat_max, cfg.lat_min])
                        ax.set_title(pair["class_name"], fontsize=9)
                        ax.axis("off")

            plt.suptitle(f"Epoch {epoch + 1}", fontsize=11)
            plt.tight_layout()
            plt.savefig(viz_dir / f"epoch_{epoch+1:04d}.png", dpi=120)
            plt.close(fig)

            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "losses": epoch_losses,
                "val_losses": val_losses,
            }, ckpt_path)
            print(f"  [ckpt] Saved → {ckpt_path}")
            model.train()

    return model


# ============================================================
# PART 7: PUTTING IT ALL TOGETHER
# ============================================================

def _cfg_hash(*parts) -> str:
    """12-char MD5 of config fields — used as cache key."""
    blob = "|".join(str(p) for p in parts)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def split_pairs_train_val(pairs: List[Dict], val_fraction: float = 0.25) -> Tuple[List[Dict], List[Dict]]:
    """Split sentence embeddings per class into train / val sets.

    Density maps are shared; only the sentence_embeddings (and derived
    avg_embedding) differ between the two splits.  The split is deterministic:
    first (1 - val_fraction) sentences → train, last val_fraction → val.
    """
    train_pairs, val_pairs = [], []
    for pair in pairs:
        sent_embs = pair["sentence_embeddings"]  # (N, dim)
        n = len(sent_embs)
        n_val = max(1, round(n * val_fraction))
        n_train = n - n_val
        train_embs = sent_embs[:n_train]
        val_embs   = sent_embs[n_train:]
        train_pairs.append({**pair,
            "sentence_embeddings": train_embs,
            "avg_embedding": train_embs.mean(0),
        })
        val_pairs.append({**pair,
            "sentence_embeddings": val_embs,
            "avg_embedding": val_embs.mean(0),
        })
    return train_pairs, val_pairs


def main():
    """Complete pipeline: rasters → distributions → training → model."""
    
    cfg = PipelineConfig()
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Step 1: Process rasters into density maps ─────────────────────────────
    print("=" * 60)
    print("STEP 1: Extract spatial distributions from rasters")
    print("=" * 60)

    dist_hash = _cfg_hash(
        cfg.r3_dir, cfg.r4_dir, cfg.corine_geojson_path,
        cfg.lat_min, cfg.lat_max, cfg.lon_min, cfg.lon_max,
        cfg.target_resolution, cfg.min_pixels_for_class,
    )
    dist_cache = cache_dir / f"distributions_{dist_hash}.pkl"

    if dist_cache.exists():
        print(f"[cache] HIT  → {dist_cache}")
        with open(dist_cache, "rb") as f:
            distributions = pickle.load(f)
        print(f"  Loaded {len(distributions)} distribution maps")
    else:
        print(f"[cache] MISS → computing and saving to {dist_cache}")
        distributions = process_all_rasters(cfg)
        with open(dist_cache, "wb") as f:
            pickle.dump(distributions, f)

    # ── Step 2: Build text-paired training data ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Pair distributions with text embeddings")
    print("=" * 60)

    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
    pairs_hash = _cfg_hash(
        dist_hash,
        cfg.text_descriptions_path, cfg.hrl_descriptions_path,
        cfg.text_emb_dim, "Qwen/Qwen3.5-4B",
    )
    pairs_cache = cache_dir / f"pairs_{pairs_hash}.pt"

    if pairs_cache.exists():
        print(f"[cache] HIT  → {pairs_cache}")
        pairs = torch.load(pairs_cache, weights_only=False)
        print(f"  Loaded {len(pairs)} pairs")
    else:
        print(f"[cache] MISS → encoding with Qwen, saving to {pairs_cache}")
        text_encoder = Qwen3EmbeddingAdapter(
            target_dim=cfg.text_emb_dim,
            freeze_encoder=True,
        ).to(device).eval()

        pairs = build_training_pairs(
            distributions,
            cfg.text_descriptions_path,
            text_encoder,
            extra_descriptions_paths=[cfg.hrl_descriptions_path],
        )

        del text_encoder
        torch.cuda.empty_cache()
        torch.save(pairs, pairs_cache)
    
    # Step 3: Create coordinate-level dataset
    print("\n" + "=" * 60)
    print("STEP 3: Build coordinate-level training dataset")
    print("=" * 60)
    train_pairs, val_pairs = split_pairs_train_val(pairs, val_fraction=0.25)
    print(f"  Train sentences: {sum(len(p['sentence_embeddings']) for p in train_pairs)}, "
          f"Val sentences: {sum(len(p['sentence_embeddings']) for p in val_pairs)}")

    dataset     = CoordinateDensityDataset(train_pairs, cfg)
    val_dataset = CoordinateDensityDataset(val_pairs, cfg, samples_per_epoch=50_000)
    print(f"Train dataset: {len(dataset)} samples/epoch, "
          f"Val dataset: {len(val_dataset)} samples/epoch")
    
    # Step 4: Build and train model
    print("\n" + "=" * 60)
    print("STEP 4: Train SpatialBasisField")
    print("=" * 60)
    model = SpatialBasisField(
        text_dim=cfg.text_emb_dim,
        n_bases=cfg.n_bases,
        n_freqs=cfg.n_fourier_freqs,
        hidden_dim=cfg.hidden_dim,
    )
    
    model = train(model, dataset, cfg, device, sample_pairs=pairs, val_dataset=val_dataset)

    # Step 5: Save model and generate sample outputs
    print("\n" + "=" * 60)
    print("STEP 5: Save model and generate samples")
    print("=" * 60)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(cfg),
        "class_names": [p["class_name"] for p in pairs],
    }, output_dir / "spatial_basis_field.pt")
    
    # Generate sample maps
    import matplotlib.pyplot as plt
    
    n_classes = len(pairs)
    fig, axes = plt.subplots(2, (n_classes + 1) // 2, figsize=(4 * ((n_classes + 1) // 2), 8))
    axes = axes.flatten()
    
    for i, pair in enumerate(pairs):
        if i >= len(axes):
            break
        density_map = model.render_map(
            pair["avg_embedding"].to(device),
            H=cfg.target_resolution, W=cfg.target_width, device=device
        )
        axes[i].imshow(density_map, cmap='YlOrRd', origin='upper',
                       extent=[cfg.lon_min, cfg.lon_max, cfg.lat_max, cfg.lat_min])
        axes[i].set_title(pair["class_name"])
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_outputs.png", dpi=150)
    plt.close()
    
    # Visualize basis maps
    basis_maps = model.get_basis_maps(H=cfg.target_resolution, W=cfg.target_width, device=device)
    n_bases = len(basis_maps)
    cols = 6
    rows = (n_bases + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()
    for i, bmap in enumerate(basis_maps):
        axes[i].imshow(bmap, cmap='viridis', origin='upper')
        axes[i].set_title(f"Basis {i}")
        axes[i].axis('off')
    for i in range(n_bases, len(axes)):
        axes[i].axis('off')
    plt.suptitle("Learned Spatial Basis Functions")
    plt.tight_layout()
    plt.savefig(output_dir / "basis_maps.png", dpi=150)
    plt.close()
    
    print(f"\n✅ Model saved to {output_dir / 'spatial_basis_field.pt'}")
    print(f"✅ Sample outputs: {output_dir / 'sample_outputs.png'}")
    print(f"✅ Basis maps: {output_dir / 'basis_maps.png'}")


if __name__ == "__main__":
    main()