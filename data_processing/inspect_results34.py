#!/usr/bin/env python3
"""
Inspect Results-3 and Results-4 — full Hungary mosaic per product.

Usage:
    .venv/bin/python data_processing/inspect_results34.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import rasterio
    from rasterio.merge import merge
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.crs import CRS
    from rasterio.transform import from_bounds
except ImportError:
    print("rasterio not available"); sys.exit(1)

ROOT = Path(__file__).parent.parent
R3   = ROOT / "data_corine" / "Results-3"
R4   = ROOT / "data_corine" / "Results-4"
OUT  = ROOT / "data_corine" / "inspect_results34"
OUT.mkdir(exist_ok=True)

# Hungary WGS84 bounds
LON_MIN, LAT_MIN, LON_MAX, LAT_MAX = 16.113, 45.737, 22.897, 48.585

PRODUCTS = [
    (R3, "CLMS_HRLVLCC_CTY",   "Crop Types 2018",            "categorical"),
    (R3, "CLMS_HRLVLCC_CPMCH", "Main Crop Harvest DOY 2018", "continuous"),
    (R3, "CLMS_HRLVLCC_HER",   "Herbaceous Cover % 2018",    "continuous"),
    (R4, "CLMS_HRLVLCC_CPBSA", "Bare Soil Before 2018",      "binary"),
    (R4, "CLMS_HRLVLCC_CPBSB", "Bare Soil After 2018",       "binary"),
]


def find_tifs(directory: Path, prefix: str) -> list[Path]:
    """Find all extracted tifs matching a product prefix."""
    return sorted(directory.rglob(f"{prefix}*.tif"))


def mosaic_to_wgs84(tif_paths: list[Path], out_h: int = 512) -> np.ndarray:
    """Merge all tiles and reproject to WGS84 over Hungary bounds."""
    wgs84 = CRS.from_epsg(4326)
    aspect = (LON_MAX - LON_MIN) / (LAT_MAX - LAT_MIN)
    out_w = int(out_h * aspect)
    dst_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, out_w, out_h)

    canvas = None
    count = 0

    for tp in tif_paths:
        with rasterio.open(tp) as src:
            src_crs = src.crs or CRS.from_epsg(3035)
            dest = np.zeros((out_h, out_w), dtype=np.float32)
            try:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src_crs,
                    dst_transform=dst_transform,
                    dst_crs=wgs84,
                    resampling=Resampling.nearest,
                    src_nodata=src.nodata,
                    dst_nodata=0,
                )
            except Exception:
                continue

        if dest.max() == 0:
            continue  # tile didn't overlap Hungary

        if canvas is None:
            canvas = dest
        else:
            # Max-merge: keep highest non-zero value per pixel
            canvas = np.where((dest > 0) & (dest > canvas), dest, canvas)
        count += 1

    print(f"  Merged {count}/{len(tif_paths)} overlapping tiles")
    return canvas if canvas is not None else np.zeros((out_h, out_w), dtype=np.float32)


def plot_and_save(data: np.ndarray, name: str, value_type: str, nodata_val: float = 0):
    valid = data[data != nodata_val]
    if valid.size == 0:
        print("  No valid data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

    plot = data.astype(float)
    plot[data == nodata_val] = np.nan

    if value_type == "categorical":
        unique = np.unique(valid).astype(int)
        print(f"  Classes present: {unique.tolist()}")
        cmap = plt.cm.get_cmap("tab20", len(unique))
        display = np.full_like(plot, np.nan)
        for i, v in enumerate(unique):
            display[data == v] = i
        im = ax.imshow(display, cmap=cmap, origin="upper",
                       extent=extent, aspect="auto", interpolation="nearest")
        cbar = fig.colorbar(im, ax=ax, fraction=0.03)
        cbar.set_ticks(range(len(unique)))
        cbar.set_ticklabels([str(v) for v in unique], fontsize=7)

    elif value_type == "binary":
        im = ax.imshow(plot, cmap="YlOrBr", vmin=0, vmax=1, origin="upper",
                       extent=extent, aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.03)
        unique, counts = np.unique(valid.astype(int), return_counts=True)
        print(f"  Values: {dict(zip(unique.tolist(), counts.tolist()))}")

    else:  # continuous
        p2, p98 = np.nanpercentile(plot, [2, 98])
        print(f"  Range: {valid.min():.1f} – {valid.max():.1f}  "
              f"(p2={p2:.1f}, p98={p98:.1f})  mean={valid.mean():.1f}")
        im = ax.imshow(plot, cmap="YlGn", vmin=p2, vmax=p98, origin="upper",
                       extent=extent, aspect="auto", interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.03, label="value")

    ax.set_title(name, fontsize=11)
    ax.set_xlabel("lon"); ax.set_ylabel("lat")

    safe = name.lower().replace(" ", "_").replace("/", "_")
    out_path = OUT / f"{safe}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    for directory, prefix, name, vtype in PRODUCTS:
        print(f"\n{'='*60}\n  {name}")
        tifs = find_tifs(directory, prefix)
        if not tifs:
            print(f"  No tifs found for {prefix} in {directory}")
            continue
        print(f"  Found {len(tifs)} tiles — mosaicing...")
        mosaic = mosaic_to_wgs84(tifs)
        plot_and_save(mosaic, name, vtype)

    # CORINE change GDB
    gdb = R3 / "U2018_CHA1218_V2020_20u1.gdb"
    print(f"\n{'='*60}\n  CORINE Land Cover Change 2012–2018")
    if gdb.exists():
        try:
            import geopandas as gpd
            gdf = gpd.read_file(str(gdb))
            print(f"  Features: {len(gdf)}")
            print(f"  Columns:  {list(gdf.columns)}")
            print(f"  CRS:      {gdf.crs}")
            if "CODE_12" in gdf.columns:
                changes = gdf.groupby(["CODE_12", "CODE_18"]).size().sort_values(ascending=False).head(10)
                print(f"  Top 10 transitions (CODE_12 → CODE_18):\n{changes.to_string()}")
        except Exception as e:
            print(f"  Could not read GDB: {e}")
    else:
        print(f"  GDB not found at {gdb}")


if __name__ == "__main__":
    main()
