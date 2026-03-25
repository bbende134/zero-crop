#!/usr/bin/env python3
"""
Plot all spatial distribution maps produced by full_flow.process_all_rasters().

Usage:
    .venv/bin/python data_processing/plot_distributions.py
    .venv/bin/python data_processing/plot_distributions.py --source corine
    .venv/bin/python data_processing/plot_distributions.py --source hrl
    .venv/bin/python data_processing/plot_distributions.py --out my_plot.png
"""

import sys
import argparse
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from full_flow import PipelineConfig, process_all_rasters


def plot_distributions(distributions: dict, cfg: PipelineConfig,
                       source_filter: str = None, out_path: str = None,
                       label: str = None):
    items = list(distributions.items())
    if source_filter:
        items = [(k, v) for k, v in items if v["source"] == source_filter]

    if not items:
        print(f"No distributions to plot (filter='{source_filter}').")
        return

    label = label or source_filter or "all"

    n = len(items)
    cols = min(6, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
    axes = np.array(axes).flatten()

    extent = [cfg.lon_min, cfg.lon_max, cfg.lat_min, cfg.lat_max]

    for i, (name, info) in enumerate(items):
        ax = axes[i]
        density = info["density"]
        im = ax.imshow(
            density,
            origin="upper",
            extent=extent,
            cmap="YlOrRd",
            vmin=0, vmax=1,
            aspect="auto",
        )
        ax.set_title(name, fontsize=7, pad=3)
        ax.set_xlabel("lon", fontsize=6)
        ax.set_ylabel("lat", fontsize=6)
        ax.tick_params(labelsize=5)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Spatial distribution maps — {label} ({n} classes)", fontsize=11)
    plt.tight_layout()

    if out_path is None:
        out_path = f"data_corine/distribution_maps_{label}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["corine", "hrl", "crop_types",
                                              "woody_veg", "small_woody",
                                              "cropping_seasons", "hrl_2018"],
                        default=None, help="Filter by data source")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    cfg = PipelineConfig()
    print("Processing rasters and CORINE vectors...")
    distributions = process_all_rasters(cfg)

    # Plot each source separately for readability
    sources = {info["source"] for info in distributions.values()}

    if args.source == "hrl":
        # Group all HRL raster sources together
        hrl_items = {k: v for k, v in distributions.items() if v["source"] != "corine"}
        plot_distributions(hrl_items, cfg, source_filter=None, label="hrl",
                           out_path=args.out or "data_corine/distribution_maps_hrl.png")
    elif args.source:
        plot_distributions(distributions, cfg, source_filter=args.source, out_path=args.out)
    else:
        for src in sorted(sources):
            plot_distributions(distributions, cfg, source_filter=src)
        plot_distributions(distributions, cfg, source_filter=None,
                           out_path=args.out or "data_corine/distribution_maps_all.png")


if __name__ == "__main__":
    main()
