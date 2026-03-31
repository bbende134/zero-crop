#!/usr/bin/env python3
"""
Inference for flow_4 — Satellite-Grounded Contrastive Text Encoder
==================================================================

Encodes text queries via Qwen (LoRA) + ProjectionHead into 64-d satellite
embedding space, then computes cosine similarity against the full satellite
grid to produce density maps.

Usage:
    uv run python inference_flow4.py --query "wheat" "sunflower fields"
    uv run python inference_flow4.py --query "danube river" --model Qwen/Qwen3-14B --multi-gpu
    uv run python inference_flow4.py --all-classes
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from flow_4 import SatelliteProjectionHead, LearnableTemperature


def load_flow4(checkpoint_path, device="cpu"):
    """Load flow_4 checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {})
    class_names = ckpt.get("class_names", [])
    sat_centroids = ckpt.get("sat_centroids", None)
    lora_state = ckpt.get("qwen_lora", None)

    # Infer dims from projection head weights
    first_weight = ckpt["proj_head"]["proj.1.weight"]
    text_dim = first_weight.shape[1]  # input to first linear after LayerNorm
    sat_dim = cfg.get("sat_emb_dim", 64)

    proj_head = SatelliteProjectionHead(
        text_dim=text_dim,
        sat_dim=sat_dim,
        hidden_1=cfg.get("proj_hidden_1", 512),
        hidden_2=cfg.get("proj_hidden_2", 128),
    )
    proj_head.load_state_dict(ckpt["proj_head"])
    proj_head = proj_head.to(device).eval()

    has_lora = lora_state is not None and len(lora_state) > 0
    epoch = ckpt.get("epoch", "final")
    print(f"  text_dim: {text_dim}, sat_dim: {sat_dim}, epoch: {epoch}")
    print(f"  Classes: {len(class_names)}")
    print(f"  LoRA weights: {'yes' if has_lora else 'NOT FOUND'}")

    return proj_head, cfg, class_names, sat_centroids, lora_state


def load_sat_grid(path):
    """Load and L2-normalize the satellite embedding grid."""
    sat_grid = np.load(path)
    H, W, D = sat_grid.shape
    flat = sat_grid.reshape(-1, D)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    flat_norm = torch.from_numpy((flat / norms).astype(np.float32))
    print(f"  Sat grid: {sat_grid.shape}, normalized to {flat_norm.shape}")
    return flat_norm, H, W


def infer_query(query, text_encoder, proj_head, sat_grid_flat, H, W, device):
    """Encode query and compute cosine similarity map."""
    with torch.no_grad():
        emb = text_encoder.encode_raw(query)
        projected = proj_head(emb.to(device))  # (1, 64)
        sim = (projected @ sat_grid_flat.T)[0]  # (H*W,)
    return sim.reshape(H, W).cpu().numpy()


def plot_result(sim_map, cfg, title, save_path, gt_density=None):
    """Plot cosine similarity map, optionally with ground truth comparison."""
    n_cols = 2 if gt_density is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    extent = [
        cfg.get("lon_min", 16.113), cfg.get("lon_max", 22.897),
        cfg.get("lat_max", 48.585), cfg.get("lat_min", 45.737),
    ]

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under(alpha=0)

    # Cosine similarity map
    density = sim_map.clip(0, 1)
    vmax = max(0.1, density.max())
    im = axes[0].imshow(density, cmap=cmap, origin="upper", extent=extent,
                        vmin=0.01, vmax=vmax)
    axes[0].set_title(f'"{title}" — cosine similarity', fontsize=11)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(True, alpha=0.2, linestyle="--")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Ground truth
    if gt_density is not None:
        im2 = axes[1].imshow(gt_density, cmap=cmap, origin="upper", extent=extent,
                             vmin=0.01, vmax=1.0)
        axes[1].set_title("Ground truth density", fontsize=11)
        axes[1].set_xlabel("Longitude")
        axes[1].grid(True, alpha=0.2, linestyle="--")
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_grid(sim_maps, titles, cfg, save_path, ncols=5):
    """Plot multiple similarity maps in a grid."""
    n = len(sim_maps)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    extent = [
        cfg.get("lon_min", 16.113), cfg.get("lon_max", 22.897),
        cfg.get("lat_max", 48.585), cfg.get("lat_min", 45.737),
    ]
    cmap = plt.cm.YlOrRd.copy()

    for idx, (sm, title) in enumerate(zip(sim_maps, titles)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        density = sm.clip(0, 1)
        ax.imshow(density, cmap=cmap, origin="upper", extent=extent,
                  vmin=0.0, vmax=max(0.1, density.max()))
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=5)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.suptitle("Flow 4 — Contrastive Similarity Maps", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference for flow_4 — contrastive text-to-satellite",
    )
    parser.add_argument("--checkpoint", type=str,
                        default="training_data_flow4/checkpoint.pt")
    parser.add_argument("--sat-grid", type=str,
                        default="pipeline_cache/sat_grid_1dffdd1c79c3.npy")
    parser.add_argument("--query", nargs="+", type=str, default=None)
    parser.add_argument("--all-classes", action="store_true",
                        help="Render similarity maps using each class name as query")
    parser.add_argument("--output-dir", type=str, default="inference_flow4_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--multi-gpu", action="store_true")
    args = parser.parse_args()

    if not args.query and not args.all_classes:
        parser.print_help()
        print("\nProvide --query or --all-classes")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load checkpoint
    proj_head, cfg, class_names, sat_centroids, lora_state = \
        load_flow4(args.checkpoint, device)

    # Load satellite grid
    print("\nLoading satellite grid...")
    sat_grid_flat, H, W = load_sat_grid(args.sat_grid)
    sat_grid_flat = sat_grid_flat.to(device)

    # Load Qwen encoder with LoRA
    model_id = args.model or cfg.get("qwen_model_id", None)
    has_lora = lora_state is not None and len(lora_state) > 0
    print(f"\nLoading Qwen encoder{f' ({model_id})' if model_id else ''}"
          f"{' + LoRA' if has_lora else ''}...")

    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
    text_encoder = Qwen3EmbeddingAdapter(
        model_id=model_id,
        freeze_encoder=True,
        lora=has_lora,
        multi_gpu=args.multi_gpu,
    )
    if has_lora:
        text_encoder.load_state_dict(lora_state, strict=False)
        print(f"  Restored {len(lora_state)} LoRA weight tensors")
    text_encoder = text_encoder.to(device).eval()

    # Load density maps for ground truth comparison
    pairs_cache = cfg.get("pairs_cache", "pipeline_cache/pairs_855d516e8bee.pt")
    pairs = torch.load(pairs_cache, map_location="cpu", weights_only=False)
    gt_maps = {p.get("desc_key", p["class_name"]): p["density_map"]
               for p in pairs}

    # --- Queries ---
    if args.query:
        sim_maps, titles = [], []
        for text in args.query:
            print(f'\nQuery: "{text}"')
            sim_map = infer_query(text, text_encoder, proj_head,
                                  sat_grid_flat, H, W, device)

            # Print stats
            pos = (sim_map > 0).sum()
            print(f"  sim range: [{sim_map.min():.3f}, {sim_map.max():.3f}], "
                  f"positive cells: {pos}/{sim_map.size}")

            # Find closest class for GT comparison
            gt = None
            text_lower = text.lower()
            for key, dm in gt_maps.items():
                if key.lower() in text_lower or text_lower in key.lower():
                    gt = dm.numpy() if hasattr(dm, 'numpy') else dm
                    break

            safe = text.lower().replace(" ", "_")[:50]
            plot_result(sim_map, cfg, text, out_dir / f"query_{safe}.png", gt_density=gt)
            sim_maps.append(sim_map)
            titles.append(text)

        if len(sim_maps) > 1:
            plot_grid(sim_maps, titles, cfg, out_dir / "query_grid.png")

    # --- All classes ---
    if args.all_classes:
        print(f"\nRendering {len(class_names)} class queries...")
        sim_maps, titles = [], []
        for name in class_names:
            sim_map = infer_query(name, text_encoder, proj_head,
                                  sat_grid_flat, H, W, device)
            gt = None
            for key, dm in gt_maps.items():
                if key.lower() == name.lower() or name.lower() in key.lower():
                    gt = dm.numpy() if hasattr(dm, 'numpy') else dm
                    break

            safe = name.lower().replace(" ", "_")[:50]
            plot_result(sim_map, cfg, name, out_dir / f"class_{safe}.png", gt_density=gt)
            sim_maps.append(sim_map)
            titles.append(name)

        plot_grid(sim_maps, titles, cfg, out_dir / "all_classes_grid.png")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
