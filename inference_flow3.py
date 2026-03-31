#!/usr/bin/env python3
"""
Inference for flow_3 — Classify, Lookup, Merge
===============================================

Loads a trained flow_3 checkpoint and renders density maps for text queries.
The classifier predicts which classes match, then merges their pre-computed
density maps via clamped weighted sum.

Usage:
    python inference_flow3.py --query "wheat"
    python inference_flow3.py --query "wheat" "maize" "sunflower fields"
    python inference_flow3.py --all-classes
    python inference_flow3.py --query "wheat and maize" --checkpoint training_data_flow3/checkpoint.pt
    python inference_flow3.py --query "wheat" --model Qwen/Qwen3.5-27B --multi-gpu
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from flow_3 import TextClassifier, merge_maps, Flow3Config


def load_flow3(checkpoint_path: str, device: str = "cpu"):
    """Load flow_3 checkpoint. Returns classifier, class_names, density_maps, config, lora_state."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg_dict = ckpt.get("config", {})
    class_names = ckpt["class_names"]
    n_classes = len(class_names)

    # density_maps may be in the final save or need loading from pairs cache
    density_maps = ckpt.get("density_maps", None)
    if density_maps is None:
        pairs_cache = cfg_dict.get("pairs_cache", "pipeline_cache/pairs_855d516e8bee.pt")
        print(f"  Loading density maps from {pairs_cache}")
        pairs = torch.load(pairs_cache, map_location="cpu", weights_only=False)
        density_maps = torch.stack([
            torch.as_tensor(p["density_map"], dtype=torch.float32) for p in pairs
        ])

    # Infer qwen_emb_dim from classifier weights
    qwen_dim = ckpt["classifier"]["head.1.weight"].shape[1]

    model = TextClassifier(n_classes=n_classes, qwen_dim=qwen_dim)
    model.load_state_dict(ckpt["classifier"])
    model = model.to(device).eval()

    lora_state = ckpt.get("qwen_lora", None)
    epoch = ckpt.get("epoch", "final")
    has_lora = lora_state is not None and len(lora_state) > 0
    print(f"  Classes: {n_classes}, qwen_dim: {qwen_dim}, epoch: {epoch}")
    print(f"  Density maps: {density_maps.shape}")
    print(f"  LoRA weights: {'yes' if has_lora else 'NOT FOUND (pre-fix checkpoint)'}")

    return model, class_names, density_maps, cfg_dict, lora_state


def plot_prediction(probs, merged, class_names, cfg_dict, title, save_path):
    """Plot top-k bar chart + merged density map side by side."""
    fig, (ax_bar, ax_map) = plt.subplots(1, 2, figsize=(14, 5))

    extent = [
        cfg_dict.get("lon_min", 16.113), cfg_dict.get("lon_max", 22.897),
        cfg_dict.get("lat_max", 48.585), cfg_dict.get("lat_min", 45.737),
    ]

    # Top-10 bar chart
    top_vals, top_idx = probs.topk(min(10, len(class_names)))
    names = [class_names[i] for i in top_idx.cpu()]
    vals = top_vals.cpu().numpy()
    colors = ["#c0392b" if v > 0.5 else "#2980b9" if v > 0.1 else "#bdc3c7" for v in vals]
    ax_bar.barh(range(len(names)), vals, color=colors)
    ax_bar.set_yticks(range(len(names)))
    ax_bar.set_yticklabels(names, fontsize=9)
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Probability")
    ax_bar.set_title("Class predictions", fontsize=11)
    ax_bar.invert_yaxis()
    ax_bar.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    # Merged map
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under(alpha=0)
    im = ax_map.imshow(merged, cmap=cmap, origin="upper", extent=extent,
                       vmin=0.01, vmax=1.0)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.set_title("Merged density map", fontsize=11)
    ax_map.grid(True, alpha=0.2, linestyle="--")
    plt.colorbar(im, ax=ax_map, shrink=0.8)

    plt.suptitle(f'"{title}"', fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_class_grid(densities, titles, cfg_dict, save_path, ncols=5):
    """Plot all class density maps in a grid."""
    n = len(densities)
    if n == 0:
        return
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes) if nrows > 1 else np.atleast_2d(axes)

    extent = [
        cfg_dict.get("lon_min", 16.113), cfg_dict.get("lon_max", 22.897),
        cfg_dict.get("lat_max", 48.585), cfg_dict.get("lat_min", 45.737),
    ]
    cmap = plt.cm.YlOrRd.copy()

    for idx, (density, title) in enumerate(zip(densities, titles)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.imshow(density, cmap=cmap, origin="upper", extent=extent, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=5)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.suptitle("Flow 3 — All Classes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid: {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference for flow_3 classifier + density map merge",
    )
    parser.add_argument("--checkpoint", type=str,
                        default="training_data_flow3/checkpoint.pt")
    parser.add_argument("--query", nargs="+", type=str, default=None)
    parser.add_argument("--all-classes", action="store_true",
                        help="Render individual density maps for all training classes")
    parser.add_argument("--output-dir", type=str, default="inference_flow3_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default=None,
                        help="Qwen model ID (e.g. Qwen/Qwen3.5-27B)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Split Qwen across GPUs via device_map='auto'")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold (default: 0.5)")
    args = parser.parse_args()

    if not args.query and not args.all_classes:
        parser.print_help()
        print("\nProvide --query or --all-classes")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # Load classifier + density maps
    model, class_names, density_maps, cfg_dict, lora_state = load_flow3(args.checkpoint, device)
    density_maps_dev = density_maps.to(device)

    # Render all-classes (no Qwen needed — just show each class's raw density map)
    if args.all_classes:
        print(f"\nRendering {len(class_names)} class density maps...")
        densities, titles = [], []
        for i, name in enumerate(class_names):
            d = density_maps[i].numpy()
            safe = name.lower().replace(" ", "_")[:50]
            plot_prediction(
                torch.zeros(len(class_names)).scatter_(0, torch.tensor(i), 1.0),
                d, class_names, cfg_dict, name, out_dir / f"class_{safe}.png",
            )
            densities.append(d)
            titles.append(name)
        plot_class_grid(densities, titles, cfg_dict, out_dir / "all_classes_grid.png")

    # Render queries (needs Qwen)
    if args.query:
        from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
        model_id = args.model or cfg_dict.get("qwen_model_id", None)
        has_lora = lora_state is not None and len(lora_state) > 0
        print(f"\nLoading Qwen encoder{f' ({model_id})' if model_id else ''}"
              f"{' + LoRA weights' if has_lora else ''}...")
        encoder = Qwen3EmbeddingAdapter(
            model_id=model_id,
            freeze_encoder=True,
            lora=has_lora,
            multi_gpu=args.multi_gpu,
        )
        if has_lora:
            encoder.load_state_dict(lora_state, strict=False)
            print(f"  Restored {len(lora_state)} LoRA weight tensors")
        encoder = encoder.to(device).eval()

        for text in args.query:
            print(f'\nQuery: "{text}"')
            emb = encoder.encode_raw(text)  # (1, dim)

            with torch.no_grad():
                logits = model(emb.to(device))
                probs = torch.sigmoid(logits)[0]  # (C,)
                merged = merge_maps(
                    probs.unsqueeze(0), density_maps_dev
                )[0].cpu().numpy()

            # Print top predictions
            above = (probs > args.threshold).sum().item()
            top5 = probs.topk(5)
            print(f"  {above} classes above {args.threshold} threshold:")
            for val, idx in zip(top5.values, top5.indices):
                marker = "*" if val > args.threshold else " "
                print(f"    {marker} {class_names[idx]:30s} {val:.3f}")

            safe = text.lower().replace(" ", "_")[:50]
            plot_prediction(probs, merged, class_names, cfg_dict, text,
                            out_dir / f"query_{safe}.png")

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
