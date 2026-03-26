#!/usr/bin/env python3
"""
Inference script for SpatialBasisFieldV2
=========================================

Loads a trained checkpoint and renders density maps for:
  - Free-text queries (any description you want)
  - All known training classes

The Qwen encoder computes text embeddings on the fly, including
any fine-tuned LoRA weights saved in the checkpoint.

Usage:
    # Render a single text query
    python inference.py --query "sunflower fields in the Great Hungarian Plain"

    # Render multiple queries
    python inference.py --query "wheat" "rapeseed" "permanent grassland"

    # Render all training classes
    python inference.py --all-classes

    # Use the training checkpoint instead of the final save
    python inference.py --checkpoint training_data/checkpoint_v2.pt --query "maize"

    # Custom output directory
    python inference.py --query "barley" --output-dir inference_outputs

    # Render at higher resolution
    python inference.py --query "sugar beet" --height 512
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load the SpatialBasisFieldV2 model from a checkpoint.

    Handles both checkpoint formats:
      - Final save: keys = model_state_dict, config, class_names, qwen_lora
      - Training checkpoint: keys = model, config, epoch, optimizer, ...
    """
    from flow_2 import SpatialBasisFieldV2

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config
    cfg = ckpt.get("config", {})
    print(f"  Config: n_bases={cfg.get('n_bases')}, n_fourier_freqs={cfg.get('n_fourier_freqs')}, "
          f"hidden_dim={cfg.get('hidden_dim')}, text_emb_dim={cfg.get('text_emb_dim')}")

    # Build model with saved config
    model = SpatialBasisFieldV2(
        text_dim=cfg.get("text_emb_dim", 2560),
        text_proj_dim=cfg.get("text_proj_dim", 128),
        n_bases=cfg.get("n_bases", 24),
        n_freqs=cfg.get("n_fourier_freqs", 32),
        hidden_dim=cfg.get("hidden_dim", 128),
        coord_hidden=cfg.get("coord_hidden", 128),
    )

    # Load weights (handle both key names)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("  Loaded from final save (model_state_dict)")
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        print(f"  Loaded from training checkpoint (epoch {epoch})")
    else:
        raise KeyError(f"Checkpoint has neither 'model_state_dict' nor 'model'. Keys: {list(ckpt.keys())}")

    model = model.to(device).eval()
    class_names = ckpt.get("class_names", [])
    qwen_lora = ckpt.get("qwen_lora", None)

    return model, cfg, class_names, qwen_lora


def load_text_encoder(device: str = "cuda", qwen_lora: dict = None):
    """Load Qwen3EmbeddingAdapter with optional LoRA weights."""
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter

    # If we have LoRA weights, load with LoRA architecture so the keys match
    has_lora = qwen_lora is not None and len(qwen_lora) > 0
    print(f"Loading Qwen text encoder (LoRA={'yes' if has_lora else 'no'})...")

    encoder = Qwen3EmbeddingAdapter(
        target_dim=2560,
        freeze_encoder=True,
        lora=has_lora,
    )

    if has_lora:
        # Inject the fine-tuned LoRA weights
        encoder._model.load_state_dict(qwen_lora, strict=False)
        print(f"  Loaded {len(qwen_lora)} LoRA weight tensors")

    encoder = encoder.to(device).eval()
    return encoder


def render_density_map(model, text_emb, cfg, device="cuda"):
    """Render a density map for a given text embedding."""
    H = cfg.get("target_resolution", 256)
    aspect = (cfg["lon_max"] - cfg["lon_min"]) / (cfg["lat_max"] - cfg["lat_min"])
    W = int(H * aspect)
    return model.render_map(text_emb, H=H, W=W, device=device)


def plot_density_map(density, cfg, title="", save_path=None, show_colorbar=True):
    """Plot a single density map with geographic extent."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Custom colormap: transparent → yellow → orange → red
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under(alpha=0)

    im = ax.imshow(
        density,
        cmap=cmap,
        origin="upper",
        extent=[cfg["lon_min"], cfg["lon_max"], cfg["lat_max"], cfg["lat_min"]],
        vmin=0.01,
        vmax=1.0,
    )

    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.2, linestyle="--")

    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Predicted Density", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_multi_grid(densities, titles, cfg, save_path=None, ncols=4):
    """Plot multiple density maps in a grid."""
    n = len(densities)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    cmap = plt.cm.YlOrRd.copy()

    for idx, (density, title) in enumerate(zip(densities, titles)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        ax.imshow(
            density,
            cmap=cmap,
            origin="upper",
            extent=[cfg["lon_min"], cfg["lon_max"], cfg["lat_max"], cfg["lat_min"]],
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.tick_params(labelsize=6)

    # Hide empty subplots
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.suptitle("SpatialBasisFieldV2 Inference", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved grid: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Inference for SpatialBasisFieldV2 — render density maps from text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="training_data/spatial_basis_field_v2.pt",
        help="Path to model checkpoint (default: training_data/spatial_basis_field_v2.pt)",
    )
    parser.add_argument(
        "--query", nargs="+", type=str, default=None,
        help="One or more free-text queries to render density maps for",
    )
    parser.add_argument(
        "--all-classes", action="store_true",
        help="Render density maps for all training classes (uses precomputed embeddings from cache)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="inference_outputs",
        help="Directory to save rendered maps (default: inference_outputs)",
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Override render height in pixels (width auto-calculated from aspect ratio)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to run on (default: cuda)",
    )
    parser.add_argument(
        "--no-lora", action="store_true",
        help="Skip loading LoRA weights even if present in checkpoint",
    )

    args = parser.parse_args()

    if not args.query and not args.all_classes:
        parser.print_help()
        print("\n❌ Provide --query or --all-classes")
        sys.exit(1)

    device = args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    model, cfg, class_names, qwen_lora = load_model(args.checkpoint, device)
    if args.height:
        cfg["target_resolution"] = args.height
    if args.no_lora:
        qwen_lora = None

    # --- Load text encoder ---
    # Put Qwen on a second GPU if available, otherwise share the device
    if torch.cuda.device_count() > 1:
        enc_device = "cuda:1"
    else:
        enc_device = device
    encoder = load_text_encoder(enc_device, qwen_lora)

    # --- Render queries ---
    if args.query:
        densities, titles = [], []
        for text in args.query:
            print(f"\nRendering: \"{text}\"")
            emb = encoder.encode_raw(text, normalize=False)  # (1, 2560)
            density = render_density_map(model, emb, cfg, device)

            # Save individual plot
            safe_name = text.lower().replace(" ", "_")[:50]
            plot_density_map(density, cfg, title=text,
                             save_path=out_dir / f"{safe_name}.png")

            densities.append(density)
            titles.append(text)

        # Save grid if multiple queries
        if len(densities) > 1:
            plot_multi_grid(densities, titles, cfg,
                            save_path=out_dir / "query_grid.png")

    # --- Render all training classes ---
    if args.all_classes:
        print(f"\nRendering all {len(class_names)} training classes...")

        # Try to load precomputed embeddings from cache for speed
        pairs_cache = list(Path("pipeline_cache").glob("pairs_*.pt"))
        precomputed = {}
        if pairs_cache:
            print(f"  Loading precomputed embeddings from {pairs_cache[0]}")
            pairs = torch.load(pairs_cache[0], map_location="cpu", weights_only=False)
            for p in pairs:
                precomputed[p["class_name"]] = p["avg_embedding"]

        densities, titles = [], []
        for name in class_names:
            print(f"  {name}...", end=" ", flush=True)

            if name in precomputed:
                emb = precomputed[name].unsqueeze(0)
            else:
                # Fall back to on-the-fly encoding
                emb = encoder.encode_raw(name, normalize=False)

            density = render_density_map(model, emb, cfg, device)

            safe_name = name.lower().replace(" ", "_")[:50]
            plot_density_map(density, cfg, title=name,
                             save_path=out_dir / f"class_{safe_name}.png")
            densities.append(density)
            titles.append(name)
            print("✓")

        # Grid of all classes
        plot_multi_grid(densities, titles, cfg,
                        save_path=out_dir / "all_classes_grid.png",
                        ncols=5)
        print(f"\n✅ All class maps saved to {out_dir}/")


if __name__ == "__main__":
    main()
