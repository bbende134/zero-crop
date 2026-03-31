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
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Inline V1 architecture (pre-FiLM, coord_trunk style)
# ---------------------------------------------------------------------------

class _FourierFeatures(nn.Module):
    def __init__(self, n_input=2, n_freqs=64, sigma=10.0):
        super().__init__()
        B = torch.randn(n_input, n_freqs) * sigma
        self.register_buffer('B', B)

    @property
    def output_dim(self):
        return 2 + 2 * self.B.shape[1]

    def forward(self, coords):
        proj = coords @ self.B
        return torch.cat([coords, torch.sin(2 * math.pi * proj),
                          torch.cos(2 * math.pi * proj)], dim=-1)


class SpatialBasisFieldV1(nn.Module):
    """Original (pre-FiLM) architecture: independent coord trunk + text factors."""

    def __init__(self, text_dim=2560, n_bases=24, n_freqs=64, hidden_dim=256):
        super().__init__()
        self.n_bases = n_bases

        self.text_to_factors = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_bases),
        )

        self.fourier = _FourierFeatures(n_input=2, n_freqs=n_freqs)
        coord_dim = self.fourier.output_dim  # 2 + 2*n_freqs

        self.coord_trunk = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.basis_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Linear(hidden_dim // 2, 1),
            ) for _ in range(n_bases)
        ])

        self.basis_scale = nn.Parameter(torch.ones(n_bases))
        self.basis_bias = nn.Parameter(torch.zeros(n_bases))

    def forward(self, coords, text_emb):
        factors = torch.sigmoid(self.text_to_factors(text_emb))          # (B, n_bases)
        h = self.coord_trunk(self.fourier(coords))                        # (B, hidden)
        basis_out = torch.cat([head(h) for head in self.basis_heads], -1) # (B, n_bases)
        basis_out = torch.sigmoid(self.basis_scale * basis_out + self.basis_bias)
        return (basis_out * factors).sum(dim=-1).clamp(0, 1)

    @torch.no_grad()
    def render_map(self, text_emb, H=256, W=480, device="cuda"):
        self.eval()
        lat_grid = torch.linspace(-1, 1, H, device=device)
        lon_grid = torch.linspace(-1, 1, W, device=device)
        grid_lat, grid_lon = torch.meshgrid(lat_grid, lon_grid, indexing='ij')
        coords = torch.stack([grid_lat.flatten(), grid_lon.flatten()], dim=-1)
        if text_emb.dim() == 1:
            text_emb = text_emb.unsqueeze(0)
        text_emb = text_emb.to(device)
        text_expanded = text_emb.expand(coords.shape[0], -1)
        parts = []
        for i in range(0, len(coords), 100_000):
            parts.append(self.forward(coords[i:i+100_000], text_expanded[i:i+100_000]))
        return torch.cat(parts).view(H, W).cpu().numpy()


def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load a SpatialBasisField model from a checkpoint.

    Auto-detects V1 (coord_trunk) vs V2 (FiLM trunk) from the state dict keys.
    Handles both save formats:
      - Final save: keys = model_state_dict, config, class_names, qwen_lora
      - Training checkpoint: keys = model, config, epoch, optimizer, ...
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {})
    state_dict = ckpt.get("model_state_dict") or ckpt.get("model", {})

    # Detect architecture from state dict keys
    is_v1 = "coord_trunk.0.weight" in state_dict

    if is_v1:
        # Infer dims from weights
        n_freqs = state_dict["fourier.B"].shape[1]
        n_bases = state_dict["basis_scale"].shape[0]
        hidden_dim = state_dict["coord_trunk.0.weight"].shape[0]
        text_dim = state_dict["text_to_factors.0.weight"].shape[1]
        print(f"  Detected V1 architecture: text_dim={text_dim}, n_bases={n_bases}, "
              f"n_freqs={n_freqs}, hidden_dim={hidden_dim}")
        model = SpatialBasisFieldV1(
            text_dim=text_dim, n_bases=n_bases, n_freqs=n_freqs, hidden_dim=hidden_dim,
        )
    else:
        from flow_2 import SpatialBasisFieldV2
        # Infer text_proj_dim from weights if absent from config
        if "text_proj_dim" not in cfg and "text_to_factors.0.weight" in state_dict:
            cfg["text_proj_dim"] = state_dict["text_to_factors.0.weight"].shape[1]
        print(f"  Detected V2 architecture: n_bases={cfg.get('n_bases')}, "
              f"n_fourier_freqs={cfg.get('n_fourier_freqs')}, hidden_dim={cfg.get('hidden_dim')}, "
              f"text_emb_dim={cfg.get('text_emb_dim')}, text_proj_dim={cfg.get('text_proj_dim')}")
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
    bridge_state = ckpt.get("bridge", None)
    class_embeddings = ckpt.get("class_embeddings", {})

    bridge = None
    if bridge_state is not None:
        from flow_2 import TextToSatBridge
        bridge_cfg = cfg if isinstance(cfg, dict) else vars(cfg)
        b = TextToSatBridge(
            text_dim=bridge_cfg.get("qwen_emb_dim", 2560),
            sat_dim=bridge_cfg.get("text_emb_dim", 66),   # sat(64)+pheno(2)=66
            hidden_dim=bridge_cfg.get("bridge_hidden_dim", 256),
        )
        b.load_state_dict(bridge_state)
        bridge = b.to(device).eval()
        print(f"  Loaded TextToSatBridge weights")

    return model, cfg, class_names, qwen_lora, bridge, class_embeddings


def load_text_encoder(device: str = "cuda", qwen_lora=None):
    """Load Qwen3EmbeddingAdapter (frozen) for bridge-based inference."""
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter

    has_lora = qwen_lora is not None and len(qwen_lora) > 0
    print(f"Loading Qwen text encoder (frozen, LoRA={'yes' if has_lora else 'no'})...")

    encoder = Qwen3EmbeddingAdapter(
        target_dim=2560,
        freeze_encoder=True,
        lora=has_lora,
    )

    if has_lora:
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
    if n == 0:
        print("  [skip] No maps to plot in grid")
        return
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
        "--gpu", type=int, default=None,
        help="GPU index shorthand — sets device to cuda:N (overrides --device)",
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

    device = f"cuda:{args.gpu}" if args.gpu is not None else args.device
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model + bridge ---
    model, cfg, class_names, qwen_lora, bridge, class_embeddings = load_model(args.checkpoint, device)
    if args.height:
        cfg["target_resolution"] = args.height
    if args.no_lora:
        qwen_lora = None

    # Determine whether this is a satellite-conditioned checkpoint (sat+pheno ≤ 128)
    is_sat_model = cfg.get("text_emb_dim", 2560) <= 128

    # --- Load Qwen encoder once if any query mode needs it ---
    enc_device = "cuda:1" if torch.cuda.device_count() > 1 else device
    encoder = None
    needs_qwen = args.query or (args.all_classes and bridge is not None)
    if needs_qwen:
        print("Loading Qwen encoder for text encoding...")
        encoder = load_text_encoder(enc_device, qwen_lora)

    # --- Render queries ---
    if args.query:
        densities, titles = [], []
        for text in args.query:
            print(f"\nRendering: \"{text}\"")

            if bridge is not None and encoder is not None:
                # Zero-shot path: Qwen → bridge → satellite space
                qwen_emb = encoder.encode_raw(text, normalize=False)  # (1, 2560)
                with torch.no_grad():
                    emb = bridge(qwen_emb.to(device))  # (1, 64)
            elif encoder is not None:
                emb = encoder.encode_raw(text, normalize=False)
            else:
                raise RuntimeError("No encoder available for query mode")

            density = render_density_map(model, emb, cfg, device)

            safe_name = text.lower().replace(" ", "_")[:50]
            plot_density_map(density, cfg, title=text,
                             save_path=out_dir / f"{safe_name}.png")
            densities.append(density)
            titles.append(text)

        if len(densities) > 1:
            plot_multi_grid(densities, titles, cfg,
                            save_path=out_dir / "query_grid.png")

    # --- Render all training classes ---
    if args.all_classes:
        # Fallback: load class names + embeddings from pairs cache if checkpoint
        # predates the class_names/class_embeddings fields.
        if not class_names:
            pairs_cache = sorted(Path("pipeline_cache").glob("pairs_*.pt"))
            if pairs_cache:
                print(f"  [fallback] Loading class names from {pairs_cache[-1]}")
                cached = torch.load(pairs_cache[-1], map_location="cpu", weights_only=False)
                class_names = [p["class_name"] for p in cached]

        print(f"\nRendering all {len(class_names)} training classes...")

        # Use class embeddings saved in checkpoint (sat+pheno centroids),
        # falling back to bridge(Qwen) for older checkpoints that lack them.
        precomputed = class_embeddings
        if precomputed:
            print(f"  Using {len(precomputed)} class embeddings from checkpoint")

        densities, titles = [], []
        for name in class_names:
            print(f"  {name}...", end=" ", flush=True)

            if name in precomputed:
                emb = precomputed[name].unsqueeze(0).to(device)
            elif encoder is not None:
                # Fallback: encode class name via Qwen → bridge (or direct)
                qwen_emb = encoder.encode_raw(name, normalize=False)
                if bridge is not None:
                    with torch.no_grad():
                        emb = bridge(qwen_emb.to(device))
                else:
                    emb = qwen_emb
            else:
                print(f"  [skip] No embedding for {name}")
                continue

            density = render_density_map(model, emb, cfg, device)

            safe_name = name.lower().replace(" ", "_")[:50]
            plot_density_map(density, cfg, title=name,
                             save_path=out_dir / f"class_{safe_name}.png")
            densities.append(density)
            titles.append(name)
            print("✓")

        plot_multi_grid(densities, titles, cfg,
                        save_path=out_dir / "all_classes_grid.png",
                        ncols=5)
        print(f"\n✅ All class maps saved to {out_dir}/")


if __name__ == "__main__":
    main()
