#!/usr/bin/env python3
"""
Inference for flow_4 — Satellite-Grounded Contrastive Text Encoder
==================================================================

Loads checkpoint + Qwen ONCE, then encodes text queries into 64-d satellite
space and computes cosine similarity maps against the full satellite grid.

Usage:
    .venv/bin/python inference_flow4.py --query "wheat"
    .venv/bin/python inference_flow4.py --query "wheat" "sunflower" "deciduous forest"
    .venv/bin/python inference_flow4.py --all-classes
    .venv/bin/python inference_flow4.py --interactive --checkpoint training_data_flow4_v6/checkpoint.pt
    .venv/bin/python inference_flow4.py --query "wheat" --checkpoint training_data_flow4_v6/checkpoint.pt
"""

import argparse
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from flow_4 import SatelliteProjectionHead


# ============================================================
# LOAD
# ============================================================

def load_checkpoint(checkpoint_path: str, device: str):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg = ckpt.get("config", {})
    class_names = ckpt.get("class_names", [])
    sat_centroids = ckpt.get("sat_centroids", None)
    lora_state = ckpt.get("qwen_lora", None)

    text_dim = ckpt["proj_head"]["proj.1.weight"].shape[1]
    sat_dim = cfg.get("sat_emb_dim", 64)

    proj_head = SatelliteProjectionHead(
        text_dim=text_dim, sat_dim=sat_dim,
        hidden_1=cfg.get("proj_hidden_1", 512),
        hidden_2=cfg.get("proj_hidden_2", 128),
    )
    proj_head.load_state_dict(ckpt["proj_head"])
    proj_head = proj_head.to(device).eval()

    # Learned temperature (1 / init_temp stored as log)
    learned_temp = None
    if "temperature" in ckpt:
        from flow_4 import LearnableTemperature
        temp_module = LearnableTemperature()
        temp_module.load_state_dict(ckpt["temperature"])
        learned_temp = temp_module().item()

    has_lora = bool(lora_state)
    print(f"  text_dim={text_dim}  sat_dim={sat_dim}  "
          f"epoch={ckpt.get('epoch','final')}  classes={len(class_names)}  "
          f"lora={'yes' if has_lora else 'no'}  "
          f"learned_temp={learned_temp:.3f}" if learned_temp else "")

    return proj_head, cfg, class_names, sat_centroids, lora_state, learned_temp


def load_sat_grid(path=None, cache_dir="pipeline_cache"):
    """Load + L2-normalize sat grid. Auto-detects newest if path missing."""
    import data_pipeline as dp

    expected_H = dp.TARGET_RES
    expected_W = dp.target_width(dp.TARGET_RES)

    if path and Path(path).exists():
        sat_grid = np.load(path)
    else:
        candidates = sorted(Path(cache_dir).glob("sat_grid_*.npy"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        sat_grid = None
        for c in candidates:
            g = np.load(c, mmap_mode="r")
            if g.shape[:2] == (expected_H, expected_W):
                sat_grid = np.array(g)
                print(f"  [auto] Sat grid: {c.name}  shape={sat_grid.shape}")
                break
            print(f"  [skip] {c.name}  shape={g.shape}")
        if sat_grid is None:
            raise FileNotFoundError(
                f"No sat_grid at {expected_H}×{expected_W} in {cache_dir}/. "
                "Run data_pipeline.py first.")

    H, W, D = sat_grid.shape
    flat = sat_grid.reshape(-1, D)
    norms = np.linalg.norm(flat, axis=1, keepdims=True) + 1e-8
    flat_norm = torch.from_numpy((flat / norms).astype(np.float32))
    print(f"  Sat grid normalized: {flat_norm.shape}")
    return flat_norm, H, W


def load_encoder(cfg, lora_state, model_id_override=None, multi_gpu=False, device="cuda"):
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
    model_id = model_id_override or cfg.get("qwen_model_id")
    has_lora = bool(lora_state)

    # Infer LoRA rank from saved weights to avoid size mismatch
    lora_r, lora_alpha = 16, 32
    if has_lora:
        for k, v in lora_state.items():
            if "lora_A" in k:
                lora_r = v.shape[0]
                lora_alpha = lora_r * 2
                break

    print(f"\nLoading Qwen encoder: {model_id}"
          f"{f' + LoRA r={lora_r}' if has_lora else ''}...")
    enc = Qwen3EmbeddingAdapter(
        model_id=model_id,
        freeze_encoder=True,
        lora=has_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        multi_gpu=multi_gpu,
    )
    if has_lora:
        enc.load_state_dict(lora_state, strict=False)
        print(f"  Restored {len(lora_state)} LoRA tensors  (r={lora_r})")
    enc = enc.to(device).eval()
    return enc


def load_hungary_mask(H, cache_dir="pipeline_cache"):
    import data_pipeline as dp
    mask_path = Path(cache_dir) / f"hungary_mask_{H}x{dp.target_width(H)}.npy"
    if mask_path.exists():
        mask = np.load(mask_path)
        print(f"  Hungary mask: {mask.sum():,}/{mask.size:,} valid cells "
              f"({mask.mean()*100:.1f}%)")
        return mask
    print("  Hungary mask not found — run data_pipeline.py to generate it")
    return None


# ============================================================
# INFER
# ============================================================

_EXPAND_LLM_URL = "http://192.168.242.180:8001"


def expand_query(query: str) -> str:
    """Expand short queries using the LLM augmentation server's /generate endpoint.

    Training data is Wikipedia-style sentences about land cover. A one-word query
    like 'wheat' produces a very different Qwen embedding than a full description —
    this bridges the gap by generating a proper description first.

    Queries already longer than 8 words are returned unchanged (assumed descriptive enough).
    Falls back to the original query if the call fails.
    """
    if len(query.split()) > 8:
        return query

    try:
        import requests
        resp = requests.post(f"{_EXPAND_LLM_URL}/generate", json={
            "class_name": query,
            "input_text": query,
            "task": "query_to_desc",
            "temperature": 0.2,
        }, timeout=30)
        resp.raise_for_status()
        expanded = resp.json().get("generated_text", "").strip()
        return expanded if expanded else query
    except Exception as e:
        print(f"  [expand] LLM call failed ({e}), using original query")
        return query


def infer(query, text_encoder, proj_head, sat_grid_flat, H, W, device,
          sat_centroids=None, class_names=None, hungary_mask=None, top_k=5,
          temperature=1.0, expand=True):
    """Encode query → cosine similarity map. Prints CLI summary.

    The map always uses raw cosine similarities (no temperature scaling).
    Temperature is only applied for top-k class ranking.
    """
    if expand:
        expanded = expand_query(query)
        if expanded != query:
            print(f"  [expanded] \"{expanded}\"")
    else:
        expanded = query

    with torch.no_grad():
        emb = text_encoder.encode_raw(expanded)
        projected = proj_head(emb.to(device))  # (1, 64)
        # Raw cosine similarity — NOT scaled by temperature — for the spatial map
        sim = (projected @ sat_grid_flat.to(device).T)[0].cpu()  # (H*W,)

    sim_map = sim.numpy().reshape(H, W)

    # CLI stats (inside Hungary only)
    if hungary_mask is not None:
        sim_inside = sim.numpy()[hungary_mask.ravel()]
    else:
        sim_inside = sim.numpy()

    pos = (sim_inside > 0).sum()
    print(f"  range [{sim_inside.min():.3f}, {sim_inside.max():.3f}]  "
          f"mean {sim_inside.mean():.3f}  "
          f"positive cells: {pos:,}/{len(sim_inside):,} "
          f"({pos/len(sim_inside)*100:.1f}%)")

    # Top-k nearest classes via sat_centroids (temperature applied here for ranking)
    if sat_centroids is not None and class_names:
        cent = sat_centroids.to(device)
        class_sims = (projected @ cent.T)[0].cpu() * temperature
        top = class_sims.topk(min(top_k, len(class_names)))
        print(f"  Top-{top_k} closest classes (temp={temperature:.1f}):")
        for val, idx in zip(top.values, top.indices):
            print(f"    {class_names[idx]:35s} {val:.3f}")

    return sim_map


# ============================================================
# PLOT
# ============================================================

def plot_result(sim_map, cfg, title, save_path, hungary_mask=None, gt_density=None):
    n_cols = 2 if gt_density is not None else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    extent = [cfg.get("lon_min", 16.113), cfg.get("lon_max", 22.897),
              cfg.get("lat_max", 48.585), cfg.get("lat_min", 45.737)]

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="lightgrey")
    cmap.set_under(alpha=0)

    density = sim_map.clip(0, 1).astype(float)
    if hungary_mask is not None:
        density[~hungary_mask] = np.nan

    vmax = max(0.1, np.nanmax(density))
    im = axes[0].imshow(density, cmap=cmap, origin="upper", extent=extent,
                        vmin=0.01, vmax=vmax)
    axes[0].set_title(f'"{title}"', fontsize=11)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(True, alpha=0.2, linestyle="--")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    if gt_density is not None:
        gt = np.asarray(gt_density).astype(float)
        if hungary_mask is not None:
            gt[~hungary_mask] = np.nan
        im2 = axes[1].imshow(gt, cmap=cmap, origin="upper", extent=extent,
                             vmin=0.01, vmax=1.0)
        axes[1].set_title("Ground truth density", fontsize=11)
        axes[1].set_xlabel("Longitude")
        axes[1].grid(True, alpha=0.2, linestyle="--")
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.suptitle(f'flow_4 — "{title}"', fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_comparison(sim_raw, sim_exp, cfg, query, save_path,
                    hungary_mask=None, gt_density=None):
    """Side-by-side: raw | expanded | ground truth (if available)."""
    n_cols = 3 if gt_density is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))

    extent = [cfg.get("lon_min", 16.113), cfg.get("lon_max", 22.897),
              cfg.get("lat_max", 48.585), cfg.get("lat_min", 45.737)]
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="lightgrey")

    for ax, sm, label in [(axes[0], sim_raw, "raw"),
                           (axes[1], sim_exp, "expanded")]:
        density = sm.clip(0, 1).astype(float)
        if hungary_mask is not None:
            density[~hungary_mask] = np.nan
        vmax = max(0.1, np.nanmax(density))
        im = ax.imshow(density, cmap=cmap, origin="upper", extent=extent,
                       vmin=0.01, vmax=vmax)
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.grid(True, alpha=0.2, linestyle="--")
        plt.colorbar(im, ax=ax, shrink=0.8)
    axes[0].set_ylabel("Latitude")

    if gt_density is not None:
        gt = np.asarray(gt_density).astype(float)
        if hungary_mask is not None:
            gt[~hungary_mask] = np.nan
        im_gt = axes[2].imshow(gt, cmap=cmap, origin="upper", extent=extent,
                               vmin=0.01, vmax=1.0)
        axes[2].set_title("ground truth", fontsize=11)
        axes[2].set_xlabel("Longitude")
        axes[2].grid(True, alpha=0.2, linestyle="--")
        plt.colorbar(im_gt, ax=axes[2], shrink=0.8)

    plt.suptitle(f'"{query}"', fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_grid(sim_maps, titles, cfg, save_path, hungary_mask=None, ncols=5):
    n = len(sim_maps)
    ncols = min(ncols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    extent = [cfg.get("lon_min", 16.113), cfg.get("lon_max", 22.897),
              cfg.get("lat_max", 48.585), cfg.get("lat_min", 45.737)]
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="lightgrey")

    for idx, (sm, title) in enumerate(zip(sim_maps, titles)):
        r, c = divmod(idx, ncols)
        ax = axes[r, c]
        density = sm.clip(0, 1).astype(float)
        if hungary_mask is not None:
            density[~hungary_mask] = np.nan
        ax.imshow(density, cmap=cmap, origin="upper", extent=extent,
                  vmin=0.0, vmax=max(0.1, np.nanmax(density)))
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=5)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    plt.suptitle("flow_4 — Contrastive Similarity Maps", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved grid: {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="flow_4 inference — text → satellite cosine similarity map",
    )
    parser.add_argument("--checkpoint", default="training_data_flow4_v3/checkpoint.pt")
    parser.add_argument("--sat-grid", default=None,
                        help="Path to sat_grid .npy (auto-detected if omitted)")
    parser.add_argument("--query", nargs="+", default=None)
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--interactive", action="store_true",
                        help="REPL: load model once, query repeatedly")
    parser.add_argument("--output-dir", default="inference_flow4_outputs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=None, help="Override Qwen model ID")
    parser.add_argument("--multi-gpu", action="store_true")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Top-k classes to print per query")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Scale class-ranking similarities (default: use learned value). "
                             "Never applied to the spatial map.")
    parser.add_argument("--no-expand", action="store_true",
                        help="Disable automatic short-query expansion")
    args = parser.parse_args()

    if not args.query and not args.all_classes and not args.interactive:
        parser.print_help()
        print("\nProvide --query, --all-classes, or --interactive")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    # ---- Load everything once ----
    proj_head, cfg, class_names, sat_centroids, lora_state, learned_temp = \
        load_checkpoint(args.checkpoint, device)

    temperature = args.temperature if args.temperature is not None else (learned_temp or 1.0)
    print(f"  Temperature: {temperature:.3f}"
          f"{'  (CLI override)' if args.temperature else '  (learned)'}")

    print("\nLoading satellite grid...")
    sat_grid_flat, H, W = load_sat_grid(args.sat_grid)
    sat_grid_flat = sat_grid_flat.to(device)

    hungary_mask = load_hungary_mask(H)

    # Zero outside-Hungary cells (matches training setup)
    if hungary_mask is not None:
        sat_grid_flat[~torch.from_numpy(hungary_mask.ravel())] = 0.0

    if sat_centroids is not None:
        sat_centroids = sat_centroids.to(device)

    # Load ground-truth density maps for GT comparison
    gt_maps = {}
    pairs_candidates = sorted(Path("pipeline_cache").glob("pairs_*.pt"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
    if pairs_candidates:
        pairs = torch.load(pairs_candidates[0], map_location="cpu", weights_only=False)
        gt_maps = {p.get("desc_key", p["class_name"]): p["density_map"] for p in pairs}

    text_encoder = load_encoder(cfg, lora_state, args.model, args.multi_gpu, device)

    def _run_query(text, expand=None, suffix=""):
        """Run inference and save plot.

        expand=None  → use CLI default (--no-expand flag)
        expand=True  → force expansion
        expand=False → force raw query
        suffix       → appended to output filename to avoid collisions
        """
        if expand is None:
            expand = not args.no_expand
        print(f'\nQuery: "{text}"  [expand={expand}]')
        sim_map = infer(text, text_encoder, proj_head, sat_grid_flat,
                        H, W, device, sat_centroids, class_names,
                        hungary_mask, args.top_k, temperature,
                        expand=expand)
        gt = None
        tl = text.lower()
        for key, dm in gt_maps.items():
            if key.lower() in tl or tl in key.lower():
                gt = np.asarray(dm)
                break
        safe = tl.replace(" ", "_")[:50]
        fname = f"query_{safe}{suffix}.png"
        plot_result(sim_map, cfg, text, out_dir / fname, hungary_mask, gt)
        return sim_map

    # ---- Interactive REPL ----
    if args.interactive:
        print("\n" + "=" * 50)
        print("Interactive mode — type a query, empty line to quit")
        print("  Each query runs BOTH raw and expanded, side-by-side")
        print("=" * 50)
        while True:
            try:
                text = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                break
            print(f'\nQuery: "{text}"')
            sm_raw = infer(text, text_encoder, proj_head, sat_grid_flat,
                           H, W, device, sat_centroids, class_names,
                           hungary_mask, args.top_k, temperature, expand=False)
            print("  --- expanded ---")
            sm_exp = infer(text, text_encoder, proj_head, sat_grid_flat,
                           H, W, device, sat_centroids, class_names,
                           hungary_mask, args.top_k, temperature, expand=True)
            # Find GT if available
            gt = None
            tl = text.lower()
            for key, dm in gt_maps.items():
                if key.lower() in tl or tl in key.lower():
                    gt = np.asarray(dm)
                    break
            safe = tl.replace(" ", "_")[:50]
            plot_comparison(sm_raw, sm_exp, cfg, text,
                            out_dir / f"query_{safe}.png", hungary_mask, gt)
        print(f"\nOutputs in {out_dir}/")
        return

    # ---- Batch queries ----
    if args.query:
        run_both = not args.no_expand  # compare raw vs expanded when expansion is on
        sim_maps_exp, sim_maps_raw, titles = [], [], []
        for text in args.query:
            sm_exp = _run_query(text, expand=True, suffix="_expanded")
            sim_maps_exp.append(sm_exp)
            sm_raw = _run_query(text, expand=False, suffix="_raw")
            sim_maps_raw.append(sm_raw)
            titles.append(text)
        if len(titles) > 1:
            plot_grid(sim_maps_exp, [f"{t} [exp]" for t in titles],
                      cfg, out_dir / "query_grid_expanded.png", hungary_mask)
            plot_grid(sim_maps_raw, [f"{t} [raw]" for t in titles],
                      cfg, out_dir / "query_grid_raw.png", hungary_mask)

    # ---- All classes ----
    if args.all_classes:
        print(f"\nRendering {len(class_names)} class queries...")
        sim_maps, titles = [], []
        for name in class_names:
            sm = _run_query(name)
            sim_maps.append(sm)
            titles.append(name)
        plot_grid(sim_maps, titles, cfg, out_dir / "all_classes_grid.png", hungary_mask)

    print(f"\nDone. Outputs in {out_dir}/")


if __name__ == "__main__":
    main()
