"""
flow_4 — Satellite-Grounded Contrastive Text Encoder
=====================================================

Text → Qwen3 (LoRA) → ProjectionHead → 64-d L2-norm
                                            ↕ InfoNCE
                satellite grid: (256, 609, 64) AlphaEarth embeddings

Inference: cosine_sim(projected_text, sat_grid) → density map.

No classifier, no lookup table. Novel queries work because the model
learns the language → spectral mapping via contrastive training against
satellite embeddings.

Training data: 40 land cover classes (HRL crops + CORINE), each with
Wikipedia-derived text descriptions and a ground-truth density map.
"""

import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flow_3 import load_raw_texts


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Flow4Config:
    # Data
    pairs_cache: str = "pipeline_cache/pairs_855d516e8bee.pt"
    sat_grid_path: str = "pipeline_cache/sat_grid_1dffdd1c79c3.npy"
    text_descriptions_path: str = "data_corine/corine_wiki_char_count.jsonl"
    hrl_descriptions_path: str = "data_corine/hrl_wiki_char_count.jsonl"
    output_dir: str = "training_data_flow4"

    # Satellite
    sat_emb_dim: int = 64

    # Projection head
    proj_hidden_1: int = 512
    proj_hidden_2: int = 128

    # Contrastive
    init_temperature: float = 0.07
    pixel_loss_weight: float = 0.25
    n_pos_pixels: int = 64
    n_neg_pixels: int = 64

    # Training
    n_epochs: int = 50
    lr: float = 1e-4
    batch_size: int = 40  # = all classes per step
    grad_accum_steps: int = 4
    val_fraction: float = 0.15
    plot_every: int = 5

    # Geo
    lat_min: float = 45.737
    lat_max: float = 48.585
    lon_min: float = 16.113
    lon_max: float = 22.897

    # Qwen
    qwen_model_id: str = "Qwen/Qwen3.5-4B"
    qwen_emb_dim: int = 0  # auto-detect


# ============================================================
# MODELS
# ============================================================

class SatelliteProjectionHead(nn.Module):
    """Projects Qwen text embeddings into 64-d satellite embedding space."""

    def __init__(self, text_dim: int, sat_dim: int = 64,
                 hidden_1: int = 512, hidden_2: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_1),
            nn.SiLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.SiLU(),
            nn.Linear(hidden_2, sat_dim),
        )

    def forward(self, text_emb):
        return F.normalize(self.proj(text_emb), dim=-1)


class LearnableTemperature(nn.Module):
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(math.log(1.0 / init_temp)))

    def forward(self):
        return self.log_temp.exp().clamp(min=0.01, max=100.0)


# ============================================================
# LOSSES
# ============================================================

def centroid_infonce_loss(text_embs, sat_centroids, temperature):
    """Symmetric InfoNCE (CLIP-style) over (text, centroid) pairs.

    text_embs:     (N, 64) L2-normalized
    sat_centroids: (N, 64) L2-normalized
    temperature:   scalar

    Returns scalar loss.
    """
    logits = (text_embs @ sat_centroids.T) * temperature  # (N, N)
    labels = torch.arange(len(text_embs), device=text_embs.device)
    loss_t2s = F.cross_entropy(logits, labels)
    loss_s2t = F.cross_entropy(logits.T, labels)
    return (loss_t2s + loss_s2t) / 2


def pixel_contrastive_loss(text_embs, sat_grid_flat, density_maps_flat,
                           class_indices, n_pos, n_neg, temperature):
    """Pixel-level contrastive: text vs sampled positive/negative grid cells.

    text_embs:        (B, 64) L2-normalized
    sat_grid_flat:    (H*W, 64) L2-normalized
    density_maps_flat: (n_classes, H*W)
    class_indices:    (B,) class index per text
    """
    B = text_embs.shape[0]
    device = text_embs.device
    losses = []

    for i in range(B):
        cls = class_indices[i]
        dm = density_maps_flat[cls]

        # Positive: cells where this class has density
        pos_mask = dm > 1e-4
        pos_indices = pos_mask.nonzero(as_tuple=True)[0]
        if len(pos_indices) == 0:
            continue
        if len(pos_indices) <= n_pos:
            pos_sample = pos_indices
        else:
            weights = dm[pos_indices]
            weights = weights / weights.sum()
            pos_sample = pos_indices[torch.multinomial(weights, n_pos, replacement=False)]

        # Negative: cells with zero density for this class
        neg_mask = dm < 1e-6
        neg_indices = neg_mask.nonzero(as_tuple=True)[0]
        if len(neg_indices) == 0:
            continue
        if len(neg_indices) > n_neg:
            neg_sample = neg_indices[torch.randint(len(neg_indices), (n_neg,), device=device)]
        else:
            neg_sample = neg_indices

        pos_embs = sat_grid_flat[pos_sample]  # (n_p, 64)
        neg_embs = sat_grid_flat[neg_sample]  # (n_n, 64)

        # For each positive: InfoNCE against all negatives
        # sim_pos: (n_p,), sim_neg: (n_p, n_n)
        sim_pos = (text_embs[i] * pos_embs).sum(dim=-1) * temperature  # (n_p,)
        sim_neg = (text_embs[i:i+1] @ neg_embs.T).squeeze(0) * temperature  # (n_n,)

        # Each positive is classified against all negatives
        for p_idx in range(len(pos_sample)):
            logits = torch.cat([sim_pos[p_idx:p_idx+1], sim_neg])  # (1+n_n,)
            losses.append(F.cross_entropy(logits.unsqueeze(0),
                                          torch.zeros(1, dtype=torch.long, device=device)))

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return torch.stack(losses).mean()


# ============================================================
# DATA
# ============================================================

def compute_sat_centroids(pairs, sat_grid):
    """Density-weighted average of sat grid per class, L2-normalized.

    Returns (n_classes, 64) tensor.
    """
    H, W, D = sat_grid.shape
    sat_flat = sat_grid.reshape(H * W, D).astype(np.float64)
    centroids = []
    for p in pairs:
        dm = p["density_map"].numpy().flatten().astype(np.float64)
        w_sum = dm.sum()
        if w_sum < 1e-8:
            c = sat_flat.mean(axis=0)
        else:
            c = (dm[:, np.newaxis] / w_sum * sat_flat).sum(axis=0)
        c = c / (np.linalg.norm(c) + 1e-8)
        centroids.append(c.astype(np.float32))
    return torch.from_numpy(np.stack(centroids))


def build_datasets(pairs, raw_texts, cfg):
    """Build class names, centroids, density maps, and text pools."""
    class_names = []
    class_to_idx = {}
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        if desc_key not in class_to_idx:
            class_to_idx[desc_key] = len(class_names)
            class_names.append(p["class_name"])
    n_classes = len(class_names)

    density_maps = torch.stack([
        torch.as_tensor(p["density_map"], dtype=torch.float32) for p in pairs
    ])

    # Build per-class text pools
    texts_by_class = {i: [] for i in range(n_classes)}
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        cls_idx = class_to_idx[desc_key]
        sents = raw_texts.get(desc_key, [f"{p['class_name']} in Hungary"])
        texts_by_class[cls_idx].extend(sents)

    # Train/val split
    rng = random.Random(42)
    train_texts = {i: [] for i in range(n_classes)}
    val_texts = {i: [] for i in range(n_classes)}
    for cls_idx in range(n_classes):
        sents = list(texts_by_class[cls_idx])
        rng.shuffle(sents)
        n_val = max(1, int(len(sents) * cfg.val_fraction))
        val_texts[cls_idx] = sents[:n_val]
        train_texts[cls_idx] = sents[n_val:]

    n_train = sum(len(v) for v in train_texts.values())
    n_val = sum(len(v) for v in val_texts.values())
    print(f"  Classes: {n_classes}")
    print(f"  Train sentences: {n_train}, Val sentences: {n_val}")
    print(f"  Density maps: {density_maps.shape}")

    return class_names, density_maps, train_texts, val_texts


def sample_contrastive_batch(texts_by_class, n_classes):
    """Sample one sentence per class for full NxN contrastive batch."""
    texts = []
    for cls_idx in range(n_classes):
        pool = texts_by_class[cls_idx]
        texts.append(random.choice(pool) if pool else f"class {cls_idx}")
    return texts, torch.arange(n_classes)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_predictions(proj_head, text_encoder, sat_grid_flat_norm,
                          density_maps, class_names, queries, cfg,
                          epoch, viz_dir, device):
    """Render cosine similarity maps for test queries."""
    proj_head.eval()
    H, W = density_maps.shape[1], density_maps.shape[2]
    n = len(queries)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    extent = [cfg.lon_min, cfg.lon_max, cfg.lat_max, cfg.lat_min]

    with torch.no_grad():
        for row, query in enumerate(queries):
            emb = text_encoder.encode_raw(query)  # (1, qwen_dim)
            projected = proj_head(emb.to(device))  # (1, 64)
            sim = (projected @ sat_grid_flat_norm.T)[0]  # (H*W,)
            sim_map = sim.reshape(H, W).cpu().numpy()
            density = sim_map.clip(0, 1)

            # Heatmap
            ax = axes[row, 0]
            im = ax.imshow(density, cmap="YlOrRd", origin="upper",
                           extent=extent, vmin=0, vmax=0.5)
            ax.set_title(f'"{query[:50]}"', fontsize=9)
            ax.axis("off")

            # Histogram of similarities
            ax2 = axes[row, 1]
            ax2.hist(sim.cpu().numpy(), bins=50, color="steelblue", alpha=0.7)
            ax2.axvline(0, color="red", linestyle="--", alpha=0.5)
            ax2.set_title(f"sim distribution (mean={sim.mean():.3f})", fontsize=9)
            ax2.set_xlim(-0.3, 0.5)

    plt.suptitle(f"Epoch {epoch}", fontsize=11)
    plt.tight_layout()
    plt.savefig(viz_dir / f"epoch_{epoch:03d}.png", dpi=120)
    plt.close(fig)
    proj_head.train()


# ============================================================
# TRAINING
# ============================================================

def validate(proj_head, text_encoder, sat_centroids, val_texts,
             n_classes, device):
    """Compute retrieval R@1 and R@5 on validation texts."""
    proj_head.eval()
    correct_1, correct_5, total = 0, 0, 0
    tokenizer = text_encoder._tokenizer
    qwen_device = text_encoder.input_device

    with torch.no_grad():
        for cls_idx in range(n_classes):
            for sent in val_texts[cls_idx]:
                inputs = tokenizer(
                    [sent], return_tensors="pt",
                    truncation=True, max_length=512, padding=True,
                )
                ids = inputs["input_ids"].to(qwen_device)
                mask = inputs["attention_mask"].to(qwen_device)

                with torch.autocast(qwen_device.type, dtype=torch.bfloat16):
                    emb = text_encoder(ids, mask)
                    projected = proj_head(emb.to(device))  # (1, 64)

                sim = (projected @ sat_centroids.T)[0]  # (n_classes,)
                top5 = sim.topk(5).indices.tolist()

                if cls_idx == top5[0]:
                    correct_1 += 1
                if cls_idx in top5:
                    correct_5 += 1
                total += 1

    proj_head.train()
    r1 = correct_1 / max(1, total) * 100
    r5 = correct_5 / max(1, total) * 100
    return r1, r5


def train(proj_head, temperature, text_encoder, sat_centroids, sat_grid_flat_norm,
          density_maps, class_names, train_texts, val_texts, cfg, device):
    """Main training loop."""

    n_classes = len(class_names)
    sat_centroids_dev = sat_centroids.to(device)
    sat_grid_flat_dev = sat_grid_flat_norm.to(device)
    H, W = density_maps.shape[1], density_maps.shape[2]
    dm_flat = density_maps.reshape(n_classes, -1).to(device)

    # Optimizer
    qwen_params = [p for p in text_encoder.parameters() if p.requires_grad]
    n_qwen = sum(p.numel() for p in qwen_params)
    n_proj = sum(p.numel() for p in proj_head.parameters())
    print(f"\n  Qwen LoRA params: {n_qwen:,}")
    print(f"  Projection head params: {n_proj:,}")

    optimizer = torch.optim.AdamW([
        {"params": qwen_params, "lr": cfg.lr},
        {"params": proj_head.parameters(), "lr": cfg.lr * 10},
        {"params": temperature.parameters(), "lr": cfg.lr * 10},
    ])

    # One "batch" = one sentence per class = 40 texts
    # n_batches = enough to cycle through most training sentences per epoch
    min_pool = min(len(v) for v in train_texts.values())
    n_batches = max(min_pool, 20)
    total_steps = n_batches * cfg.n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)

    tokenizer = text_encoder._tokenizer
    qwen_device = text_encoder.input_device

    viz_dir = Path(cfg.output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    test_queries = [
        "wheat", "maize", "sunflower", "water bodies",
        "deciduous forest", "urban residential areas",
        "grassland near rivers", "quantum physics textbook",
    ]

    train_losses, val_r1s, val_r5s, temps = [], [], [], []

    # Resume
    ckpt_path = Path(cfg.output_dir) / "checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        print(f"[resume] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        proj_head.load_state_dict(ckpt["proj_head"])
        temperature.load_state_dict(ckpt["temperature"])
        lora_state = ckpt.get("qwen_lora", {})
        if lora_state:
            text_encoder.load_state_dict(lora_state, strict=False)
            print(f"  Restored {len(lora_state)} LoRA weight tensors")
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        train_losses = ckpt.get("train_losses", [])
        val_r1s = ckpt.get("val_r1s", [])
        val_r5s = ckpt.get("val_r5s", [])
        temps = ckpt.get("temps", [])
        print(f"  Resuming from epoch {start_epoch}")

    print(f"\nTraining: {n_classes} classes, {n_batches} batches/epoch")
    print(f"  Grad accum: {cfg.grad_accum_steps}, effective batch: {n_classes * cfg.grad_accum_steps}")

    proj_head.train()
    text_encoder.train()

    for epoch in range(start_epoch, cfg.n_epochs):
        ep_loss = 0.0
        ep_centroid_loss = 0.0
        ep_pixel_loss = 0.0

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1:3d}",
                    leave=False, ncols=100)

        for batch_i in pbar:
            texts, class_indices = sample_contrastive_batch(train_texts, n_classes)
            class_indices = class_indices.to(device)

            inputs = tokenizer(
                texts, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            ids = inputs["input_ids"].to(qwen_device)
            mask = inputs["attention_mask"].to(qwen_device)

            with torch.autocast(qwen_device.type, dtype=torch.bfloat16):
                embs = text_encoder(ids, mask)  # (40, qwen_dim)
                projected = proj_head(embs.to(device))  # (40, 64)
                temp = temperature()

                loss_c = centroid_infonce_loss(projected, sat_centroids_dev, temp)
                loss_p = pixel_contrastive_loss(
                    projected, sat_grid_flat_dev, dm_flat,
                    class_indices, cfg.n_pos_pixels, cfg.n_neg_pixels, temp)

                loss = (loss_c + cfg.pixel_loss_weight * loss_p) / cfg.grad_accum_steps

            loss.backward()

            if (batch_i + 1) % cfg.grad_accum_steps == 0 or batch_i == n_batches - 1:
                torch.nn.utils.clip_grad_norm_(qwen_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            ep_loss += loss.item() * cfg.grad_accum_steps
            ep_centroid_loss += loss_c.item()
            ep_pixel_loss += loss_p.item()

            pbar.set_postfix({
                "loss": f"{ep_loss/(batch_i+1):.4f}",
                "temp": f"{temp.item():.2f}",
            })

        avg_loss = ep_loss / n_batches
        avg_c = ep_centroid_loss / n_batches
        avg_p = ep_pixel_loss / n_batches
        cur_temp = temperature().item()
        train_losses.append(avg_loss)
        temps.append(cur_temp)

        # Validation
        r1, r5 = validate(proj_head, text_encoder, sat_centroids_dev,
                          val_texts, n_classes, device)
        val_r1s.append(r1)
        val_r5s.append(r5)

        print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f} (centroid={avg_c:.4f}, "
              f"pixel={avg_p:.4f}), R@1={r1:.1f}%, R@5={r5:.1f}%, temp={cur_temp:.2f}")

        # --- Progress plot ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs_x = list(range(1, len(train_losses) + 1))

        axes[0].plot(epochs_x, train_losses, color="coral", marker="o", markersize=3)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Total Loss")
        axes[0].set_title("Training Loss")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_x, val_r1s, label="R@1", color="steelblue", marker="o", markersize=3)
        axes[1].plot(epochs_x, val_r5s, label="R@5", color="coral", marker="o", markersize=3)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Retrieval %")
        axes[1].set_title("Validation Retrieval")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs_x, temps, color="green", marker="o", markersize=3)
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Temperature")
        axes[2].set_title("Learned Temperature")
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("Flow 4 — Contrastive Training")
        plt.tight_layout()
        plt.savefig(viz_dir / "progress.png", dpi=120)
        plt.close(fig)

        # --- Visualization ---
        if cfg.plot_every > 0 and ((epoch + 1) % cfg.plot_every == 0 or epoch == 0):
            visualize_predictions(
                proj_head, text_encoder, sat_grid_flat_dev,
                density_maps, class_names, test_queries,
                cfg, epoch + 1, viz_dir, device)

        # --- Checkpoint ---
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        lora_state = {k: v.cpu() for k, v in text_encoder.state_dict().items()
                      if "lora" in k.lower()}
        torch.save({
            "epoch": epoch,
            "proj_head": proj_head.state_dict(),
            "temperature": temperature.state_dict(),
            "qwen_lora": lora_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "sat_centroids": sat_centroids.cpu(),
            "class_names": class_names,
            "config": vars(cfg),
            "train_losses": train_losses,
            "val_r1s": val_r1s,
            "val_r5s": val_r5s,
            "temps": temps,
        }, ckpt_path)

    # Final save
    lora_state = {k: v.cpu() for k, v in text_encoder.state_dict().items()
                  if "lora" in k.lower()}
    torch.save({
        "proj_head": proj_head.state_dict(),
        "temperature": temperature.state_dict(),
        "qwen_lora": lora_state,
        "sat_centroids": sat_centroids.cpu(),
        "class_names": class_names,
        "config": vars(cfg),
    }, Path(cfg.output_dir) / "flow4_model.pt")
    print(f"\n  Saved to {Path(cfg.output_dir) / 'flow4_model.pt'}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default=None,
                        help="Qwen model ID, e.g. Qwen/Qwen3-14B")
    parser.add_argument("--multi-gpu", action="store_true")
    args = parser.parse_args()

    cfg = Flow4Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.model:
        cfg.qwen_model_id = args.model

    multi_gpu = args.multi_gpu
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, multi_gpu: {multi_gpu}")

    # --- Step 1: Load pairs ---
    print("\n" + "=" * 60)
    print("STEP 1: Load cached training pairs")
    print("=" * 60)
    pairs = torch.load(cfg.pairs_cache, map_location="cpu", weights_only=False)
    print(f"  {len(pairs)} pairs loaded")

    # --- Step 2: Load satellite grid ---
    print("\n" + "=" * 60)
    print("STEP 2: Load satellite embedding grid")
    print("=" * 60)
    sat_grid = np.load(cfg.sat_grid_path)
    print(f"  Grid shape: {sat_grid.shape}")
    H, W, D = sat_grid.shape

    # L2-normalize per cell (once, for cosine similarity)
    sat_grid_flat = sat_grid.reshape(-1, D)
    norms = np.linalg.norm(sat_grid_flat, axis=1, keepdims=True) + 1e-8
    sat_grid_flat_norm = torch.from_numpy((sat_grid_flat / norms).astype(np.float32))
    print(f"  Normalized: {sat_grid_flat_norm.shape}, to {device}")

    # --- Step 3: Compute centroids ---
    print("\n" + "=" * 60)
    print("STEP 3: Compute satellite centroids")
    print("=" * 60)
    sat_centroids = compute_sat_centroids(pairs, sat_grid)
    print(f"  Centroids: {sat_centroids.shape}")

    # --- Step 4: Load texts & build datasets ---
    print("\n" + "=" * 60)
    print("STEP 4: Load text descriptions & build datasets")
    print("=" * 60)
    raw_texts = load_raw_texts(
        cfg.text_descriptions_path,
        extra_paths=[cfg.hrl_descriptions_path],
    )
    class_names, density_maps, train_texts, val_texts = \
        build_datasets(pairs, raw_texts, cfg)

    # --- Step 5: Load Qwen ---
    print("\n" + "=" * 60)
    print("STEP 5: Load Qwen text encoder")
    print("=" * 60)
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
    text_encoder = Qwen3EmbeddingAdapter(
        model_id=cfg.qwen_model_id,
        freeze_encoder=True,
        lora=True,
        lora_r=32 if multi_gpu else 16,
        lora_alpha=64 if multi_gpu else 32,
        multi_gpu=multi_gpu,
    )
    text_encoder = text_encoder.to(device)
    cfg.qwen_emb_dim = text_encoder.target_dim
    n_lora = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    print(f"  Qwen loaded, emb_dim={cfg.qwen_emb_dim}")
    print(f"  LoRA trainable params: {n_lora:,}")

    # --- Step 6: Build projection head ---
    print("\n" + "=" * 60)
    print("STEP 6: Build projection head + temperature")
    print("=" * 60)
    proj_head = SatelliteProjectionHead(
        text_dim=cfg.qwen_emb_dim,
        sat_dim=cfg.sat_emb_dim,
        hidden_1=cfg.proj_hidden_1,
        hidden_2=cfg.proj_hidden_2,
    ).to(device)
    temperature = LearnableTemperature(cfg.init_temperature).to(device)
    print(f"  ProjectionHead: {cfg.qwen_emb_dim} → {cfg.proj_hidden_1} → {cfg.proj_hidden_2} → {cfg.sat_emb_dim}")

    # --- Step 7: Train ---
    print("\n" + "=" * 60)
    print("STEP 7: Contrastive training")
    print("=" * 60)
    train(proj_head, temperature, text_encoder, sat_centroids,
          sat_grid_flat_norm, density_maps, class_names,
          train_texts, val_texts, cfg, device=device)

    print("\nDone.")


if __name__ == "__main__":
    main()
