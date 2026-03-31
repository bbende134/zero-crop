"""
flow_3 — Classify, Lookup, Merge
================================

Text → multi-label classifier → weighted sum of pre-computed density maps.

No learned spatial field. The 40 ground-truth density maps (256×609) are
frozen lookup tables. Qwen3 (LoRA fine-tuned) classifies which classes
the input text refers to, then we merge their maps via clamped sum.

Training data: ~45% single-class sentences, ~50% synthetic multi-class
(concatenated with random separators), ~5% OOD negatives (all-zeros target).
"""

import json
import random
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Flow3Config:
    # Data (reuse flow_2 caches)
    pairs_cache: str = "pipeline_cache/pairs_855d516e8bee.pt"
    text_descriptions_path: str = "data_corine/corine_wiki_char_count.jsonl"
    hrl_descriptions_path: str = "data_corine/hrl_wiki_char_count.jsonl"
    output_dir: str = "training_data_flow3"

    # Training
    n_epochs: int = 30
    lr: float = 1e-4
    batch_size: int = 32
    val_fraction: float = 0.15
    multi_label_ratio: float = 0.50
    negative_ratio: float = 0.05
    max_classes_per_sample: int = 4
    plot_every: int = 5

    # Geo (for visualization)
    lat_min: float = 45.737
    lat_max: float = 48.585
    lon_min: float = 16.113
    lon_max: float = 22.897

    # Qwen
    qwen_emb_dim: int = 2560


# ============================================================
# NEGATIVE SENTENCE POOL (OOD examples — all-zeros target)
# ============================================================

_NEGATIVE_SENTENCES = [
    "The history of quantum mechanics began in the early 20th century.",
    "Albert Einstein was born in Ulm, Germany in 1879.",
    "The stock market experienced a major crash in October 1929.",
    "JavaScript is a programming language commonly used for web development.",
    "The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    "Mozart composed his first symphony at the age of eight.",
    "The human body contains approximately 206 bones.",
    "Basketball was invented by James Naismith in 1891.",
    "The periodic table organizes chemical elements by atomic number.",
    "Shakespeare wrote 37 plays during his lifetime.",
    "The speed of light in vacuum is approximately 299,792 km/s.",
    "DNA was first identified by Friedrich Miescher in 1869.",
    "The Great Wall of China stretches over 21,000 kilometers.",
    "Python was created by Guido van Rossum and released in 1991.",
    "The Titanic sank on April 15, 1912 during her maiden voyage.",
    "Neural networks are inspired by the structure of the human brain.",
    "The Olympics originated in ancient Greece around 776 BC.",
    "Beethoven composed his Ninth Symphony while nearly completely deaf.",
    "The Amazon River is the largest river by discharge volume.",
    "TCP/IP is the fundamental communication protocol of the internet.",
]

_MULTI_SEPARATORS = [". ", " and ", " combined with ", ". Also features "]


# ============================================================
# TEXT LOADING (reuse from flow_2)
# ============================================================

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
    filtered = []
    for s in sentences:
        s_lower = s.lower()
        if any(drop in s_lower for drop in _DROP_PATTERNS):
            continue
        if any(keep in s_lower for keep in _KEEP_PATTERNS):
            filtered.append(s)
    return filtered


def load_raw_texts(descriptions_path, extra_paths=None, filter_relevance=True):
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
                        sents = [s.strip() for s in article.replace("\n", " ").split(". ")
                                 if len(s.strip()) > 30]
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

    for key, sents in list(descriptions.items()):
        if len(sents) < 3:
            class_label = key.replace("_", " ")
            descriptions[key].extend([
                f"Land cover characterized by {class_label}",
                f"Areas of {class_label} as observed from satellite imagery",
                f"Spatial distribution of {class_label} in Hungary",
            ])

    total = sum(len(v) for v in descriptions.values())
    print(f"  Loaded texts: {len(descriptions)} classes, {total} total sentences")
    return descriptions


# ============================================================
# MODEL
# ============================================================

class TextClassifier(nn.Module):
    def __init__(self, n_classes: int = 40, qwen_dim: int = 2560):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.SiLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, text_emb):
        return self.head(text_emb)  # (B, n_classes) raw logits


def merge_maps(probs, density_maps):
    """Clamped sum merge.
    probs: (B, C), density_maps: (C, H, W) → (B, H, W)
    """
    C, H, W = density_maps.shape
    flat = density_maps.view(C, -1)       # (C, H*W)
    out = (probs @ flat).clamp(0, 1)      # (B, H*W)
    return out.view(-1, H, W)


# ============================================================
# DATA PIPELINE
# ============================================================

def build_datasets(pairs, raw_texts, cfg):
    """Build train/val sentence pools with class mappings.

    Returns:
        class_names:  List[str] of length n_classes
        density_maps: (n_classes, H, W) tensor
        train_pool:   List[(sentence, class_idx)]
        val_pool:     List[(sentence, class_idx)]
        texts_by_class: Dict[int, List[str]]  — for synthetic multi-label
    """
    class_names = []
    class_to_idx = {}
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        if desc_key not in class_to_idx:
            class_to_idx[desc_key] = len(class_names)
            class_names.append(p["class_name"])
    n_classes = len(class_names)

    # Stack density maps
    density_maps = torch.stack([
        torch.as_tensor(p["density_map"], dtype=torch.float32) for p in pairs
    ])  # (n_classes, H, W)

    # Build sentence pool
    sentence_pool = []
    texts_by_class = {i: [] for i in range(n_classes)}
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        cls_idx = class_to_idx[desc_key]
        sents = raw_texts.get(desc_key, [f"{p['class_name']} in Hungary"])
        for s in sents:
            sentence_pool.append((s, cls_idx))
            texts_by_class[cls_idx].append(s)

    # Train/val split: hold out val_fraction per class
    rng = random.Random(42)
    train_pool, val_pool = [], []
    for cls_idx in range(n_classes):
        sents = [(s, cls_idx) for s in texts_by_class[cls_idx]]
        rng.shuffle(sents)
        n_val = max(1, int(len(sents) * cfg.val_fraction))
        val_pool.extend(sents[:n_val])
        train_pool.extend(sents[n_val:])

    print(f"  Classes: {n_classes}")
    print(f"  Train sentences: {len(train_pool)}, Val sentences: {len(val_pool)}")
    print(f"  Density maps: {density_maps.shape}, range [{density_maps.min():.3f}, {density_maps.max():.3f}]")

    return class_names, density_maps, train_pool, val_pool, texts_by_class


def sample_batch(train_pool, texts_by_class, n_classes, cfg):
    """Build a mixed batch: single-class + synthetic multi-class + negatives.

    Returns:
        texts:  List[str] of length batch_size
        labels: (batch_size, n_classes) float tensor
    """
    bs = cfg.batch_size
    n_neg = max(1, int(bs * cfg.negative_ratio))
    n_multi = int(bs * cfg.multi_label_ratio)
    n_single = bs - n_multi - n_neg

    texts = []
    labels = torch.zeros(bs, n_classes)

    # Single-class samples
    for i in range(n_single):
        sent, cls_idx = random.choice(train_pool)
        texts.append(sent)
        labels[i, cls_idx] = 1.0

    # Synthetic multi-class samples
    for i in range(n_multi):
        k = random.randint(2, cfg.max_classes_per_sample)
        chosen_classes = random.sample(range(n_classes), min(k, n_classes))
        parts = []
        for cls_idx in chosen_classes:
            pool = texts_by_class[cls_idx]
            parts.append(random.choice(pool) if pool else f"class {cls_idx}")
            labels[n_single + i, cls_idx] = 1.0
        sep = random.choice(_MULTI_SEPARATORS)
        texts.append(sep.join(parts))

    # Negative (OOD) samples
    for i in range(n_neg):
        texts.append(random.choice(_NEGATIVE_SENTENCES))
        # labels row stays all-zeros

    return texts, labels


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_predictions(model, text_encoder, density_maps, class_names,
                          queries, cfg, epoch, viz_dir):
    """Render predictions for a set of test queries."""
    model.eval()
    n = len(queries)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    extent = [cfg.lon_min, cfg.lon_max, cfg.lat_max, cfg.lat_min]

    with torch.no_grad():
        for row, query in enumerate(queries):
            # Encode
            emb = text_encoder.encode_raw(query)
            logits = model(emb)
            probs = torch.sigmoid(logits)  # (1, C)
            merged = merge_maps(probs.to(density_maps.device),
                                density_maps)  # (1, H, W)

            # Top-5 bar chart
            top_vals, top_idx = probs[0].topk(min(5, len(class_names)))
            ax_bar = axes[row, 0]
            names = [class_names[i] for i in top_idx.cpu()]
            vals = top_vals.cpu().numpy()
            ax_bar.barh(range(len(names)), vals, color="steelblue")
            ax_bar.set_yticks(range(len(names)))
            ax_bar.set_yticklabels(names, fontsize=8)
            ax_bar.set_xlim(0, 1)
            ax_bar.set_title(f'"{query[:50]}"', fontsize=9)
            ax_bar.invert_yaxis()

            # Merged map
            ax_map = axes[row, 1]
            ax_map.imshow(merged[0].cpu().numpy(), cmap="YlOrRd",
                          origin="upper", extent=extent, vmin=0, vmax=1)
            ax_map.set_title("merged output", fontsize=9)
            ax_map.axis("off")

    plt.suptitle(f"Epoch {epoch}", fontsize=11)
    plt.tight_layout()
    plt.savefig(viz_dir / f"epoch_{epoch:03d}.png", dpi=120)
    plt.close(fig)
    model.train()


# ============================================================
# TRAINING
# ============================================================

def train(model, text_encoder, density_maps, class_names,
          train_pool, val_pool, texts_by_class, cfg, device="cuda"):
    """Main training loop."""

    n_classes = len(class_names)
    density_maps_dev = density_maps.to(device)

    # Optimizer: LoRA params + classifier head
    qwen_params = [p for p in text_encoder.parameters() if p.requires_grad]
    n_qwen = sum(p.numel() for p in qwen_params)
    n_head = sum(p.numel() for p in model.parameters())
    print(f"\n  Qwen LoRA params: {n_qwen:,}")
    print(f"  Classifier head params: {n_head:,}")

    optimizer = torch.optim.AdamW([
        {"params": qwen_params, "lr": cfg.lr},
        {"params": model.parameters(), "lr": cfg.lr * 10},
    ])

    n_batches = len(train_pool) // cfg.batch_size + 1
    total_steps = n_batches * cfg.n_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6)

    tokenizer = text_encoder._tokenizer
    qwen_device = next(text_encoder.parameters()).device

    viz_dir = Path(cfg.output_dir) / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Test queries for visualization
    test_queries = [
        "wheat",
        "maize",
        "water bodies",
        "deciduous forest",
        "wheat and maize",
        "urban residential areas",
        "grassland near rivers",
        "quantum physics textbook",  # OOD — should produce near-zero
    ]

    loss_fn = nn.BCEWithLogitsLoss()
    train_losses, val_accs = [], []

    # Resume
    ckpt_path = Path(cfg.output_dir) / "checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        print(f"[resume] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["classifier"])
        start_epoch = ckpt.get("epoch", 0) + 1
        train_losses = ckpt.get("train_losses", [])
        val_accs = ckpt.get("val_accs", [])
        print(f"  Resuming from epoch {start_epoch}")

    print(f"\nTraining: {n_classes} classes, {len(train_pool)} train sentences")
    print(f"  {n_batches} batches/epoch × {cfg.batch_size} = ~{n_batches * cfg.batch_size} samples")
    print(f"  Mix: {1 - cfg.multi_label_ratio - cfg.negative_ratio:.0%} single, "
          f"{cfg.multi_label_ratio:.0%} multi, {cfg.negative_ratio:.0%} negative")

    model.train()
    text_encoder.train()

    for epoch in range(start_epoch, cfg.n_epochs):
        random.shuffle(train_pool)
        ep_loss = 0.0
        n_correct, n_total = 0, 0

        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch+1:3d}",
                    leave=False, ncols=100)
        for batch_i in pbar:
            texts, labels = sample_batch(train_pool, texts_by_class,
                                         n_classes, cfg)
            labels = labels.to(qwen_device)

            inputs = tokenizer(
                texts, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            input_ids = inputs["input_ids"].to(qwen_device)
            attn_mask = inputs["attention_mask"].to(qwen_device)

            with torch.autocast(qwen_device.type, dtype=torch.bfloat16):
                embs = text_encoder(input_ids, attn_mask)
                logits = model(embs)
                loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qwen_params, 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            # Accuracy on single-class samples only
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                single_mask = (labels.sum(dim=-1) == 1)
                if single_mask.any():
                    n_correct += (preds[single_mask].argmax(dim=-1) ==
                                  labels[single_mask].argmax(dim=-1)).sum().item()
                    n_total += single_mask.sum().item()

            pbar.set_postfix({
                "loss": f"{ep_loss/(batch_i+1):.4f}",
                "acc": f"{n_correct/max(1,n_total)*100:.1f}%",
            })

        avg_loss = ep_loss / n_batches
        train_acc = n_correct / max(1, n_total) * 100
        train_losses.append(avg_loss)

        # --- Validation ---
        model.eval()
        text_encoder.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for i in range(0, len(val_pool), cfg.batch_size):
                batch = val_pool[i:i + cfg.batch_size]
                texts_v = [s for s, _ in batch]
                labels_v = torch.zeros(len(batch), n_classes, device=qwen_device)
                for j, (_, cls_idx) in enumerate(batch):
                    labels_v[j, cls_idx] = 1.0

                inputs_v = tokenizer(
                    texts_v, return_tensors="pt",
                    truncation=True, max_length=512, padding=True,
                )
                input_ids_v = inputs_v["input_ids"].to(qwen_device)
                attn_mask_v = inputs_v["attention_mask"].to(qwen_device)

                with torch.autocast(qwen_device.type, dtype=torch.bfloat16):
                    embs_v = text_encoder(input_ids_v, attn_mask_v)
                    logits_v = model(embs_v)

                preds_v = logits_v.argmax(dim=-1)
                targets_v = labels_v.argmax(dim=-1)
                v_correct += (preds_v == targets_v).sum().item()
                v_total += len(batch)

        val_acc = v_correct / max(1, v_total) * 100
        val_accs.append(val_acc)
        model.train()
        text_encoder.train()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: loss={avg_loss:.4f}, "
                  f"train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")

        # --- Visualization + checkpoint ---
        if cfg.plot_every > 0 and ((epoch + 1) % cfg.plot_every == 0 or epoch == 0):
            visualize_predictions(
                model, text_encoder, density_maps_dev, class_names,
                test_queries, cfg, epoch + 1, viz_dir)

            # Loss + accuracy plot
            fig, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(train_losses, label="train loss", color="coral")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("BCE Loss")
            ax1.legend(loc="upper left")
            ax2 = ax1.twinx()
            ax2.plot(val_accs, label="val acc", color="steelblue", linestyle="--")
            ax2.set_ylabel("Accuracy %")
            ax2.legend(loc="upper right")
            plt.title("Training Progress")
            plt.tight_layout()
            plt.savefig(viz_dir / "progress.png", dpi=120)
            plt.close(fig)

            # Checkpoint
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "classifier": model.state_dict(),
                "train_losses": train_losses,
                "val_accs": val_accs,
                "class_names": class_names,
                "config": vars(cfg),
            }, ckpt_path)

    # Final save
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    torch.save({
        "classifier": model.state_dict(),
        "class_names": class_names,
        "density_maps": density_maps.cpu(),
        "config": vars(cfg),
    }, Path(cfg.output_dir) / "flow3_model.pt")
    print(f"\n  Saved to {Path(cfg.output_dir) / 'flow3_model.pt'}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = Flow3Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Step 1: Load cached pairs ---
    print("\n" + "=" * 60)
    print("STEP 1: Load cached training pairs")
    print("=" * 60)
    pairs = torch.load(cfg.pairs_cache, map_location="cpu", weights_only=False)
    print(f"  {len(pairs)} pairs loaded")

    # --- Step 2: Load raw texts ---
    print("\n" + "=" * 60)
    print("STEP 2: Load text descriptions")
    print("=" * 60)
    raw_texts = load_raw_texts(
        cfg.text_descriptions_path,
        extra_paths=[cfg.hrl_descriptions_path],
    )

    # --- Step 3: Build datasets ---
    print("\n" + "=" * 60)
    print("STEP 3: Build train/val datasets")
    print("=" * 60)
    class_names, density_maps, train_pool, val_pool, texts_by_class = \
        build_datasets(pairs, raw_texts, cfg)

    # --- Step 4: Load Qwen ---
    print("\n" + "=" * 60)
    print("STEP 4: Load Qwen3 text encoder")
    print("=" * 60)
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
    text_encoder = Qwen3EmbeddingAdapter(
        target_dim=cfg.qwen_emb_dim,
        freeze_encoder=True,    # freeze base, LoRA adapters are trainable
        lora=True,
    )
    text_encoder = text_encoder.to(device)
    print(f"  Qwen3 loaded on {device}")
    n_lora = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    print(f"  LoRA trainable params: {n_lora:,}")

    # --- Step 5: Train classifier ---
    print("\n" + "=" * 60)
    print("STEP 5: Train multi-label classifier")
    print("=" * 60)
    n_classes = len(class_names)
    model = TextClassifier(n_classes=n_classes, qwen_dim=cfg.qwen_emb_dim)
    model = model.to(device)

    train(model, text_encoder, density_maps, class_names,
          train_pool, val_pool, texts_by_class, cfg, device=device)

    print("\nDone.")


if __name__ == "__main__":
    main()
