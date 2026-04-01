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
    query_aug_path: str = "data_corine/query_augmentations.jsonl"
    output_dir: str = "training_data_flow3"

    # Training
    n_epochs: int = 30
    lr: float = 1e-4
    batch_size: int = 32
    grad_accum_steps: int = 4       # effective batch = batch_size * grad_accum_steps
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
    qwen_model_id: str = "Qwen/Qwen3.5-4B"
    qwen_emb_dim: int = 0  # 0 = auto-detect from model


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

    # Build sentence pool — separate wiki and query-style sentences
    # Short query-style sentences (< 50 chars) are oversampled 3x
    sentence_pool = []
    texts_by_class = {i: [] for i in range(n_classes)}
    n_query_style = 0
    for p in pairs:
        desc_key = p.get("desc_key", p["class_name"])
        cls_idx = class_to_idx[desc_key]
        sents = raw_texts.get(desc_key, [f"{p['class_name']} in Hungary"])
        for s in sents:
            sentence_pool.append((s, cls_idx))
            texts_by_class[cls_idx].append(s)
            # Oversample short query-style sentences (< 50 chars)
            if len(s) < 50:
                for _ in range(2):  # 3x total
                    sentence_pool.append((s, cls_idx))
                    texts_by_class[cls_idx].append(s)
                n_query_style += 1

    print(f"  Short query-style sentences oversampled: {n_query_style} (3x each)")

    # Train/val split: hold out val_fraction per class
    # Val includes both wiki and query-style sentences for realistic eval
    rng = random.Random(42)
    train_pool, val_pool = [], []
    for cls_idx in range(n_classes):
        # Deduplicate for val split to avoid inflated metrics
        unique_sents = list(set(texts_by_class[cls_idx]))
        rng.shuffle(unique_sents)
        n_val = max(1, int(len(unique_sents) * cfg.val_fraction))
        val_set = set(unique_sents[:n_val])
        val_pool.extend([(s, cls_idx) for s in unique_sents[:n_val]])
        # Train keeps oversampled duplicates (minus val sentences)
        train_pool.extend([(s, cls_idx) for s in texts_by_class[cls_idx]
                           if s not in val_set])

    print(f"  Classes: {n_classes}")
    print(f"  Train sentences: {len(train_pool)}, Val sentences: {len(val_pool)}")
    print(f"  Density maps: {density_maps.shape}, range [{density_maps.min():.3f}, {density_maps.max():.3f}]")

    return class_names, density_maps, train_pool, val_pool, texts_by_class


def _make_short_query(sentence):
    """Extract a short 1-5 word query from a sentence (on-the-fly augmentation)."""
    words = sentence.split()
    if len(words) <= 2:
        return sentence
    n = random.randint(1, min(5, len(words)))
    start = random.randint(0, len(words) - n)
    return " ".join(words[start:start + n])


def sample_batch(train_pool, texts_by_class, n_classes, cfg):
    """Build a mixed batch: single-class + synthetic multi-class + negatives.

    ~20% of single-class samples are converted to short keyword queries
    to bridge the gap between wiki-style training text and real user queries.

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

    # Single-class samples (20% converted to short keyword queries)
    for i in range(n_single):
        sent, cls_idx = random.choice(train_pool)
        if random.random() < 0.2:
            sent = _make_short_query(sent)
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
            logits = model(emb.to(next(model.parameters()).device))
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
    qwen_device = text_encoder.input_device

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
    train_losses, train_accs, val_losses, val_accs = [], [], [], []

    # Resume
    ckpt_path = Path(cfg.output_dir) / "checkpoint.pt"
    start_epoch = 0
    if ckpt_path.exists():
        print(f"[resume] Loading {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["classifier"])
        # Restore LoRA weights
        lora_state = ckpt.get("qwen_lora", {})
        if lora_state:
            text_encoder.load_state_dict(lora_state, strict=False)
            print(f"  Restored {len(lora_state)} LoRA weight tensors")
        else:
            print("  WARNING: no LoRA weights in checkpoint (pre-fix checkpoint)")
        start_epoch = ckpt.get("epoch", 0) + 1
        train_losses = ckpt.get("train_losses", [])
        train_accs = ckpt.get("train_accs", [])
        val_losses = ckpt.get("val_losses", [])
        val_accs = ckpt.get("val_accs", [])
        # Restore optimizer state
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print(f"  Restored optimizer state")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"  Restored scheduler state")
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
            labels = labels.to(device)

            inputs = tokenizer(
                texts, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            input_ids = inputs["input_ids"].to(qwen_device)
            attn_mask = inputs["attention_mask"].to(qwen_device)

            with torch.autocast(qwen_device.type, dtype=torch.bfloat16):
                embs = text_encoder(input_ids, attn_mask)
                logits = model(embs.to(device))
                loss = loss_fn(logits, labels.to(logits.device)) / cfg.grad_accum_steps

            loss.backward()

            if (batch_i + 1) % cfg.grad_accum_steps == 0 or batch_i == n_batches - 1:
                torch.nn.utils.clip_grad_norm_(qwen_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            ep_loss += loss.item() * cfg.grad_accum_steps
            # Accuracy on single-class samples only
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                labels_dev = labels.to(logits.device)
                single_mask = (labels_dev.sum(dim=-1) == 1)
                if single_mask.any():
                    n_correct += (preds[single_mask].argmax(dim=-1) ==
                                  labels_dev[single_mask].argmax(dim=-1)).sum().item()
                    n_total += single_mask.sum().item()

            pbar.set_postfix({
                "loss": f"{ep_loss/(batch_i+1):.4f}",
                "acc": f"{n_correct/max(1,n_total)*100:.1f}%",
            })

        avg_loss = ep_loss / n_batches
        train_acc = n_correct / max(1, n_total) * 100
        train_losses.append(avg_loss)
        train_accs.append(train_acc)

        # --- Validation ---
        model.eval()
        text_encoder.eval()
        v_correct, v_total = 0, 0
        v_loss_sum, v_batches = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(val_pool), cfg.batch_size):
                batch = val_pool[i:i + cfg.batch_size]
                texts_v = [s for s, _ in batch]
                labels_v = torch.zeros(len(batch), n_classes, device=device)
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
                    logits_v = model(embs_v.to(device))
                    v_loss_sum += loss_fn(logits_v, labels_v.to(logits_v.device)).item()
                    v_batches += 1

                preds_v = logits_v.argmax(dim=-1)
                targets_v = labels_v.to(logits_v.device).argmax(dim=-1)
                v_correct += (preds_v == targets_v).sum().item()
                v_total += len(batch)

        val_loss = v_loss_sum / max(1, v_batches)
        val_acc = v_correct / max(1, v_total) * 100
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        model.train()
        text_encoder.train()

        print(f"Epoch {epoch+1:3d}: train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
              f"train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%")

        # --- Loss + accuracy plot (every epoch) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_x = list(range(1, len(train_losses) + 1))

        ax1.plot(epochs_x, train_losses, label="train", color="coral", marker="o", markersize=3)
        ax1.plot(epochs_x, val_losses, label="val", color="steelblue", marker="o", markersize=3)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("BCE Loss")
        ax1.set_title("Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_x, train_accs, label="train", color="coral", marker="o", markersize=3)
        ax2.plot(epochs_x, val_accs, label="val", color="steelblue", marker="o", markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy %")
        ax2.set_title("Accuracy")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("Training Progress")
        plt.tight_layout()
        plt.savefig(viz_dir / "progress.png", dpi=120)
        plt.close(fig)

        # --- Visualization (every plot_every epochs) ---
        if cfg.plot_every > 0 and ((epoch + 1) % cfg.plot_every == 0 or epoch == 0):
            visualize_predictions(
                model, text_encoder, density_maps_dev, class_names,
                test_queries, cfg, epoch + 1, viz_dir)

        # --- Checkpoint (every epoch) ---
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        lora_state = {k: v.cpu() for k, v in text_encoder.state_dict().items()
                      if "lora" in k.lower()}
        torch.save({
            "epoch": epoch,
            "classifier": model.state_dict(),
            "qwen_lora": lora_state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_losses": train_losses,
            "train_accs": train_accs,
            "val_losses": val_losses,
            "val_accs": val_accs,
            "class_names": class_names,
            "config": vars(cfg),
        }, ckpt_path)

    # Final save
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    lora_state = {k: v.cpu() for k, v in text_encoder.state_dict().items()
                  if "lora" in k.lower()}
    torch.save({
        "classifier": model.state_dict(),
        "qwen_lora": lora_state,
        "class_names": class_names,
        "density_maps": density_maps.cpu(),
        "config": vars(cfg),
    }, Path(cfg.output_dir) / "flow3_model.pt")
    print(f"\n  Saved to {Path(cfg.output_dir) / 'flow3_model.pt'}")


# ============================================================
# MAIN
# ============================================================

def predict(text, model, text_encoder, density_maps, class_names, top_k=5):
    """Run inference on a single text query.

    Returns:
        merged: (H, W) density map
        top_classes: list of (class_name, probability) tuples
        all_probs: (C,) tensor of all class probabilities
    """
    model.eval()
    text_encoder.eval()
    with torch.no_grad():
        emb = text_encoder.encode_raw(text)
        logits = model(emb.to(next(model.parameters()).device))
        probs = torch.sigmoid(logits)  # (1, C)
        merged = merge_maps(probs.to(density_maps.device), density_maps)  # (1, H, W)
        top_vals, top_idx = probs[0].topk(min(top_k, len(class_names)))
        top_classes = [(class_names[i], p.item()) for i, p in zip(top_idx.cpu(), top_vals)]
    return merged[0], top_classes, probs[0]


def load_for_inference(checkpoint_path, device="cuda", multi_gpu=False):
    """Load a trained flow_3 model for inference.

    Args:
        checkpoint_path: path to flow3_model.pt or checkpoint.pt
        device: torch device
        multi_gpu: split Qwen across GPUs

    Returns:
        model, text_encoder, density_maps, class_names, cfg
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    cfg = Flow3Config(**{k: v for k, v in config.items()
                         if k in Flow3Config.__dataclass_fields__})
    class_names = ckpt["class_names"]

    # Density maps: in final save or need to load from pairs cache
    if "density_maps" in ckpt:
        density_maps = ckpt["density_maps"].to(device)
    else:
        pairs = torch.load(cfg.pairs_cache, map_location="cpu", weights_only=False)
        density_maps = torch.stack([p["density_map"] for p in pairs]).to(device)

    # Detect LoRA rank from checkpoint weights
    lora_state = ckpt.get("qwen_lora", {})
    lora_r = 16  # default
    for k, v in lora_state.items():
        if "lora_A" in k:
            lora_r = v.shape[0]
            break
    lora_alpha = lora_r * 2
    print(f"  Detected LoRA rank from checkpoint: r={lora_r}, alpha={lora_alpha}")

    # Qwen encoder
    from fine_tune.qwen3_adapter import Qwen3EmbeddingAdapter
    text_encoder = Qwen3EmbeddingAdapter(
        model_id=cfg.qwen_model_id,
        freeze_encoder=True,
        lora=True,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        multi_gpu=multi_gpu,
    )
    print(f"  Adapter created with lora_r={lora_r}")
    # Restore LoRA weights
    if lora_state:
        text_encoder.load_state_dict(lora_state, strict=False)
        print(f"  Restored {len(lora_state)} LoRA weight tensors (r={lora_r})")
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    # Classifier head
    n_classes = len(class_names)
    qwen_dim = cfg.qwen_emb_dim or text_encoder.target_dim
    model = TextClassifier(n_classes=n_classes, qwen_dim=qwen_dim)
    model.load_state_dict(ckpt["classifier"])
    model = model.to(device)
    model.eval()

    print(f"Loaded flow_3 model: {n_classes} classes, maps {density_maps.shape}")
    return model, text_encoder, density_maps, class_names, cfg


def render_query(query, model, text_encoder, density_maps, class_names, cfg,
                 viz_dir=None, show_in_terminal=True):
    """Process a single query: print results, show map in terminal, optionally save."""
    import subprocess
    import tempfile

    merged, top_classes, _ = predict(
        query, model, text_encoder, density_maps, class_names, top_k=10)

    print(f"\n  Top classes:")
    for name, prob in top_classes:
        bar = "█" * int(prob * 30)
        print(f"    {prob:.3f} {bar} {name}")

    # Render map image
    extent = [cfg.lon_min, cfg.lon_max, cfg.lat_max, cfg.lat_min]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    names = [n for n, _ in top_classes[:10]]
    vals = [p for _, p in top_classes[:10]]
    ax1.barh(names[::-1], vals[::-1], color="steelblue")
    ax1.set_xlim(0, 1)
    ax1.set_title("Class probabilities")

    ax2.imshow(merged.cpu().numpy(), extent=extent,
               cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax2.set_title("Merged density map")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")

    slug = query[:40].replace(" ", "_").replace("/", "_")
    fig.suptitle(f'"{query}"', fontsize=12)
    plt.tight_layout()

    # Save to viz_dir if requested
    if viz_dir:
        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        save_path = viz_dir / f"infer_{slug}.png"
        plt.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")

    # Display in terminal via chafa
    if show_in_terminal:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
            plt.savefig(tmp.name, dpi=150)
            plt.close(fig)
            try:
                subprocess.run(["chafa", "--size=120x30", tmp.name], check=True)
            except FileNotFoundError:
                print("  (install chafa for inline terminal images)")
    else:
        plt.close(fig)


def run_inference(args):
    """Interactive CLI chat mode. Load model once, then accept queries in a loop."""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model, text_encoder, density_maps, class_names, cfg = \
        load_for_inference(args.checkpoint, device=device, multi_gpu=args.multi_gpu)

    viz_dir = args.save_viz

    # If queries provided on command line, run those and exit
    if args.query:
        for query in args.query:
            print(f"\nQuery: \"{query}\"")
            render_query(query, model, text_encoder, density_maps, class_names, cfg, viz_dir)
        return

    # Interactive mode
    print(f"\nReady. Type a query and press Enter. Commands: :quit, :viz <dir>")
    print(f"  Saving viz to: {viz_dir or '(off, use :viz <dir> to enable)'}")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query in (":quit", ":q", ":exit"):
            break
        if query.startswith(":viz "):
            viz_dir = query[5:].strip() or None
            print(f"  Viz dir: {viz_dir or '(off)'}")
            continue
        if query == ":viz":
            viz_dir = None
            print("  Viz disabled")
            continue
        if query == ":classes":
            for i, name in enumerate(class_names):
                print(f"  {i:2d}  {name}")
            continue

        render_query(query, model, text_encoder, density_maps, class_names, cfg, viz_dir)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    # --- train mode (default) ---
    train_p = sub.add_parser("train", help="Train classifier")
    train_p.add_argument("--device", default=None)
    train_p.add_argument("--output-dir", default=None)
    train_p.add_argument("--model", default=None,
                         help="Qwen model ID, e.g. Qwen/Qwen3.5-27B")
    train_p.add_argument("--multi-gpu", action="store_true")

    # --- infer mode ---
    infer_p = sub.add_parser("infer", help="Run inference on text queries")
    infer_p.add_argument("query", nargs="*", help="Text queries (omit for interactive mode)")
    infer_p.add_argument("--checkpoint", required=True,
                         help="Path to flow3_model.pt or checkpoint.pt")
    infer_p.add_argument("--device", default=None)
    infer_p.add_argument("--multi-gpu", action="store_true")
    infer_p.add_argument("--save-viz", default=None,
                         help="Directory to save visualization PNGs")

    args = parser.parse_args()

    # Default to train if no subcommand
    if args.mode == "infer":
        run_inference(args)
        return

    # --- Training ---
    if args.mode is None:
        # Backwards compat: no subcommand = train
        args.device = getattr(args, "device", None)
        args.output_dir = getattr(args, "output_dir", None)
        args.model = getattr(args, "model", None)
        args.multi_gpu = getattr(args, "multi_gpu", False)

    cfg = Flow3Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.model:
        cfg.qwen_model_id = args.model

    multi_gpu = args.multi_gpu
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, multi_gpu: {multi_gpu}")

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
        extra_paths=[cfg.hrl_descriptions_path, cfg.query_aug_path],
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
        model_id=cfg.qwen_model_id,
        freeze_encoder=True,    # freeze base, LoRA adapters are trainable
        lora=True,
        lora_r=32 if multi_gpu else 16,
        lora_alpha=64 if multi_gpu else 32,
        multi_gpu=multi_gpu,
    )
    text_encoder = text_encoder.to(device)
    # Auto-detect embedding dim from loaded model
    cfg.qwen_emb_dim = text_encoder.target_dim
    n_lora = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    print(f"  Qwen3 loaded, emb_dim={cfg.qwen_emb_dim}")
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
