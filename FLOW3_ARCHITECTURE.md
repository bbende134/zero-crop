# Flow 3 — Classify, Lookup, Merge

## Overview

Flow 3 is a zero-shot land cover mapping system for Hungary. Given an arbitrary text query (e.g. "sunflower fields near the Danube"), it produces a 256x609 density map showing where that land cover type likely occurs.

The core insight: instead of learning a spatial field from scratch (as flow_2 does), flow 3 treats the problem as **multi-label classification over a fixed set of 40 known land cover classes**, then merges their pre-computed ground-truth density maps.

```
                                    ┌──────────────┐
                                    │  40 frozen    │
"sunflower fields" ──► Qwen3 ──► Classifier ──► │ density maps │ ──► merged output
                      (LoRA)     (MLP head)      │ (256 x 609)  │
                                    └──────────────┘
```

## Components

### 1. Text Encoder — `Qwen3EmbeddingAdapter`

**File:** `fine_tune/qwen3_adapter.py`

The text encoder converts natural language into a fixed-dimensional embedding vector. It wraps a pretrained Qwen3 causal language model and repurposes it as a text encoder via last-token pooling.

**Architecture:**
- Base model: Qwen3-14B (or any Qwen3/3.5 variant) loaded in bf16
- Pooling: last non-pad token's hidden state (similar to how GPT-style models use the final token for classification)
- Output: `(B, hidden_dim)` — 5120-d for Qwen3-14B, 2560-d for Qwen3.5-4B

**LoRA fine-tuning:**
- The base model weights are frozen; only LoRA adapter layers are trainable
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` (all attention projections)
- Default rank: 32 (multi-GPU) or 16 (single GPU), alpha: 64/32
- Gradient checkpointing enabled to reduce activation memory
- Total trainable params: ~40-80M out of 14B+ total (~0.25%)

**Why LoRA and not frozen embeddings?**
A frozen encoder produces generic text representations. LoRA adapts the attention layers so that semantically similar land cover descriptions (e.g. "wheat fields in the lowlands" and "cereal crops on the Great Hungarian Plain") produce similar embeddings, while dissimilar ones (e.g. "wheat" vs "deciduous forest") are pushed apart. This adaptation is critical for the downstream classifier to generalize beyond exact training sentences.

**Multi-GPU support:**
- `device_map='auto'` splits transformer layers across available GPUs
- Custom `.to()` override prevents moving the already-placed model
- `input_device` property returns the correct GPU for input tensors

### 2. Classifier Head — `TextClassifier`

**File:** `flow_3.py`, line 180

A lightweight MLP that maps text embeddings to 40 class logits.

```
LayerNorm(hidden_dim)
  → Linear(hidden_dim, 512) → SiLU → Dropout(0.1)
  → Linear(512, 128) → SiLU
  → Linear(128, 40)
```

**Design choices:**
- **LayerNorm first:** normalizes the varying-magnitude embeddings from different Qwen model sizes
- **Bottleneck (512 → 128 → 40):** prevents overfitting given the small number of classes; the 128-d bottleneck forces the model to learn compressed class-relevant features
- **SiLU activation:** smooth non-linearity, matches Qwen's internal activation function
- **Dropout only in first layer:** regularization where the representation is richest; later layers are too narrow to benefit
- **Raw logits output:** BCE loss with logits is numerically more stable than sigmoid + BCE

**Parameter count:** ~2.7M (tiny compared to the encoder)

### 3. Density Maps — Frozen Lookup Table

**Source:** pre-computed in flow_2's pipeline, cached in `pipeline_cache/pairs_*.pt`

Each of the 40 land cover classes has a ground-truth density map of shape `(256, 609)` covering Hungary. These maps are derived from:
- **HRL (High Resolution Layer) crops:** wheat, barley, maize, sunflower, etc. — 20 classes
- **CORINE land cover:** forests, urban areas, water bodies, etc. — 20 classes

The maps are normalized to [0, 1] and represent the spatial probability of finding that land cover type at each grid cell. They are **never modified during training** — they serve as a fixed lookup table.

### 4. Merge Function — `merge_maps`

```python
def merge_maps(probs, density_maps):
    out = (probs @ density_maps.view(C, -1)).clamp(0, 1)
    return out.view(-1, H, W)
```

A simple weighted sum: each class's density map is scaled by its predicted probability, then summed and clamped to [0, 1].

**Why clamped sum and not softmax-weighted?**
- Softmax would force the model to distribute probability mass, making multi-label queries (e.g. "wheat and maize") produce diluted maps
- Clamped sum allows overlapping classes to reinforce each other in regions where they co-occur
- The clamp prevents values > 1 when many classes overlap

## Training

### Data Pipeline

**Text sources:** Wikipedia-derived sentences for each class, filtered for geographic/botanical relevance via keyword matching. ~8,300 training sentences across 40 classes.

**Batch composition** (each batch of 32):
- **45% single-class** — one sentence from one class, label is a one-hot vector
- **50% synthetic multi-label** — 2-4 class sentences concatenated with random separators (". ", " and ", " combined with ", ". Also features "), label is multi-hot
- **5% OOD negatives** — unrelated sentences (quantum mechanics, Shakespeare, etc.), label is all-zeros

**Why this mix matters:**
- Single-class teaches the model to recognize individual land cover types
- Multi-label teaches compositionality — the model must activate multiple classes simultaneously
- Negatives teach the model to output near-zero for irrelevant queries, preventing false positives on arbitrary text

### Loss and Optimization

- **Loss:** `BCEWithLogitsLoss` — binary cross-entropy per class, treating each class as an independent binary decision. This is the natural loss for multi-label classification.
- **Optimizer:** AdamW with two learning rate groups:
  - LoRA params: `lr` (1e-4)
  - Classifier head: `lr × 10` (1e-3) — the head needs to learn faster since it starts from random initialization while LoRA starts near the pretrained solution
- **Schedule:** Cosine annealing over all steps to `eta_min=1e-6`
- **Gradient accumulation:** 4 steps (effective batch size = 128)
- **Gradient clipping:** max norm 1.0 on LoRA params to prevent instability

### Checkpointing

Every epoch saves:
- Classifier head state dict
- LoRA weight tensors (filtered by `"lora" in key_name`)
- Optimizer + scheduler state (for clean resume)
- Training metrics history

**Previous bug (now fixed):** LoRA weights were not saved in earlier versions, causing a train/inference mismatch where the classifier expected LoRA-modified embeddings but received base model embeddings.

## Inference

**File:** `inference_flow3.py`

```
Query text → Qwen3 (+ LoRA weights from checkpoint)
           → Classifier head → sigmoid → 40 probabilities
           → weighted sum of density maps → clamped [0, 1]
           → rendered map of Hungary
```

For `--all-classes` mode, Qwen is not needed — each class's raw density map is displayed directly.

## Comparison with Flow 2

| Aspect | Flow 2 | Flow 3 |
|--------|--------|--------|
| Spatial model | Learned SpatialBasisFieldV2 (FiLM-conditioned) | None — frozen density map lookup |
| Text conditioning | 66-d (sat centroid + phenology) | Full Qwen hidden dim (5120-d) |
| Zero-shot path | Qwen → Bridge → 66-d → spatial field | Qwen → Classifier → merge maps |
| Training signal | Pixel-level density reconstruction | Class-level BCE |
| Novel queries | Generates new spatial patterns | Can only combine known class maps |
| Training cost | 400 epochs, 2M samples/epoch | 10 epochs, ~8K sentences |

**Key trade-off:** Flow 3 cannot generate spatial patterns for classes it hasn't seen — it can only mix existing ones. But it trains orders of magnitude faster and produces cleaner maps for known classes, since the ground-truth density maps are used directly rather than approximated by a learned field.
