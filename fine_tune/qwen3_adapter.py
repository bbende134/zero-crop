"""
Qwen3EmbeddingAdapter
=====================
Wraps Qwen/Qwen3.5-4B as a text encoder via last-token pooling.
Output dim = 2560 (matches Qwen3.5-4B hidden size).

Supports:
  - Frozen mode (precompute embeddings)
  - LoRA fine-tuning (trainable adapter layers)
  - Batch encoding with gradients (for fine-tuning)

Usage:
    # Frozen (for precomputing)
    encoder = Qwen3EmbeddingAdapter(freeze_encoder=True).eval()
    emb = encoder.encode_raw("Winter wheat fields.", normalize=False)

    # LoRA fine-tune
    encoder = Qwen3EmbeddingAdapter(freeze_encoder=False, lora=True)
    embs = encoder.encode_batch(["wheat", "barley"])  # (2, 2560) with grads
"""

import torch
import torch.nn as nn
from typing import Optional, List


MODEL_ID = "Qwen/Qwen3.5-4B"
HIDDEN_DIM = 2560



class Qwen3EmbeddingAdapter(nn.Module):
    def __init__(
        self,
        target_dim: int = HIDDEN_DIM,
        pretrained_encoder_path: Optional[str] = None,
        model_id: Optional[str] = None,
        freeze_encoder: bool = True,
        lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        multi_gpu: bool = False,
    ):
        super().__init__()
        self.freeze_encoder = freeze_encoder
        self._multi_gpu = multi_gpu
        resolved_model_id = model_id or pretrained_encoder_path or MODEL_ID

        from transformers import AutoTokenizer, AutoModel
        print(f"[Qwen3EmbeddingAdapter] Loading {resolved_model_id}...")
        self._tokenizer = AutoTokenizer.from_pretrained(resolved_model_id, trust_remote_code=True)

        load_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        if multi_gpu:
            load_kwargs["device_map"] = "auto"
            print(f"[Qwen3EmbeddingAdapter] Using device_map='auto' (multi-GPU)")

        self._model = AutoModel.from_pretrained(resolved_model_id, **load_kwargs)

        self._is_fp8 = any(p.dtype == torch.float8_e4m3fn for p in self._model.parameters())

        cfg = self._model.config
        raw_dim = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size
        self.hidden_dim = raw_dim
        self.target_dim = target_dim if target_dim != HIDDEN_DIM else raw_dim
        print(f"[Qwen3EmbeddingAdapter] hidden_dim={raw_dim}, target_dim={self.target_dim}")

        self._projection: Optional[nn.Linear] = None
        if raw_dim != self.target_dim:
            self._projection = nn.Linear(raw_dim, self.target_dim, bias=False)

        if freeze_encoder and not lora:
            for p in self._model.parameters():
                p.requires_grad_(False)

        if lora:
            from peft import get_peft_model, LoraConfig, TaskType
            # FP8 models can't do dropout on quantized tensors
            if self._is_fp8 and lora_dropout > 0:
                print(f"[Qwen3EmbeddingAdapter] FP8 detected — forcing lora_dropout=0.0")
                lora_dropout = 0.0
            print(f"[Qwen3EmbeddingAdapter] Applying LoRA (r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout})")
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self._model = get_peft_model(self._model, lora_config)
            self._model.print_trainable_parameters()
            # Gradient checkpointing: trade compute for memory
            # Skip for FP8 — recomputation triggers ufunc_add on FP8 tensors
            if self._is_fp8:
                print(f"[Qwen3EmbeddingAdapter] FP8 model — skipping gradient checkpointing")
            else:
                self._model.gradient_checkpointing_enable()
                print(f"[Qwen3EmbeddingAdapter] Gradient checkpointing enabled")

    @property
    def tokenizer(self):
        return self._tokenizer

    def to(self, *args, **kwargs):
        """Skip .to(device) when using device_map='auto' (model already placed)."""
        if self._multi_gpu:
            # Only move projection layer if it exists
            if self._projection is not None:
                self._projection = self._projection.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)

    @property
    def input_device(self):
        """Device where input tensors should be placed."""
        if self._multi_gpu:
            # With device_map, first layer's device is the input device
            return next(self._model.parameters()).device
        return next(self.parameters()).device

    def _last_token_pool(self, last_hidden_state, attention_mask):
        """Pool by selecting the last non-pad token per sequence."""
        seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        emb = last_hidden_state[batch_idx, seq_lens]  # (B, dim)
        return emb

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass on tokenized tensors. DataParallel splits these across GPUs.

        Args:
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) attention mask

        Returns: (B, target_dim) embeddings
        """
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        emb = self._last_token_pool(out.last_hidden_state, attention_mask)
        if self._projection is not None:
            emb = self._projection(emb)
        return emb

    def encode_raw(self, text: str, normalize: bool = False) -> torch.Tensor:
        """Encode a single string. Returns (1, target_dim) tensor. No gradients."""
        device = self.input_device
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            emb = self.forward(input_ids, attn_mask)
            if normalize:
                emb = nn.functional.normalize(emb, dim=-1)
        return emb.float()

    def encode_batch(self, texts: List[str], chunk_size: int = 16,
                     normalize: bool = False) -> torch.Tensor:
        """Tokenize + encode a batch of strings, chunked to limit VRAM.

        Args:
            texts: list of strings to encode
            chunk_size: max sequences per forward pass (limits VRAM usage)
            normalize: whether to L2-normalize the output

        Returns: (B, target_dim) embeddings with gradients if unfrozen
        """
        device = self.input_device
        all_embs = []
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            inputs = self._tokenizer(
                chunk_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            )
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)
            emb = self.forward(input_ids, attn_mask)  # goes through DataParallel
            if normalize:
                emb = nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb)
        return torch.cat(all_embs, dim=0)
