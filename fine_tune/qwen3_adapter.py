"""
Qwen3EmbeddingAdapter
=====================
Wraps a Qwen model as a text encoder via last-token pooling.

Supports:
  - bfloat16 full-precision loading (smaller models, e.g. 9B)
  - 4-bit QLoRA loading (large models, e.g. 35B) via bitsandbytes NF4
  - Frozen mode (precompute embeddings)
  - LoRA / QLoRA fine-tuning (trainable adapter layers)
  - Batch encoding with gradients (for fine-tuning)

Usage:
    # Frozen bfloat16 (precomputing with 9B)
    encoder = Qwen3EmbeddingAdapter(freeze_encoder=True).eval()
    emb = encoder.encode_raw("Winter wheat fields.", normalize=False)

    # QLoRA fine-tune (35B)
    encoder = Qwen3EmbeddingAdapter(
        pretrained_encoder_path="Qwen/Qwen3.5-35B",
        use_4bit=True, lora=True,
    )
    embs = encoder.encode_batch(["wheat", "barley"])
"""

import torch
import torch.nn as nn
from typing import Optional, List


MODEL_ID = "Qwen/Qwen3.5-35B"
TARGET_DIM = 4096   # adapter output dim (projection added if model hidden ≠ this)


class Qwen3EmbeddingAdapter(nn.Module):
    def __init__(
        self,
        target_dim: int = TARGET_DIM,
        pretrained_encoder_path: Optional[str] = None,
        freeze_encoder: bool = True,
        use_4bit: bool = False,
        lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.target_dim = target_dim
        self.freeze_encoder = freeze_encoder
        self._use_4bit = use_4bit
        model_id = pretrained_encoder_path or MODEL_ID

        from transformers import AutoTokenizer, AutoModel, AutoConfig
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # Check whether the model ships with its own quantization config (e.g. FP8).
        # In that case we must not pass a conflicting torch_dtype or BitsAndBytesConfig.
        _model_cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        _has_builtin_quant = getattr(_model_cfg, "quantization_config", None) is not None

        if use_4bit and _has_builtin_quant:
            raise ValueError(
                f"{model_id} already has a built-in quantization config "
                f"({_model_cfg.quantization_config.get('quant_type', 'unknown')}). "
                "Set use_4bit=False — the model handles its own quantization."
            )

        print(f"[Qwen3EmbeddingAdapter] Loading {model_id} "
              f"({'4-bit NF4' if use_4bit else 'built-in FP8' if _has_builtin_quant else 'bfloat16'})...")

        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self._model = AutoModel.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        elif _has_builtin_quant:
            # FP8 or other pre-quantized model: the built-in quantizer handles weight
            # storage, but we must override torch_dtype to bfloat16 because PyTorch
            # can't set FP8 as the default dtype (used during model init).
            self._model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self._model = AutoModel.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        cfg = self._model.config
        raw_dim = getattr(cfg, "hidden_size", None) or cfg.text_config.hidden_size

        self._projection: Optional[nn.Linear] = None
        if raw_dim != target_dim:
            self._projection = nn.Linear(raw_dim, target_dim, bias=False)

        if freeze_encoder and not lora:
            for p in self._model.parameters():
                p.requires_grad_(False)

        if lora:
            from peft import get_peft_model, LoraConfig, TaskType
            print(f"[Qwen3EmbeddingAdapter] Applying {'Q' if use_4bit else ''}LoRA (r={lora_r}, alpha={lora_alpha})")

            if use_4bit:
                from peft import prepare_model_for_kbit_training
                self._model = prepare_model_for_kbit_training(self._model)

            # FP8/quantized tensors don't support fused_dropout — disable it
            effective_dropout = 0.0 if (use_4bit or _has_builtin_quant) else lora_dropout
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=effective_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self._model = get_peft_model(self._model, lora_config)
            self._model.print_trainable_parameters()
            # Gradient checkpointing: trade compute for memory.
            # Must disable KV cache — incompatible with gradient checkpointing.
            self._model.config.use_cache = False
            self._model.gradient_checkpointing_enable()
            print(f"[Qwen3EmbeddingAdapter] Gradient checkpointing enabled")

    def to(self, *args, **kwargs):
        """All loading paths use device_map='auto', so the backbone is device-bound.
        Only move the projection layer; never try to move the backbone."""
        if self._projection is not None:
            self._projection = self._projection.to(*args, **kwargs)
        return self

    @property
    def tokenizer(self):
        return self._tokenizer

    def _last_token_pool(self, last_hidden_state, attention_mask):
        """Pool by selecting the last non-pad token per sequence."""
        seq_lens = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        emb = last_hidden_state[batch_idx, seq_lens]  # (B, dim)
        return emb

    def _get_model_device(self) -> torch.device:
        return next(self._model.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass on tokenized tensors.

        Args:
            input_ids: (B, seq_len) token IDs
            attention_mask: (B, seq_len) attention mask

        Returns: (B, target_dim) embeddings
        """
        out = self._model(input_ids=input_ids, attention_mask=attention_mask)
        emb = self._last_token_pool(out.last_hidden_state, attention_mask)
        if self._projection is not None:
            emb = self._projection(
                emb.to(self._projection.weight.device).to(self._projection.weight.dtype)
            )
        return emb

    def encode_raw(self, text: str, normalize: bool = False) -> torch.Tensor:
        """Encode a single string. Returns (1, target_dim) tensor. No gradients."""
        device = self._get_model_device()
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
        device = self._get_model_device()
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
            emb = self.forward(input_ids, attn_mask)
            if normalize:
                emb = nn.functional.normalize(emb, dim=-1)
            all_embs.append(emb)
        return torch.cat(all_embs, dim=0)
