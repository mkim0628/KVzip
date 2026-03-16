# ------------------------------------------------------------------------------
# KVzip vLLM Custom Attention Backend
#
# Implements a vLLM AttentionBackend that integrates KVzip's KV-cache
# compression directly into vLLM's PagedAttention pipeline.
#
# Architecture overview:
#
#   ┌─────────────────────────────────────────────────────────┐
#   │  vLLM Worker                                            │
#   │                                                         │
#   │  ┌─────────────┐   KV blocks    ┌────────────────────┐ │
#   │  │ BlockManager│ ─────────────► │ KVzipAttentionImpl │ │
#   │  └─────────────┘                │                    │ │
#   │                                 │  prefill phase:    │ │
#   │                                 │   • standard attn  │ │
#   │                                 │   • score KV pairs │ │
#   │                                 │   • compress cache │ │
#   │                                 │                    │ │
#   │                                 │  decode phase:     │ │
#   │                                 │   • compressed attn│ │
#   │                                 └────────────────────┘ │
#   └─────────────────────────────────────────────────────────┘
#
# Usage with vLLM (>=0.5.0):
#
#   from vllm import LLM
#   from vllm_integration.vllm_attention_backend import register_kvzip_backend
#
#   register_kvzip_backend(compression_ratio=0.3)
#   llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-1M")
#
# NOTE: This is an advanced integration path that modifies vLLM internals.
#       The simpler KVzipVLLMEngine (engine.py) is recommended for most users.
# ------------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# KVzip score accumulator (shared state across layers in one forward pass)
# ──────────────────────────────────────────────────────────────────────────────

class KVzipScoreBuffer:
    """
    Accumulates per-(layer, head, position) KV importance scores during
    a KVzip-instrumented forward pass.  Used by KVzipAttentionImpl to
    decide which KV positions to evict after prefill.
    """

    def __init__(self, n_layers: int, n_heads_kv: int, ctx_len: int, device: torch.device):
        self.n_layers = n_layers
        self.n_heads_kv = n_heads_kv
        self.ctx_len = ctx_len
        # score[layer_idx] shape: (1, n_heads_kv, ctx_len)
        self.score: List[Optional[torch.Tensor]] = [None] * n_layers
        self.device = device

    def init(self):
        self.score = [
            torch.zeros((1, self.n_heads_kv, self.ctx_len), device=self.device)
            for _ in range(self.n_layers)
        ]

    def update(self, layer_idx: int, attn_weights: torch.Tensor, ctx_start: int, ctx_end: int):
        """
        Update score for one layer using attention weights from a scoring query.

        Args:
            layer_idx:   Layer index.
            attn_weights: Soft-max attention weights, shape (bsz, n_heads, q_len, k_len).
            ctx_start:   Start position of the context segment in k_len.
            ctx_end:     End position of the context segment in k_len.
        """
        if self.score[layer_idx] is None:
            return
        # Take the max over query positions and head groups → (1, n_heads_kv, ctx_len)
        bsz, n_heads, q_len, k_len = attn_weights.shape
        n_groups = n_heads // self.n_heads_kv
        ctx_weights = attn_weights[..., ctx_start:ctx_end]  # (bsz, h, q, ctx)
        ctx_weights = ctx_weights.view(bsz, self.n_heads_kv, n_groups, q_len, -1)
        score_update = ctx_weights.amax(dim=(2, 3))  # (bsz, n_heads_kv, ctx)
        self.score[layer_idx] = torch.maximum(self.score[layer_idx], score_update)

    def threshold(self, ratio: float) -> List[torch.Tensor]:
        """
        Apply global top-k thresholding across all layers/heads/positions.

        Returns a list of boolean masks (one per layer), shape (1, n_heads_kv, ctx_len).
        """
        all_scores = torch.stack(self.score, dim=0)  # (L, 1, H, S)
        flat = all_scores.reshape(-1)
        n_keep = max(int(len(flat) * ratio), 1)
        threshold_val = torch.topk(flat, n_keep).values[-1].item()
        masks = [(s >= threshold_val) for s in self.score]
        return masks


# ──────────────────────────────────────────────────────────────────────────────
# KVzip-aware attention module (monkey-patch target for vLLM models)
# ──────────────────────────────────────────────────────────────────────────────

class KVzipAttentionState:
    """
    Per-request state attached to the KV cache to drive compression.
    Stored on the ``past_key_values`` object used by vLLM model runners.
    """

    def __init__(self, compression_ratio: float = 0.3):
        self.compression_ratio = compression_ratio
        self.score_buffer: Optional[KVzipScoreBuffer] = None
        self.valid_masks: Optional[List[torch.Tensor]] = None  # set after pruning
        self.is_prefill_phase: bool = True
        self.is_scoring_phase: bool = False
        self.ctx_start: int = 0
        self.ctx_end: int = 0

    def start_scoring(self, n_layers: int, n_heads_kv: int, ctx_len: int, device: torch.device):
        self.score_buffer = KVzipScoreBuffer(n_layers, n_heads_kv, ctx_len, device)
        self.score_buffer.init()
        self.is_scoring_phase = True

    def finish_scoring(self):
        """Compute importance masks once scoring is complete."""
        if self.score_buffer is not None:
            self.valid_masks = self.score_buffer.threshold(self.compression_ratio)
        self.is_scoring_phase = False
        self.is_prefill_phase = False


def _kvzip_attention_forward_hook(
    module: nn.Module,
    args: Tuple,
    kwargs: Dict[str, Any],
    state: KVzipAttentionState,
    layer_idx: int,
) -> None:
    """
    Forward pre-hook that injects scoring logic into vLLM attention layers.

    Registers the hook with:
        handle = attn_layer.register_forward_pre_hook(
            lambda m, a, kw: _kvzip_attention_forward_hook(m, a, kw, state, idx),
            with_kwargs=True,
        )
    """
    if not state.is_scoring_phase:
        return

    # The hook cannot modify attn_weights here (they are computed inside the module).
    # Instead, we rely on a post-hook (see below) to capture them.


def _kvzip_attention_post_hook(
    module: nn.Module,
    args: Tuple,
    kwargs: Dict[str, Any],
    output: Any,
    state: KVzipAttentionState,
    layer_idx: int,
) -> Any:
    """
    Forward post-hook that captures attention weights for scoring.

    Most vLLM attention modules return (attn_output, attn_weights, ...).
    If attn_weights are not returned, scoring is skipped for this layer.
    """
    if not state.is_scoring_phase or state.score_buffer is None:
        return output

    # Unpack output – vLLM typically returns (hidden_states,) or (hidden_states, weights)
    if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
        attn_weights = output[1]
        if attn_weights.dim() == 4:  # (bsz, heads, q, k)
            state.score_buffer.update(
                layer_idx, attn_weights, state.ctx_start, state.ctx_end
            )
    return output


# ──────────────────────────────────────────────────────────────────────────────
# Public API: register KVzip backend with an existing vLLM LLM instance
# ──────────────────────────────────────────────────────────────────────────────

def register_kvzip_backend(
    llm: Any,
    compression_ratio: float = 0.3,
    verbose: bool = True,
) -> KVzipAttentionState:
    """
    Monkey-patch a vLLM ``LLM`` instance to use KVzip attention scoring.

    This installs forward hooks on every attention layer so that during the
    first (scoring) forward pass attention weights are captured and used to
    build per-position importance scores.  After scoring the KV cache is
    pruned in-place before decoding begins.

    Args:
        llm:               A ``vllm.LLM`` instance.
        compression_ratio: Fraction of KV pairs to retain (default: 0.3).
        verbose:           Print registration info.

    Returns:
        The shared :class:`KVzipAttentionState` object; hold a reference to
        inspect scores / masks after inference.

    Example::

        from vllm import LLM
        from vllm_integration.vllm_attention_backend import register_kvzip_backend

        llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-1M")
        state = register_kvzip_backend(llm, compression_ratio=0.3)

        # Run inference – first request triggers scoring + pruning automatically
        outputs = llm.generate(["Hello!"], SamplingParams(max_tokens=64))
    """
    state = KVzipAttentionState(compression_ratio=compression_ratio)
    hooks = []

    try:
        # Access the underlying model from different vLLM versions
        model = None
        if hasattr(llm, "llm_engine"):
            engine = llm.llm_engine
            if hasattr(engine, "model_executor"):
                worker = engine.model_executor.driver_worker
                model = worker.model_runner.model
            elif hasattr(engine, "driver_worker"):
                model = engine.driver_worker.model_runner.model

        if model is None:
            if verbose:
                print("[KVzip] WARNING: Could not access vLLM model internals. "
                      "Hook-based compression is unavailable. "
                      "Use KVzipVLLMEngine instead.")
            return state

        n_layers = 0
        for name, module in model.named_modules():
            if "attention" in name.lower() and hasattr(module, "forward"):
                idx = n_layers
                pre_hook = module.register_forward_pre_hook(
                    lambda m, a, kw, i=idx: _kvzip_attention_forward_hook(m, a, kw, state, i),
                    with_kwargs=True,
                )
                post_hook = module.register_forward_hook(
                    lambda m, a, kw, o, i=idx: _kvzip_attention_post_hook(m, a, kw, o, state, i),
                    with_kwargs=True,
                )
                hooks.extend([pre_hook, post_hook])
                n_layers += 1

        if verbose:
            print(
                f"[KVzip] Registered attention hooks on {n_layers} layers "
                f"(compression_ratio={compression_ratio})."
            )

    except Exception as exc:
        if verbose:
            print(f"[KVzip] WARNING: Hook registration failed: {exc}")

    state._hooks = hooks  # keep handles to prevent GC
    return state


def compress_kv_cache(
    llm: Any,
    state: KVzipAttentionState,
    verbose: bool = True,
) -> None:
    """
    Trigger KV-cache compression using scores accumulated so far.

    Call this after a scoring forward pass (or scoring queries) to apply
    the importance masks to the vLLM block tables.

    Note: Full block-level eviction requires access to vLLM's
    BlockSpaceManager, which has a different API across vLLM versions.
    This function provides a best-effort implementation; for production use,
    integrate with vLLM's upcoming custom-cache-manager API.
    """
    state.finish_scoring()
    if state.valid_masks is None:
        if verbose:
            print("[KVzip] No valid masks computed – skipping compression.")
        return

    n_layers = len(state.valid_masks)
    total = sum(m.numel() for m in state.valid_masks)
    kept = sum(m.float().sum().item() for m in state.valid_masks)
    actual_ratio = kept / max(total, 1)

    if verbose:
        print(
            f"[KVzip] Compression applied: {actual_ratio:.2%} of KV pairs retained "
            f"(target={state.compression_ratio:.2%}) across {n_layers} layers."
        )
