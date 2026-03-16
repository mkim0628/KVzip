# ------------------------------------------------------------------------------
# KVzip-vLLM Integration Engine
#
# Provides a vLLM-compatible inference engine backed by KVzip for
# KV-cache compression of long contexts.
#
# Supported backends:
#   "kvzip"  - Pure KVzip (HuggingFace model + monkey-patched attention).
#              Works out-of-the-box with any KVzip-supported model.
#   "vllm"   - KVzip compression preprocessing followed by vLLM generation.
#              Requires `pip install vllm`.  KVzip identifies the most
#              important token positions; those tokens are forwarded to vLLM
#              as a shorter, compressed prompt, allowing vLLM to handle fast
#              PagedAttention-based decoding on the reduced sequence.
# ------------------------------------------------------------------------------

import os
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import torch

# Allow running from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

# ── Graceful fallback: if the CUDA extension is missing, install the pure-Python
#    stub so that the rest of the imports succeed.  The stub emits an ImportWarning
#    to remind the user to build the real extension.
try:
    import tiny_api_cuda  # noqa: F401  (compiled CUDA extension)
except ModuleNotFoundError:
    import importlib.util, types
    _stub_path = os.path.join(_REPO_ROOT, "tiny_api_cuda.py")
    _spec = importlib.util.spec_from_file_location("tiny_api_cuda", _stub_path)
    _stub = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_stub)
    sys.modules["tiny_api_cuda"] = _stub

from model import ModelKVzip
from attention.kvcache import EvictCache, RetainCache
from .types import CompletionOutput, RequestOutput, SamplingParams

# ── Detect whether EvictCache decode is safe (needs CUDA extension) ───────────
_CUDA_EXT_AVAILABLE = "tiny_api_cuda" in sys.modules and not hasattr(
    sys.modules["tiny_api_cuda"], "__file__"
) or os.path.exists(os.path.join(_REPO_ROOT, "tiny_api_cuda.py")) is False

def _safe_kv_type(requested: str) -> str:
    """
    'evict' requires the compiled CUDA extension for efficient flatten-view
    updates during decode.  Fall back to 'retain' when the stub is active.
    """
    try:
        import tiny_api_cuda as _m
        # If the real .so is loaded it has no __file__ attribute set to .py
        if hasattr(_m, "__file__") and str(_m.__file__).endswith(".py"):
            if requested == "evict":
                print(
                    "[KVzip] INFO: CUDA extension not compiled – "
                    "switching kv_type 'evict' → 'retain'.\n"
                    "         Build for full performance:  cd csrc && python build.py install"
                )
                return "retain"
    except Exception:
        pass
    return requested


class KVzipVLLMEngine:
    """
    vLLM-compatible inference engine with KVzip KV-cache compression.

    Example (pure KVzip backend)::

        engine = KVzipVLLMEngine("Qwen/Qwen2.5-7B-Instruct-1M", compression_ratio=0.3)

        # Standard generate – accepts a list of prompts just like vLLM
        outputs = engine.generate(["Tell me about transformers."])
        print(outputs[0].text)

        # Long-context mode – compress once, query many times
        outputs = engine.generate_with_context(
            context=long_document,
            queries=["Who wrote this?", "What is the main topic?"],
            compression_ratio=0.3,
        )
        for o in outputs:
            print(o.text)

    Example (vLLM backend)::

        engine = KVzipVLLMEngine(
            "Qwen/Qwen2.5-7B-Instruct-1M",
            compression_ratio=0.3,
            backend="vllm",
        )
        outputs = engine.generate_with_context(context, queries)
    """

    def __init__(
        self,
        model: str,
        compression_ratio: float = 0.3,
        kv_type: str = "evict",
        backend: str = "kvzip",
        max_new_tokens: int = 512,
        **vllm_kwargs,
    ):
        """
        Args:
            model: HuggingFace model ID or abbreviated name (e.g. "qwen2.5-7b").
            compression_ratio: Fraction of KV pairs to *retain* (0 < r ≤ 1).
                               E.g. 0.3 keeps 30 % of the context KV cache.
            kv_type: Cache eviction strategy – "evict" (physical eviction,
                     faster decode) or "retain" (logical masking, useful for
                     multi-ratio evaluation).
            backend: "kvzip" or "vllm".
            max_new_tokens: Default token budget for generation.
            **vllm_kwargs: Extra keyword arguments forwarded to vLLM's LLM().
        """
        self.model_id = model
        self.compression_ratio = compression_ratio
        self.kv_type = _safe_kv_type(kv_type)
        self.backend = backend
        self.max_new_tokens = max_new_tokens

        print(f"[KVzipVLLMEngine] model={model}, ratio={compression_ratio}, "
              f"kv_type={self.kv_type}, backend={backend}")

        # KVzip (HuggingFace) model – always required
        self.kvzip_model = ModelKVzip(model, kv_type=self.kv_type)
        self.kvzip_model.gen_kwargs["max_new_tokens"] = max_new_tokens

        # Optional vLLM engine
        self.vllm_engine = None
        if backend == "vllm":
            self._init_vllm(model, **vllm_kwargs)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_vllm(self, model: str, **kwargs):
        """Initialise a vLLM LLM engine.  Falls back to 'kvzip' backend on failure."""
        try:
            from vllm import LLM as VllmLLM

            print("[KVzipVLLMEngine] Initialising vLLM engine …")
            self.vllm_engine = VllmLLM(model=model, dtype="auto", **kwargs)
            print("[KVzipVLLMEngine] vLLM engine ready.")
        except ImportError:
            print(
                "[KVzipVLLMEngine] WARNING: vllm not installed – "
                "falling back to 'kvzip' backend.  Install with: pip install vllm"
            )
            self.backend = "kvzip"
        except Exception as exc:
            print(f"[KVzipVLLMEngine] WARNING: vLLM init failed ({exc}) – falling back to 'kvzip'.")
            self.backend = "kvzip"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_sampling_params(self, params: SamplingParams):
        """Push SamplingParams into the underlying HF gen_kwargs."""
        hf = params.to_hf_kwargs()
        for k, v in hf.items():
            self.kvzip_model.gen_kwargs[k] = v

    def _make_output(
        self,
        request_id: str,
        prompt: str,
        text: str,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed: float,
        compression_ratio: Optional[float] = None,
    ) -> RequestOutput:
        return RequestOutput(
            request_id=request_id,
            prompt=prompt,
            outputs=[CompletionOutput(index=0, text=text, finish_reason="stop")],
            finished=True,
            metrics={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_time_s": round(elapsed, 3),
                "tokens_per_second": round(completion_tokens / max(elapsed, 1e-6), 1),
                "compression_ratio": compression_ratio,
            },
        )

    # ------------------------------------------------------------------
    # Public API  (vLLM-compatible)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        use_tqdm: bool = False,
    ) -> List[RequestOutput]:
        """
        Generate completions for a list of prompts (vLLM-compatible interface).

        The full prompt is treated as context and prefilled without compression.
        For long-context use-cases, prefer :meth:`generate_with_context`.

        Args:
            prompts: Single prompt string or list of prompt strings.
            sampling_params: Generation hyper-parameters.
            use_tqdm: Ignored (kept for API compatibility with vLLM).

        Returns:
            List of :class:`RequestOutput` objects, one per prompt.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if sampling_params is None:
            sampling_params = SamplingParams()

        self._apply_sampling_params(sampling_params)

        outputs = []
        for prompt in prompts:
            request_id = str(uuid.uuid4())
            t0 = time.time()

            # Apply the model's chat template so the prompt is well-formed,
            # then call model.generate() directly – no separate KV-cache object
            # needed for short, single-turn prompts.
            tokenizer = self.kvzip_model.tokenizer
            messages = [{"role": "user", "content": prompt}]
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.kvzip_model.device)

            prompt_tokens = input_ids.shape[1]

            # Build generation kwargs from current model defaults
            gen_kwargs = {k: v for k, v in self.kvzip_model.gen_kwargs.items()
                          if k != "cache_implementation"}

            raw_output = self.kvzip_model.model.generate(input_ids, **gen_kwargs)
            a_ids = raw_output[0, prompt_tokens:]
            # Strip trailing eos if present
            eos = tokenizer.eos_token_id
            if a_ids.shape[0] > 0 and a_ids[-1].item() == eos:
                a_ids = a_ids[:-1]
            output_text = tokenizer.decode(a_ids, skip_special_tokens=True)
            completion_tokens = a_ids.shape[0]

            outputs.append(
                self._make_output(
                    request_id=request_id,
                    prompt=prompt,
                    text=output_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed=time.time() - t0,
                )
            )
        return outputs

    def generate_with_context(
        self,
        context: str,
        queries: List[str],
        compression_ratio: Optional[float] = None,
        sampling_params: Optional[SamplingParams] = None,
        load_score: bool = False,
        level: str = "pair",
    ) -> List[RequestOutput]:
        """
        KVzip's primary use-case: compress a long context once, answer many queries.

        Pipeline:
            1. **Prefill**   – tokenise & run the model over *context*, building the full KV cache.
            2. **Score**     – compute per-(layer, head, position) importance via context-reconstruction.
            3. **Prune**     – physically evict low-importance KV pairs (EvictCache) or mask them
                               (RetainCache), reaching the target ``compression_ratio``.
            4. **Generate**  – decode each query reusing the compressed cache.

        Args:
            context: Long document / system context (string).
            queries: List of question strings.
            compression_ratio: Fraction of KV pairs to keep (default: ``self.compression_ratio``).
            sampling_params: Generation hyper-parameters.
            load_score: If True, skip self-task scoring and use pre-computed head scores from
                        ``utils/head_score/``.  Much faster but slightly less accurate.
            level: Pruning granularity – ``"pair"`` (non-uniform across heads),
                   ``"pair-uniform"`` (equal budget per head), or ``"head"`` (whole heads).

        Returns:
            List of :class:`RequestOutput` objects, one per query.
        """
        ratio = compression_ratio if compression_ratio is not None else self.compression_ratio
        if sampling_params is None:
            sampling_params = SamplingParams()
        self._apply_sampling_params(sampling_params)

        # ── Stage 1: Prefill ──────────────────────────────────────────────────
        print(f"\n[KVzip] Stage 1 – Prefill  (context length: {len(context):,} chars)")
        t_prefill = time.time()
        kv = self.kvzip_model.prefill(context, load_score=load_score)
        prefill_elapsed = time.time() - t_prefill
        kv_size_before = kv._mem()
        print(
            f"[KVzip] Prefill done: {prefill_elapsed:.2f}s | "
            f"ctx_len={kv.ctx_len:,} tokens | KV={kv_size_before} GB"
        )

        # ── Stage 2: Prune ────────────────────────────────────────────────────
        if ratio < 1.0:
            print(f"\n[KVzip] Stage 2 – Prune  (target ratio={ratio})")
            t_prune = time.time()
            thres, actual_ratio = kv.prune(ratio=ratio, level=level)
            prune_elapsed = time.time() - t_prune
            kv_size_after = kv._mem()
            print(
                f"[KVzip] Prune done: {prune_elapsed:.2f}s | "
                f"actual_ratio={actual_ratio:.2f} | KV {kv_size_before} → {kv_size_after} GB"
            )
        else:
            print("\n[KVzip] ratio=1.0 – skipping pruning (full KV cache)")

        # ── Stage 3: Generate ─────────────────────────────────────────────────
        print(f"\n[KVzip] Stage 3 – Generate  ({len(queries)} queries)")
        outputs = []

        for i, query in enumerate(queries):
            request_id = str(uuid.uuid4())
            t0 = time.time()

            query_ids = self.kvzip_model.apply_template(query)
            prompt_tokens = kv._seen_tokens + query_ids.shape[1]

            if self.backend == "vllm" and self.vllm_engine is not None:
                output_text = self._generate_vllm_compressed(context, query, ratio, kv)
            else:
                output_text = self.kvzip_model.generate(query_ids, kv=kv, update_cache=False)

            completion_tokens = len(self.kvzip_model.encode(output_text)[0])
            elapsed = time.time() - t0

            print(
                f"  [{i+1}/{len(queries)}] Q: {query[:60].strip()!r}\n"
                f"            A: {output_text[:100].strip()!r}  "
                f"({completion_tokens} tok, {elapsed:.2f}s)"
            )

            outputs.append(
                self._make_output(
                    request_id=request_id,
                    prompt=query,
                    text=output_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed=elapsed,
                    compression_ratio=ratio,
                )
            )

        return outputs

    # ------------------------------------------------------------------
    # vLLM hybrid backend: compressed-token extraction
    # ------------------------------------------------------------------

    def _generate_vllm_compressed(
        self,
        context: str,
        query: str,
        ratio: float,
        kv,
    ) -> str:
        """
        Extract the tokens retained by KVzip and feed them to vLLM as a
        shorter prompt.  vLLM then handles fast PagedAttention decoding.

        Token-level importance is derived by averaging the per-(layer, head)
        validity mask across all layers and heads.  The top-k tokens (where
        k = ratio * ctx_len) are kept; the rest are discarded.
        """
        ctx_ids = kv.ctx_ids  # (1, ctx_len)

        if getattr(kv, "valid", None) is not None:
            valid = kv.valid  # list[layer] of (1, n_heads_kv, ctx_len) OR stacked tensor
            if isinstance(valid, list):
                valid = torch.stack(valid, dim=0)  # (n_layers, 1, n_heads_kv, ctx_len)

            # Average across layers and heads → per-token importance
            token_importance = valid.float().mean(dim=(0, 1, 2))  # (ctx_len,)
            n_keep = max(int(ctx_ids.shape[1] * ratio), 1)
            _, top_idx = torch.topk(token_importance, n_keep)
            top_idx = top_idx.sort().values
            compressed_ids = ctx_ids[:, top_idx]
            compressed_context = self.kvzip_model.decode(compressed_ids)
        else:
            # No validity mask available (ratio == 1.0 or unscored cache)
            compressed_context = context

        # Build the full prompt using the same template KVzip uses
        sys_prefix = self.kvzip_model.decode(self.kvzip_model.sys_prompt_ids)
        sys_postfix = self.kvzip_model.decode(self.kvzip_model.postfix_ids)
        full_prompt = f"{sys_prefix}{compressed_context}\n\n{query.strip()}{sys_postfix}"

        from vllm import SamplingParams as VllmSamplingParams

        vllm_params = VllmSamplingParams(
            max_tokens=self.kvzip_model.gen_kwargs.get("max_new_tokens", 512),
            temperature=0.0,
        )
        vllm_out = self.vllm_engine.generate([full_prompt], vllm_params)
        return vllm_out[0].outputs[0].text

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_cache_stats(self, kv) -> Dict[str, Any]:
        """Return a dict of cache statistics for logging / debugging."""
        stats: Dict[str, Any] = {
            "kv_size_gb": kv._mem(),
            "ctx_len": kv.ctx_len,
            "pruned": kv.pruned,
        }
        if getattr(kv, "valid", None) is not None:
            if isinstance(kv.valid, torch.Tensor):
                stats["retention_ratio"] = kv.valid.float().mean().item()
            elif isinstance(kv.valid, list) and len(kv.valid) > 0:
                valid_stack = torch.stack(kv.valid, dim=0)
                stats["retention_ratio"] = valid_stack.float().mean().item()
        return stats
