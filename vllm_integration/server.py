# ------------------------------------------------------------------------------
# KVzip OpenAI-compatible HTTP server
#
# Exposes the same REST API surface as `vllm serve`, so existing OpenAI SDK
# clients work without modification.
#
# Endpoints:
#   GET  /health                    – liveness probe
#   GET  /v1/models                 – list available models
#   POST /v1/chat/completions       – OpenAI Chat Completions
#   POST /v1/completions            – OpenAI Completions (legacy)
#   POST /v1/kvzip/generate         – KVzip batch long-context endpoint
#
# Usage:
#   python -m vllm_integration.server \
#       --model Qwen/Qwen2.5-7B-Instruct-1M \
#       --compression-ratio 0.3 \
#       --host 0.0.0.0 \
#       --port 8000
# ------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import time
import uuid
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
    _fastapi_available = True
except ImportError:
    _fastapi_available = False

from .engine import KVzipVLLMEngine
from .types import SamplingParams

# Global engine instance (populated at startup)
_engine: Optional[KVzipVLLMEngine] = None


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic request / response schemas (OpenAI-compatible)
# ──────────────────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    # KVzip extensions
    compression_ratio: Optional[float] = Field(None, description="KV-cache compression ratio (0, 1]")
    context: Optional[str] = Field(None, description="Long document context to compress")


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    compression_ratio: Optional[float] = None


class KVzipBatchRequest(BaseModel):
    context: str = Field(..., description="Long context document")
    queries: List[str] = Field(..., description="Questions / instructions over the context")
    compression_ratio: Optional[float] = None
    max_tokens: int = 512
    load_score: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────

def _build_app() -> "FastAPI":
    if not _fastapi_available:
        raise RuntimeError(
            "FastAPI and uvicorn are required for the KVzip server.\n"
            "Install them with:  pip install fastapi uvicorn[standard]"
        )

    app = FastAPI(
        title="KVzip API Server",
        description="OpenAI-compatible API server backed by KVzip KV-cache compression.",
        version="0.1.0",
    )

    # ── Health / model listing ─────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok", "engine": "kvzip"}

    @app.get("/v1/models")
    async def list_models():
        _require_engine()
        return {
            "object": "list",
            "data": [
                {
                    "id": _engine.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "kvzip",
                    "permission": [],
                }
            ],
        }

    # ── Chat Completions ───────────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        _require_engine()

        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        # Separate system/context from user query
        system_text = ""
        user_text = ""
        for msg in request.messages:
            if msg.role == "system":
                system_text = msg.content
            elif msg.role == "user":
                user_text = msg.content

        context = request.context or system_text
        query = user_text

        try:
            if context and query:
                outputs = _engine.generate_with_context(
                    context=context,
                    queries=[query],
                    compression_ratio=request.compression_ratio,
                    sampling_params=params,
                )
                answer = outputs[0].text
                prompt_tok = outputs[0].metrics.get("prompt_tokens", 0)
                completion_tok = outputs[0].metrics.get("completion_tokens", 0)
            else:
                prompt = user_text or system_text
                outputs = _engine.generate([prompt], sampling_params=params)
                answer = outputs[0].text
                prompt_tok = outputs[0].metrics.get("prompt_tokens", 0)
                completion_tok = outputs[0].metrics.get("completion_tokens", 0)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "total_tokens": prompt_tok + completion_tok,
            },
        }

    # ── Completions (legacy) ───────────────────────────────────────────────

    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        _require_engine()
        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
        try:
            outputs = _engine.generate([request.prompt], sampling_params=params)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        out = outputs[0]
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": _engine.model_id,
            "choices": [
                {
                    "text": out.text,
                    "index": 0,
                    "finish_reason": out.outputs[0].finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": out.metrics.get("prompt_tokens", 0),
                "completion_tokens": out.metrics.get("completion_tokens", 0),
                "total_tokens": (
                    out.metrics.get("prompt_tokens", 0) + out.metrics.get("completion_tokens", 0)
                ),
            },
        }

    # ── KVzip batch endpoint ───────────────────────────────────────────────

    @app.post("/v1/kvzip/generate")
    async def kvzip_generate(request: KVzipBatchRequest):
        """
        KVzip-specific endpoint: compress a long context and answer multiple
        queries in a single request.  More efficient than N separate
        /v1/chat/completions calls for the same document.
        """
        _require_engine()

        if not request.context or not request.queries:
            raise HTTPException(status_code=400, detail="`context` and `queries` are required.")

        params = SamplingParams(max_tokens=request.max_tokens)
        try:
            outputs = _engine.generate_with_context(
                context=request.context,
                queries=request.queries,
                compression_ratio=request.compression_ratio,
                sampling_params=params,
                load_score=request.load_score,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "object": "kvzip.batch",
            "model": _engine.model_id,
            "results": [
                {
                    "index": i,
                    "query": request.queries[i],
                    "answer": out.text,
                    "metrics": out.metrics,
                }
                for i, out in enumerate(outputs)
            ],
        }

    return app


def _require_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def serve(
    model: str,
    compression_ratio: float = 0.3,
    kv_type: str = "evict",
    backend: str = "kvzip",
    host: str = "0.0.0.0",
    port: int = 8000,
    **vllm_kwargs,
):
    """Start the KVzip HTTP server (blocking)."""
    global _engine
    _engine = KVzipVLLMEngine(
        model=model,
        compression_ratio=compression_ratio,
        kv_type=kv_type,
        backend=backend,
        **vllm_kwargs,
    )
    app = _build_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KVzip OpenAI-compatible API server")
    parser.add_argument("--model", required=True, help="HuggingFace model ID or abbreviated name")
    parser.add_argument("--compression-ratio", type=float, default=0.3,
                        help="KV-cache compression ratio (default: 0.3)")
    parser.add_argument("--kv-type", default="evict", choices=["evict", "retain"],
                        help="Cache type: 'evict' (default) or 'retain'")
    parser.add_argument("--backend", default="kvzip", choices=["kvzip", "vllm"],
                        help="Inference backend (default: kvzip)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    serve(
        model=args.model,
        compression_ratio=args.compression_ratio,
        kv_type=args.kv_type,
        backend=args.backend,
        host=args.host,
        port=args.port,
    )
