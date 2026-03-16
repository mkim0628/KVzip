#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# KVzip × vLLM Integration Demo
#
# Demonstrates four usage patterns:
#   1. Basic generation with KVzipVLLMEngine (vLLM-compatible API)
#   2. Long-context Q&A with KV-cache compression
#   3. Multi-ratio benchmarking (latency vs quality trade-off)
#   4. vLLM hybrid backend (KVzip compression → vLLM decoding)
#
# Quick start:
#   python demo_vllm.py --mode basic     --model qwen2.5-7b
#   python demo_vllm.py --mode context   --model qwen2.5-7b --ratio 0.3
#   python demo_vllm.py --mode benchmark --model qwen2.5-7b
#   python demo_vllm.py --mode vllm      --model qwen2.5-7b   # requires: pip install vllm
#
# To start the OpenAI-compatible server and query it:
#   # Terminal 1
#   python -m vllm_integration.server --model qwen2.5-7b --port 8000
#   # Terminal 2
#   python demo_vllm.py --mode client --port 8000
# ------------------------------------------------------------------------------

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from pathlib import Path
from typing import List

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ── Demo contexts ─────────────────────────────────────────────────────────────

_SHORT_CONTEXT = textwrap.dedent("""\
    Alice is a software engineer at TechCorp. She joined the company in 2019 after
    completing her PhD in computer science at MIT. Her specialisation is distributed
    systems and she has published 12 peer-reviewed papers on the topic.
    Bob is a product manager at TechCorp. He has been with the company for five years
    and previously worked at three different start-ups. He holds an MBA from Stanford.
    TechCorp was founded in 2010 and is headquartered in San Francisco. The company
    focuses on cloud-infrastructure software and has approximately 3,000 employees.
""")

_SHORT_QUERIES = [
    "What is Alice's educational background?",
    "How long has Bob worked at TechCorp?",
    "In which city is TechCorp headquartered?",
]
_SHORT_ANSWERS = [
    "PhD in computer science from MIT",
    "Five years",
    "San Francisco",
]


def _load_long_context() -> str:
    """Load the repo context bundled with KVzip, or fall back to a synthetic long text."""
    repo_txt = ROOT / "data" / "repo.txt"
    if repo_txt.exists():
        return repo_txt.read_text(encoding="utf-8")

    # Synthetic long document (repeat the short context to reach ~4 k tokens)
    block = _SHORT_CONTEXT * 30
    filler = (
        "\n\nThe following paragraphs are filler content added to extend the document length "
        "for benchmarking purposes.\n\n" + ("Lorem ipsum dolor sit amet. " * 100 + "\n\n") * 10
    )
    return block + filler


_LONG_QUERIES = [
    "What must max_num_tokens be a multiple of when creating a cache?",
    "What bit ranges are allowed for keys and values in quantized cache layers?",
    "Which C++/CUDA file handles the implementation of dequant_cache_paged?",
]
_LONG_ANSWERS = [
    "256",
    "From 2 to 8 bits",
    "exllamav3/exllamav3_ext/cache/q_cache.cu",
]


# ──────────────────────────────────────────────────────────────────────────────
# Demo 1 – Basic vLLM-compatible generate()
# ──────────────────────────────────────────────────────────────────────────────

def demo_basic(model: str, max_tokens: int):
    """
    Shows that KVzipVLLMEngine.generate() is a drop-in replacement for
    vLLM's LLM.generate() when prompts are short.
    """
    from vllm_integration import KVzipVLLMEngine, SamplingParams

    print("\n" + "=" * 70)
    print(" Demo 1 – Basic generation (vLLM-compatible API)")
    print("=" * 70)

    engine = KVzipVLLMEngine(model=model, max_new_tokens=max_tokens)

    prompts = [
        "Explain KV-cache compression in one sentence.",
        "What is flash attention?",
    ]
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = engine.generate(prompts, sampling_params=params)

    for prompt, out in zip(prompts, outputs):
        print(f"\n  Prompt  : {prompt}")
        print(f"  Response: {out.text.strip()}")
        print(f"  Metrics : {out.metrics}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 2 – Long-context Q&A with KV compression
# ──────────────────────────────────────────────────────────────────────────────

def demo_context(model: str, ratio: float, max_tokens: int):
    """
    Compress a long context once and answer multiple queries against it.
    Compares full-KV output with compressed-KV output.
    """
    from vllm_integration import KVzipVLLMEngine, SamplingParams

    print("\n" + "=" * 70)
    print(f" Demo 2 – Long-context Q&A  (ratio={ratio})")
    print("=" * 70)

    context = _load_long_context()
    queries = _LONG_QUERIES if (ROOT / "data" / "repo.txt").exists() else _SHORT_QUERIES
    answers = _LONG_ANSWERS if (ROOT / "data" / "repo.txt").exists() else _SHORT_ANSWERS

    engine = KVzipVLLMEngine(model=model, compression_ratio=ratio, max_new_tokens=max_tokens)
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # ── Full KV cache (baseline) ──────────────────────────────────────────
    print("\n── Baseline (ratio=1.0, no compression) ──")
    baseline_outputs = engine.generate_with_context(
        context=context,
        queries=queries,
        compression_ratio=1.0,
        sampling_params=params,
    )

    # ── Compressed KV cache ───────────────────────────────────────────────
    print(f"\n── KVzip (ratio={ratio}) ──")
    compressed_outputs = engine.generate_with_context(
        context=context,
        queries=queries,
        compression_ratio=ratio,
        sampling_params=params,
    )

    # ── Comparison ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f" Results (ratio={ratio})")
    print("=" * 70)
    for i, (q, a_gt, b_out, c_out) in enumerate(
        zip(queries, answers, baseline_outputs, compressed_outputs)
    ):
        print(f"\n  Q[{i+1}]: {q}")
        print(f"  Ground-truth : {a_gt}")
        print(f"  Baseline     : {b_out.text.strip()}")
        print(f"  KVzip-{ratio:.1f}   : {c_out.text.strip()}")
        print(
            f"  Timing (baseline / kvzip): "
            f"{b_out.metrics['elapsed_time_s']:.2f}s  /  {c_out.metrics['elapsed_time_s']:.2f}s"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Demo 3 – Multi-ratio benchmarking
# ──────────────────────────────────────────────────────────────────────────────

def demo_benchmark(model: str, ratios: List[float], max_tokens: int):
    """
    Run the full KVzipBenchmark across multiple compression ratios and print
    a summary table.
    """
    from vllm_integration.benchmark import KVzipBenchmark

    print("\n" + "=" * 70)
    print(" Demo 3 – Multi-ratio benchmark")
    print("=" * 70)

    context = _load_long_context()
    queries = _LONG_QUERIES if (ROOT / "data" / "repo.txt").exists() else _SHORT_QUERIES

    bench = KVzipBenchmark(model=model, max_new_tokens=max_tokens)
    results = bench.run(context=context, queries=queries, ratios=ratios)
    bench.print_report(results, queries=queries)


# ──────────────────────────────────────────────────────────────────────────────
# Demo 4 – vLLM hybrid backend
# ──────────────────────────────────────────────────────────────────────────────

def demo_vllm_backend(model: str, ratio: float, max_tokens: int):
    """
    Use KVzip for importance scoring and token selection, then feed the
    compressed token sequence to vLLM for fast PagedAttention decoding.

    Requires:  pip install vllm
    """
    from vllm_integration import KVzipVLLMEngine, SamplingParams

    print("\n" + "=" * 70)
    print(f" Demo 4 – vLLM hybrid backend  (ratio={ratio})")
    print("=" * 70)

    context = _load_long_context()
    queries = _LONG_QUERIES if (ROOT / "data" / "repo.txt").exists() else _SHORT_QUERIES

    engine = KVzipVLLMEngine(
        model=model,
        compression_ratio=ratio,
        backend="vllm",       # KVzip compression → vLLM generation
        max_new_tokens=max_tokens,
    )

    if engine.backend != "vllm":
        print("  [INFO] vLLM not available – engine fell back to 'kvzip' backend.")

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = engine.generate_with_context(
        context=context,
        queries=queries,
        compression_ratio=ratio,
        sampling_params=params,
    )

    print("\n── Answers ──")
    for i, (q, out) in enumerate(zip(queries, outputs)):
        print(f"  Q[{i+1}]: {q}")
        print(f"  A     : {out.text.strip()}")
        print(f"  Metrics: {out.metrics}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 5 – OpenAI client against the running server
# ──────────────────────────────────────────────────────────────────────────────

def demo_client(host: str, port: int):
    """
    Send requests to the KVzip server using the openai Python SDK
    (same interface as vLLM's `vllm serve`).

    Requires the server to be running:
        python -m vllm_integration.server --model <model_id> --port <port>
    Requires:  pip install openai
    """
    try:
        import openai
    except ImportError:
        print("openai package not installed.  Run:  pip install openai")
        return

    print("\n" + "=" * 70)
    print(f" Demo 5 – OpenAI client → KVzip server  (http://{host}:{port})")
    print("=" * 70)

    client = openai.OpenAI(base_url=f"http://{host}:{port}/v1", api_key="kvzip")

    # ── /v1/models ────────────────────────────────────────────────────────
    models = client.models.list()
    print(f"\n  Available models: {[m.id for m in models.data]}")

    # ── /v1/chat/completions ──────────────────────────────────────────────
    context = _SHORT_CONTEXT
    query = _SHORT_QUERIES[0]

    print(f"\n  Context: {context[:80].strip()}…")
    print(f"  Query  : {query}")

    response = client.chat.completions.create(
        model=models.data[0].id,
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": query},
        ],
        max_tokens=128,
        extra_body={"compression_ratio": 0.3},  # KVzip extension
    )
    print(f"\n  Answer : {response.choices[0].message.content.strip()}")
    print(f"  Usage  : {response.usage}")

    # ── /v1/kvzip/generate (batch) ────────────────────────────────────────
    print("\n  Calling /v1/kvzip/generate (batch endpoint) …")
    import urllib.request
    import json

    payload = {
        "context": _SHORT_CONTEXT,
        "queries": _SHORT_QUERIES,
        "compression_ratio": 0.3,
        "max_tokens": 128,
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/kvzip/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read())

    for item in result["results"]:
        print(f"\n  Q[{item['index']+1}]: {item['query']}")
        print(f"  A     : {item['answer'].strip()}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="KVzip × vLLM integration demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Modes:
              basic      – vLLM-compatible generate() on short prompts
              context    – Long-context Q&A with KV compression
              benchmark  – Multi-ratio latency/memory benchmark
              vllm       – vLLM hybrid backend (requires: pip install vllm)
              client     – OpenAI SDK client against a running KVzip server
        """),
    )
    parser.add_argument(
        "--mode",
        default="context",
        choices=["basic", "context", "benchmark", "vllm", "client"],
        help="Demo mode (default: context)",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-7b",
        help="Model name / HuggingFace ID (default: qwen2.5-7b)",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.3,
        help="KV compression ratio – fraction of KV pairs to keep (default: 0.3)",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.3, 0.5, 0.7, 1.0],
        help="Compression ratios for benchmark mode (default: 0.3 0.5 0.7 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)",
    )
    parser.add_argument("--host", default="localhost", help="Server host for client mode")
    parser.add_argument("--port", type=int, default=8000, help="Server port for client mode")

    args = parser.parse_args()

    if args.mode == "basic":
        demo_basic(args.model, args.max_tokens)
    elif args.mode == "context":
        demo_context(args.model, args.ratio, args.max_tokens)
    elif args.mode == "benchmark":
        demo_benchmark(args.model, args.ratios, args.max_tokens)
    elif args.mode == "vllm":
        demo_vllm_backend(args.model, args.ratio, args.max_tokens)
    elif args.mode == "client":
        demo_client(args.host, args.port)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
