# ------------------------------------------------------------------------------
# KVzip-vLLM Benchmarking Utilities
#
# Compare KVzip compression against baseline (no compression) across:
#   - Generation quality  (exact match / keyword match)
#   - Latency             (time-to-first-token, total time)
#   - Memory usage        (GPU KV-cache footprint)
# ------------------------------------------------------------------------------

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .engine import KVzipVLLMEngine
from .types import SamplingParams


@dataclass
class BenchmarkResult:
    ratio: float
    kv_size_gb: float
    prefill_time_s: float
    prune_time_s: float
    avg_gen_time_s: float
    avg_toks_per_sec: float
    total_time_s: float
    answers: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class KVzipBenchmark:
    """
    Benchmark KVzip compression ratios against a full-cache baseline.

    Example::

        bench = KVzipBenchmark("Qwen/Qwen2.5-7B-Instruct-1M")
        results = bench.run(
            context=long_document,
            queries=["Q1?", "Q2?"],
            ratios=[0.3, 0.5, 0.7, 1.0],
        )
        bench.print_report(results)
    """

    def __init__(self, model: str, kv_type: str = "evict", max_new_tokens: int = 256):
        self.engine = KVzipVLLMEngine(
            model=model,
            kv_type=kv_type,
            max_new_tokens=max_new_tokens,
        )

    def run(
        self,
        context: str,
        queries: List[str],
        ratios: Optional[List[float]] = None,
        load_score: bool = False,
        level: str = "pair",
    ) -> List[BenchmarkResult]:
        """
        Run the benchmark for each compression ratio.

        ``ratio=1.0`` corresponds to the full-cache baseline (no eviction).

        Args:
            context: Long document used as context.
            queries: List of questions about the document.
            ratios: Compression ratios to evaluate (default: [0.3, 0.5, 0.7, 1.0]).
            load_score: Use pre-computed head scores instead of self-task scoring.
            level: Pruning granularity – "pair", "pair-uniform", or "head".

        Returns:
            List of :class:`BenchmarkResult` objects, one per ratio.
        """
        if ratios is None:
            ratios = [0.3, 0.5, 0.7, 1.0]

        # Pre-compute scores once (shared across all ratios when using RetainCache)
        print(f"\n{'='*70}")
        print("KVzip Benchmark")
        print(f"  Model      : {self.engine.model_id}")
        print(f"  Context    : {len(context):,} chars")
        print(f"  Queries    : {len(queries)}")
        print(f"  Ratios     : {ratios}")
        print(f"{'='*70}\n")

        results: List[BenchmarkResult] = []
        for ratio in ratios:
            result = self._eval_ratio(context, queries, ratio, load_score, level)
            results.append(result)

        return results

    def _eval_ratio(
        self,
        context: str,
        queries: List[str],
        ratio: float,
        load_score: bool,
        level: str,
    ) -> BenchmarkResult:
        print(f"\n── ratio = {ratio} {'(baseline)' if ratio == 1.0 else ''} {'─'*50}")

        params = SamplingParams(max_tokens=self.engine.max_new_tokens)
        t_total_start = time.time()

        # Stage 1: Prefill
        t_prefill = time.time()
        kv = self.engine.kvzip_model.prefill(context, load_score=load_score)
        prefill_time = time.time() - t_prefill
        kv_size_before = kv._mem()

        # Stage 2: Prune
        t_prune = time.time()
        if ratio < 1.0:
            kv.prune(ratio=ratio, level=level)
        prune_time = time.time() - t_prune
        kv_size_after = kv._mem()

        # Stage 3: Generate
        gen_times: List[float] = []
        answers: List[str] = []
        for query in queries:
            query_ids = self.engine.kvzip_model.apply_template(query)
            t_gen = time.time()
            answer = self.engine.kvzip_model.generate(query_ids, kv=kv, update_cache=False)
            gen_elapsed = time.time() - t_gen
            answers.append(answer)
            gen_times.append(gen_elapsed)

        total_time = time.time() - t_total_start
        avg_gen = sum(gen_times) / max(len(gen_times), 1)

        # Estimate tokens/s (completion tokens only)
        total_comp_toks = sum(
            len(self.engine.kvzip_model.encode(a)[0]) for a in answers
        )
        avg_tps = total_comp_toks / max(sum(gen_times), 1e-9)

        result = BenchmarkResult(
            ratio=ratio,
            kv_size_gb=kv_size_after,
            prefill_time_s=round(prefill_time, 3),
            prune_time_s=round(prune_time, 3),
            avg_gen_time_s=round(avg_gen, 3),
            avg_toks_per_sec=round(avg_tps, 1),
            total_time_s=round(total_time, 3),
            answers=answers,
            metrics={
                "kv_before_gb": kv_size_before,
                "kv_after_gb": kv_size_after,
                "kv_reduction_gb": round(kv_size_before - kv_size_after, 2),
            },
        )
        print(
            f"  KV: {kv_size_before} → {kv_size_after} GB  |  "
            f"prefill {prefill_time:.2f}s  prune {prune_time:.2f}s  "
            f"gen {avg_gen:.2f}s/q  ({avg_tps:.0f} tok/s)"
        )
        return result

    @staticmethod
    def print_report(results: List[BenchmarkResult], queries: Optional[List[str]] = None):
        """Pretty-print a comparison table."""
        print(f"\n{'='*80}")
        print(f"{'BENCHMARK REPORT':^80}")
        print(f"{'='*80}")
        print(
            f"{'Ratio':>8}  {'KV (GB)':>9}  "
            f"{'Prefill':>9}  {'Prune':>8}  {'Gen/q':>7}  {'Tok/s':>7}  {'Total':>8}"
        )
        print("-" * 80)
        for r in results:
            tag = " (baseline)" if r.ratio == 1.0 else ""
            print(
                f"{r.ratio:>8.2f}  {r.kv_size_gb:>9.2f}  "
                f"{r.prefill_time_s:>8.2f}s  {r.prune_time_s:>7.2f}s  "
                f"{r.avg_gen_time_s:>6.2f}s  {r.avg_toks_per_sec:>7.0f}  "
                f"{r.total_time_s:>7.2f}s{tag}"
            )
        print("=" * 80)

        # Memory savings vs baseline
        baseline = next((r for r in results if r.ratio == 1.0), None)
        if baseline:
            print("\nMemory savings vs. baseline (ratio=1.0):")
            for r in results:
                if r.ratio < 1.0:
                    saved = baseline.kv_size_gb - r.kv_size_gb
                    pct = saved / max(baseline.kv_size_gb, 1e-6) * 100
                    print(f"  ratio={r.ratio:.2f}: saved {saved:.2f} GB ({pct:.0f}%)")

        if queries:
            print("\nGenerated answers:")
            baseline_answers = baseline.answers if baseline else [""] * len(queries)
            for i, q in enumerate(queries):
                print(f"\n  Q[{i+1}]: {q.strip()}")
                for r in results:
                    if i < len(r.answers):
                        tag = "(baseline)" if r.ratio == 1.0 else f"ratio={r.ratio}"
                        print(f"    [{tag}]: {r.answers[i].strip()}")
