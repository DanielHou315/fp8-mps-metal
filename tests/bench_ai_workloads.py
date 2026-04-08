"""
FP8 Metal kernel benchmarks for AI inference workloads.

Measures throughput (GFLOPS), latency, and relative performance across
matrix shapes from real transformer architectures (Llama-2/Qwen3 style).

Compares:
  - FP8 fused kernel (in-register decode, no tiling)
  - FP8 fast path (dequant to FP16 + native matmul)
  - FP8 auto selector
  - FP16 native matmul (hardware-optimized baseline)
  - FP32 native matmul

Run:
    uv run pytest bench_ai_workloads.py -v
    uv run pytest bench_ai_workloads.py -v -m benchmark  # only benchmarks
    uv run python bench_ai_workloads.py                   # standalone report
"""

import sys
import os
import time
from dataclasses import dataclass, asdict

import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fp8_mps_native
from conftest import make_fp8_pair


# ── Helpers ──────────────────────────────────────────────────────────────────

def sync_and_time(fn, warmup=5, iters=20):
    """Run fn with warmup, return median time in ms."""
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()

    times = []
    for _ in range(iters):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def gflops(M, N, K, ms):
    if ms <= 0:
        return float('inf')
    return 2 * M * N * K / (ms / 1000) / 1e9


# ── AI workload matrix shapes ───────────────────────────────────────────────

# (label, M, K, N) — shapes from transformer linear projections
DECODE_SHAPES = [
    ("decode/qkv_proj",  1, 4096, 4096),
    ("decode/attn_out",  1, 4096, 4096),
    ("decode/ffn_gate",  1, 4096, 14336),
    ("decode/ffn_down",  1, 14336, 4096),
]

BATCH_SHAPES = [
    ("batch4/qkv_proj",  4, 4096, 4096),
    ("batch4/ffn_gate",  4, 4096, 14336),
]

PREFILL_SHAPES = [
    ("prefill128/qkv",   128, 4096, 4096),
    ("prefill128/ffn_g", 128, 4096, 14336),
    ("prefill128/ffn_d", 128, 14336, 4096),
    ("prefill512/qkv",   512, 4096, 4096),
    ("prefill512/ffn_g", 512, 4096, 14336),
    ("prefill2k/qkv",    2048, 4096, 4096),
]

ALL_SHAPES = DECODE_SHAPES + BATCH_SHAPES + PREFILL_SHAPES


# ── Pytest benchmarks ────────────────────────────────────────────────────────

@pytest.mark.benchmark
class TestMatmulBenchmarks:
    """Per-shape latency and GFLOPS across all kernel paths."""

    @pytest.mark.parametrize("label,M,K,N", ALL_SHAPES, ids=[s[0] for s in ALL_SHAPES])
    def test_shape_comparison(self, label, M, K, N):
        A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(M, K, N)
        A_f16 = torch.randn(M, K, dtype=torch.float16, device="mps")
        B_f16 = torch.randn(K, N, dtype=torch.float16, device="mps")

        fused_ms = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s))
        fast_ms = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s))
        auto_ms = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm_auto(A_q, B_q, A_s, B_s))
        fp16_ms = sync_and_time(lambda: A_f16 @ B_f16)

        best_fp8 = min(fused_ms, fast_ms)
        ratio = best_fp8 / fp16_ms

        print(f"\n  {label:>24}  M={M:<5} K={K:<5} N={N:<5}")
        print(f"    fused: {fused_ms:8.3f}ms  ({gflops(M,N,K,fused_ms):8.1f} GFLOPS)")
        print(f"    fast:  {fast_ms:8.3f}ms  ({gflops(M,N,K,fast_ms):8.1f} GFLOPS)")
        print(f"    auto:  {auto_ms:8.3f}ms  (selects {'fused' if M <= 16 else 'fast'})")
        print(f"    fp16:  {fp16_ms:8.3f}ms  ({gflops(M,N,K,fp16_ms):8.1f} GFLOPS)")
        print(f"    best_fp8/fp16: {ratio:.2f}x {'(FP8 wins)' if ratio < 1 else '(FP16 wins)'}")


@pytest.mark.benchmark
class TestPreparedWeight:
    """Benchmark prepared (cached) vs unprepared weight dequant."""

    @pytest.mark.parametrize("label,M,K,N", ALL_SHAPES, ids=[s[0] for s in ALL_SHAPES])
    def test_prepared_vs_unprepared(self, label, M, K, N):
        A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(M, K, N)
        B_prepared = fp8_mps_native.fp8_prepare_weight(B_q, B_s)

        unprepared_ms = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm_auto(A_q, B_q, A_s, B_s))
        prepared_ms = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm_auto(A_q, B_prepared, A_s, B_s))

        A_f16 = torch.randn(M, K, dtype=torch.float16, device="mps")
        B_f16 = torch.randn(K, N, dtype=torch.float16, device="mps")
        fp16_ms = sync_and_time(lambda: A_f16 @ B_f16)

        ratio = prepared_ms / fp16_ms
        speedup = unprepared_ms / prepared_ms

        print(f"\n  {label:>24}  M={M:<5} K={K:<5} N={N:<5}")
        print(f"    unprepared: {unprepared_ms:8.3f}ms")
        print(f"    prepared:   {prepared_ms:8.3f}ms  ({speedup:.2f}x faster)")
        print(f"    fp16:       {fp16_ms:8.3f}ms")
        print(f"    prepared/fp16: {ratio:.2f}x {'(FP8 wins)' if ratio < 1 else '(FP16 wins)'}")


@pytest.mark.benchmark
class TestCrossover:
    """Find where fused kernel loses to fast path as M grows."""

    @pytest.mark.parametrize("K,N", [(4096, 4096), (4096, 14336)])
    def test_crossover(self, K, N):
        print(f"\n  Crossover analysis K={K}, N={N}:")
        print(f"  {'M':>6}  {'fused':>10}  {'fast':>10}  {'winner':>8}  {'ratio':>8}")

        for M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(M, K, N)
            fused_ms = sync_and_time(
                lambda: fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s),
                warmup=3, iters=15,
            )
            fast_ms = sync_and_time(
                lambda: fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s),
                warmup=3, iters=15,
            )
            winner = "fused" if fused_ms < fast_ms else "fast"
            ratio = fused_ms / fast_ms
            print(f"  {M:>6}  {fused_ms:>9.3f}ms  {fast_ms:>9.3f}ms  {winner:>8}  {ratio:>7.2f}x")


@pytest.mark.benchmark
class TestDequantOverhead:
    """Measure per-call FP8->FP16 dequant cost (paid by fast path every call)."""

    @pytest.mark.parametrize("rows,cols", [
        (1, 4096),
        (1, 14336),
        (4096, 4096),
        (4096, 14336),
        (14336, 4096),
    ])
    def test_dequant_cost(self, rows, cols):
        data = torch.randint(0, 128, (rows, cols), dtype=torch.uint8, device="mps")
        scale = torch.tensor([0.01], device="mps")

        ms = sync_and_time(lambda: fp8_mps_native.fp8_dequantize(data, scale))
        elements = rows * cols
        bandwidth_gbps = (elements * 3) / (ms / 1000) / 1e9

        print(f"\n  {rows}x{cols}: {ms:.3f}ms  ({elements:,} elements, {bandwidth_gbps:.1f} GB/s)")


@pytest.mark.benchmark
class TestTransformerLayer:
    """Simulate one full transformer layer (all linear projections)."""

    @pytest.mark.parametrize("M", [1, 128, 512])
    def test_layer_latency(self, M):
        H, FFN = 4096, 14336

        # Pre-quantize weights
        weights = {}
        weights_f16 = {}
        for name, out_dim, in_dim in [
            ("qkv", 3 * H, H), ("out", H, H),
            ("gate", FFN, H), ("up", FFN, H), ("down", H, FFN),
        ]:
            W = torch.randn(out_dim, in_dim)
            wq, ws = fp8_mps_native.fp8_quantize(W)
            weights[name] = (wq, ws)
            weights_f16[name] = W.half().to("mps")

        # Activation
        x_f32 = torch.randn(M, H)
        x_q, x_s = fp8_mps_native.fp8_quantize(x_f32)
        x_f16 = x_f32.half().to("mps")

        def fp8_layer():
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["qkv"][0], x_s, weights["qkv"][1])
            mid_q, mid_s = fp8_mps_native.fp8_quantize(torch.randn(M, H, device="mps"))
            fp8_mps_native.fp8_scaled_mm_auto(mid_q, weights["out"][0], mid_s, weights["out"][1])
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["gate"][0], x_s, weights["gate"][1])
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["up"][0], x_s, weights["up"][1])
            ffn_q, ffn_s = fp8_mps_native.fp8_quantize(torch.randn(M, FFN, device="mps"))
            fp8_mps_native.fp8_scaled_mm_auto(ffn_q, weights["down"][0], ffn_s, weights["down"][1])

        def fp16_layer():
            x_f16 @ weights_f16["qkv"].T
            torch.randn(M, H, dtype=torch.float16, device="mps") @ weights_f16["out"].T
            x_f16 @ weights_f16["gate"].T
            x_f16 @ weights_f16["up"].T
            torch.randn(M, FFN, dtype=torch.float16, device="mps") @ weights_f16["down"].T

        fp8_ms = sync_and_time(fp8_layer, warmup=3, iters=10)
        fp16_ms = sync_and_time(fp16_layer, warmup=3, iters=10)

        total_flops = 2 * M * (H * 3 * H + H * H + H * FFN + H * FFN + FFN * H)
        fp8_tflops = total_flops / (fp8_ms / 1000) / 1e12
        fp16_tflops = total_flops / (fp16_ms / 1000) / 1e12
        ratio = fp8_ms / fp16_ms

        label = {1: "decode", 128: "prefill-128", 512: "prefill-512"}[M]
        print(f"\n  {label} (M={M}) — full layer (5 projections):")
        print(f"    FP8  auto:   {fp8_ms:8.2f}ms  ({fp8_tflops:.3f} TFLOPS)")
        print(f"    FP16 native: {fp16_ms:8.2f}ms  ({fp16_tflops:.3f} TFLOPS)")
        print(f"    ratio:       {ratio:.2f}x {'(FP8 wins)' if ratio < 1 else '(FP16 wins)'}")


@pytest.mark.benchmark
class TestMemoryTraffic:
    """Theoretical memory traffic analysis: untiled vs optimal."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 4096, 4096),
        (4, 4096, 4096),
        (16, 4096, 4096),
        (128, 4096, 4096),
        (512, 4096, 4096),
        (128, 4096, 14336),
        (512, 4096, 14336),
    ])
    def test_traffic_amplification(self, M, K, N):
        # Untiled: each of M*N threads reads 2*K bytes
        actual = 2 * M * N * K
        # Optimal (tiled): read A once + B once + write C
        optimal = M * K + N * K + M * N * 4
        amp = actual / optimal

        print(f"\n  M={M:<5} K={K:<5} N={N:<5}  "
              f"untiled={actual/1e6:.0f}MB  optimal={optimal/1e6:.0f}MB  "
              f"amplification={amp:.1f}x")


@pytest.mark.benchmark
class TestAccuracy:
    """Quantization error comparison: FP8 fused vs fast vs FP16."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 4096, 4096),
        (128, 4096, 4096),
        (128, 4096, 14336),
        (512, 4096, 4096),
    ])
    def test_relative_rmse(self, M, K, N):
        A_q, B_q, A_s, B_s, A_f32, B_f32 = make_fp8_pair(M, K, N)
        ref = A_f32 @ B_f32.T
        ref_rms = torch.sqrt((ref ** 2).mean()).item()

        fused = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s).cpu().float()
        fast = fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s).cpu().float()
        fp16 = (A_f32.half() @ B_f32.half().T).float()

        fused_rmse = torch.sqrt(((fused - ref) ** 2).mean()).item() / ref_rms
        fast_rmse = torch.sqrt(((fast - ref) ** 2).mean()).item() / ref_rms
        fp16_rmse = torch.sqrt(((fp16 - ref) ** 2).mean()).item() / ref_rms

        print(f"\n  M={M:<5} K={K:<5} N={N:<5}  "
              f"fused={fused_rmse:.4%}  fast={fast_rmse:.4%}  fp16={fp16_rmse:.4%}")


# ── Standalone mode ──────────────────────────────────────────────────────────

def print_report():
    """Run all benchmarks and print a consolidated report."""
    print(f"PyTorch {torch.__version__}")
    print(f"MPS: {torch.backends.mps.is_available()}")
    print()

    # Shape comparison table
    print("=" * 90)
    print("  MATMUL BENCHMARKS: FP8 vs FP16 across AI workload shapes")
    print("=" * 90)
    hdr = (f"  {'label':>24}  {'M':>5} {'K':>5} {'N':>5}  "
           f"{'fused_ms':>9} {'fast_ms':>9} {'fp16_ms':>9} {'best/fp16':>9} {'GFLOPS':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for label, M, K, N in ALL_SHAPES:
        A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(M, K, N)
        A_f16 = torch.randn(M, K, dtype=torch.float16, device="mps")
        B_f16 = torch.randn(K, N, dtype=torch.float16, device="mps")

        fused_ms = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s))
        fast_ms = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s))
        fp16_ms = sync_and_time(lambda: A_f16 @ B_f16)

        best = min(fused_ms, fast_ms)
        ratio = best / fp16_ms
        best_gf = max(gflops(M, N, K, fused_ms), gflops(M, N, K, fast_ms))

        print(f"  {label:>24}  {M:>5} {K:>5} {N:>5}  "
              f"{fused_ms:>8.3f}ms {fast_ms:>8.3f}ms {fp16_ms:>8.3f}ms "
              f"{ratio:>8.2f}x {best_gf:>8.1f}")

    # Crossover
    for K, N in [(4096, 4096), (4096, 14336)]:
        print(f"\n{'='*90}")
        print(f"  CROSSOVER: fused vs fast at K={K}, N={N}")
        print(f"{'='*90}")
        print(f"  {'M':>6}  {'fused':>10}  {'fast':>10}  {'winner':>8}")
        for M in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(M, K, N)
            fu = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s), warmup=3, iters=15)
            fa = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s), warmup=3, iters=15)
            print(f"  {M:>6}  {fu:>9.3f}ms  {fa:>9.3f}ms  {'fused' if fu < fa else 'fast':>8}")

    # Memory traffic
    print(f"\n{'='*90}")
    print("  MEMORY TRAFFIC AMPLIFICATION (untiled fused kernel)")
    print(f"{'='*90}")
    for M, K, N in [(1,4096,4096),(16,4096,4096),(128,4096,4096),(512,4096,4096),(128,4096,14336)]:
        actual = 2 * M * N * K
        optimal = M * K + N * K + M * N * 4
        print(f"  M={M:<5} K={K:<5} N={N:<5}  untiled={actual/1e6:>8.0f}MB  optimal={optimal/1e6:>6.0f}MB  amp={actual/optimal:.0f}x")

    # Accuracy
    print(f"\n{'='*90}")
    print("  ACCURACY: relative RMSE vs FP32 reference")
    print(f"{'='*90}")
    for M, K, N in [(1,4096,4096),(128,4096,4096),(128,4096,14336)]:
        A_q, B_q, A_s, B_s, A_f, B_f = make_fp8_pair(M, K, N)
        ref = A_f @ B_f.T
        rms = torch.sqrt((ref**2).mean()).item()
        fu_e = torch.sqrt(((fp8_mps_native.fp8_scaled_mm(A_q,B_q,A_s,B_s).cpu().float()-ref)**2).mean()).item()/rms
        fa_e = torch.sqrt(((fp8_mps_native.fp8_scaled_mm_fast(A_q,B_q,A_s,B_s).cpu().float()-ref)**2).mean()).item()/rms
        f16_e = torch.sqrt((((A_f.half()@B_f.half().T).float()-ref)**2).mean()).item()/rms
        print(f"  M={M:<5} K={K:<5} N={N:<5}  fused={fu_e:.4%}  fast={fa_e:.4%}  fp16={f16_e:.4%}")

    # Prepared weight comparison
    print(f"\n{'='*90}")
    print("  PREPARED WEIGHT: cached FP16 weights vs per-call dequant")
    print(f"{'='*90}")
    print(f"  {'label':>24}  {'unprepared':>12} {'prepared':>12} {'fp16':>12} {'prep/fp16':>10} {'speedup':>8}")
    for label, M, K, N in ALL_SHAPES:
        A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(M, K, N)
        B_prep = fp8_mps_native.fp8_prepare_weight(B_q, B_s)
        A_f16 = torch.randn(M, K, dtype=torch.float16, device="mps")
        B_f16 = torch.randn(K, N, dtype=torch.float16, device="mps")

        unprep = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm_auto(A_q, B_q, A_s, B_s))
        prep = sync_and_time(lambda: fp8_mps_native.fp8_scaled_mm_auto(A_q, B_prep, A_s, B_s))
        f16 = sync_and_time(lambda: A_f16 @ B_f16)
        print(f"  {label:>24}  {unprep:>11.3f}ms {prep:>11.3f}ms {f16:>11.3f}ms {prep/f16:>9.2f}x {unprep/prep:>7.2f}x")

    print(f"\n{'='*90}")
    print("  DONE")
    print(f"{'='*90}")


if __name__ == "__main__":
    if not torch.backends.mps.is_available():
        print("MPS not available")
        sys.exit(1)
    print_report()
