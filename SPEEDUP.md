# FP8 Metal Kernel Performance Analysis

Benchmark results from M4 Pro (48GB), PyTorch 2.11.0, MPS backend.
Tests and benchmarks live in `tests/` — run with `cd tests && uv run python bench_ai_workloads.py`.

## Benchmark Results

### FP8 vs FP16 Latency (AI workload shapes)

| Shape | M | K | N | Fused (ms) | Fast (ms) | FP16 (ms) | Best FP8 / FP16 |
|---|---|---|---|---|---|---|---|
| decode/qkv_proj | 1 | 4096 | 4096 | 0.832 | 0.882 | 0.463 | 1.79x slower |
| decode/attn_out | 1 | 4096 | 4096 | 0.561 | 0.914 | 0.291 | 1.93x slower |
| decode/ffn_gate | 1 | 4096 | 14336 | 0.553 | 2.456 | 0.650 | **0.85x (FP8 wins)** |
| decode/ffn_down | 1 | 14336 | 4096 | 0.568 | 2.422 | 0.630 | **0.90x (FP8 wins)** |
| batch4/qkv_proj | 4 | 4096 | 4096 | 0.692 | 0.976 | 0.408 | 1.70x slower |
| batch4/ffn_gate | 4 | 4096 | 14336 | 1.624 | 2.382 | 0.714 | 2.28x slower |
| prefill128/qkv | 128 | 4096 | 4096 | 12.528 | 0.995 | 0.498 | 2.00x slower |
| prefill128/ffn_g | 128 | 4096 | 14336 | 43.180 | 2.583 | 0.788 | 3.28x slower |
| prefill128/ffn_d | 128 | 14336 | 4096 | 43.513 | 2.646 | 0.794 | 3.33x slower |
| prefill512/qkv | 512 | 4096 | 4096 | 49.198 | 1.408 | 0.752 | 1.87x slower |
| prefill512/ffn_g | 512 | 4096 | 14336 | 171.399 | 4.103 | 2.090 | 1.96x slower |
| prefill2k/qkv | 2048 | 4096 | 4096 | 195.631 | 3.371 | 2.358 | 1.43x slower |

### Fused vs Fast Crossover Point

At K=N=4096, fused wins only at M=1-2, fast wins from M=4+.
At K=4096, N=14336, fused wins at M=1-4, fast wins from M=8+.
The current auto-selector threshold `M<=16` is too generous.

| M | Fused (ms) | Fast (ms) | Winner | (K=N=4096) |
|---|---|---|---|---|
| 1 | 0.267 | 0.900 | fused | |
| 2 | 0.501 | 1.875 | fused | |
| 4 | 0.958 | 0.936 | fast | |
| 8 | 1.074 | 0.937 | fast | |
| 16 | 1.820 | 1.486 | fast | |
| 32 | 3.368 | 1.154 | fast | |
| 128 | 12.572 | 0.945 | fast | |
| 256 | 24.716 | 1.107 | fast | |

### Memory Traffic Amplification (untiled fused kernel)

| M | K | N | Untiled reads | Optimal reads | Amplification |
|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 34 MB | 17 MB | 2x |
| 16 | 4096 | 4096 | 537 MB | 17 MB | 31x |
| 128 | 4096 | 4096 | 4,295 MB | 19 MB | 221x |
| 512 | 4096 | 4096 | 17,180 MB | 27 MB | 630x |
| 128 | 4096 | 14336 | 15,032 MB | 67 MB | 226x |

### Accuracy (relative RMSE vs FP32 reference)

| M | K | N | Fused | Fast | FP16 |
|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 3.93% | 3.93% | 0.04% |
| 128 | 4096 | 4096 | 3.99% | 3.99% | 0.04% |
| 128 | 4096 | 14336 | 4.00% | 3.99% | 0.04% |

~4% RMSE is expected for 8-bit quantization. Both kernel paths produce equivalent accuracy.

---

## Problems Found

### P1: Untiled 2D matmul kernel (CRITICAL for prefill)

**File:** `fp8_matmul.metal:95-143` (`fp8_scaled_matmul_kernel`)

Each thread reads entire rows of A and B from global memory independently. No threadgroup shared memory. For M=128, K=N=4096, this produces 4.3 GB of global reads vs 19 MB with tiling — a 221x memory traffic amplification. The kernel is 12-43x slower than the fast path at prefill sizes and 2-3x slower than native FP16.

**Proposed fix:** Implement register-tiled blocked GEMM. Load `TILE_M x TILE_K` and `TILE_K x TILE_N` submatrices into threadgroup memory, barrier, then let all threads in the group reuse those tiles. A 32x32x32 tile scheme reduces global reads from `O(M*N*K)` to `O(M*K + N*K)`. This is the standard GPU matmul optimization and should close most of the gap to native FP16.

### P2: Branchy per-element FP8 decode with `exp2()`

**File:** `fp8_matmul.metal:19-40` (`fp8_e4m3fn_to_float`)

Two branches (NaN check, subnormal vs normal) + `exp2()` transcendental per element, called 8x per 4-element unroll step. Causes SIMD divergence on subnormal values and ALU bottleneck from `exp2`.

**Proposed fix:** 256-entry lookup table in threadgroup memory (256 x 4 bytes = 1 KB). Lane 0 fills it once, `threadgroup_barrier`, then all lanes replace the decode function with a single indexed load. Eliminates all branches and the `exp2` call.

### P3: Uncoalesced memory access in vecmat kernel

**File:** `fp8_matmul.metal:173` (`fp8_scaled_vecmat_kernel`)

SIMD lanes stride by 128 bytes (`simd_lane * 4`, step `32 * 4`). Consecutive lanes access bytes at offsets 0, 16, 32, ... instead of 0, 1, 2, ... — preventing the memory controller from coalescing into single cache line fetches.

**Proposed fix:** Restructure the inner loop so consecutive lanes access consecutive bytes:
```metal
for (uint k = 0; k < K; k += 32) {
    float xi = fp8_lut[x[k + simd_lane]];
    float wi = fp8_lut[W[row_offset + k + simd_lane]];
    sum += xi * wi;
}
```
This puts lanes 0-31 on 32 contiguous bytes per iteration — one cache line.

### P4: Fast path re-dequantizes weights every call

**File:** `fp8_mps_native.py:204-231` (`fp8_scaled_mm_fast`)

Weight matrix B is dequantized from FP8 to FP16 on every call. During inference, B doesn't change between tokens. For a 32-layer transformer with 5 projections per layer, that's 160 redundant dequant operations per forward pass. Each dequant of B at N=K=4096 processes 16M elements.

**Proposed fix:** Add a weight cache API. Pre-dequantize and cache B_f16 once per model load. The matmul call then accepts pre-dequantized FP16 weights directly, skipping the dequant dispatch entirely. This is the single largest optimization opportunity for repeated inference.

### P5: Extra kernel dispatches in fast path

**File:** `fp8_mps_native.py:224-233`

The fast path dispatches 6 GPU operations per matmul: dequant A, dequant B, scale A, scale B, matmul, cast to f32. The two scale multiplies (`A_f16 * sa`, `B_f16 * sb`) could be fused into the dequant kernel. The f32 cast at the end is unnecessary if the caller accepts FP16 (which transformers typically can until the final logit layer).

**Proposed fix:** Write a `fp8_to_scaled_half_kernel` that applies `output[gid] = half(fp8_to_float(input[gid]) * scale)` in one pass. Optionally return FP16 output directly. Reduces 6 dispatches to 3 (dequant_A, dequant_B, matmul).

### P6: Wrong auto-selector threshold

**File:** `fp8_mps_native.py:174` (`fp8_scaled_mm_auto`)

The `M<=16` threshold routes M=2..16 to the fused 2D kernel, which is already slower than the fast path at M=4 for K=N=4096. Only M=1 (vecmat kernel) genuinely benefits from the fused path.

**Proposed fix:** Use vecmat for M=1 only, fast path for M>=2. Once the tiled kernel (P1) is implemented, re-benchmark and add a third tier for large M where the tiled fused kernel beats dequant+native.

### P7: C++ bridge CPU round-trips (low priority)

**File:** `fp8_bridge.cpp:180-258`

The C++ bridge forces MPS→CPU→Metal→CPU→MPS copies for all tensors. The native Python path (`fp8_mps_native.py`) avoids this entirely via `torch.mps.compile_shader()`. The C++ bridge exists for compatibility but should be considered deprecated for performance-sensitive use.

**Proposed fix:** Deprecate in favor of the native path. Only keep for testing or environments where `torch.mps.compile_shader()` is unavailable.

---

## Implementation Results

### P6: Auto-selector threshold M<=16 → M<=4 (MERGED)

Changed `fp8_scaled_mm_auto` threshold from `M<=16` to `M<=4`. Crossover benchmarks showed the fused kernel only beats the fast path at M=1-4; from M=8+ the fast path wins at both K=N=4096 and K=4096,N=14336.

| M | K | N | Before (fused) | After (fast) | Improvement |
|---|---|---|---|---|---|
| 8 | 4096 | 4096 | 1.05ms | 0.93ms | 12% faster |
| 16 | 4096 | 4096 | 1.83ms | 0.89ms | **51% faster** |
| 8 | 4096 | 14336 | 2.95ms | 2.37ms | 20% faster |
| 16 | 4096 | 14336 | 5.64ms | 2.39ms | **58% faster** |

No regression for M=1, M=4 (still use fused), or M>=128 (already used fast path).

---

## Proposed Implementation Order

1. ~~**P6** (threshold fix)~~ — DONE
2. **P2** (LUT decode) — moderate effort, speeds up all kernel paths
3. **P5** (fused scale+dequant kernel) — moderate effort, speeds up fast path
4. **P3** (coalesced vecmat) — moderate effort, speeds up M=1 decode
5. **P4** (weight cache) — API change, biggest win for inference loops
6. **P1** (tiled 2D kernel) — significant effort, needed for prefill parity with FP16
7. **P7** (deprecate C++ bridge) — cleanup
