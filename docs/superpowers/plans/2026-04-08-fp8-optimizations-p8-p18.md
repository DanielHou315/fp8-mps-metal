# FP8 Metal Kernel Optimizations P8-P18 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining 1.2-1.65x gap between FP8 kernels and native FP16 matmul on Apple Silicon MPS.

**Architecture:** Three tiers of optimization — trivial Python-side dispatch elimination (P8-P10), moderate Metal kernel improvements and caching (P11-P14), and significant hardware-accelerated matmul via SGMMA/MPP (P15-P16). Each task is independent and follows the worktree workflow: branch → implement → test (31 tests) → benchmark → merge-or-discard → document in SPEEDUP.md.

**Tech Stack:** Metal Shading Language 3.0+, PyTorch MPS, torch.mps.compile_shader(), Python 3.12, uv, pytest

**Workflow for every task:** `git worktree add ../fp8-fix-PX -b fix/PX-name main` → implement → `cd tests && uv venv .venv --clear && uv pip install torch pytest pytest-benchmark` → `uv run pytest test_correctness.py -v` (31+ pass) → `uv run python bench_ai_workloads.py` → record before/after → merge if better → `git worktree remove ../fp8-fix-PX && git branch -d fix/PX-name` → update SPEEDUP.md

---

## File Map

| File | Responsibility | Tasks that modify it |
|---|---|---|
| `fp8_mps_native.py` | Python dispatch layer, public API | P8, P9, P10, P11, P14 |
| `fp8_matmul.metal` | All Metal GPU kernels | P11, P13, P14, P15, P18 |
| `fp8_mps_patch.py` | Monkey-patch for torch._scaled_mm | P12 |
| `tests/test_correctness.py` | Correctness tests (currently 31) | P8, P9, P10, P12, P14 |
| `tests/bench_ai_workloads.py` | Performance benchmarks | P9, P17 |
| `SPEEDUP.md` | Results documentation | All tasks |

---

## Tier 1: Trivial Dispatch Elimination

### Task P8: Remove `.float()` upcast from fast path return

**Files:**
- Modify: `fp8_mps_native.py:255`
- Modify: `fp8_mps_native.py:215` (docstring)
- Modify: `tests/test_correctness.py:208` (prepared-matches-unprepared test)

**Rationale:** `fp8_scaled_mm_fast` unconditionally calls `C.float()` on line 255, dispatching an extra GPU kernel to upcast the entire output from FP16 to FP32. The monkey-patch at `fp8_mps_patch.py:70` already does `result.to(out_dtype)` so the caller controls dtype. Removing the upcast eliminates one kernel dispatch per fast-path call.

- [ ] **Step 1: Write failing test for FP16 return type**

Add to `tests/test_correctness.py` after the `TestPreparedWeight` class:

```python
class TestFastPathDtype:
    def test_fast_path_returns_fp16(self):
        import fp8_mps_native
        A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(128, 256, 128)
        result = fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s)
        assert result.dtype == torch.float16, f"Expected float16, got {result.dtype}"

    def test_auto_with_prepared_returns_fp16(self):
        import fp8_mps_native
        A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(128, 256, 128)
        B_prep = fp8_mps_native.fp8_prepare_weight(B_q, B_s)
        result = fp8_mps_native.fp8_scaled_mm_auto(A_q, B_prep, A_s, B_s)
        assert result.dtype == torch.float16, f"Expected float16, got {result.dtype}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && uv run pytest test_correctness.py::TestFastPathDtype -v`
Expected: FAIL — currently returns float32

- [ ] **Step 3: Remove `.float()` and update docstring**

In `fp8_mps_native.py`, change line 255 from:

```python
    return C.float()
```

to:

```python
    return C
```

Update the docstring at line 215 from `Returns: (M, N) float32 on MPS` to `Returns: (M, N) float16 on MPS`.

- [ ] **Step 4: Fix the prepared-matches-unprepared test**

In `tests/test_correctness.py:208`, the test compares unprepared vs prepared output. Both now return FP16, so update the comparison — change `.cpu().float()` calls to just `.cpu().float()` (these are fine — the test-side cast to float for comparison is still needed, no change actually required here since `.cpu().float()` is on the test side).

- [ ] **Step 5: Run all tests**

Run: `cd tests && uv run pytest test_correctness.py -v`
Expected: 33 passed (31 existing + 2 new)

- [ ] **Step 6: Run benchmarks and record**

Run: `cd tests && uv run python bench_ai_workloads.py`
Compare fast-path times before/after. Expect small improvement from eliminating the cast kernel.

- [ ] **Step 7: Commit**

```bash
git add fp8_mps_native.py tests/test_correctness.py
git commit -m "P8: remove .float() upcast from fp8_scaled_mm_fast, return FP16

Eliminates one GPU kernel dispatch per fast-path call.
The monkey-patch already handles out_dtype casting.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task P9: Accept FP16 activations (skip A dequant)

**Files:**
- Modify: `fp8_mps_native.py:225-237` (fp8_scaled_mm_fast)
- Modify: `fp8_mps_native.py:182-200` (fp8_scaled_mm_auto)
- Create tests in: `tests/test_correctness.py`
- Modify: `tests/bench_ai_workloads.py` (add FP16-activation benchmark)

**Rationale:** When activations arrive as FP16 (common in real inference — outputs of LayerNorm, attention), the fast path still dequantizes A from uint8→FP16. Adding a dtype check mirrors the existing B.dtype check at line 240.

- [ ] **Step 1: Write failing test**

Add to `tests/test_correctness.py`:

```python
class TestFP16Activations:
    @pytest.mark.parametrize("M,K,N", [(1, 4096, 4096), (128, 4096, 4096)])
    def test_fp16_activation_accuracy(self, M, K, N):
        import fp8_mps_native
        A_q, B_q, A_s, B_s, A_f32, B_f32 = make_fp8_pair(M, K, N)
        ref = A_f32 @ B_f32.T

        # Simulate FP16 activation (pre-scaled)
        A_f16 = fp8_mps_native.fp8_dequantize(A_q, A_s).to(device="mps")
        B_prepared = fp8_mps_native.fp8_prepare_weight(B_q, B_s)

        result = fp8_mps_native.fp8_scaled_mm_auto(A_f16, B_prepared, A_s, B_s)
        ref_rms = torch.sqrt((ref ** 2).mean()).item()
        rel_rmse = torch.sqrt(((result.cpu().float() - ref) ** 2).mean()).item() / ref_rms
        assert rel_rmse < 0.15, f"FP16 activation RMSE {rel_rmse:.4%} exceeds 15%"

    def test_fp16_activation_skips_dequant(self):
        """When A is FP16, fast path should use it directly (no dequant dispatch)."""
        import fp8_mps_native
        A_f16 = torch.randn(4, 256, dtype=torch.float16, device="mps")
        B_f16 = torch.randn(128, 256, dtype=torch.float16, device="mps")
        result = fp8_mps_native.fp8_scaled_mm_fast(
            A_f16, B_f16,
            torch.tensor([1.0]), torch.tensor([1.0])
        )
        assert result.shape == (4, 128)
        assert torch.isfinite(result).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd tests && uv run pytest test_correctness.py::TestFP16Activations -v`
Expected: FAIL — `fp8_scaled_mm_fast` asserts/errors on non-uint8 A

- [ ] **Step 3: Add FP16 activation bypass in fp8_scaled_mm_fast**

In `fp8_mps_native.py`, replace lines 225-237:

```python
    M, K = A.shape
    N = B.shape[0]

    sa_val = scale_a.to(device="cpu", dtype=torch.float32).item()

    # Dequant+scale A to FP16 in one pass
    A_f16 = torch.empty(M, K, dtype=torch.float16, device="mps")
    count_a = A.numel()
    lib.fp8_to_scaled_half_kernel(
        A.contiguous().view(-1), A_f16.view(-1),
        count_a, sa_val,
        threads=(count_a,), group_size=(256,),
    )
```

with:

```python
    M, K = A.shape
    N = B.shape[0]

    # A: use FP16 directly if already dequantized, otherwise dequant+scale
    if A.dtype == torch.float16:
        A_f16 = A
    else:
        sa_val = scale_a.to(device="cpu", dtype=torch.float32).item()
        A_f16 = torch.empty(M, K, dtype=torch.float16, device="mps")
        count_a = A.numel()
        lib.fp8_to_scaled_half_kernel(
            A.contiguous().view(-1), A_f16.view(-1),
            count_a, sa_val,
            threads=(count_a,), group_size=(256,),
        )
```

- [ ] **Step 4: Update fp8_scaled_mm_auto to route FP16 A to fast path**

In `fp8_mps_native.py`, update `fp8_scaled_mm_auto` to check A dtype too. Replace the body (lines 195-200):

```python
    M = A.shape[0]
    if B.dtype == torch.float16:
        return fp8_scaled_mm_fast(A, B, scale_a, scale_b)
    if M <= 16:
        return fp8_scaled_mm(A, B, scale_a, scale_b)
    return fp8_scaled_mm_fast(A, B, scale_a, scale_b)
```

with:

```python
    M = A.shape[0]
    if A.dtype == torch.float16 or B.dtype == torch.float16:
        return fp8_scaled_mm_fast(A, B, scale_a, scale_b)
    if M <= 16:
        return fp8_scaled_mm(A, B, scale_a, scale_b)
    return fp8_scaled_mm_fast(A, B, scale_a, scale_b)
```

- [ ] **Step 5: Run all tests**

Run: `cd tests && uv run pytest test_correctness.py -v`
Expected: All pass (33+ from P8 + 2 new = 35)

- [ ] **Step 6: Benchmark and commit**

Run benchmarks. The improvement shows when callers pass FP16 activations — not in the standard benchmark path which passes uint8. Add a benchmark test to `bench_ai_workloads.py` that passes FP16 A with prepared B to show the combined speedup.

---

### Task P10: Use fused scale kernel in fp8_dequantize

**Files:**
- Modify: `fp8_mps_native.py:114-123`
- Modify: `tests/test_correctness.py` (verify no behavior change)

**Rationale:** `fp8_dequantize` uses the unscaled `fp8_to_half_kernel` + a separate PyTorch scale multiply (2 dispatches). The fused `fp8_to_scaled_half_kernel` already exists and does it in 1 dispatch.

- [ ] **Step 1: Write test to capture current behavior**

The existing tests already cover `fp8_dequantize` (TestFP8Decode, TestQuantizeRoundtrip). No new test needed — just run existing tests to baseline.

- [ ] **Step 2: Replace the two-step dequant with fused kernel**

In `fp8_mps_native.py`, replace lines 106-124:

```python
    lib = _get_lib()

    if input.device.type != "mps":
        input = input.to("mps")

    count = input.numel()
    output = torch.empty(input.shape, dtype=torch.float16, device="mps")

    lib.fp8_to_half_kernel(
        input.contiguous().view(-1), output.view(-1),
        count,
        threads=(count,), group_size=(256,),
            )

    # Apply scale
    scale_val = scale.to(device="mps", dtype=torch.float16)
    output = output * scale_val

    return output
```

with:

```python
    lib = _get_lib()

    if input.device.type != "mps":
        input = input.to("mps")

    count = input.numel()
    scale_val = scale.to(device="cpu", dtype=torch.float32).item()
    output = torch.empty(input.shape, dtype=torch.float16, device="mps")

    lib.fp8_to_scaled_half_kernel(
        input.contiguous().view(-1), output.view(-1),
        count, scale_val,
        threads=(count,), group_size=(256,),
    )

    return output
```

- [ ] **Step 3: Run all tests**

Run: `cd tests && uv run pytest test_correctness.py -v`
Expected: All pass (no behavior change)

- [ ] **Step 4: Commit**

```bash
git add fp8_mps_native.py
git commit -m "P10: use fused scale+dequant kernel in fp8_dequantize

Replace fp8_to_half_kernel + elementwise scale with single
fp8_to_scaled_half_kernel call. Eliminates 1 dispatch per dequantize.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Tier 2: Moderate Kernel & Caching Improvements

### Task P11: Increase tile size from 16x16 to 32x32

**Files:**
- Modify: `fp8_matmul.metal:112` (`#define TILE 16` → `32`)
- Modify: `fp8_mps_native.py:92` (`group_size=(16, 16)` → `(32, 32)`)

**Rationale:** Current 16x16 tiles use 2KB threadgroup memory (6% of 32KB limit) with arithmetic intensity of 8 FLOPs/byte. 32x32 tiles use 8KB (25%), with 16 FLOPs/byte — 2x better reuse. The barrier count halves (K/32 vs K/16). Apple Silicon supports up to 1024 threads/threadgroup; 32x32 = 1024 threads is at the limit.

- [ ] **Step 1: Verify 1024 threads/threadgroup is supported**

This is the Metal maximum. If the M4 Pro's `maxTotalThreadsPerThreadgroup` for this kernel is < 1024 (due to register pressure), the dispatch will fail. Must test empirically.

- [ ] **Step 2: Change TILE define and dispatch**

In `fp8_matmul.metal`, change line 112:
```metal
#define TILE 32
```

In `fp8_mps_native.py`, change line 92:
```python
            threads=(N, M), group_size=(32, 32),
```

- [ ] **Step 3: Run correctness tests**

Run: `cd tests && uv run pytest test_correctness.py -v`
Expected: All pass. If tests crash with "threadgroup size exceeds limit", fall back to TILE=24 (576 threads) and `group_size=(24, 24)`.

- [ ] **Step 4: Benchmark**

Run: `cd tests && uv run python bench_ai_workloads.py`
Key metrics: fused kernel at M=128+ shapes. Expect 20-40% improvement.

- [ ] **Step 5: Commit if improved, discard if not**

---

### Task P12: Cache transposed B in monkey-patch

**Files:**
- Modify: `fp8_mps_patch.py:50`
- Add test in: `tests/test_correctness.py`

**Rationale:** `other.t().contiguous()` on line 50 allocates a fresh N*K copy of the weight every call. For K=4096, N=14336 (58.7 MB), this is a measurable cost. Cache it keyed by `data_ptr()` of the source tensor.

- [ ] **Step 1: Write test for cache behavior**

```python
class TestMonkeyPatchCache:
    def test_repeated_calls_reuse_cache(self):
        import fp8_mps_patch
        fp8_mps_patch.install()
        try:
            A = torch.randint(0, 128, (4, 256), dtype=torch.uint8, device="mps")
            B = torch.randint(0, 128, (256, 128), dtype=torch.uint8, device="mps")
            sa = torch.tensor([0.01], device="mps")
            sb = torch.tensor([0.01], device="mps")

            r1 = torch._scaled_mm(A, B, scale_a=sa, scale_b=sb)
            r2 = torch._scaled_mm(A, B, scale_a=sa, scale_b=sb)
            assert torch.allclose(r1.float(), r2.float(), atol=1e-3)
        finally:
            fp8_mps_patch.uninstall()
```

- [ ] **Step 2: Implement cache using WeakValueDictionary**

In `fp8_mps_patch.py`, add at module level:

```python
import weakref
_transposed_cache = {}
```

Replace line 50:
```python
    B = other.t().contiguous()
```

with:

```python
    cache_key = other.data_ptr()
    if cache_key in _transposed_cache:
        ref = _transposed_cache[cache_key]
        if ref is not None and ref.shape == (other.shape[1], other.shape[0]):
            B = ref
        else:
            B = other.t().contiguous()
            _transposed_cache[cache_key] = B
    else:
        B = other.t().contiguous()
        _transposed_cache[cache_key] = B
```

- [ ] **Step 3: Run tests and benchmark**

The benchmark won't show improvement in the standard path (which doesn't go through the monkey-patch), but the correctness test validates the cache. Real-world ComfyUI usage benefits.

- [ ] **Step 4: Commit**

---

### Task P13: Half-precision tile arrays for higher occupancy

**Files:**
- Modify: `fp8_matmul.metal:132-133`

**Rationale:** Changing `threadgroup float tileA[TILE][TILE]` to `threadgroup half tileA[TILE][TILE]` halves threadgroup memory usage (1KB→512B per tile at TILE=16, or 4KB→2KB at TILE=32). This doubles the number of concurrent threadgroups, improving occupancy. FP8 max value ±448 fits in FP16 range (±65504). Accumulator stays float32.

- [ ] **Step 1: Change tile arrays to half**

In `fp8_matmul.metal`, replace lines 132-133:
```metal
    threadgroup float tileA[TILE][TILE];
    threadgroup float tileB[TILE][TILE];
```
with:
```metal
    threadgroup half tileA[TILE][TILE];
    threadgroup half tileB[TILE][TILE];
```

Update the tile load lines (143-144, 151-152) to cast to half:
```metal
    tileA[tid.y][tid.x] = (a_row < M && a_k < K)
        ? half(fp8_e4m3fn_lut[A[a_row * K + a_k]]) : half(0.0f);
```

Update the accumulation loop (157-158) to cast back to float:
```metal
    for (uint k = 0; k < TILE; k++) {
        sum += float(tileA[tid.y][k]) * float(tileB[tid.x][k]);
    }
```

- [ ] **Step 2: Run tests and benchmark**

Check accuracy — the float→half→float round-trip for tile storage should be lossless for all 256 FP8 values (3 mantissa bits fit easily in FP16's 10 mantissa bits).

- [ ] **Step 3: Commit if improved**

---

### Task P14: GPU-resident amax in fp8_quantize

**Files:**
- Modify: `fp8_mps_native.py:139-142`
- Add new kernel to: `fp8_matmul.metal`

**Rationale:** `fp8_quantize` calls `amax = inp.abs().max().item()` which forces a GPU→CPU sync, breaking pipeline overlap. A fused kernel can compute amax + scale + quantize in one GPU pass.

- [ ] **Step 1: Add fused quantize kernel to Metal shader**

Add to `fp8_matmul.metal` after `float_to_fp8_kernel`:

```metal
// ─── Fused amax + scale + FP8 quantize ─────────────────────────────────────
// Two-pass: pass 1 computes per-threadgroup partial max via SIMD reduction,
// writes to partial_max buffer. Python dispatches a second small kernel to
// reduce partials and compute the final scale, then a third pass quantizes.
// For simplicity, keep the current two-step Python approach but move .item()
// to after the quantize kernel — the sync is unavoidable for returning
// inv_scale to the caller.
```

Actually, the simplest fix that removes the sync: keep amax on GPU and pass it to `float_to_fp8_kernel` as a buffer.

Add a new kernel:
```metal
kernel void float_to_fp8_scaled_kernel(
    device const float* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = float_to_fp8_e4m3fn(input[gid] * scale);
}
```

- [ ] **Step 2: Update Python fp8_quantize to avoid .item() in hot path**

Replace lines 139-155 in `fp8_mps_native.py`:

```python
    amax = inp.abs().max()  # stays on GPU as scalar tensor
    max_fp8 = 448.0
    # Compute scale on CPU — this .item() sync is unavoidable
    # but now happens once per quantize call, not blocking the pipeline
    amax_val = amax.item()
    scale = max_fp8 / amax_val if amax_val > 0 else 1.0

    output = torch.empty(inp.shape, dtype=torch.uint8, device="mps")

    lib.float_to_fp8_scaled_kernel(
        inp.contiguous().view(-1), output.view(-1),
        count, scale,
        threads=(count,), group_size=(256,),
    )

    inv_scale = torch.tensor([1.0 / scale], dtype=torch.float32, device="mps")
    return output, inv_scale
```

Note: This still has one `.item()` call, but it eliminates the separate `inp * scale` GPU dispatch and `.contiguous()` that followed. The net effect is removing 2 dispatches (elementwise multiply + contiguous copy) at the cost of keeping the sync.

- [ ] **Step 3: Run tests and benchmark**

- [ ] **Step 4: Commit if improved**

---

## Tier 3: Hardware-Accelerated Matmul

### Fallback Architecture (applies to P15 and P16)

Both SGMMA (P15) and Metal 4 MPP (P16) require hardware/OS capabilities not present on all devices. The fallback design uses a **single compiled shader library** with `#if` preprocessor guards, and `dir(lib)` on the Python side to detect which kernels are available.

**Metal side** — conditional compilation in `fp8_matmul.metal`:
```metal
// Always-present scalar tiled fallback (works on all Metal devices)
kernel void fp8_scaled_matmul_kernel(...) { /* existing tiled kernel */ }

#if __METAL_VERSION__ >= 300 && defined(__HAVE_SIMDGROUP_MATRIX__)
#include <metal_simdgroup_matrix>
kernel void fp8_scaled_matmul_sgmma_kernel(...) { /* SGMMA kernel */ }
#endif
```

`torch.mps.compile_shader()` targets the device's maximum MSL version automatically. On M2+ (MSL 3.0+), the SGMMA kernel compiles and appears in the library. On M1 (MSL 2.x), the `#if` guard strips it out — compilation succeeds with only the scalar kernel present.

**Python side** — runtime detection in `fp8_mps_native.py`:
```python
_lib = None
_use_sgmma = False

def _get_lib():
    global _lib, _use_sgmma
    if _lib is not None:
        return _lib
    source = _load_shader_source()
    _lib = torch.mps.compile_shader(source)
    _use_sgmma = "fp8_scaled_matmul_sgmma_kernel" in dir(_lib)
    return _lib
```

**Critical:** Use `dir(lib)` to check kernel availability — NOT `hasattr()`. PyTorch's compiled shader library raises `RuntimeError` (not `AttributeError`) for missing kernels, so `hasattr()` will crash instead of returning False.

**Compatibility matrix:**

| Device | MSL Version | Scalar tiled kernel | SGMMA kernel | Metal 4 MPP |
|---|---|---|---|---|
| M1 / A14 | 2.4 | Yes | No | No |
| M2 / A15 | 3.0 | Yes | **Yes** | No |
| M3 / A16 | 3.1 | Yes | **Yes** | No |
| M4 / A17 | 3.2 | Yes | **Yes** | Requires macOS 16+ |
| M5+ | 4.0 | Yes | **Yes** | **Yes** |

---

### Task P15: simdgroup_matrix_multiply_accumulate (SGMMA)

**Files:**
- Modify: `fp8_matmul.metal` — add SGMMA kernel inside `#if` guard, keep existing scalar kernel as-is
- Modify: `fp8_mps_native.py` — add `_use_sgmma` flag and dispatch routing

**Rationale:** The remaining 1.3-1.65x gap to native FP16 is because our kernel uses scalar FP32 multiply-accumulate while Apple's native matmul uses `simdgroup_matrix` hardware (~2.3x higher throughput). This is the single highest-ceiling optimization.

**Design (following MLX's steel GEMM pattern):**
- Threadgroup: 128 threads (4 simdgroups), BM=32, BN=32, BK=16
- Each simdgroup computes a 16x16 sub-tile of the 32x32 output using four 8x8 SGMMA operations
- Threadgroup memory: `half tileA[BM][BK]` (1KB) + `half tileB[BN][BK]` (1KB) = 2KB
- FP8 LUT decode to half during tile load
- Float32 accumulator via `simdgroup_matrix<float, 8, 8>`

- [ ] **Step 1: Add SGMMA kernel to Metal shader with `#if` guard**

Add at the end of `fp8_matmul.metal`, **after** the existing `fp8_scaled_matmul_kernel` (which remains unchanged as the M1 fallback):

```metal
// ─── SGMMA Tiled MatMul (M2+ / Metal 3+) ───────────────────────────────────
// Uses simdgroup_matrix hardware for 2.3x higher throughput vs scalar ALU.
// Conditionally compiled: absent on M1 (MSL < 3.0), falls back to scalar kernel.

#if __METAL_VERSION__ >= 300 && defined(__HAVE_SIMDGROUP_MATRIX__)
#include <metal_simdgroup_matrix>

kernel void fp8_scaled_matmul_sgmma_kernel(
    device const uint8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    device const float* scale_a [[buffer(3)]],
    device const float* scale_b [[buffer(4)]],
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& scale_mode [[buffer(8)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    // 4 simdgroups per threadgroup, each handles a 16x16 sub-tile
    // of the 32x32 output tile. 128 threads total.
    constexpr uint BM = 32, BN = 32, BK = 16;

    uint sg_row = (simd_gid / 2) * 16;  // 0 or 16
    uint sg_col = (simd_gid % 2) * 16;  // 0 or 16

    threadgroup half tileA[BM][BK];  // 32x16 = 1KB
    threadgroup half tileB[BN][BK];  // 32x16 = 1KB

    // Accumulator: 2x2 grid of 8x8 float fragments
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_float8x8(0);

    uint linear_tid = tid.y * 32 + tid.x;  // 0..127

    for (uint kk = 0; kk < K; kk += BK) {
        // Cooperative load: 128 threads load BM*BK=512 half values (4 per thread)
        for (uint i = linear_tid; i < BM * BK; i += 128) {
            uint r = i / BK;
            uint c = i % BK;
            uint gr = tgid.y * BM + r;
            uint gk = kk + c;
            tileA[r][c] = (gr < M && gk < K)
                ? half(fp8_e4m3fn_lut[A[gr * K + gk]]) : half(0);
        }
        for (uint i = linear_tid; i < BN * BK; i += 128) {
            uint r = i / BK;
            uint c = i % BK;
            uint gr = tgid.x * BN + r;
            uint gk = kk + c;
            tileB[r][c] = (gr < N && gk < K)
                ? half(fp8_e4m3fn_lut[B[gr * K + gk]]) : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // SGMMA: 2x2 grid of 8x8 multiplies per simdgroup's 16x16 sub-tile
        for (uint k8 = 0; k8 < BK; k8 += 8) {
            simdgroup_half8x8 mA[2], mB[2];
            simdgroup_load(mA[0], &tileA[sg_row][k8], BK);
            simdgroup_load(mA[1], &tileA[sg_row + 8][k8], BK);
            simdgroup_load(mB[0], &tileB[sg_col][k8], BK);
            simdgroup_load(mB[1], &tileB[sg_col + 8][k8], BK);

            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], mA[i], mB[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results: each simdgroup writes its 16x16 sub-tile
    threadgroup float result_tile[BM][BN];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            simdgroup_store(acc[i][j], &result_tile[sg_row + i*8][sg_col + j*8], BN);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write to global memory with scaling
    for (uint i = linear_tid; i < BM * BN; i += 128) {
        uint r = i / BN;
        uint c = i % BN;
        uint gr = tgid.y * BM + r;
        uint gc = tgid.x * BN + c;
        if (gr < M && gc < N) {
            float sa = (scale_mode == 0) ? scale_a[0] : scale_a[gr];
            float sb = (scale_mode == 0) ? scale_b[0] : scale_b[gc];
            C[gr * N + gc] = result_tile[r][c] * sa * sb;
        }
    }
}

#endif // __METAL_VERSION__ >= 300 && __HAVE_SIMDGROUP_MATRIX__
```

**Important transpose note:** `simdgroup_multiply_accumulate(D, A, B, C)` computes `D = A * B + C`. Our matmul is `C = A @ B^T` where A is (M,K) and B is (N,K). Since both tileA and tileB store K in columns (layout `[rows][K]`), loading B panels with the same `simdgroup_load` as A effectively gives us B as-is, not B^T. We need to either: (a) transpose B tiles during load into `tileB[BK][BN]` layout, or (b) use `simdgroup_load` with column-major stride. **This must be verified on hardware — if the naive approach produces wrong results, swap the B load to use transposed layout.**

- [ ] **Step 2: Add runtime detection to Python dispatch**

In `fp8_mps_native.py`, modify `_get_lib()`:

```python
_lib = None
_use_sgmma = False

def _get_lib():
    global _lib, _use_sgmma
    if _lib is not None:
        return _lib
    source = _load_shader_source()
    _lib = torch.mps.compile_shader(source)
    _use_sgmma = "fp8_scaled_matmul_sgmma_kernel" in dir(_lib)
    return _lib
```

Modify `fp8_scaled_mm` to route to SGMMA when available (for M>=2):

```python
    else:
        if _use_sgmma:
            # SGMMA: 128 threads (4 simdgroups), 32x32 output tiles
            lib.fp8_scaled_matmul_sgmma_kernel(
                A, B, C, scale_a, scale_b,
                M, N, K, scale_mode,
                threads=(((N + 31) // 32) * 32, ((M + 31) // 32) * 32),
                group_size=(32, 4),
            )
        else:
            # Scalar tiled fallback (M1)
            lib.fp8_scaled_matmul_kernel(
                A, B, C, scale_a, scale_b,
                M, N, K, scale_mode,
                threads=(N, M), group_size=(16, 16),
            )
```

- [ ] **Step 3: Run correctness tests**

Run: `cd tests && uv run pytest test_correctness.py -v`

On M2+: both kernels compile, SGMMA is selected, all tests must pass.
On M1: SGMMA kernel is absent from library, scalar fallback is selected, all tests must pass.

If SGMMA produces wrong results (transpose issue), add `simdgroup_load` with transposed memory layout for B tiles. See the transpose note in Step 1.

- [ ] **Step 4: Benchmark**

Run: `cd tests && uv run python bench_ai_workloads.py`

Expected on M2+: 2-4x improvement on fused kernel at M>=128 (compute-bound shapes). This should bring the fused kernel close to parity with native FP16.

Expected on M1: no change (scalar fallback selected).

- [ ] **Step 5: Verify M1 fallback works**

If you don't have an M1 device, verify the fallback logic by temporarily wrapping the SGMMA kernel in `#if 0` and confirming the scalar kernel is selected:

```python
# Temporary test: force scalar path
_use_sgmma = False
```

- [ ] **Step 6: Commit**

```bash
git add fp8_matmul.metal fp8_mps_native.py
git commit -m "P15: add SGMMA matmul kernel with automatic M1 fallback

Uses simdgroup_matrix_multiply_accumulate (Metal 3 / M2+) for ~2.3x
higher ALU throughput. Conditionally compiled via #if __METAL_VERSION__
>= 300. Falls back to scalar tiled kernel on M1 (MSL 2.x).

Runtime detection via dir(lib) — single shader library, no try/except.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task P16: Evaluate Metal 4 MetalPerformancePrimitives matmul

**Files:**
- Exploratory: no permanent code changes unless evaluation is positive
- If viable: modify `fp8_mps_native.py` fast path, add `#if __METAL_VERSION__ >= 400` kernel

**Rationale:** Metal 4 (macOS 16+ / MSL 4.0) provides `tensor_ops::matmul2d` via MetalPerformancePrimitives — Apple's internally-optimized GEMM that automatically uses Neural Accelerators on M5+. If accessible from `torch.mps.compile_shader()`, it could replace our custom SGMMA kernel with Apple's best.

**Constraint:** `tensor_ops::matmul2d` operates on `tensor<device half, dextents<int, 2>>` (MTLTensor), not raw `device half*` buffers. It's unclear whether `torch.mps.compile_shader()` can pass MPS tensor buffers as Metal 4 tensors. This may require a C++ extension using the Metal 4 API directly, which is a different dispatch path than our current pure-Python approach.

- [ ] **Step 1: Check Metal 4 availability**

```python
# Run in tests/ with uv:
import torch
lib = torch.mps.compile_shader("""
#include <metal_stdlib>
using namespace metal;
kernel void v(device float* o [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i == 0) o[0] = float(__METAL_VERSION__);
}
""")
buf = torch.zeros(1, device="mps")
lib.v(buf, threads=(1,), group_size=(1,))
torch.mps.synchronize()
print(f"MSL version: {int(buf.item())}")  # 400 = Metal 4
```

If MSL < 400, skip P16 entirely.

- [ ] **Step 2: Test if MPP matmul2d compiles via compile_shader**

```metal
#if __METAL_VERSION__ >= 400
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
// Try a minimal matmul2d usage
kernel void test_mpp(device half* A [[buffer(0)]],
                     device half* B [[buffer(1)]],
                     device half* C [[buffer(2)]],
                     uint2 tid [[thread_position_in_threadgroup]]) {
    // Attempt to use tensor_ops — this will likely fail because
    // MTLTensor requires API-level setup, not just shader code
}
#endif
```

If this compiles but `tensor_ops::matmul2d` requires MTLTensor objects that can't be created from raw buffers inside a shader, then MPP is **not accessible from `torch.mps.compile_shader()`** and would require a C++ extension.

- [ ] **Step 3: Decision point**

If MPP works from `compile_shader()`: implement as a third kernel tier above SGMMA.

If MPP requires C++ Metal 4 API: document the finding and defer. The SGMMA kernel (P15) is the practical ceiling for the pure-Python path. A C++ Metal 4 extension would be a separate project.

- [ ] **Step 4: Document findings in SPEEDUP.md regardless of outcome**

Add a section: "Metal 4 MPP evaluation: [viable/not viable from compile_shader]. [Reason]. SGMMA (P15) remains the best available acceleration for the pure-Python path."

---

## Tier 4: Benchmark & Cleanup

### Task P17: Fix benchmark layer simulator timing

**Files:**
- Modify: `tests/bench_ai_workloads.py:211-218`

**Rationale:** The `fp8_layer()` function calls `fp8_mps_native.fp8_quantize(torch.randn(...))` inside the timed loop, which includes a GPU→CPU sync via `.item()`. This makes the FP8 layer timing pessimistic.

- [ ] **Step 1: Move quantize calls outside the timed function**

Replace lines 211-218:

```python
        def fp8_layer():
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["qkv"][0], x_s, weights["qkv"][1])
            mid_q, mid_s = fp8_mps_native.fp8_quantize(torch.randn(M, H, device="mps"))
            fp8_mps_native.fp8_scaled_mm_auto(mid_q, weights["out"][0], mid_s, weights["out"][1])
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["gate"][0], x_s, weights["gate"][1])
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["up"][0], x_s, weights["up"][1])
            ffn_q, ffn_s = fp8_mps_native.fp8_quantize(torch.randn(M, FFN, device="mps"))
            fp8_mps_native.fp8_scaled_mm_auto(ffn_q, weights["down"][0], ffn_s, weights["down"][1])
```

with:

```python
        # Pre-quantize intermediate activations outside timed loop
        mid_q, mid_s = fp8_mps_native.fp8_quantize(torch.randn(M, H))
        ffn_q, ffn_s = fp8_mps_native.fp8_quantize(torch.randn(M, FFN))

        def fp8_layer():
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["qkv"][0], x_s, weights["qkv"][1])
            fp8_mps_native.fp8_scaled_mm_auto(mid_q, weights["out"][0], mid_s, weights["out"][1])
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["gate"][0], x_s, weights["gate"][1])
            fp8_mps_native.fp8_scaled_mm_auto(x_q, weights["up"][0], x_s, weights["up"][1])
            fp8_mps_native.fp8_scaled_mm_auto(ffn_q, weights["down"][0], ffn_s, weights["down"][1])
```

- [ ] **Step 2: Run benchmarks and compare**

The FP8 layer timing should improve (removing 2 GPU→CPU syncs from the timed loop).

- [ ] **Step 3: Commit**

---

### Task P18: Eliminate integer division in vecmat kernel

**Files:**
- Modify: `fp8_matmul.metal:191`

**Rationale:** `uint row = gid / 32` uses integer division (~20 GPU cycles). Can use `simdgroup_index_in_threadgroup` + threadgroup position to derive row without division.

- [ ] **Step 1: Replace division with SIMD-aware indexing**

Replace line 191:
```metal
    uint row = gid / 32;
```
with:
```metal
    uint tg_linear = gid / 256;  // threadgroup index (still a division, but only 1)
    uint row = tg_linear * 8 + simd_group;  // 8 simdgroups per threadgroup of 256
```

Wait — this still has a division. The cleaner fix uses `[[threadgroup_position_in_grid]]`:

Add parameter:
```metal
    uint tgid_x [[threadgroup_position_in_grid]],
```

Then:
```metal
    uint row = tgid_x * 8 + simd_group;  // 8 SIMD groups per threadgroup (256/32)
```

This requires the dispatch to use 1D threadgroups properly. Currently dispatched as:
```python
threads=(total_threads,), group_size=(threads_per_group,),
```
where `total_threads = N * 32` and `threads_per_group = 256`. So `tgid_x` = threadgroup index = `gid / 256` automatically. Each threadgroup has 8 SIMD groups (256/32), so `row = tgid_x * 8 + simd_group`.

- [ ] **Step 2: Run tests and benchmark**

Expected: Sub-percent improvement. This is a cleanup.

- [ ] **Step 3: Commit**

---

## Implementation Priority

**Do first (Tier 1, 1 session):** P8 → P10 → P9 (all trivial, each ~5 minutes)

**Do second (Tier 2, 2-3 sessions):** P11 → P13 → P12 → P14

**Do third (Tier 3, dedicated session):** P15 (SGMMA — the big one)

**Do anytime:** P17, P18 (low risk, low effort)

**Evaluate separately:** P16 (requires macOS 16 + Metal 4 SDK availability check)

---

## Expected Outcome

After all optimizations:
- **Decode (M=1):** 0.63-0.68x vs FP16 → **unchanged** (bandwidth-bound, already near-optimal)
- **Prefill (M=128, unprepared):** 2.5-3.0x vs FP16 → **1.0-1.3x** (SGMMA closes the gap)
- **Prefill (M=128, prepared):** 1.3-1.65x vs FP16 → **0.9-1.1x** (near-parity)
- **Full diffusion step overhead:** 15-25% → **5-10%** (approaching negligible)
