# fp8-mps-metal

FP8 (e4m3fn) quantized matrix multiplication kernels for Apple Silicon via Metal GPU.

## Architecture

Two execution paths, both using the same Metal shader (`fp8_matmul.metal`):

- **Native path** (`fp8_mps_native.py`) — preferred. Uses `torch.mps.compile_shader()` for zero-copy dispatch directly on MPS tensor buffers.
- **C++ bridge** (`fp8_bridge.cpp` + `setup.py`) — legacy. Requires compilation, forces CPU round-trips. Slower.

Monkey-patch (`fp8_mps_patch.py`) replaces `torch._scaled_mm` to transparently route FP8 on MPS through the native path.

## Key files

| File | Purpose |
|---|---|
| `fp8_matmul.metal` | All Metal GPU kernels: 2D matmul, vecmat (M=1), dequant, quant |
| `fp8_mps_native.py` | Python API: `fp8_scaled_mm`, `fp8_scaled_mm_fast`, `fp8_scaled_mm_auto`, `fp8_quantize`, `fp8_dequantize` |
| `fp8_mps_patch.py` | Monkey-patch for `torch._scaled_mm` |
| `fp8_bridge.cpp` | C++ pybind11 extension (legacy path) |
| `setup.py` | Builds C++ extension, auto-downloads metal-cpp headers |

## Tests and benchmarks

Located in `tests/` with a dedicated venv:

```sh
cd tests
uv venv .venv    # one-time setup
uv pip install torch pytest pytest-benchmark
uv run pytest test_correctness.py -v          # correctness (25 tests)
uv run python bench_ai_workloads.py           # performance report
uv run pytest bench_ai_workloads.py -v -m benchmark  # benchmarks via pytest
```

## Performance status

See [SPEEDUP.md](SPEEDUP.md) for full benchmark results, identified performance problems (P1-P7), and proposed implementation order.

**TL;DR:** FP8 wins at M=1 for large FFN layers (0.85x vs FP16), but loses at prefill sizes (2-3x slower) due to untiled memory access and per-call weight dequantization overhead.

## Dev notes

- Metal has no native FP8 type — stored as `uint8_t`, decoded in-register via IEEE-754 bit extraction
- FP8 e4m3fn format: `[sign:1][exponent:4][mantissa:3]`, bias=7, no inf, NaN=0x7F/0xFF mapped to 0
- The `auto` selector routes M=1 to vecmat (fused), M>16 to dequant+native-FP16-matmul (fast path)
- All Python operations in `tests/` use `uv` with `tests/pyproject.toml`
