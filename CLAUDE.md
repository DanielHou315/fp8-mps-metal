# fp8-mps-metal

FP8 (e4m3fn) quantized matrix multiplication kernels for Apple Silicon via Metal GPU.

## Architecture

Single execution path via `torch.mps.compile_shader()` — zero-copy dispatch directly on MPS tensor buffers. No compilation step required.

Monkey-patch (`fp8_mps_patch.py`) replaces `torch._scaled_mm` to transparently route FP8 on MPS through the native path.

## Key files

| File | Purpose |
|---|---|
| `fp8_matmul.metal` | All Metal GPU kernels: 2D matmul, vecmat (M=1), dequant, quant |
| `fp8_mps_native.py` | Python API: `fp8_scaled_mm`, `fp8_scaled_mm_fast`, `fp8_scaled_mm_auto`, `fp8_quantize`, `fp8_dequantize` |
| `fp8_mps_patch.py` | Monkey-patch for `torch._scaled_mm` |

## ComfyUI integration

Installed as a custom node at `~/ComfyUI/custom_nodes/fp8-mps-metal/__init__.py`.

On startup, the node adds this repo to `sys.path` and calls `fp8_mps_patch.install()`, transparently routing all FLUX/SD3.5 FP8 `_scaled_mm` calls through the Metal kernel. No ComfyUI source changes needed.

To reinstall or update after moving the repo:
```sh
# Path is hardcoded in the __init__.py
cat ~/ComfyUI/custom_nodes/fp8-mps-metal/__init__.py
```

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
