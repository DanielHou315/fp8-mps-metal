"""
Monkey-patch torch._scaled_mm and torch.nn.functional.scaled_mm to route FP8 MPS
tensors through our Metal kernel. Also patches comfy_kitchen's dequantize path.

Usage:
    import fp8_mps_patch
    fp8_mps_patch.install()   # patches torch._scaled_mm + scaled_mm + comfy_kitchen
    fp8_mps_patch.uninstall() # restores originals

ComfyUI integration: import this before loading models, and all
FLUX/SD3.5 FP8 scaled_mm calls will transparently use Metal GPU.
"""

import logging
import torch

_original_scaled_mm = None
_original_scaled_mm_v2 = None
_original_ck_dequantize = None
_installed = False
_transposed_cache = {}

logger = logging.getLogger(__name__)


def _to_uint8(t: torch.Tensor) -> torch.Tensor:
    """View an FP8 tensor as uint8, or pass through if already uint8."""
    if t.dtype != torch.uint8:
        return t.view(torch.uint8)
    return t


def _make_mps_fp8_result(input, B, scale_a, scale_b, bias, out_dtype):
    """Run our Metal FP8 matmul and apply bias / output dtype."""
    import fp8_mps_native

    input = _to_uint8(input)
    B = _to_uint8(B)

    if scale_a is None:
        scale_a = torch.ones(1, device=input.device)
    if scale_b is None:
        scale_b = torch.ones(1, device=input.device)

    result = fp8_mps_native.fp8_scaled_mm_auto(input, B, scale_a, scale_b)

    if bias is not None:
        result = result + bias
    if out_dtype is not None:
        result = result.to(out_dtype)

    return result


def _transpose_cached(other: torch.Tensor) -> torch.Tensor:
    """Return contiguous transpose, caching by data pointer."""
    key = other.data_ptr()
    expected = (other.shape[1], other.shape[0])
    cached = _transposed_cache.get(key)
    if cached is not None and cached.shape == expected:
        return cached
    t = other.t().contiguous()
    _transposed_cache[key] = t
    return t


# ---------------------------------------------------------------------------
# Patch 1: torch._scaled_mm
# ---------------------------------------------------------------------------

def _metal_scaled_mm(input, other, *, out_dtype=None, scale_a=None, scale_b=None,
                     bias=None, scale_result=None, use_fast_accum=False):
    """
    Drop-in replacement for torch._scaled_mm that handles FP8 on MPS.

    torch._scaled_mm signature: (input, other, *, out_dtype, scale_a, scale_b, bias, scale_result, use_fast_accum)
    - input: (M, K) — activation tensor (FP8 or float)
    - other: (K, N) — weight tensor (FP8 or float), column-major (NOT transposed like our kernel)
    - scale_a: per-tensor or per-row scale for input
    - scale_b: per-tensor or per-row scale for other
    """
    is_mps = input.device.type == "mps"
    is_fp8 = input.dtype in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)

    if not (is_mps and is_fp8):
        return _original_scaled_mm(
            input, other, out_dtype=out_dtype, scale_a=scale_a,
            scale_b=scale_b, bias=bias, scale_result=scale_result,
            use_fast_accum=use_fast_accum,
        )

    # torch._scaled_mm expects other as (K, N); our kernel wants B as (N, K)
    B = _transpose_cached(other)

    result = _make_mps_fp8_result(input, B, scale_a, scale_b, bias, out_dtype)

    if scale_result is not None:
        result = result * scale_result

    return result


# ---------------------------------------------------------------------------
# Patch 2: torch.nn.functional.scaled_mm  (PyTorch 2.10+ preferred path)
# ---------------------------------------------------------------------------

def _metal_scaled_mm_v2(mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b,
                         swizzle_a=None, swizzle_b=None, bias=None,
                         output_dtype=torch.bfloat16, contraction_dim=(),
                         use_fast_accum=False):
    """
    Drop-in for torch.nn.functional.scaled_mm (PyTorch 2.10 new API).

    comfy_kitchen uses this when has_scaled_mm_v2() returns True.
    mat_a: (M, K) FP8, mat_b: (K, N) FP8 — same layout as _scaled_mm.
    scale_a / scale_b may be tensors or lists of tensors (block-wise).
    """
    is_mps = mat_a.device.type == "mps"
    is_fp8 = mat_a.dtype in (torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2)

    if not (is_mps and is_fp8):
        return _original_scaled_mm_v2(
            mat_a, mat_b, scale_a, scale_recipe_a, scale_b, scale_recipe_b,
            swizzle_a=swizzle_a, swizzle_b=swizzle_b, bias=bias,
            output_dtype=output_dtype, contraction_dim=contraction_dim,
            use_fast_accum=use_fast_accum,
        )

    # Unwrap list scales (block-wise) — use first element (tensor-wise scale)
    sa = scale_a[0] if isinstance(scale_a, list) else scale_a
    sb = scale_b[0] if isinstance(scale_b, list) else scale_b

    B = _transpose_cached(mat_b)
    return _make_mps_fp8_result(mat_a, B, sa, sb, bias, output_dtype)


# ---------------------------------------------------------------------------
# Patch 3: comfy_kitchen dequantize (fallback when scaled_mm fails)
# ---------------------------------------------------------------------------

def _metal_ck_dequantize(x: torch.Tensor, scale: torch.Tensor,
                          output_type: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    MPS-aware replacement for comfy_kitchen's dequantize_per_tensor_fp8.

    comfy_kitchen calls x.to(dtype=output_type) which MPS rejects for FP8.
    We view the tensor as uint8 and run our Metal dequantize kernel instead.
    """
    if x.device.type == "mps" and x.dtype in (torch.float8_e4m3fn, torch.float8_e5m2,
                                                torch.uint8):
        import fp8_mps_native
        x_u8 = _to_uint8(x)
        result = fp8_mps_native.fp8_dequantize(x_u8, scale)
        return result.to(output_type)

    return _original_ck_dequantize(x, scale, output_type)


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------

def install():
    """Monkey-patch torch scaled_mm APIs and comfy_kitchen dequantize to use Metal FP8 kernel on MPS."""
    global _original_scaled_mm, _original_scaled_mm_v2, _original_ck_dequantize, _installed
    if _installed:
        return

    # Patch 1: legacy _scaled_mm
    if hasattr(torch, "_scaled_mm"):
        _original_scaled_mm = torch._scaled_mm
        torch._scaled_mm = _metal_scaled_mm
    else:
        raise RuntimeError("torch._scaled_mm not found — requires PyTorch 2.4+")

    # Patch 2: new scaled_mm (PyTorch 2.10+)
    if hasattr(torch.nn.functional, "scaled_mm"):
        _original_scaled_mm_v2 = torch.nn.functional.scaled_mm
        torch.nn.functional.scaled_mm = _metal_scaled_mm_v2
        logger.info("fp8-mps-metal: patched torch.nn.functional.scaled_mm")

    # Patch 3: comfy_kitchen dequantize (optional — only if installed)
    try:
        import comfy_kitchen.backends.eager.quantization as _ck_q
        _original_ck_dequantize = _ck_q.dequantize_per_tensor_fp8
        _ck_q.dequantize_per_tensor_fp8 = _metal_ck_dequantize
        logger.info("fp8-mps-metal: patched comfy_kitchen dequantize_per_tensor_fp8")
    except ImportError:
        pass

    _installed = True


def uninstall():
    """Restore original implementations."""
    global _original_scaled_mm, _original_scaled_mm_v2, _original_ck_dequantize, _installed, _transposed_cache
    if not _installed:
        return

    if _original_scaled_mm is not None:
        torch._scaled_mm = _original_scaled_mm
        _original_scaled_mm = None

    if _original_scaled_mm_v2 is not None:
        torch.nn.functional.scaled_mm = _original_scaled_mm_v2
        _original_scaled_mm_v2 = None

    if _original_ck_dequantize is not None:
        try:
            import comfy_kitchen.backends.eager.quantization as _ck_q
            _ck_q.dequantize_per_tensor_fp8 = _original_ck_dequantize
        except ImportError:
            pass
        _original_ck_dequantize = None

    _installed = False
    _transposed_cache = {}


def is_installed():
    """Check if the monkey-patch is active."""
    return _installed
