"""
FP8 Metal kernels using PyTorch's native torch.mps.compile_shader() API.

Zero-copy dispatch: kernels run directly on MPS tensor buffers.
No C++ extension needed — pure Python + Metal shader source.

This replaces the C++ bridge approach for dramatically better performance
since it avoids MPS→CPU→Metal→CPU→MPS buffer copies.
"""

import torch
import os

_lib = None
_SHADER_SOURCE = None


def _load_shader_source():
    """Load the Metal shader source from fp8_matmul.metal."""
    global _SHADER_SOURCE
    if _SHADER_SOURCE is not None:
        return _SHADER_SOURCE

    shader_path = os.path.join(os.path.dirname(__file__), "fp8_matmul.metal")
    with open(shader_path, "r") as f:
        _SHADER_SOURCE = f.read()
    return _SHADER_SOURCE


def _get_lib():
    """Get or create the compiled shader library (singleton)."""
    global _lib
    if _lib is not None:
        return _lib

    source = _load_shader_source()
    _lib = torch.mps.compile_shader(source)
    return _lib


def fp8_scaled_mm(A: torch.Tensor, B: torch.Tensor,
                  scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """
    FP8 scaled matrix multiplication on Metal GPU.

    A: (M, K) uint8 — FP8 e4m3fn encoded, row-major
    B: (N, K) uint8 — FP8 e4m3fn encoded, row-major (B is pre-transposed)
    scale_a: per-tensor [1] or per-row [M] float32
    scale_b: per-tensor [1] or per-row [N] float32

    Returns: (M, N) float32 on MPS
    """
    lib = _get_lib()

    assert A.dtype == torch.uint8 and B.dtype == torch.uint8
    assert A.is_contiguous() and B.is_contiguous()

    M, K = A.shape
    N = B.shape[0]
    assert B.shape[1] == K

    # Ensure tensors are on MPS
    if A.device.type != "mps":
        A = A.to("mps")
    if B.device.type != "mps":
        B = B.to("mps")

    # Ensure scales are on MPS and float32
    scale_a = scale_a.to(device="mps", dtype=torch.float32).contiguous()
    scale_b = scale_b.to(device="mps", dtype=torch.float32).contiguous()

    # Determine scale mode: 0=per-tensor, 1=per-channel
    scale_mode = 0 if (scale_a.numel() == 1 and scale_b.numel() == 1) else 1

    # Output tensor
    C = torch.empty(M, N, dtype=torch.float32, device="mps")

    if M == 1:
        # Vecmat kernel: SIMD reduction, 32 threads per output row
        total_threads = N * 32
        threads_per_group = 256
        lib.fp8_scaled_vecmat_kernel(
            A, B, C, scale_a, scale_b,
            N, K, scale_mode,
            threads=(total_threads,), group_size=(threads_per_group,),
                    )
    else:
        # General 2D matmul
        lib.fp8_scaled_matmul_kernel(
            A, B, C, scale_a, scale_b,
            M, N, K, scale_mode,
            threads=(N, M), group_size=(16, 16),
                    )

    return C


def fp8_dequantize(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    FP8 → half dequantization on Metal GPU.

    input: uint8 tensor (FP8 e4m3fn encoded)
    scale: scalar float32 tensor
    Returns: float16 tensor on MPS, scaled
    """
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


def fp8_quantize(input: torch.Tensor):
    """
    Float → FP8 quantization on Metal GPU.

    input: float32 tensor
    Returns: (uint8 tensor on MPS, inverse_scale on MPS)
    """
    lib = _get_lib()

    inp = input.to(device="mps", dtype=torch.float32).contiguous()
    count = inp.numel()

    # Compute scale: max_fp8 / max(abs(input))
    amax_val = inp.abs().max().item()
    max_fp8 = 448.0
    scale = max_fp8 / amax_val if amax_val > 0 else 1.0

    # Fused scale + quantize in one GPU pass
    output = torch.empty(inp.shape, dtype=torch.uint8, device="mps")

    lib.float_to_fp8_scaled_kernel(
        inp.view(-1), output.view(-1),
        count, scale,
        threads=(count,), group_size=(256,),
    )

    inv_scale = torch.tensor([1.0 / scale], dtype=torch.float32, device="mps")
    return output, inv_scale


def fp8_prepare_weight(B_q: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """
    Pre-dequantize FP8 weight matrix to scaled FP16. Cache the result
    for repeated inference to avoid re-dequantizing every call.

    B_q: (N, K) uint8 — FP8 e4m3fn encoded weight matrix
    scale_b: per-tensor scale
    Returns: (N, K) float16 tensor on MPS (scaled, ready for matmul)
    """
    lib = _get_lib()
    if B_q.device.type != "mps":
        B_q = B_q.to("mps")
    sb_val = scale_b.to(device="cpu", dtype=torch.float32).item()
    B_f16 = torch.empty(B_q.shape, dtype=torch.float16, device="mps")
    count = B_q.numel()
    lib.fp8_to_scaled_half_kernel(
        B_q.contiguous().view(-1), B_f16.view(-1),
        count, sb_val,
        threads=(count,), group_size=(256,),
    )
    return B_f16


def fp8_scaled_mm_auto(A: torch.Tensor, B: torch.Tensor,
                       scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """
    Auto-select best FP8 matmul strategy based on dimensions.

    If B is pre-prepared (float16 from fp8_prepare_weight), always uses
    the fast path since the B dequant cost is eliminated.

    Otherwise: M<=16 uses the tiled fused kernel (vecmat for M=1,
    tiled 2D for M=2-16). The tiled kernel's shared-memory reuse
    beats dequant+native-matmul up to M=16 at both K=N=4096 and
    K=4096,N=14336. Crossover is between M=16 and M=32.
    """
    M = A.shape[0]
    if A.dtype == torch.float16 or B.dtype == torch.float16:
        return fp8_scaled_mm_fast(A, B, scale_a, scale_b)
    if M <= 16:
        return fp8_scaled_mm(A, B, scale_a, scale_b)
    return fp8_scaled_mm_fast(A, B, scale_a, scale_b)


def fp8_scaled_mm_fast(A: torch.Tensor, B: torch.Tensor,
                       scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
    """
    Fast FP8 scaled matmul: dequant to FP16 on GPU, then native FP16 matmul.

    This is faster than the fused kernel because it leverages the hardware's
    optimized FP16 matrix multiply engine (AMX/GPU matmul units).

    A: (M, K) uint8 — FP8 e4m3fn encoded
    B: (N, K) uint8 — FP8 e4m3fn encoded (pre-transposed)
    scale_a: per-tensor scale
    scale_b: per-tensor scale
    Returns: (M, N) float16 on MPS
    """
    lib = _get_lib()

    # Ensure on MPS
    if A.device.type != "mps":
        A = A.to("mps")
    if B.device.type != "mps":
        B = B.to("mps")

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

    # B: use pre-prepared FP16 if available, otherwise dequant+scale
    if B.dtype == torch.float16:
        B_f16 = B
    else:
        sb_val = scale_b.to(device="cpu", dtype=torch.float32).item()
        B_f16 = torch.empty(N, K, dtype=torch.float16, device="mps")
        count_b = B.numel()
        lib.fp8_to_scaled_half_kernel(
            B.contiguous().view(-1), B_f16.view(-1),
            count_b, sb_val,
            threads=(count_b,), group_size=(256,),
        )

    # Native FP16 matmul: A @ B^T = (M, K) @ (K, N) = (M, N)
    C = A_f16 @ B_f16.T

    return C
