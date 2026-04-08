"""Shared fixtures for FP8 Metal tests and benchmarks."""

import sys
import os
import pytest
import torch

# Add the parent directory so we can import fp8_mps_native / fp8_mps_patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session", autouse=True)
def check_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")


def fp8_e4m3fn_decode_reference(bits: int) -> float:
    """Pure Python reference decode for e4m3fn format."""
    if (bits & 0x7F) == 0x7F:
        return 0.0
    sign = (bits >> 7) & 1
    exp_bits = (bits >> 3) & 0xF
    mant_bits = bits & 0x7
    if exp_bits == 0:
        value = (mant_bits / 8.0) * (2.0 ** -6)
    else:
        mantissa = 1.0 + mant_bits / 8.0
        exponent = exp_bits - 7
        value = mantissa * (2.0 ** exponent)
    return -value if sign else value


def make_fp8_pair(M, K, N):
    """Create FP8-quantized A(M,K) and B(N,K) with scales on MPS."""
    import fp8_mps_native
    A_f32 = torch.randn(M, K)
    B_f32 = torch.randn(N, K)
    A_q, A_s = fp8_mps_native.fp8_quantize(A_f32)
    B_q, B_s = fp8_mps_native.fp8_quantize(B_f32)
    return A_q, B_q, A_s, B_s, A_f32, B_f32
