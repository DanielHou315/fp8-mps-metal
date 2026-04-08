"""
Correctness tests for FP8 Metal kernels.

Tests:
  - Exhaustive FP8 decode: all 256 bit patterns vs Python reference
  - Matmul accuracy: fused and fast paths vs FP32 reference
  - Quantize/dequantize roundtrip
  - Vecmat (M=1) kernel path
  - Monkey-patch install/uninstall
  - Per-channel vs per-tensor scaling
  - Edge cases: zero matrices, identity-like, large values
"""

import torch
import pytest
from conftest import fp8_e4m3fn_decode_reference, make_fp8_pair


class TestFP8Decode:
    """Verify every FP8 bit pattern decodes correctly."""

    def test_exhaustive_256_patterns(self):
        import fp8_mps_native

        all_bits = torch.arange(256, dtype=torch.uint8)
        scale = torch.tensor([1.0])
        decoded = fp8_mps_native.fp8_dequantize(all_bits, scale).cpu().float()
        ref = torch.tensor([fp8_e4m3fn_decode_reference(i) for i in range(256)])

        max_abs_err = (decoded - ref).abs().max().item()
        assert max_abs_err < 0.5, f"Max decode error {max_abs_err} exceeds 0.5"

    def test_nan_maps_to_zero(self):
        """NaN encodings (0x7F, 0xFF) should decode to 0."""
        import fp8_mps_native

        nans = torch.tensor([0x7F, 0xFF], dtype=torch.uint8)
        scale = torch.tensor([1.0])
        decoded = fp8_mps_native.fp8_dequantize(nans, scale).cpu().float()
        assert decoded[0].item() == 0.0
        assert decoded[1].item() == 0.0

    def test_zero_decodes_to_zero(self):
        import fp8_mps_native

        zeros = torch.tensor([0x00], dtype=torch.uint8)
        scale = torch.tensor([1.0])
        decoded = fp8_mps_native.fp8_dequantize(zeros, scale).cpu().float()
        assert decoded[0].item() == 0.0

    def test_max_positive_value(self):
        """0x7E (0_1111_110) should decode to 448.0."""
        import fp8_mps_native

        val = torch.tensor([0x7E], dtype=torch.uint8)
        scale = torch.tensor([1.0])
        decoded = fp8_mps_native.fp8_dequantize(val, scale).cpu().float()
        assert abs(decoded[0].item() - 448.0) < 1.0


class TestMatmulAccuracy:
    """FP8 matmul accuracy against FP32 reference."""

    @pytest.mark.parametrize("M,K,N", [
        (64, 256, 128),
        (1, 4096, 4096),
        (128, 4096, 4096),
        (128, 4096, 14336),
    ])
    def test_fused_kernel(self, M, K, N):
        import fp8_mps_native

        A_q, B_q, A_s, B_s, A_f32, B_f32 = make_fp8_pair(M, K, N)
        ref = A_f32 @ B_f32.T
        result = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s).cpu().float()

        ref_rms = torch.sqrt((ref ** 2).mean()).item()
        rel_rmse = torch.sqrt(((result - ref) ** 2).mean()).item() / ref_rms
        assert rel_rmse < 0.15, f"Fused kernel RMSE {rel_rmse:.4%} exceeds 15%"

    @pytest.mark.parametrize("M,K,N", [
        (64, 256, 128),
        (128, 4096, 4096),
        (128, 4096, 14336),
        (512, 4096, 4096),
    ])
    def test_fast_path(self, M, K, N):
        import fp8_mps_native

        A_q, B_q, A_s, B_s, A_f32, B_f32 = make_fp8_pair(M, K, N)
        ref = A_f32 @ B_f32.T
        result = fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s).cpu().float()

        ref_rms = torch.sqrt((ref ** 2).mean()).item()
        rel_rmse = torch.sqrt(((result - ref) ** 2).mean()).item() / ref_rms
        assert rel_rmse < 0.15, f"Fast path RMSE {rel_rmse:.4%} exceeds 15%"

    def test_auto_matches_component_paths(self):
        """Auto selector should produce results consistent with the path it selects."""
        import fp8_mps_native

        for M in [1, 4, 16, 32, 128]:
            K, N = 256, 128
            A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(M, K, N)
            auto_result = fp8_mps_native.fp8_scaled_mm_auto(A_q, B_q, A_s, B_s).cpu().float()

            if M <= 4:
                expected = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s).cpu().float()
            else:
                expected = fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s).cpu().float()

            diff = (auto_result - expected).abs().max().item()
            assert diff < 1e-3, f"Auto result differs from expected path at M={M}: {diff}"


class TestQuantizeRoundtrip:

    @pytest.mark.parametrize("values", [
        [0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 448.0],
        [0.001, -0.001, 0.1, -0.1],
    ])
    def test_roundtrip(self, values):
        import fp8_mps_native

        x = torch.tensor(values)
        q, scale = fp8_mps_native.fp8_quantize(x)
        d = fp8_mps_native.fp8_dequantize(q, scale).cpu().float()
        max_err = (d - x).abs().max().item()
        assert max_err < 50.0, f"Roundtrip max error {max_err} exceeds 50"

    def test_zero_tensor(self):
        import fp8_mps_native

        x = torch.zeros(16)
        q, scale = fp8_mps_native.fp8_quantize(x)
        d = fp8_mps_native.fp8_dequantize(q, scale).cpu().float()
        assert (d.abs() < 1e-6).all()

    def test_shape_preserved(self):
        import fp8_mps_native

        x = torch.randn(4, 8, 16)
        q, scale = fp8_mps_native.fp8_quantize(x)
        assert q.shape == x.shape
        d = fp8_mps_native.fp8_dequantize(q, scale)
        assert d.shape == x.shape


class TestVecmat:
    """M=1 vecmat kernel path."""

    @pytest.mark.parametrize("K,N", [
        (256, 128),
        (512, 256),
        (4096, 4096),
        (4096, 14336),
    ])
    def test_vecmat_accuracy(self, K, N):
        import fp8_mps_native

        A_q, B_q, A_s, B_s, A_f32, B_f32 = make_fp8_pair(1, K, N)
        ref = A_f32 @ B_f32.T
        result = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s).cpu().float()

        ref_rms = torch.sqrt((ref ** 2).mean()).item()
        rel_rmse = torch.sqrt(((result - ref) ** 2).mean()).item() / ref_rms
        assert rel_rmse < 0.15, f"Vecmat RMSE {rel_rmse:.4%} exceeds 15%"

    def test_output_shape(self):
        import fp8_mps_native

        A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(1, 512, 256)
        result = fp8_mps_native.fp8_scaled_mm(A_q, B_q, A_s, B_s)
        assert result.shape == (1, 256)


class TestPreparedWeight:
    """Test fp8_prepare_weight + prepared fast path."""

    @pytest.mark.parametrize("M,K,N", [
        (1, 4096, 4096),
        (1, 4096, 14336),
        (4, 4096, 4096),
        (128, 4096, 4096),
        (128, 4096, 14336),
    ])
    def test_prepared_accuracy(self, M, K, N):
        import fp8_mps_native

        A_q, B_q, A_s, B_s, A_f32, B_f32 = make_fp8_pair(M, K, N)
        ref = A_f32 @ B_f32.T

        B_prepared = fp8_mps_native.fp8_prepare_weight(B_q, B_s)
        assert B_prepared.dtype == torch.float16
        assert B_prepared.shape == B_q.shape

        result = fp8_mps_native.fp8_scaled_mm_auto(A_q, B_prepared, A_s, B_s).cpu().float()
        ref_rms = torch.sqrt((ref ** 2).mean()).item()
        rel_rmse = torch.sqrt(((result - ref) ** 2).mean()).item() / ref_rms
        assert rel_rmse < 0.15, f"Prepared weight RMSE {rel_rmse:.4%} exceeds 15%"

    def test_prepared_matches_unprepared(self):
        import fp8_mps_native

        M, K, N = 64, 256, 128
        A_q, B_q, A_s, B_s, _, _ = make_fp8_pair(M, K, N)

        unprepared = fp8_mps_native.fp8_scaled_mm_fast(A_q, B_q, A_s, B_s).cpu().float()
        B_prepared = fp8_mps_native.fp8_prepare_weight(B_q, B_s)
        prepared = fp8_mps_native.fp8_scaled_mm_fast(A_q, B_prepared, A_s, B_s).cpu().float()

        diff = (unprepared - prepared).abs().max().item()
        assert diff < 1e-2, f"Prepared vs unprepared diff {diff} exceeds 0.01"


class TestMonkeyPatch:

    def test_install_uninstall(self):
        import fp8_mps_patch

        assert not fp8_mps_patch.is_installed()

        fp8_mps_patch.install()
        assert fp8_mps_patch.is_installed()
        assert torch._scaled_mm is not fp8_mps_patch._original_scaled_mm

        fp8_mps_patch.install()  # idempotent
        assert fp8_mps_patch.is_installed()

        fp8_mps_patch.uninstall()
        assert not fp8_mps_patch.is_installed()

    def test_uninstall_when_not_installed(self):
        import fp8_mps_patch

        fp8_mps_patch.uninstall()  # should be a no-op
        assert not fp8_mps_patch.is_installed()


class TestPerChannelScaling:

    def test_per_row_scales(self):
        import fp8_mps_native

        M, K, N = 4, 64, 8
        A_q = torch.randint(0, 128, (M, K), dtype=torch.uint8, device="mps")
        B_q = torch.randint(0, 128, (N, K), dtype=torch.uint8, device="mps")
        scale_a = torch.rand(M, device="mps", dtype=torch.float32) * 0.1
        scale_b = torch.rand(N, device="mps", dtype=torch.float32) * 0.1

        result = fp8_mps_native.fp8_scaled_mm(A_q, B_q, scale_a, scale_b)
        assert result.shape == (M, N)
        assert torch.isfinite(result).all()
