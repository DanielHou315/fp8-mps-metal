/**
 * FP8 (e4m3fn) Dequantization + Matrix Multiplication on Metal
 *
 * Metal has no native FP8 type, so we store FP8 as uint8_t and decode
 * via a 256-entry constant LUT (zero branches, zero transcendentals).
 *
 * e4m3fn format: [sign:1][exponent:4][mantissa:3], bias=7, no inf, NaN=0x7F/0xFF
 *
 * Patterns from:
 *   metalQwen3/quantized_matmul_optimized.metal  — SIMD reduction, 4-element unroll
 *   metalQwen3/quantize.metal                    — group-wise quant/dequant
 */

#include <metal_stdlib>
using namespace metal;

// ─── FP8 e4m3fn decode LUT ─────────────────────────────────────────────────
// All 256 possible uint8 → float32 mappings, precomputed.
// Constant address space: cached per GPU core, zero init cost.
// NaN values (0x7F, 0xFF) map to 0.0f.

constant float fp8_e4m3fn_lut[256] = {
    0.0000000000f, 0.0019531250f, 0.0039062500f, 0.0058593750f, 0.0078125000f, 0.0097656250f, 0.0117187500f, 0.0136718750f,
    0.0156250000f, 0.0175781250f, 0.0195312500f, 0.0214843750f, 0.0234375000f, 0.0253906250f, 0.0273437500f, 0.0292968750f,
    0.0312500000f, 0.0351562500f, 0.0390625000f, 0.0429687500f, 0.0468750000f, 0.0507812500f, 0.0546875000f, 0.0585937500f,
    0.0625000000f, 0.0703125000f, 0.0781250000f, 0.0859375000f, 0.0937500000f, 0.1015625000f, 0.1093750000f, 0.1171875000f,
    0.1250000000f, 0.1406250000f, 0.1562500000f, 0.1718750000f, 0.1875000000f, 0.2031250000f, 0.2187500000f, 0.2343750000f,
    0.2500000000f, 0.2812500000f, 0.3125000000f, 0.3437500000f, 0.3750000000f, 0.4062500000f, 0.4375000000f, 0.4687500000f,
    0.5000000000f, 0.5625000000f, 0.6250000000f, 0.6875000000f, 0.7500000000f, 0.8125000000f, 0.8750000000f, 0.9375000000f,
    1.0000000000f, 1.1250000000f, 1.2500000000f, 1.3750000000f, 1.5000000000f, 1.6250000000f, 1.7500000000f, 1.8750000000f,
    2.0000000000f, 2.2500000000f, 2.5000000000f, 2.7500000000f, 3.0000000000f, 3.2500000000f, 3.5000000000f, 3.7500000000f,
    4.0000000000f, 4.5000000000f, 5.0000000000f, 5.5000000000f, 6.0000000000f, 6.5000000000f, 7.0000000000f, 7.5000000000f,
    8.0000000000f, 9.0000000000f, 10.0000000000f, 11.0000000000f, 12.0000000000f, 13.0000000000f, 14.0000000000f, 15.0000000000f,
    16.0000000000f, 18.0000000000f, 20.0000000000f, 22.0000000000f, 24.0000000000f, 26.0000000000f, 28.0000000000f, 30.0000000000f,
    32.0000000000f, 36.0000000000f, 40.0000000000f, 44.0000000000f, 48.0000000000f, 52.0000000000f, 56.0000000000f, 60.0000000000f,
    64.0000000000f, 72.0000000000f, 80.0000000000f, 88.0000000000f, 96.0000000000f, 104.0000000000f, 112.0000000000f, 120.0000000000f,
    128.0000000000f, 144.0000000000f, 160.0000000000f, 176.0000000000f, 192.0000000000f, 208.0000000000f, 224.0000000000f, 240.0000000000f,
    256.0000000000f, 288.0000000000f, 320.0000000000f, 352.0000000000f, 384.0000000000f, 416.0000000000f, 448.0000000000f, 0.0000000000f,
    -0.0000000000f, -0.0019531250f, -0.0039062500f, -0.0058593750f, -0.0078125000f, -0.0097656250f, -0.0117187500f, -0.0136718750f,
    -0.0156250000f, -0.0175781250f, -0.0195312500f, -0.0214843750f, -0.0234375000f, -0.0253906250f, -0.0273437500f, -0.0292968750f,
    -0.0312500000f, -0.0351562500f, -0.0390625000f, -0.0429687500f, -0.0468750000f, -0.0507812500f, -0.0546875000f, -0.0585937500f,
    -0.0625000000f, -0.0703125000f, -0.0781250000f, -0.0859375000f, -0.0937500000f, -0.1015625000f, -0.1093750000f, -0.1171875000f,
    -0.1250000000f, -0.1406250000f, -0.1562500000f, -0.1718750000f, -0.1875000000f, -0.2031250000f, -0.2187500000f, -0.2343750000f,
    -0.2500000000f, -0.2812500000f, -0.3125000000f, -0.3437500000f, -0.3750000000f, -0.4062500000f, -0.4375000000f, -0.4687500000f,
    -0.5000000000f, -0.5625000000f, -0.6250000000f, -0.6875000000f, -0.7500000000f, -0.8125000000f, -0.8750000000f, -0.9375000000f,
    -1.0000000000f, -1.1250000000f, -1.2500000000f, -1.3750000000f, -1.5000000000f, -1.6250000000f, -1.7500000000f, -1.8750000000f,
    -2.0000000000f, -2.2500000000f, -2.5000000000f, -2.7500000000f, -3.0000000000f, -3.2500000000f, -3.5000000000f, -3.7500000000f,
    -4.0000000000f, -4.5000000000f, -5.0000000000f, -5.5000000000f, -6.0000000000f, -6.5000000000f, -7.0000000000f, -7.5000000000f,
    -8.0000000000f, -9.0000000000f, -10.0000000000f, -11.0000000000f, -12.0000000000f, -13.0000000000f, -14.0000000000f, -15.0000000000f,
    -16.0000000000f, -18.0000000000f, -20.0000000000f, -22.0000000000f, -24.0000000000f, -26.0000000000f, -28.0000000000f, -30.0000000000f,
    -32.0000000000f, -36.0000000000f, -40.0000000000f, -44.0000000000f, -48.0000000000f, -52.0000000000f, -56.0000000000f, -60.0000000000f,
    -64.0000000000f, -72.0000000000f, -80.0000000000f, -88.0000000000f, -96.0000000000f, -104.0000000000f, -112.0000000000f, -120.0000000000f,
    -128.0000000000f, -144.0000000000f, -160.0000000000f, -176.0000000000f, -192.0000000000f, -208.0000000000f, -224.0000000000f, -240.0000000000f,
    -256.0000000000f, -288.0000000000f, -320.0000000000f, -352.0000000000f, -384.0000000000f, -416.0000000000f, -448.0000000000f, 0.0000000000f
};

// ─── float32 → FP8 e4m3fn encode ───────────────────────────────────────────

inline uint8_t float_to_fp8_e4m3fn(float val) {
    uint sign = 0;
    if (val < 0.0f) {
        sign = 1;
        val = -val;
    }

    // Max representable: 448.0 (1111_110 = exp=14, bias=7 → 2^8*(1+6/8)=448)
    // Clamp to max
    if (val >= 448.0f) {
        return (sign << 7) | 0x7E;  // 0_1111_110 = max normal
    }

    // Min subnormal: 2^(-9) = 1/512
    if (val < (1.0f / 512.0f)) {
        return 0;  // flush to zero
    }

    // Try subnormal first: val = mant/8 * 2^(-6)
    // mant = val * 8 * 64 = val * 512
    if (val < (1.0f / 64.0f)) {
        uint mant = uint(val * 512.0f + 0.5f);
        mant = min(mant, 7u);
        return (sign << 7) | uint8_t(mant);
    }

    // Normal: find exponent
    int exp_val = int(floor(log2(val)));
    // Clamp exponent to [0-7, 14-7] = [-7, 7]
    exp_val = clamp(exp_val, -6, 8);

    float mantissa = val / exp2(float(exp_val));  // 1.xxx
    uint mant = uint((mantissa - 1.0f) * 8.0f + 0.5f);
    mant = min(mant, 7u);

    uint exp_bits = uint(exp_val + 7);  // add bias
    exp_bits = clamp(exp_bits, 1u, 15u);

    // Avoid NaN encoding (exp=15, mant=7)
    if (exp_bits == 15 && mant == 7) {
        mant = 6;  // clamp to max normal
    }

    return (sign << 7) | uint8_t(exp_bits << 3) | uint8_t(mant);
}

// ─── Tiled MxN Scaled MatMul ────────────────────────────────────────────────
// A: (M,K) uint8 FP8, B: (N,K) uint8 FP8 (transposed), out: (M,N) float32
// scale_mode: 0=per-tensor, 1=per-channel(row)
// Threadgroup: 16x16, tiled over K dimension to reduce global memory traffic.
// Each tile loads 16x16 blocks of A and B into threadgroup memory,
// reducing reads from O(M*N*K) to O(M*K + N*K).

#define TILE 16

kernel void fp8_scaled_matmul_kernel(
    device const uint8_t* A [[buffer(0)]],       // (M, K) row-major FP8
    device const uint8_t* B [[buffer(1)]],       // (N, K) row-major FP8 (B is transposed)
    device float* C [[buffer(2)]],               // (M, N) output
    device const float* scale_a [[buffer(3)]],   // per-tensor [1] or per-row [M]
    device const float* scale_b [[buffer(4)]],   // per-tensor [1] or per-row [N]
    constant uint& M [[buffer(5)]],
    constant uint& N [[buffer(6)]],
    constant uint& K [[buffer(7)]],
    constant uint& scale_mode [[buffer(8)]],     // 0=per-tensor, 1=per-channel
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Output element this thread computes
    uint row = tgid.y * TILE + tid.y;
    uint col = tgid.x * TILE + tid.x;

    threadgroup half tileA[TILE][TILE];   // 16x16 = 512B
    threadgroup half tileB[TILE][TILE];   // 16x16 = 512B

    float sum = 0.0f;

    // Tile over K dimension
    for (uint kk = 0; kk < K; kk += TILE) {
        // Cooperatively load tiles: each of 256 threads loads one A and one B element
        // tileA[tid.y][tid.x] = decoded A[row, kk + tid.x]
        uint a_row = tgid.y * TILE + tid.y;
        uint a_k = kk + tid.x;
        tileA[tid.y][tid.x] = (a_row < M && a_k < K)
            ? half(fp8_e4m3fn_lut[A[a_row * K + a_k]]) : half(0.0h);

        // tileB[tid.x][tid.y] = decoded B[col_for_tid.x, kk + tid.y]
        // tid.x indexes the N dimension, tid.y indexes the K dimension
        // Threads with consecutive tid.y access B[same_row, consecutive_k] → coalesced
        uint b_row = tgid.x * TILE + tid.x;
        uint b_k = kk + tid.y;
        tileB[tid.x][tid.y] = (b_row < N && b_k < K)
            ? half(fp8_e4m3fn_lut[B[b_row * K + b_k]]) : half(0.0h);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Accumulate partial dot product from tile
        for (uint k = 0; k < TILE; k++) {
            sum += float(tileA[tid.y][k]) * float(tileB[tid.x][k]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write result
    if (row < M && col < N) {
        float sa = (scale_mode == 0) ? scale_a[0] : scale_a[row];
        float sb = (scale_mode == 0) ? scale_b[0] : scale_b[col];
        C[row * N + col] = sum * sa * sb;
    }
}


// ─── Optimized Vec-Mat for Single Token (M=1) ──────────────────────────────
// x: (K,) uint8 FP8, W: (N,K) uint8 FP8, out: (N,) float32
// Uses SIMD reduction across K dimension
// Threadgroup: 256

kernel void fp8_scaled_vecmat_kernel(
    device const uint8_t* x [[buffer(0)]],       // (K,) input vector FP8
    device const uint8_t* W [[buffer(1)]],       // (N, K) weight matrix FP8 row-major
    device float* output [[buffer(2)]],           // (N,) output
    device const float* scale_x [[buffer(3)]],   // per-tensor scale for x
    device const float* scale_w [[buffer(4)]],   // per-tensor [1] or per-row [N]
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& scale_mode [[buffer(7)]],
    uint gid [[thread_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    uint row = gid / 32;  // Each SIMD group handles one output row
    if (row >= N) return;

    uint row_offset = row * K;

    float sum = 0.0f;

    // Coalesced access: consecutive lanes read consecutive bytes.
    // Each iteration, 32 lanes cover 32 contiguous elements (one cache line).
    // 4x unroll in the outer loop for throughput.
    uint K128 = (K / 128) * 128;
    uint k = 0;
    for (; k < K128; k += 128) {
        // 4 coalesced reads of 32 elements each
        float x0 = fp8_e4m3fn_lut[x[k + simd_lane]];
        float w0 = fp8_e4m3fn_lut[W[row_offset + k + simd_lane]];
        float x1 = fp8_e4m3fn_lut[x[k + 32 + simd_lane]];
        float w1 = fp8_e4m3fn_lut[W[row_offset + k + 32 + simd_lane]];
        float x2 = fp8_e4m3fn_lut[x[k + 64 + simd_lane]];
        float w2 = fp8_e4m3fn_lut[W[row_offset + k + 64 + simd_lane]];
        float x3 = fp8_e4m3fn_lut[x[k + 96 + simd_lane]];
        float w3 = fp8_e4m3fn_lut[W[row_offset + k + 96 + simd_lane]];
        sum += x0 * w0 + x1 * w1 + x2 * w2 + x3 * w3;
    }
    // Handle remaining elements in coalesced 32-wide steps
    for (; k + 32 <= K; k += 32) {
        sum += fp8_e4m3fn_lut[x[k + simd_lane]] * fp8_e4m3fn_lut[W[row_offset + k + simd_lane]];
    }
    // Handle tail (< 32 elements)
    if (k + simd_lane < K) {
        sum += fp8_e4m3fn_lut[x[k + simd_lane]] * fp8_e4m3fn_lut[W[row_offset + k + simd_lane]];
    }

    // SIMD reduction — hardware-accelerated sum across 32 lanes
    sum = simd_sum(sum);

    // First lane writes result
    if (simd_lane == 0) {
        float sx = scale_x[0];
        float sw = (scale_mode == 0) ? scale_w[0] : scale_w[row];
        output[row] = sum * sx * sw;
    }
}


// ─── Standalone FP8 → half dequantize ───────────────────────────────────────

kernel void fp8_to_half_kernel(
    device const uint8_t* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = half(fp8_e4m3fn_lut[input[gid]]);
}


// ─── FP8 → scaled half dequantize (fused scale multiply) ───────────────────

kernel void fp8_to_scaled_half_kernel(
    device const uint8_t* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant float& scale [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = half(fp8_e4m3fn_lut[input[gid]] * scale);
}


// ─── Standalone float → FP8 quantize ───────────────────────────────────────

kernel void float_to_fp8_kernel(
    device const float* input [[buffer(0)]],
    device uint8_t* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    output[gid] = float_to_fp8_e4m3fn(input[gid]);
}
