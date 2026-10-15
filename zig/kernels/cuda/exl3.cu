// Native EXL3 reference projections; arithmetic follows core/exl3_format.zig and the upstream Metal path.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

struct Exl3Args {
  uint32_t k, n, rows, halves;
  uint32_t codebook, multiplier, su_signs, sv_signs;
  uint32_t input_f32, output_f32;
  uint64_t tile_k_stride, tile_n_stride;
};

__device__ float scale_at(const uint16_t* data, uint32_t i, uint32_t signs) {
  return signs ? ((data[i / 16] >> (i % 16)) & 1 ? -1.0f : 1.0f)
               : __half2float(__ushort_as_half(data[i]));
}

__device__ uint32_t word_at(const uint16_t* tile, uint32_t i) {
  return uint32_t(tile[2 * i]) | (uint32_t(tile[2 * i + 1]) << 16);
}

__device__ uint16_t state_at(const uint16_t* tile, uint32_t halves, uint32_t p) {
  const uint32_t ring = 128 * halves;
  const uint32_t end = ((p + 1) * halves - ((halves & 1) ? ((p + 1) & 1) : 0)) / 2;
  const uint32_t start = (end + ring - 16) % ring, wi = start / 32, off = start % 32;
  const uint32_t lo = word_at(tile, wi);
  if (off <= 16) return uint16_t(lo >> (16 - off));
  return uint16_t((lo << (off - 16)) | (word_at(tile, (wi + 1) % (ring / 32)) >> (48 - off)));
}

__device__ float code_value(uint16_t state, uint32_t codebook, uint32_t multiplier) {
  uint32_t x = uint32_t(state) * multiplier;
  if (codebook == 2) {
    const uint32_t h = 1024 + (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
    return __half2float(__float2half_rn(fmaf(float(h), __half2float(__ushort_as_half(0x1EEE)),
                                          __half2float(__ushort_as_half(0xC931)))));
  }
  if (codebook == 0) x += 64248484u;
  x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
  return __half2float(__float2half_rn(__half2float(__ushort_as_half(uint16_t(x))) +
                                    __half2float(__ushort_as_half(uint16_t(x >> 16)))));
}

__device__ void hadamard(float* tile, unsigned lane) {
  for (unsigned d = 1; d < 128; d *= 2) {
    const float own = tile[lane], peer = tile[lane ^ d];
    __syncthreads();
    tile[lane] = (lane & d) ? peer - own : own + peer;
    __syncthreads();
  }
}

extern "C" __global__ void tf_exl3_rotate_in(const void* x, const uint16_t* su, half* out, Exl3Args a) {
  __shared__ float tile[128];
  const unsigned lane = threadIdx.x, k = blockIdx.x * 128 + lane, row = blockIdx.y;
  const uint64_t at = uint64_t(row) * a.k + k;
  const float v = a.input_f32 ? static_cast<const float*>(x)[at]
                            : __bfloat162float(static_cast<const __nv_bfloat16*>(x)[at]);
  tile[lane] = v * scale_at(su, k, a.su_signs);
  __syncthreads();
  hadamard(tile, lane);
  out[at] = __float2half_rn(tile[lane] * 0.08838834764831845f);
}

extern "C" __global__ void tf_exl3_matmul(const half* x, const uint8_t* trellis, float* sums, Exl3Args a) {
  const unsigned col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (col >= a.n) return;
  float high = 0, low = 0;
  for (unsigned k = 0; k < a.k; ++k) {
    const uint16_t* tile = reinterpret_cast<const uint16_t*>(
        trellis + uint64_t(k / 16) * a.tile_k_stride + uint64_t(col / 16) * a.tile_n_stride);
    const unsigned r = k % 16, c = col % 16;
    const unsigned lane = 4 * (c % 8) + (r % 8) / 2;
    const unsigned j = (r & 1) + 2 * (r / 8) + 4 * (c / 8);
    const float w = code_value(state_at(tile, a.halves, lane * 8 + j), a.codebook, a.multiplier);
    const float product = __half2float(x[uint64_t(row) * a.k + k]) * w;
    const float next = high + product, borrowed = next - high;
    low += (high - (next - borrowed)) + (product - borrowed);
    high = next;
  }
  sums[uint64_t(row) * a.n + col] = high + low;
}

extern "C" __global__ void tf_exl3_rotate_out(const float* sums, const uint16_t* sv,
                                             const half* bias, void* out, Exl3Args a) {
  __shared__ float tile[128];
  const unsigned lane = threadIdx.x, n = blockIdx.x * 128 + lane, row = blockIdx.y;
  const uint64_t at = uint64_t(row) * a.n + n;
  tile[lane] = sums[at];
  __syncthreads();
  hadamard(tile, lane);
  float v = (tile[lane] * 0.08838834764831845f) * scale_at(sv, n, a.sv_signs);
  if (bias) v += __half2float(bias[n]);
  if (a.output_f32) static_cast<float*>(out)[at] = v;
  else static_cast<__nv_bfloat16*>(out)[at] = __float2bfloat16_rn(v);
}

extern "C" __global__ void tf_exl3_codebook_check(half* out) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 3 * 65536) return;
  const unsigned cb = i / 65536;
  const uint32_t multiplier = cb == 0 ? 89226354u : cb == 1 ? 0xCBAC1FEDu : 0x83DCD12Du;
  out[i] = __float2half_rn(code_value(uint16_t(i), cb, multiplier));
}
