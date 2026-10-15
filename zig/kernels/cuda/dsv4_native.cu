// DeepSeek's packed E2M1 and block-scaled E4M3 projections, retaining source weight bytes on device.
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

static __device__ float e4m3(uint8_t v) {
  return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

static __device__ float e2m1(uint8_t v) {
  const float table[8] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  return (v & 8) ? -table[v & 7] : table[v & 7];
}

static __device__ float ue8m0(uint8_t v) {
  return v == 255 ? NAN : ldexpf(1.0f, int(v) - 127);
}

extern "C" __global__ void tf_dsv4_native_quant(const void* x, uint8_t* q, uint8_t* scales,
    uint32_t k, uint32_t input_f32) {
  __shared__ float maxima[128];
  const uint32_t j = threadIdx.x;
  const uint64_t at = uint64_t(blockIdx.y) * k + blockIdx.x * 128 + j;
  const float v = input_f32 ? static_cast<const float*>(x)[at] :
      __bfloat162float(static_cast<const __nv_bfloat16*>(x)[at]);
  maxima[j] = fabsf(v);
  __syncthreads();
  for (uint32_t step = 64; step; step >>= 1) {
    if (j < step) maxima[j] = fmaxf(maxima[j], maxima[j + step]);
    __syncthreads();
  }
  const uint32_t bits = __float_as_uint(fmaxf(maxima[0], 1e-4f) * (1.0f / 448.0f));
  const uint8_t exponent = uint8_t((bits >> 23) + ((bits & 0x7fffff) != 0));
  const float scale = ue8m0(exponent);
  q[at] = __nv_cvt_float_to_fp8(fminf(fmaxf(v / scale, -448.0f), 448.0f), __NV_SATFINITE, __NV_E4M3);
  if (j == 0) scales[uint64_t(blockIdx.y) * (k / 128) + blockIdx.x] = exponent;
}

extern "C" __global__ void tf_dsv4_native_matmul(const uint8_t* x, const uint8_t* xs,
    const uint8_t* w, const uint8_t* ws, const float* bias, void* y,
    uint32_t k, uint32_t n, uint32_t fp4, uint32_t output_f32) {
  const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (col >= n) return;
  const uint32_t group = fp4 ? 32 : 128;
  float total = 0;
  for (uint32_t first = 0; first < k; first += group) {
    float sum = 0;
    for (uint32_t j = 0; j < group; ++j) {
      const uint32_t p = first + j;
      const uint64_t at = uint64_t(col) * (fp4 ? k / 2 : k) + (fp4 ? p / 2 : p);
      const float weight = fp4 ? e2m1((w[at] >> ((p & 1) * 4)) & 15) : e4m3(w[at]);
      sum += e4m3(x[uint64_t(row) * k + p]) * weight;
    }
    const float act_scale = ue8m0(xs[uint64_t(row) * (k / 128) + first / 128]);
    const float weight_scale = ue8m0(ws[uint64_t(fp4 ? col : col / 128) * (k / group) + first / group]);
    total += (sum * act_scale) * weight_scale;
  }
  if (bias) total += bias[col];
  const uint64_t at = uint64_t(row) * n + col;
  if (output_f32) static_cast<float*>(y)[at] = total;
  else static_cast<__nv_bfloat16*>(y)[at] = __float2bfloat16_rn(total);
}

extern "C" __global__ void tf_dsv4_native_dequant(const uint8_t* w, const uint8_t* ws,
    __nv_bfloat16* y, uint32_t k, uint32_t n) {
  const uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (at >= uint64_t(k) * n) return;
  const uint32_t row = at / k, col = at % k;
  y[at] = __float2bfloat16_rn(e4m3(w[at]) * ue8m0(ws[uint64_t(row / 128) * (k / 128) + col / 128]));
}

extern "C" __global__ void tf_dsv4_native_decode(float* out) {
  const unsigned v = threadIdx.x;
  out[v] = e4m3(v);
  out[256 + v] = ue8m0(v);
  if (v < 16) out[512 + v] = e2m1(v);
}
