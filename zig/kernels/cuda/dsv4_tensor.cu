// V4 tensor primitives use explicit activation casts and FP32 accumulation, without hidden quantization.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

__device__ float read_value(const void* p, uint64_t i, unsigned kind) {
  if (kind == 0) return static_cast<const float*>(p)[i];
  if (kind == 1) return __half2float(static_cast<const half*>(p)[i]);
  return __bfloat162float(static_cast<const __nv_bfloat16*>(p)[i]);
}

extern "C" __global__ void tf_dsv4_dense(const void* x, const void* w, const float* bias, void* y,
    uint32_t k, uint32_t n, uint32_t xkind, uint32_t wkind, uint32_t ykind, uint32_t normalize, float eps) {
  const unsigned row = blockIdx.y, col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= n) return;
  float sum = 0, error = 0, square = 0;
  for (unsigned i = 0; i < k; ++i) {
    const float input = read_value(x, uint64_t(row) * k + i, xkind);
    const float product = input * read_value(w, uint64_t(col) * k + i, wkind);
    const float next = sum + product, borrowed = next - sum;
    error += (sum - (next - borrowed)) + (product - borrowed);
    sum = next;
    if (normalize) square += input * input;
  }
  sum += error;
  if (normalize) sum *= rsqrtf(square / k + eps);
  if (bias) sum += bias[col];
  const uint64_t at = uint64_t(row) * n + col;
  if (ykind == 0) static_cast<float*>(y)[at] = sum;
  else static_cast<__nv_bfloat16*>(y)[at] = __float2bfloat16_rn(sum);
}

extern "C" __global__ void tf_dsv4_embed(const void* embedding, const uint32_t* ids,
    __nv_bfloat16* out, uint32_t dim, uint32_t vocab, uint32_t weight_kind, uint32_t copies) {
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (d >= dim) return;
  const uint32_t token = ids[row];
  const float v = token < vocab ? read_value(embedding, uint64_t(token) * dim + d, weight_kind) : 0;
  for (unsigned i = 0; i < copies; ++i) out[(uint64_t(row) * copies + i) * dim + d] = __float2bfloat16_rn(v);
}

extern "C" __global__ void tf_dsv4_expand(const __nv_bfloat16* x, __nv_bfloat16* out, uint32_t dim) {
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (d >= dim) return;
  for (unsigned i = 0; i < 4; ++i) out[(uint64_t(row) * 4 + i) * dim + d] = x[uint64_t(row) * dim + d];
}

extern "C" __global__ void tf_dsv4_tap(const __nv_bfloat16* h, __nv_bfloat16* out,
    uint32_t dim, uint32_t taps, uint32_t tap) {
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (d >= dim) return;
  float v = 0;
  for (unsigned i = 0; i < 4; ++i) v += __bfloat162float(h[(uint64_t(row) * 4 + i) * dim + d]);
  out[(uint64_t(row) * taps + tap) * dim + d] = __float2bfloat16_rn(v / 4);
}

extern "C" __global__ void tf_dsv4_collapse(const __nv_bfloat16* h, const float* mixes,
    const float* scale, const float* base, __nv_bfloat16* out, uint32_t dim, float eps) {
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (d >= dim) return;
  float v = 0;
  for (unsigned i = 0; i < 4; ++i) {
    const float pre = 1.0f / (1.0f + expf(-(mixes[row * 4 + i] * scale[0] + base[i]))) + eps;
    v += pre * __bfloat162float(h[(uint64_t(row) * 4 + i) * dim + d]);
  }
  out[uint64_t(row) * dim + d] = __float2bfloat16_rn(v);
}

extern "C" __global__ void tf_dsv4_add(__nv_bfloat16* residual, const __nv_bfloat16* update, uint32_t count) {
  const uint32_t at = blockIdx.x * blockDim.x + threadIdx.x;
  if (at < count) residual[at] = __float2bfloat16_rn(__bfloat162float(residual[at]) + __bfloat162float(update[at]));
}

extern "C" __global__ void tf_dsv4_gelu(__nv_bfloat16* x, uint32_t count) {
  const uint32_t at = blockIdx.x * blockDim.x + threadIdx.x;
  if (at < count) {
    const float v = __bfloat162float(x[at]);
    x[at] = __float2bfloat16_rn(0.5f * v * (1.0f + erff(v * 0.7071067811865475f)));
  }
}

extern "C" __global__ void tf_dsv4_vision_rope(__nv_bfloat16* x, uint32_t grid_w,
    uint32_t heads, uint32_t head_dim, float theta) {
  const unsigned row = blockIdx.x, head = blockIdx.y, j = threadIdx.x;
  if (j >= head_dim / 2) return;
  const unsigned quarter = head_dim / 4;
  const float position = float(j < quarter ? row / grid_w : row % grid_w);
  const float angle = position / powf(theta, float(2 * (j % quarter)) / float(head_dim / 2));
  const float c = cosf(angle), s = sinf(angle);
  const uint64_t at = (uint64_t(row) * heads + head) * head_dim + j;
  const float a = __bfloat162float(x[at]), b = __bfloat162float(x[at + head_dim / 2]);
  x[at] = __float2bfloat16_rn(a * c - b * s);
  x[at + head_dim / 2] = __float2bfloat16_rn(b * c + a * s);
}
