// DeepSeek-V4 Flash Vision operators, reimplemented from DeepSeek's MIT reference at 6821d6ad.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <math.h>

__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

extern "C" __global__ void tf_dsv4_route(const float* logits, const uint32_t* ids, const int64_t* table,
    const half* bias, const float* bias_vl, uint32_t* picks, float* weights,
    uint32_t vocab, uint32_t hashed, float route_scale) {
  __shared__ float original[256], score[256];
  const unsigned e = threadIdx.x, row = blockIdx.x;
  const uint32_t token = ids[row];
  const bool image = token >= vocab;
  const float v = logits[row * 256 + e];
  original[e] = sqrtf(v > 20 ? v : log1pf(expf(v)));
  score[e] = original[e] + (image ? bias_vl[e] : __half2float(bias[e]));
  __syncthreads();
  if (e != 0) return;
  float total = 0;
  for (unsigned slot = 0; slot < 6; ++slot) {
    unsigned best = 0;
    if (hashed && !image) best = unsigned(table[uint64_t(token) * 6 + slot]);
    else {
      for (unsigned i = 1; i < 256; ++i) if (score[i] > score[best]) best = i;
      score[best] = -INFINITY;
    }
    picks[row * 6 + slot] = best;
    total += original[best];
  }
  for (unsigned slot = 0; slot < 6; ++slot)
    weights[row * 6 + slot] = original[picks[row * 6 + slot]] / total * route_scale;
}

extern "C" __global__ void tf_dsv4_sinkhorn(const float* mixes, const float* scale, const float* base,
    float* pre, float* post, float* comb, uint32_t iterations, float eps) {
  if (threadIdx.x != 0) return;
  const unsigned row = blockIdx.x;
  float matrix[16];
  for (unsigned j = 0; j < 4; ++j) {
    pre[row * 4 + j] = sigmoid(mixes[row * 24 + j] * scale[0] + base[j]) + eps;
    post[row * 4 + j] = 2 * sigmoid(mixes[row * 24 + j + 4] * scale[1] + base[j + 4]);
    float largest = -INFINITY, sum = 0;
    for (unsigned k = 0; k < 4; ++k) {
      const unsigned at = j * 4 + k;
      matrix[at] = mixes[row * 24 + 8 + at] * scale[2] + base[8 + at];
      largest = fmaxf(largest, matrix[at]);
    }
    for (unsigned k = 0; k < 4; ++k) { matrix[j * 4 + k] = expf(matrix[j * 4 + k] - largest); sum += matrix[j * 4 + k]; }
    for (unsigned k = 0; k < 4; ++k) matrix[j * 4 + k] = matrix[j * 4 + k] / sum + eps;
  }
  for (unsigned it = 0; it < iterations; ++it) {
    if (it) for (unsigned j = 0; j < 4; ++j) {
      float sum = eps;
      for (unsigned k = 0; k < 4; ++k) sum += matrix[j * 4 + k];
      for (unsigned k = 0; k < 4; ++k) matrix[j * 4 + k] /= sum;
    }
    for (unsigned k = 0; k < 4; ++k) {
      float sum = eps;
      for (unsigned j = 0; j < 4; ++j) sum += matrix[j * 4 + k];
      for (unsigned j = 0; j < 4; ++j) matrix[j * 4 + k] /= sum;
    }
  }
  for (unsigned i = 0; i < 16; ++i) comb[row * 16 + i] = matrix[i];
}

extern "C" __global__ void tf_dsv4_hc_pre(const __nv_bfloat16* residual, const float* pre,
    __nv_bfloat16* out, uint32_t width) {
  const unsigned row = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= width) return;
  float v = 0;
  for (unsigned j = 0; j < 4; ++j) v += pre[row * 4 + j] * __bfloat162float(residual[(row * 4 + j) * width + d]);
  out[row * width + d] = __float2bfloat16_rn(v);
}

extern "C" __global__ void tf_dsv4_hc_post(const __nv_bfloat16* x, const __nv_bfloat16* residual,
    const float* post, const float* comb, __nv_bfloat16* out, uint32_t width) {
  const unsigned row = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= width) return;
  float before[4];
  for (unsigned i = 0; i < 4; ++i) before[i] = __bfloat162float(residual[(row * 4 + i) * width + d]);
  for (unsigned j = 0; j < 4; ++j) {
    float v = 0;
    for (unsigned i = 0; i < 4; ++i) v += comb[row * 16 + i * 4 + j] * before[i];
    v += post[row * 4 + j] * __bfloat162float(x[row * width + d]);
    out[(row * 4 + j) * width + d] = __float2bfloat16_rn(v);
  }
}

extern "C" __global__ void tf_dsv4_qat(__nv_bfloat16* x, uint32_t width, uint32_t group,
    uint32_t fp4, uint32_t stride) {
  __shared__ float values[128];
  const unsigned j = threadIdx.x, row = blockIdx.y, first = blockIdx.x * group;
  float v = j < group ? __bfloat162float(x[uint64_t(row) * stride + first + j]) : 0;
  values[j] = fabsf(v);
  __syncthreads();
  for (unsigned d = 64; d > 0; d >>= 1) {
    if (j < d) values[j] = fmaxf(values[j], values[j + d]);
    __syncthreads();
  }
  const float amax = fmaxf(values[0], fp4 ? 6 * 0x1p-126f : 1e-4f);
  const float raw = amax * (fp4 ? 1.0f / 6.0f : 1.0f / 448.0f);
  const uint32_t bits = __float_as_uint(raw);
  const int exponent = int((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0);
  const float scale = __uint_as_float(uint32_t(exponent + 127) << 23);
  if (j >= group || first + j >= width) return;
  v = fminf(fmaxf(v / scale, fp4 ? -6.0f : -448.0f), fp4 ? 6.0f : 448.0f);
  if (fp4) {
    const auto q = __nv_cvt_float_to_fp4(v, __NV_E2M1, cudaRoundNearest);
    v = __half2float(__nv_cvt_fp4_to_halfraw(q, __NV_E2M1));
  } else {
    const auto q = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    v = __half2float(__nv_cvt_fp8_to_halfraw(q, __NV_E4M3));
  }
  x[uint64_t(row) * stride + first + j] = __float2bfloat16_rn(v * scale);
}

extern "C" __global__ void tf_dsv4_rope(__nv_bfloat16* x, const float* cos, const float* sin,
    const uint32_t* positions, uint32_t heads, uint32_t dim, uint32_t rope_dim, uint32_t inverse) {
  const unsigned row = blockIdx.x, head = blockIdx.y, j = threadIdx.x;
  if (j >= rope_dim / 2) return;
  const uint64_t at = (uint64_t(row) * heads + head) * dim + dim - rope_dim + j * 2;
  const float a = __bfloat162float(x[at]), b = __bfloat162float(x[at + 1]);
  const uint64_t freq = uint64_t(positions[row]) * (rope_dim / 2) + j;
  const float c = cos[freq], s = inverse ? -sin[freq] : sin[freq];
  x[at] = __float2bfloat16_rn(a * c - b * s);
  x[at + 1] = __float2bfloat16_rn(b * c + a * s);
}

extern "C" __global__ void tf_dsv4_pool(const float* kv, const float* scores,
    __nv_bfloat16* out, uint32_t slots, uint32_t dim) {
  const unsigned channel = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (channel >= dim) return;
  float largest = -INFINITY, sum = 0, value = 0;
  for (unsigned i = 0; i < slots; ++i) largest = fmaxf(largest, scores[(uint64_t(row) * slots + i) * dim + channel]);
  for (unsigned i = 0; i < slots; ++i) {
    const uint64_t at = (uint64_t(row) * slots + i) * dim + channel;
    const float w = expf(scores[at] - largest);
    sum += w;
    value += w * kv[at];
  }
  out[uint64_t(row) * dim + channel] = __float2bfloat16_rn(value / sum);
}

extern "C" __global__ void tf_dsv4_sparse_attn(const __nv_bfloat16* q, const __nv_bfloat16* kv,
    const int32_t* indices, const float* sink, __nv_bfloat16* out,
    uint32_t heads, uint32_t selected, uint32_t dim, float scale) {
  extern __shared__ float shared[];
  float* logits = shared;
  float* reduction = shared + selected;
  const unsigned row = blockIdx.x, head = blockIdx.y, thread = threadIdx.x;
  const uint64_t qbase = (uint64_t(row) * heads + head) * dim;
  for (unsigned key = 0; key < selected; ++key) {
    const int32_t index = indices[uint64_t(row) * selected + key];
    float dot = 0;
    if (index >= 0) for (unsigned d = thread; d < dim; d += blockDim.x)
      dot += __bfloat162float(q[qbase + d]) * __bfloat162float(kv[uint64_t(index) * dim + d]);
    reduction[thread] = dot;
    __syncthreads();
    for (unsigned d = blockDim.x / 2; d > 0; d >>= 1) {
      if (thread < d) reduction[thread] += reduction[thread + d];
      __syncthreads();
    }
    if (thread == 0) logits[key] = index < 0 ? -INFINITY : reduction[0] * scale;
    __syncthreads();
  }
  if (thread == 0) {
    float largest = sink[head];
    for (unsigned i = 0; i < selected; ++i) largest = fmaxf(largest, logits[i]);
    float denom = expf(sink[head] - largest);
    for (unsigned i = 0; i < selected; ++i) { logits[i] = expf(logits[i] - largest); denom += logits[i]; }
    for (unsigned i = 0; i < selected; ++i) logits[i] /= denom;
  }
  __syncthreads();
  for (unsigned d = thread; d < dim; d += blockDim.x) {
    float value = 0;
    for (unsigned key = 0; key < selected; ++key) {
      const int32_t index = indices[uint64_t(row) * selected + key];
      if (index >= 0) value += logits[key] * __bfloat162float(kv[uint64_t(index) * dim + d]);
    }
    out[qbase + d] = __float2bfloat16_rn(value);
  }
}

extern "C" __global__ void tf_dsv4_swiglu(const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const float* weights, __nv_bfloat16* out, uint32_t width, float limit) {
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (d >= width) return;
  const uint64_t at = uint64_t(row) * width + d;
  const float g = fminf(__bfloat162float(gate[at]), limit);
  const float u = fminf(fmaxf(__bfloat162float(up[at]), -limit), limit);
  out[at] = __float2bfloat16_rn((g * sigmoid(g) * u) * (weights ? weights[row] : 1.0f));
}

extern "C" __global__ void tf_dsv4_unfold(const __nv_bfloat16* x, __nv_bfloat16* out,
    uint32_t height, uint32_t width, uint32_t channels) {
  const unsigned j = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (j >= channels * 9) return;
  const unsigned cols = (width + 2) / 3;
  const unsigned h = (row / cols) * 3 + (j % 9) / 3, w = (row % cols) * 3 + j % 3;
  out[uint64_t(row) * channels * 9 + j] = (h < height && w < width)
      ? x[(uint64_t(h) * width + w) * channels + j / 9] : __float2bfloat16_rn(0);
}
