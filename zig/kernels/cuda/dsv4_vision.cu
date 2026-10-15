// Vision-only arithmetic keeps BF16 operation boundaries from DeepSeek's PyTorch reference.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

// Preserve small FP32 terms when attention products cancel, before the final BF16 cast.
__device__ void add_compensated(float value, float& sum, float& error) {
  const float next = sum + value, borrowed = next - sum;
  error += (sum - (next - borrowed)) + (value - borrowed);
  sum = next;
}

extern "C" __global__ void tf_dsv4_vision_norm(const __nv_bfloat16* x, const void* weight,
    __nv_bfloat16* y, uint32_t dim, uint32_t kind) {
  __shared__ float squares[256];
  const unsigned row = blockIdx.x, thread = threadIdx.x;
  float square = 0;
  for (unsigned d = thread; d < dim; d += 256) {
    const float v = __bfloat162float(x[uint64_t(row) * dim + d]);
    square += v * v;
  }
  squares[thread] = square;
  __syncthreads();
  for (unsigned step = 128; step; step >>= 1) {
    if (thread < step) squares[thread] += squares[thread + step];
    __syncthreads();
  }
  const float inv = 1.0f / sqrtf(squares[0] / dim + 1e-6f);
  for (unsigned d = thread; d < dim; d += 256) {
    const float w = kind == 0 ? static_cast<const float*>(weight)[d] : kind == 1 ?
        __half2float(static_cast<const half*>(weight)[d]) : __bfloat162float(static_cast<const __nv_bfloat16*>(weight)[d]);
    y[uint64_t(row) * dim + d] = __float2bfloat16_rn((__bfloat162float(x[uint64_t(row) * dim + d]) * inv) * w);
  }
}

extern "C" __global__ void tf_dsv4_vision_split(const __nv_bfloat16* fused, __nv_bfloat16* a,
    __nv_bfloat16* b, __nv_bfloat16* c, uint32_t dim, uint32_t parts) {
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (d >= dim) return;
  const uint64_t source = uint64_t(row) * dim * parts + d, target = uint64_t(row) * dim + d;
  a[target] = fused[source];
  b[target] = fused[source + dim];
  if (parts == 3) c[target] = fused[source + 2 * dim];
}

extern "C" __global__ void tf_dsv4_vision_attention(const __nv_bfloat16* q, const __nv_bfloat16* k,
    const __nv_bfloat16* v, __nv_bfloat16* y, uint32_t rows, uint32_t heads, uint32_t dim) {
  extern __shared__ float scores[];
  __shared__ float largest[128], sums[128];
  const unsigned query = blockIdx.x, head = blockIdx.y, thread = threadIdx.x;
  const uint64_t qbase = (uint64_t(query) * heads + head) * dim;
  float max_value = -INFINITY;
  for (unsigned key = thread; key < rows; key += 128) {
    float dot = 0, dot_error = 0;
    const uint64_t base = (uint64_t(key) * heads + head) * dim;
    for (unsigned d = 0; d < dim; ++d)
      add_compensated(__bfloat162float(q[qbase + d]) * __bfloat162float(k[base + d]), dot, dot_error);
    const float score = (dot + dot_error) / sqrtf(float(dim));
    scores[key] = score;
    max_value = fmaxf(max_value, score);
  }
  largest[thread] = max_value;
  __syncthreads();
  for (unsigned step = 64; step; step >>= 1) {
    if (thread < step) largest[thread] = fmaxf(largest[thread], largest[thread + step]);
    __syncthreads();
  }
  float sum = 0;
  for (unsigned key = thread; key < rows; key += 128) {
    scores[key] = expf(scores[key] - largest[0]);
    sum += scores[key];
  }
  sums[thread] = sum;
  __syncthreads();
  for (unsigned step = 64; step; step >>= 1) {
    if (thread < step) sums[thread] += sums[thread + step];
    __syncthreads();
  }
  for (unsigned d = thread; d < dim; d += 128) {
    float value = 0, value_error = 0;
    for (unsigned key = 0; key < rows; ++key) add_compensated((scores[key] / sums[0]) *
        __bfloat162float(v[(uint64_t(key) * heads + head) * dim + d]), value, value_error);
    y[qbase + d] = __float2bfloat16_rn(value + value_error);
  }
}

extern "C" __global__ void tf_dsv4_vision_swiglu(const __nv_bfloat16* gate,
    const __nv_bfloat16* up, __nv_bfloat16* out, uint32_t width) {
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
  if (d >= width) return;
  const uint64_t at = uint64_t(row) * width + d;
  const float g = __bfloat162float(gate[at]);
  const float silu = __bfloat162float(__float2bfloat16_rn(g / (1 + expf(-g))));
  out[at] = __float2bfloat16_rn(silu * __bfloat162float(up[at]));
}

extern "C" __global__ void tf_dsv4_vision_span(const __nv_bfloat16* raster, const float* special,
    const int32_t* map, __nv_bfloat16* out, uint32_t dim) {
  const unsigned d = blockIdx.x * blockDim.x + threadIdx.x, token = blockIdx.y;
  if (d >= dim) return;
  const int32_t index = map[token];
  out[uint64_t(token) * dim + d] = index >= 0 ? raster[uint64_t(index) * dim + d] :
      __float2bfloat16_rn(special[uint64_t(-index - 1) * dim + d]);
}
