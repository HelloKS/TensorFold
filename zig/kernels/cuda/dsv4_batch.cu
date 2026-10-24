// Batched lane rounds for the DeepSeek V4 reference: one row table staged per round, one FP32 rank exchange per round.
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __global__ void tf_dsv4_batch_ids(const int32_t* table, uint32_t* ids, uint32_t rows) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<rows)ids[i]=(uint32_t)table[i];
}

extern "C" __global__ void tf_dsv4_batch_rows(const float* gathered, const int32_t* sampled, float* out,
    uint32_t rows, uint32_t half, uint32_t width) {
  unsigned k=blockIdx.x,at=(uint32_t)sampled[k];
  for(unsigned j=threadIdx.x;j<width;j+=blockDim.x)
    out[k*width+j]=j<half?gathered[at*half+j]:gathered[(rows+at)*half+(j-half)];
}

extern "C" __global__ void tf_dsv4_batch_merge(const float* gathered, float* out, uint32_t count) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=gathered[i]+gathered[count+i];
}
