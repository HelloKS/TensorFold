// Text graph glue for the pinned DeepSeek V4 reference; activations stay BF16, accumulations stay FP32.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
__device__ float get(const void* p, unsigned i, unsigned kind) {
  return kind == 0 ? ((const float*)p)[i] : kind == 1 ? __half2float(((const half*)p)[i]) : __bfloat162float(((const __nv_bfloat16*)p)[i]);
}
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
extern "C" __global__ void tf_dsv4_text_norm(const __nv_bfloat16* x, const void* w,
    __nv_bfloat16* y, uint32_t dim, uint32_t kind, float eps, uint32_t qnorm) {
  __shared__ float sums[256];
  unsigned t=threadIdx.x, row=blockIdx.x; float sum=0;
  for(unsigned j=t;j<dim;j+=256) { float v=__bfloat162float(x[row*dim+j]); sum+=qnorm?bf(v*v):v*v; }
  sums[t]=sum; __syncthreads();
  for(unsigned s=128;s;s>>=1) { if(t<s)sums[t]+=sums[t+s]; __syncthreads(); }
  float inv=qnorm?bf(rsqrtf(bf(bf(sums[0]/dim)+eps))):rsqrtf(sums[0]/dim+eps);
  for(unsigned j=t;j<dim;j+=256) y[row*dim+j]=__float2bfloat16_rn(__bfloat162float(x[row*dim+j])*inv*(w?get(w,j,kind):1));
}
extern "C" __global__ void tf_dsv4_text_rope(__nv_bfloat16* x, const float* freq,
    uint32_t position, uint32_t dim, uint32_t rd, uint32_t inverse) {
  unsigned j=threadIdx.x; if(j>=rd/2)return;
  unsigned at=blockIdx.x*dim+dim-rd+2*j; float angle=float(position)*freq[j];
  float c=cosf(angle),s=sinf(angle)*(inverse?-1:1),a=__bfloat162float(x[at]),b=__bfloat162float(x[at+1]);
  x[at]=__float2bfloat16_rn(a*c-b*s);x[at+1]=__float2bfloat16_rn(b*c+a*s);
}
extern "C" __global__ void tf_dsv4_text_rotate(__nv_bfloat16* x, uint32_t dim) {
  __shared__ float v[128];unsigned t=threadIdx.x,row=blockIdx.x;v[t]=__bfloat162float(x[row*dim+t]);__syncthreads();
  for(unsigned s=1;s<128;s*=2) {float a=v[t&~s],b=v[t|s];__syncthreads();v[t]=(t&s)?a-b:a+b;__syncthreads();}
  x[row*dim+t]=__float2bfloat16_rn(v[t]*0.08838834764831844f);
}
extern "C" __global__ void tf_dsv4_text_merge(const float* gathered, void* out, uint32_t count, uint32_t cast) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;float v=gathered[i]+gathered[count+i];
  if(cast==1)((__nv_bfloat16*)out)[i]=__float2bfloat16_rn(v);else((float*)out)[i]=cast==2?bf(v):v;
}
extern "C" __global__ void tf_dsv4_text_accumulate(float* out,const __nv_bfloat16* x,uint32_t count) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]+=__bfloat162float(x[i]);
}
extern "C" __global__ void tf_dsv4_text_store(const float* kv,const float* scores,const float* ape,
    float* state,float* score_state,uint32_t position,uint32_t ratio,uint32_t dim) {
  unsigned d=blockIdx.x*blockDim.x+threadIdx.x,channels=dim*(ratio==4?2:1);if(d>=channels)return;
  unsigned slot=position%(ratio*(ratio==4?2:1));state[slot*channels+d]=kv[d];
  score_state[slot*channels+d]=scores[d]+ape[(position%ratio)*channels+d];
}
extern "C" __global__ void tf_dsv4_text_pool(const float* state,const float* scores,__nv_bfloat16* out,
    uint32_t position,uint32_t ratio,uint32_t dim) {
  unsigned d=blockIdx.x*blockDim.x+threadIdx.x;if(d>=dim)return;unsigned slots=ratio==4?2*ratio:ratio,channels=ratio==4?2*dim:dim;
  float largest=-INFINITY,total=0,value=0;unsigned start=position+1>=slots?position+1-slots:0;
  for(unsigned i=0;i<slots;i++) {if(position+1<slots && i<ratio)continue;
    unsigned token=start+(position+1<slots?i-ratio:i),part=(ratio==4&&i>=ratio)?dim:0;
    largest=fmaxf(largest,scores[(token%slots)*channels+part+d]);}
  for(unsigned i=0;i<slots;i++) {if(position+1<slots && i<ratio)continue;
    unsigned token=start+(position+1<slots?i-ratio:i),part=(ratio==4&&i>=ratio)?dim:0,at=(token%slots)*channels+part+d;
    float w=expf(scores[at]-largest);total+=w;value+=w*state[at];}
  out[d]=__float2bfloat16_rn(value/total);
}
extern "C" __global__ void tf_dsv4_text_index(const __nv_bfloat16* q,const __nv_bfloat16* kv,
    const __nv_bfloat16* weights,float* out,uint32_t count,uint32_t heads,uint32_t dim) {
  unsigned key=blockIdx.x*blockDim.x+threadIdx.x;if(key>=count)return;float sum=0;
  float scale=rsqrtf(float(dim))*rsqrtf(float(heads*2));
  for(unsigned h=0;h<heads;h++) {float dot=0;for(unsigned d=0;d<dim;d++)dot+=__bfloat162float(q[h*dim+d])*__bfloat162float(kv[key*dim+d]);
    sum+=bf(fmaxf(bf(dot),0)*bf(__bfloat162float(weights[h])*scale));}
  out[key]=bf(sum);
}
extern "C" __global__ void tf_dsv4_text_cast(const void* x,float* y,uint32_t count,uint32_t kind) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=get(x,i,kind);
}
extern "C" __global__ void tf_dsv4_text_route(const float* logits,const uint32_t* ids,const int64_t* table,
    const float* bias,uint32_t* picks,float* weights,uint32_t hashed,float scale) {
  if(threadIdx.x)return;float original[256],scores[256];
  for(unsigned e=0;e<256;e++){float v=logits[e];original[e]=sqrtf(v>20?v:log1pf(expf(v)));scores[e]=original[e]+bias[e];}
  float total=0;for(unsigned j=0;j<6;j++){unsigned best=0;
    if(hashed)best=unsigned(table[uint64_t(ids[0])*6+j]);else {for(unsigned i=1;i<256;i++)if(scores[i]>scores[best])best=i;scores[best]=-INFINITY;}
    picks[j]=best;total+=original[best];}
  for(unsigned j=0;j<6;j++)weights[j]=original[picks[j]]/total*scale;
}
extern "C" __global__ void tf_dsv4_text_ids(const int32_t* x,int64_t* y,uint32_t count) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=x[i];
}

extern "C" __global__ void tf_dsv4_text_round(const float* x,__nv_bfloat16* y,uint32_t count) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)y[i]=__float2bfloat16_rn(x[i]);
}
