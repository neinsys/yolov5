#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <vector>

int div_round_up(int A, int B) { return (A + B - 1) / B; }

template <typename src_t, typename dest_t>
__global__ void preprocessing_kernel(src_t *src, dest_t *dest, const int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    dest[idx] = (dest_t)src[idx] / (dest_t)255.0;
  }
}

void preprocessing_cuda(torch::Tensor src, torch::Tensor buf, torch::Tensor dst,
                        bool half) {
  const auto color_size = src.size(0);
  const auto width_size = src.size(1);
  const auto height_size = src.size(2);
  const int total_size = color_size * width_size * height_size;

  const int threads = 1024;
  const int blocks = div_round_up(total_size, threads);

  cudaMemcpyAsync(buf.data_ptr<uint8_t>(), src.data_ptr<uint8_t>(),
                  total_size * sizeof(uint8_t), cudaMemcpyHostToDevice, 0);

  if (half) {
    // preprocessing_kernel<uint8_t,half><<<blocks,threads,0>>>(
    //     buf.data_ptr<uint8_t>(),
    //     dst.data_ptr<__half>(),
    //     total_size
    // );
  } else {
    preprocessing_kernel<uint8_t, float><<<blocks, threads, 0>>>(
        buf.data_ptr<uint8_t>(), dst.data_ptr<float>(), total_size);
  }
}