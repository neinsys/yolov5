#include <torch/extension.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <vector>

static int div_round_up(int A, int B) { return (A + B - 1) / B; }

const int N_THREAD = 32;
const int SHFL_MASK = (1LL << N_THREAD) - 1;

__device__ float iou_calc(float4 A, float4 B) {
  float A_w = A.z - A.x;
  float A_h = A.w - A.y;
  float B_w = B.z - B.x;
  float B_h = B.w - B.y;

  float A_area = A_w * A_h;
  float B_area = B_w * B_h;

  float left = min(A.x, B.x);
  float bottom = min(A.y, B.y);
  float right = max(A.z, B.z);
  float top = max(A.w, B.w);
  float width = max(right - left, 0.f);
  float height = max(top - bottom, 0.f);

  float inter = height * width;

  return inter / (A_area + B_area - inter);
}

__global__ void nms_kernel(float4* boxes, float* score, float iou_threshold,
                           int size, int* result, int* check) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  int checkWarp = false;

  if (x < size && y < size && score[y] < score[x]) {
    float iou = iou_calc(boxes[y], boxes[x]);
    checkWarp = (iou > iou_threshold);
  }
  if (y < size) {
    // reduce
    for (int offset = 1; offset < N_THREAD; offset <<= 1) {
      checkWarp |= __shfl_xor_sync(SHFL_MASK, checkWarp, offset);
    }
    atomicAnd(&check[y], checkWarp);
  }
}

int nms_cuda(torch::Tensor boxes, torch::Tensor score, float iou_threshold,
             torch::Tensor result, torch::Tensor check,
             torch::Tensor score_temp) {
  const auto box_size = boxes.size(0);
  float4* boxes_ptr = (float4*)boxes.data_ptr();
  float* score_ptr = score.data_ptr<float>();
  int* result_ptr = (int*)result.data_ptr();
  int* check_ptr = result.data_ptr<int>();
  float* score_temp_ptr = score_temp.data_ptr<float>();

  int n = div_round_up(box_size, N_THREAD);

  dim3 blocks(n, n);
  dim3 threads(N_THREAD, N_THREAD);

  nms_kernel<<<blocks, threads>>>(boxes_ptr, score_ptr, iou_threshold, box_size,
                                  result_ptr, check_ptr);

  thrust::sequence(thrust::device, result_ptr, result_ptr + box_size);
  auto end =
      thrust::remove_if(thrust::device, result_ptr, result_ptr + box_size,
                        check_ptr, thrust::identity<int>());
  int result_size = end - result_ptr;
  thrust::gather(thrust::device, result_ptr, result_ptr + result_size,
                 score_ptr, score_temp_ptr);
  thrust::sort_by_key(thrust::device, score_temp_ptr,
                      score_temp_ptr + result_size, result_ptr,
                      thrust::greater<float>());

  return result_size;
}
