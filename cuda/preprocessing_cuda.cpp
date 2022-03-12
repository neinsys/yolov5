#include <torch/extension.h>

#include <iostream>
#include <vector>

#define CHECK_CUDA(x) \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void preprocessing_cuda(torch::Tensor src, torch::Tensor buf, torch::Tensor dst,
                        bool half);

void preprocessing(torch::Tensor src, torch::Tensor buf, torch::Tensor dst,
                   bool half) {
  CHECK_INPUT(buf);
  CHECK_INPUT(dst);

  preprocessing_cuda(src, buf, dst, half);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocessing", &preprocessing, "preprocessing (CUDA)");
}