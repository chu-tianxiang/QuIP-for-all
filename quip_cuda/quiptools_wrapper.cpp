#include <torch/extension.h>

#include <iostream>
#include <cassert>

at::Tensor d4_mm_cuda(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& CB);

at::Tensor e8_mm_cuda(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& CB);

void decompress_d4_origorder(
    torch::Tensor YIs,      // m x (n/4)
    torch::Tensor CB,       // 256 x 4
    torch::Tensor Y         // m x n
);

void decompress_e8p_origorder(
    torch::Tensor YIs,      // m x (n/8)
    torch::Tensor CB,       // 256 x 8
    torch::Tensor CB_even_flips, // 256
    torch::Tensor &Y         // m x n
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("d4_mm_cuda", &d4_mm_cuda, "d4_mm_cuda");
  m.def("e8p_mm_cuda", &e8_mm_cuda, "e8_mm_cuda");
  m.def("decompress_d4", &decompress_d4_origorder, "decompress_d4_origorder");
  m.def("decompress_e8p", &decompress_e8p_origorder, "decompress_e8p_origorder");
}
