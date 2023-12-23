#include <torch/extension.h>

#include <iostream>
#include <cassert>

at::Tensor d4_mm_origorder(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& CB);

at::Tensor e8p_mm_origorder(
    const at::Tensor& A,
    const at::Tensor& B,
    const at::Tensor& CB);

void decompress_d4_origorder(
    torch::Tensor YIs,
    torch::Tensor CB,
    torch::Tensor Y
);

void decompress_e8p_origorder(
    torch::Tensor YIs,
    torch::Tensor CB,
    torch::Tensor &Y
);

torch::Tensor decompress_packed_e8p(
    torch::Tensor weights_compressed,
    torch::Tensor codebook_abs
);

torch::Tensor decode_matvec_e8p(
    torch::Tensor x,
    torch::Tensor weights_compressed,
    torch::Tensor codebook_abs
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("d4_mm_origorder", &d4_mm_origorder, "d4_mm_origorder");
  m.def("e8p_mm_origorder", &e8p_mm_origorder, "e8p_mm_origorder");
  m.def("decompress_d4_origorder", &decompress_d4_origorder, "decompress_d4_origorder");
  m.def("decompress_e8p_origorder", &decompress_e8p_origorder, "decompress_e8p_origorder");
  m.def("decompress_packed_e8p", &decompress_packed_e8p, "decompress_packed_e8p");
  m.def("decode_matvec_e8p", &decode_matvec_e8p, "decode_matvec_e8p");
}