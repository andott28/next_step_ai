#include <torch/extension.h>

#include <stdexcept>
#include <vector>

std::vector<torch::Tensor> spatial_gate_cuda_launcher(
    torch::Tensor q,
    torch::Tensor block_centers,
    torch::Tensor refractory_until,
    int64_t step,
    double sigma,
    int64_t top_k);

torch::Tensor sparse_adapter_cuda_launcher(
    torch::Tensor hidden,
    torch::Tensor active_idx,
    torch::Tensor down_w,
    torch::Tensor down_b,
    torch::Tensor up_w,
    torch::Tensor up_b);

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<torch::Tensor> spatial_gate_cuda(
    torch::Tensor q,
    torch::Tensor block_centers,
    torch::Tensor refractory_until,
    int64_t step,
    double sigma,
    int64_t top_k) {
  CHECK_CUDA(q);
  CHECK_CUDA(block_centers);
  CHECK_CUDA(refractory_until);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(block_centers);
  CHECK_CONTIGUOUS(refractory_until);

  return spatial_gate_cuda_launcher(
      q, block_centers, refractory_until, step, sigma, top_k);
}

torch::Tensor sparse_adapter_cuda(
    torch::Tensor hidden,
    torch::Tensor active_idx,
    torch::Tensor down_w,
    torch::Tensor down_b,
    torch::Tensor up_w,
    torch::Tensor up_b) {
  CHECK_CUDA(hidden);
  CHECK_CUDA(active_idx);
  CHECK_CUDA(down_w);
  CHECK_CUDA(down_b);
  CHECK_CUDA(up_w);
  CHECK_CUDA(up_b);

  CHECK_CONTIGUOUS(hidden);
  CHECK_CONTIGUOUS(active_idx);
  CHECK_CONTIGUOUS(down_w);
  CHECK_CONTIGUOUS(down_b);
  CHECK_CONTIGUOUS(up_w);
  CHECK_CONTIGUOUS(up_b);

  return sparse_adapter_cuda_launcher(hidden, active_idx, down_w, down_b, up_w, up_b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "spatial_gate_cuda",
      &spatial_gate_cuda,
      "Coordinate-driven spatial Top-K gating (CUDA)");
  m.def(
      "sparse_adapter_cuda",
      &sparse_adapter_cuda,
      "Block-sparse adapter compute (CUDA)");
}
