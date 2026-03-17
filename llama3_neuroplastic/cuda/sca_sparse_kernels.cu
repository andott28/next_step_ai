#include <torch/extension.h>

#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

__global__ void spatial_gate_kernel(
    const float* q,
    const float* block_centers,
    const int32_t* refractory_until,
    int64_t step,
    float sigma_coeff,
    int num_blocks,
    int top_k,
    int64_t* out_idx,
    float* out_score) {
  __shared__ float scores[256];

  const int row = blockIdx.x;
  const int tid = threadIdx.x;

  if (tid < num_blocks) {
    const float qx = q[row * 3 + 0];
    const float qy = q[row * 3 + 1];
    const float qz = q[row * 3 + 2];

    const float cx = block_centers[tid * 3 + 0];
    const float cy = block_centers[tid * 3 + 1];
    const float cz = block_centers[tid * 3 + 2];

    const float dx = qx - cx;
    const float dy = qy - cy;
    const float dz = qz - cz;
    const float dist2 = dx * dx + dy * dy + dz * dz;

    float score = expf(-dist2 * sigma_coeff);
    if (step >= 0 && refractory_until[tid] > step) {
      score = -INFINITY;
    }

    scores[tid] = score;
  }
  __syncthreads();

  if (tid == 0) {
    for (int k = 0; k < top_k; ++k) {
      int best_idx = -1;
      float best_score = -INFINITY;

      for (int b = 0; b < num_blocks; ++b) {
        const float s = scores[b];
        if (s > best_score || (s == best_score && (best_idx < 0 || b < best_idx))) {
          best_score = s;
          best_idx = b;
        }
      }

      const int out_offset = row * top_k + k;
      if (best_idx < 0 || !isfinite(best_score)) {
        out_idx[out_offset] = -1;
        out_score[out_offset] = 0.0f;
      } else {
        out_idx[out_offset] = static_cast<int64_t>(best_idx);
        out_score[out_offset] = best_score;
        scores[best_idx] = -INFINITY;
      }
    }
  }
}

__global__ void sparse_adapter_kernel(
    const __half* hidden,
    const int64_t* active_idx,
    const __half* down_w,
    const __half* down_b,
    const __half* up_w,
    const __half* up_b,
    __half* out,
    int rows,
    int hidden_size,
    int block_size,
    int block_rank,
    int top_k) {
  const int pair = blockIdx.x;
  if (pair >= rows * top_k || threadIdx.x != 0) {
    return;
  }

  if (block_rank > 32 || block_size > 64) {
    return;
  }

  const int row = pair / top_k;
  const int slot = pair % top_k;

  const int64_t block_idx_i64 = active_idx[row * top_k + slot];
  if (block_idx_i64 < 0) {
    return;
  }
  const int block_idx = static_cast<int>(block_idx_i64);

  const int block_offset = block_idx * block_size;
  const __half* x_ptr = hidden + row * hidden_size + block_offset;
  __half* out_ptr = out + row * hidden_size + block_offset;

  float rank_vals[32];
  for (int r = 0; r < block_rank; ++r) {
    float v = __half2float(down_b[block_idx * block_rank + r]);
    for (int i = 0; i < block_size; ++i) {
      const int w_idx = ((block_idx * block_size + i) * block_rank) + r;
      v += __half2float(x_ptr[i]) * __half2float(down_w[w_idx]);
    }
    rank_vals[r] = silu(v);
  }

  for (int o = 0; o < block_size; ++o) {
    float y = __half2float(up_b[block_idx * block_size + o]);
    for (int r = 0; r < block_rank; ++r) {
      const int w_idx = ((block_idx * block_rank + r) * block_size) + o;
      y += rank_vals[r] * __half2float(up_w[w_idx]);
    }
    out_ptr[o] = __float2half(y);
  }
}

}  // namespace

std::vector<torch::Tensor> spatial_gate_cuda_launcher(
    torch::Tensor q,
    torch::Tensor block_centers,
    torch::Tensor refractory_until,
    int64_t step,
    double sigma,
    int64_t top_k) {
  TORCH_CHECK(q.is_cuda(), "q must be CUDA");
  TORCH_CHECK(block_centers.is_cuda(), "block_centers must be CUDA");
  TORCH_CHECK(refractory_until.is_cuda(), "refractory_until must be CUDA");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32, "q must be float32");
  TORCH_CHECK(block_centers.scalar_type() == torch::kFloat32, "block_centers must be float32");
  TORCH_CHECK(refractory_until.scalar_type() == torch::kInt32, "refractory_until must be int32");

  const auto rows = static_cast<int>(q.size(0));
  const auto num_blocks = static_cast<int>(block_centers.size(0));
  TORCH_CHECK(q.size(1) == 3, "q must have shape [N, 3]");
  TORCH_CHECK(block_centers.size(1) == 3, "block_centers must have shape [B, 3]");
  TORCH_CHECK(num_blocks <= 256, "num_blocks > 256 not supported in current kernel");
  TORCH_CHECK(top_k > 0, "top_k must be > 0");
  TORCH_CHECK(top_k <= num_blocks, "top_k must be <= num_blocks");

  auto idx_options = torch::TensorOptions().device(q.device()).dtype(torch::kInt64);
  auto score_options = torch::TensorOptions().device(q.device()).dtype(torch::kFloat32);
  auto active_idx = torch::full({rows, top_k}, -1, idx_options);
  auto active_score = torch::zeros({rows, top_k}, score_options);

  const float sigma_coeff = 1.0f / (2.0f * static_cast<float>(sigma) * static_cast<float>(sigma));

  const int threads = 256;
  spatial_gate_kernel<<<rows, threads>>>(
      q.data_ptr<float>(),
      block_centers.data_ptr<float>(),
      refractory_until.data_ptr<int32_t>(),
      step,
      sigma_coeff,
      num_blocks,
      static_cast<int>(top_k),
      active_idx.data_ptr<int64_t>(),
      active_score.data_ptr<float>());

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "spatial_gate_kernel launch failed: ", cudaGetErrorString(err));

  return {active_idx, active_score};
}

torch::Tensor sparse_adapter_cuda_launcher(
    torch::Tensor hidden,
    torch::Tensor active_idx,
    torch::Tensor down_w,
    torch::Tensor down_b,
    torch::Tensor up_w,
    torch::Tensor up_b) {
  TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
  TORCH_CHECK(active_idx.is_cuda(), "active_idx must be CUDA");
  TORCH_CHECK(down_w.is_cuda(), "down_w must be CUDA");
  TORCH_CHECK(down_b.is_cuda(), "down_b must be CUDA");
  TORCH_CHECK(up_w.is_cuda(), "up_w must be CUDA");
  TORCH_CHECK(up_b.is_cuda(), "up_b must be CUDA");

  TORCH_CHECK(hidden.scalar_type() == torch::kFloat16, "hidden must be float16");
  TORCH_CHECK(down_w.scalar_type() == torch::kFloat16, "down_w must be float16");
  TORCH_CHECK(down_b.scalar_type() == torch::kFloat16, "down_b must be float16");
  TORCH_CHECK(up_w.scalar_type() == torch::kFloat16, "up_w must be float16");
  TORCH_CHECK(up_b.scalar_type() == torch::kFloat16, "up_b must be float16");
  TORCH_CHECK(active_idx.scalar_type() == torch::kInt64, "active_idx must be int64");

  TORCH_CHECK(hidden.dim() == 2, "hidden must be [N, H]");
  TORCH_CHECK(active_idx.dim() == 2, "active_idx must be [N, K]");
  TORCH_CHECK(down_w.dim() == 3, "down_w must be [B, BS, R]");
  TORCH_CHECK(down_b.dim() == 2, "down_b must be [B, R]");
  TORCH_CHECK(up_w.dim() == 3, "up_w must be [B, R, BS]");
  TORCH_CHECK(up_b.dim() == 2, "up_b must be [B, BS]");

  const int rows = static_cast<int>(hidden.size(0));
  const int hidden_size = static_cast<int>(hidden.size(1));
  const int top_k = static_cast<int>(active_idx.size(1));
  const int num_blocks = static_cast<int>(down_w.size(0));
  const int block_size = static_cast<int>(down_w.size(1));
  const int block_rank = static_cast<int>(down_w.size(2));

  TORCH_CHECK(active_idx.size(0) == rows, "active_idx first dimension mismatch");
  TORCH_CHECK(down_b.size(0) == num_blocks && down_b.size(1) == block_rank, "down_b shape mismatch");
  TORCH_CHECK(up_w.size(0) == num_blocks && up_w.size(1) == block_rank && up_w.size(2) == block_size, "up_w shape mismatch");
  TORCH_CHECK(up_b.size(0) == num_blocks && up_b.size(1) == block_size, "up_b shape mismatch");
  TORCH_CHECK(num_blocks * block_size == hidden_size, "num_blocks * block_size must equal hidden_size");

  auto out = torch::zeros_like(hidden);

  const int blocks = rows * top_k;
  sparse_adapter_kernel<<<blocks, 1>>>(
      reinterpret_cast<__half*>(hidden.data_ptr<at::Half>()),
      active_idx.data_ptr<int64_t>(),
      reinterpret_cast<__half*>(down_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(down_b.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(up_w.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(up_b.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(out.data_ptr<at::Half>()),
      rows,
      hidden_size,
      block_size,
      block_rank,
      top_k);

  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "sparse_adapter_kernel launch failed: ", cudaGetErrorString(err));

  return out;
}
