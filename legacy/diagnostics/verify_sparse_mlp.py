from __future__ import annotations

import traceback
from collections.abc import Callable
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime


def _make_runtime_stub() -> StreamingLlamaRuntime:
    runtime = StreamingLlamaRuntime.__new__(StreamingLlamaRuntime)
    runtime.device = torch.device("cpu")
    runtime.dtype = torch.float32
    runtime.config = SimpleNamespace(hidden_size=4, intermediate_size=4)
    runtime._sparse_routing = {}
    runtime._sparse_top_k_by_layer = {}
    runtime._sparse_basis_top_k_by_layer = {}
    runtime._sparse_basis_bias_mode = "selected"
    runtime._sparse_top_k = 2
    runtime._sparse_runtime_top_k = 2
    runtime._sparse_block_size = 2
    runtime._sparse_num_blocks = 2
    runtime._sparse_checkpoint_basis_rank = 2
    runtime._sparse_semantic_block_score_normalized = False
    runtime._sparse_mlp_execution_request = "output_basis_surrogate"
    runtime._sparse_basis_execution = "output_basis_surrogate"
    runtime._sparse_mlp_prefill_mode = "sparse"
    runtime._traffic_current_phase = "decode"
    runtime._mlp_hot_blocks_by_layer = {}
    runtime._session_sparse_route_layers = set()
    runtime._upper_decode_guard_layers = set()
    runtime._lm_head_gpu_attempted = False
    runtime._prefer_gpu_lm_head = False
    return runtime


def verify_linear_latent_uses_absolute_topk() -> None:
    runtime = _make_runtime_stub()
    runtime._sparse_routing[0] = {
        "enc_w": torch.eye(2, dtype=torch.float32),
        "enc_b": torch.zeros(2, dtype=torch.float32),
        "basis_top_k": 1,
    }

    flat_hidden = torch.tensor([[2.0, -3.0]], dtype=torch.float32)
    latent = runtime._compute_sparse_basis_latent(flat_hidden, 0)
    expected = torch.tensor([[0.0, -3.0]], dtype=torch.float32)

    assert torch.allclose(latent, expected), f"expected {expected.tolist()}, got {latent.tolist()}"


def verify_intermediate_router_scores_use_linear_reconstruction() -> None:
    runtime = _make_runtime_stub()
    runtime._sparse_routing[0] = {
        "enc_w": torch.eye(2, dtype=torch.float32),
        "enc_b": torch.zeros(2, dtype=torch.float32),
        "route_score_weight": torch.tensor([[0.0, 1.0], [3.0, 0.0]], dtype=torch.float32),
        "route_score_bias": torch.zeros(2, dtype=torch.float32),
        "block_domain": "intermediate",
        "top_k": 1,
        "basis_top_k": 2,
    }

    hidden = torch.tensor([[[1.0, 4.0]]], dtype=torch.float32)
    active_blocks = runtime._route_sparse_mlp(hidden, 0)

    assert active_blocks.shape == (1, 1)
    assert int(active_blocks.item()) == 0, f"expected block 0, got {active_blocks.tolist()}"


def verify_intermediate_block_scores_apply_silu_gate() -> None:
    runtime = _make_runtime_stub()
    runtime._mlp_proj_staging = torch.empty(16, dtype=torch.float32)
    runtime._mlp_proj_staging_numel = 16
    runtime._dense_mlp_staging_warned = False
    runtime._record_h2d_bytes = lambda *args, **kwargs: None
    runtime._load_optional_bias = lambda *args, **kwargs: None

    class MockMLP(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = torch.nn.Linear(4, 4, bias=False)
            self.up_proj = torch.nn.Linear(4, 4, bias=False)
            self.down_proj = torch.nn.Linear(4, 4, bias=False)
            self.gate_proj.weight.data.copy_(torch.eye(4, dtype=torch.float32))
            self.up_proj.weight.data.copy_(torch.eye(4, dtype=torch.float32))
            self.down_proj.weight.data.copy_(torch.eye(4, dtype=torch.float32))

    mlp = MockMLP()

    def load_parameter(name: str) -> torch.Tensor:
        if name.endswith("gate_proj.weight"):
            return mlp.gate_proj.weight.detach().clone()
        if name.endswith("up_proj.weight"):
            return mlp.up_proj.weight.detach().clone()
        if name.endswith("down_proj.weight"):
            return mlp.down_proj.weight.detach().clone()
        raise KeyError(name)

    runtime.loader = SimpleNamespace(load_parameter=load_parameter)
    hidden = torch.tensor([[[1.0, -2.0, 0.5, 3.0]]], dtype=torch.float32)
    out, block_scores = runtime._dense_mlp_forward_streaming_fast_details(
        0,
        mlp,
        hidden,
        capture_intermediate_block_scores=True,
    )

    activated = F.silu(hidden.view(1, 4)) * hidden.view(1, 4)
    expected_scores = activated.view(1, 2, 2).abs().mean(dim=-1)

    assert block_scores is not None
    assert torch.allclose(block_scores, expected_scores), (
        f"expected SiLU-gated scores {expected_scores.tolist()}, got {block_scores.tolist()}"
    )
    assert torch.allclose(out, activated.view_as(hidden))


def verify_dense_prefill_bypass() -> None:
    runtime = _make_runtime_stub()
    runtime._traffic_current_phase = "prefill"
    runtime._sparse_mlp_prefill_mode = "dense"
    mlp_input = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    runtime._dense_guard_mlp_forward_exact_chunked_4bit = lambda layer_idx, mlp, hidden: hidden * 2.0

    out = runtime._mlp_forward_dispatch(0, SimpleNamespace(mlp=SimpleNamespace()), mlp_input)

    assert torch.allclose(out, mlp_input * 2.0), "prefill did not use dense guard MLP"


def verify_oracle_dispatch_selects_dense_block_scores() -> None:
    runtime = _make_runtime_stub()
    runtime._sparse_mlp_execution_request = "exact_intermediate_sparse_oracle"
    runtime._sparse_basis_execution = "exact_intermediate_sparse_oracle"
    runtime._sparse_routing[0] = {
        "block_domain": "intermediate",
        "top_k": 1,
        "basis_top_k": 2,
    }
    mlp_input = torch.zeros((1, 1, 4), dtype=torch.float32)
    runtime._dense_mlp_forward_streaming_fast_details = lambda *args, **kwargs: (
        torch.zeros_like(mlp_input),
        torch.tensor([[0.1, 3.0]], dtype=torch.float32),
    )

    seen: list[torch.Tensor] = []

    def sparse_fast(layer_idx, mlp, hidden, active_blocks):
        seen.append(active_blocks.clone())
        return torch.full_like(hidden, 5.0)

    runtime._sparse_mlp_forward_fast = sparse_fast
    out = runtime._mlp_forward_dispatch(0, SimpleNamespace(mlp=SimpleNamespace()), mlp_input)

    assert torch.all(out == 5.0)
    assert seen, "oracle dispatch did not call sparse executor"
    assert torch.equal(seen[0], torch.tensor([[1]], dtype=torch.long)), f"unexpected oracle blocks {seen[0].tolist()}"


def verify_fast_executor_accepts_oracle_mode() -> None:
    runtime = _make_runtime_stub()
    runtime._sparse_mlp_execution_request = "exact_intermediate_sparse_oracle"
    runtime._sparse_basis_execution = "exact_intermediate_sparse_oracle"
    runtime._sparse_routing[0] = {"block_domain": "intermediate"}
    active_blocks = torch.tensor([[0, 1]], dtype=torch.long)
    hidden = torch.zeros((1, 1, 4), dtype=torch.float32)
    seen: list[torch.Tensor] = []

    def sparse_reference(layer_idx, mlp, sparse_hidden, sparse_active_blocks):
        seen.append(sparse_active_blocks.clone())
        return torch.full_like(sparse_hidden, 7.0)

    runtime._sparse_mlp_forward = sparse_reference
    out = runtime._sparse_mlp_forward_fast(0, SimpleNamespace(), hidden, active_blocks)

    assert torch.all(out == 7.0)
    assert seen, "fast executor did not reach sparse fallback"
    assert torch.equal(seen[0], active_blocks)


def main() -> int:
    checks: list[tuple[str, Callable[[], None]]] = [
        ("linear latent absolute top-k", verify_linear_latent_uses_absolute_topk),
        ("intermediate router score reconstruction", verify_intermediate_router_scores_use_linear_reconstruction),
        ("SiLU-gated intermediate block scores", verify_intermediate_block_scores_apply_silu_gate),
        ("dense prefill bypass", verify_dense_prefill_bypass),
        ("oracle dispatch", verify_oracle_dispatch_selects_dense_block_scores),
        ("oracle fast executor acceptance", verify_fast_executor_accepts_oracle_mode),
    ]
    failures = 0
    for name, check in checks:
        try:
            check()
            print(f"PASS: {name}")
        except Exception:
            failures += 1
            print(f"FAIL: {name}")
            traceback.print_exc()
    if failures:
        print(f"{failures} sparse MLP verification check(s) failed.")
        return 1
    print("All sparse MLP verification checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
