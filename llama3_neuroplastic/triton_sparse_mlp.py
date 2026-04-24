from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import bitsandbytes.functional as bnb_functional
except ImportError:
    bnb_functional = None

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

_DEFAULT_MAX_FUSED_TOPK = 32


def triton_sparse_mlp_available() -> bool:
    if not _TRITON_AVAILABLE:
        return False
    return True


def _next_power_of_two(value: int) -> int:
    value = max(1, int(value))
    return 1 << (value - 1).bit_length()


def _env_int(name: str, default: int, lower: int, upper: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return int(max(lower, min(upper, value)))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _cuda_capability(device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return (0, 0)
    index = device.index if device.index is not None else torch.cuda.current_device()
    return tuple(int(x) for x in torch.cuda.get_device_capability(index))


def _is_pre_ampere(device: torch.device) -> bool:
    major, _minor = _cuda_capability(device)
    return major != 0 and major < 8


def _preferred_compute_dtype(tensor: torch.Tensor) -> torch.dtype:
    dtype = tensor.dtype
    if tensor.device.type == "cuda" and _is_pre_ampere(tensor.device) and dtype == torch.bfloat16:
        return torch.float16
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    return torch.float16


def _resolve_weight_dtype(tensor: torch.Tensor) -> torch.dtype:
    return _preferred_compute_dtype(tensor)


def _prepare_activation_tensor(x: torch.Tensor) -> torch.Tensor:
    target_dtype = _preferred_compute_dtype(x)
    if x.dtype != target_dtype:
        x = x.to(dtype=target_dtype)
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def _prepare_active_idx(active_idx: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if active_idx.device != device or active_idx.dtype != torch.int32 or not active_idx.is_contiguous():
        active_idx = active_idx.to(device=device, dtype=torch.int32).contiguous()
    return active_idx


def _prepare_mask(mask: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if mask.device != device or not mask.is_contiguous():
        mask = mask.to(device=device).contiguous()
    return mask


def _topk_fused_limit() -> int:
    value = _env_int("SCA_TRITON_MAX_FUSED_TOPK", _DEFAULT_MAX_FUSED_TOPK, 1, 128)
    return int(value)


def _should_use_fused_input_kernel(
    *,
    rows: int,
    out_features: int,
    top_k: int,
    block_size: int,
    device: torch.device,
) -> bool:
    if top_k <= 0 or top_k > _topk_fused_limit():
        return False
    if _env_flag("SCA_TRITON_DISABLE_FUSED_INPUT", False):
        return False
    if _env_flag("SCA_TRITON_FORCE_FUSED_INPUT", False):
        return True

    major, _minor = _cuda_capability(device)
    if major == 0:
        return top_k <= 8

    if major < 8:
        if rows < 8:
            return False
        if top_k > 8:
            return False
        return not out_features < 1024

    return top_k <= 16 or (rows >= 4 and block_size <= 256)


def materialize_linear_weight(linear: torch.nn.Module, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    weight = getattr(linear, "weight", None)
    if torch.is_tensor(weight) and weight.dim() == 2 and weight.is_floating_point():
        return weight.detach().to(device=device, dtype=dtype).contiguous()

    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None or bnb_functional is None:
        raise TypeError(f"Unsupported linear weight type for Triton sparse path: {type(weight)!r}")
    if device.type != "cuda":
        raise RuntimeError("Triton sparse path for quantized weights requires CUDA tensors")

    dense = bnb_functional.dequantize_4bit(weight.t(), quant_state)
    return dense.t().detach().to(device=device, dtype=dtype).contiguous()


def materialize_linear_bias(
    linear: torch.nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    bias = getattr(linear, "bias", None)
    if bias is None:
        return None
    return bias.detach().to(device=device, dtype=dtype).contiguous()


def linear_has_4bit_weight(linear: torch.nn.Module) -> bool:
    weight = getattr(linear, "weight", None)
    if not torch.is_tensor(weight):
        return False
    quant_state = getattr(weight, "quant_state", None)
    if quant_state is None:
        return False
    if getattr(quant_state, "shape", None) is None:
        return False
    return not int(getattr(quant_state, "blocksize", 0)) <= 0


def materialize_linear_4bit_params(
    linear: torch.nn.Module,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, str]:
    if not linear_has_4bit_weight(linear):
        raise TypeError(
            f"Unsupported linear weight type for 4-bit Triton sparse path: {type(getattr(linear, 'weight', None))!r}"
        )
    if bnb_functional is None:
        raise RuntimeError("bitsandbytes is required for 4-bit Triton sparse path")
    if device.type != "cuda":
        raise RuntimeError("4-bit Triton sparse path requires CUDA tensors")

    weight = linear.weight
    quant_state = weight.quant_state
    packed = weight.detach().view(-1).to(device=device, dtype=torch.uint8).contiguous()

    absmax = quant_state.absmax
    if quant_state.nested:
        absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
        absmax = absmax + quant_state.offset
    absmax = absmax.to(device=device, dtype=torch.float32).contiguous()

    code = quant_state.code.to(device=device, dtype=torch.float32).contiguous()
    out_features = int(quant_state.shape[0])
    in_features = int(quant_state.shape[1])
    quant_block_size = int(quant_state.blocksize)
    quant_type = str(quant_state.quant_type)
    return packed, absmax, code, out_features, in_features, quant_block_size, quant_type


def _dequantize_4bit_weight_for_backward(
    quant_weight_ref: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if quant_weight_ref is None:
        raise RuntimeError("4-bit Triton backward requires a quantized weight reference")
    quant_state = getattr(quant_weight_ref, "quant_state", None)
    if quant_state is None or bnb_functional is None:
        raise RuntimeError("bitsandbytes quant_state is required for 4-bit Triton backward")
    dense = bnb_functional.dequantize_4bit(quant_weight_ref.t(), quant_state)
    return dense.t().to(device=device, dtype=dtype).contiguous()


if _TRITON_AVAILABLE:
    _INPUT_TILE_CONFIGS = [
        triton.Config({"BLOCK_OUT": 64}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_OUT": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_OUT": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OUT": 256}, num_warps=8, num_stages=1),
    ]
    _OUTPUT_TILE_CONFIGS = [
        triton.Config({"BLOCK_OUT": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_OUT": 64, "BLOCK_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_OUT": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_OUT": 128, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OUT": 128, "BLOCK_K": 128}, num_warps=8, num_stages=1),
    ]
    _INPUT_4BIT_TILE_CONFIGS = [
        triton.Config({"BLOCK_OUT": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_OUT": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_OUT": 128}, num_warps=4, num_stages=1),
    ]
    _OUTPUT_4BIT_TILE_CONFIGS = [
        triton.Config({"BLOCK_OUT": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_OUT": 64, "BLOCK_K": 32}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_OUT": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_OUT": 128, "BLOCK_K": 64}, num_warps=4, num_stages=1),
    ]

    @triton.autotune(configs=_INPUT_TILE_CONFIGS, key=["out_features", "block_size", "top_k"])
    @triton.jit
    def _sparse_input_linear_tiled_kernel(
        x_ptr,
        active_idx_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows,
        out_features,
        top_k,
        block_size,
        stride_x_row,
        stride_idx_row,
        stride_idx_col,
        stride_w_row,
        stride_w_col,
        stride_out_row,
        TOPK_MAX: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_n = tl.program_id(1)
        if row >= rows:
            return

        offs_out = pid_n * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        offs_in = tl.arange(0, BLOCK_IN)
        out_mask = offs_out < out_features

        if HAS_BIAS:
            acc = tl.load(bias_ptr + offs_out, mask=out_mask, other=0.0).to(tl.float32)
        else:
            acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

        for slot in range(TOPK_MAX):
            block_idx = tl.load(
                active_idx_ptr + row * stride_idx_row + slot * stride_idx_col,
                mask=slot < top_k,
                other=-1,
            )
            if block_idx >= 0:
                block_base = block_idx * block_size
                x_ptrs = x_ptr + row * stride_x_row + block_base + offs_in
                x = tl.load(x_ptrs, mask=offs_in < block_size, other=0.0)

                w_ptrs = weight_ptr + offs_out[:, None] * stride_w_row + (block_base + offs_in[None, :]) * stride_w_col
                w_mask = out_mask[:, None] & (offs_in[None, :] < block_size)
                w = tl.load(w_ptrs, mask=w_mask, other=0.0)
                acc += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)

        out_ptrs = out_ptr + row * stride_out_row + offs_out
        tl.store(out_ptrs, acc, mask=out_mask)


    @triton.jit
    def _sparse_input_linear_atomic_kernel(
        x_ptr,
        active_idx_ptr,
        weight_ptr,
        out_ptr,
        rows,
        out_features,
        top_k,
        block_size,
        stride_x_row,
        stride_idx_row,
        stride_idx_col,
        stride_w_row,
        stride_w_col,
        stride_out_row,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        pid_pair = tl.program_id(0)
        pid_tile = tl.program_id(1)

        row = pid_pair // top_k
        if row >= rows:
            return

        slot = pid_pair % top_k
        block_idx = tl.load(active_idx_ptr + row * stride_idx_row + slot * stride_idx_col)
        if block_idx < 0:
            return

        offs_in = tl.arange(0, BLOCK_IN)
        offs_out = pid_tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        block_base = block_idx * block_size

        x_ptrs = x_ptr + row * stride_x_row + block_base + offs_in
        x = tl.load(x_ptrs, mask=offs_in < block_size, other=0.0).to(tl.float32)

        w_ptrs = weight_ptr + offs_out[:, None] * stride_w_row + (block_base + offs_in[None, :]) * stride_w_col
        w_mask = (offs_out[:, None] < out_features) & (offs_in[None, :] < block_size)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)

        acc = tl.sum(w * x[None, :], axis=1)
        out_ptrs = out_ptr + row * stride_out_row + offs_out
        tl.atomic_add(out_ptrs, acc, mask=offs_out < out_features)


    @triton.autotune(configs=_OUTPUT_TILE_CONFIGS, key=["input_dim", "block_size"])
    @triton.jit
    def _sparse_output_linear_kernel(
        x_ptr,
        active_idx_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        rows,
        input_dim,
        top_k,
        block_size,
        stride_x_row,
        stride_idx_row,
        stride_idx_col,
        stride_w_row,
        stride_w_col,
        stride_out_row,
        BLOCK_OUT: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_pair = tl.program_id(0)
        pid_tile = tl.program_id(1)

        row = pid_pair // top_k
        if row >= rows:
            return

        slot = pid_pair % top_k
        block_idx = tl.load(active_idx_ptr + row * stride_idx_row + slot * stride_idx_col)
        if block_idx < 0:
            return

        offs_out = pid_tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        output_idx = block_idx * block_size + offs_out
        valid_out = offs_out < block_size

        if HAS_BIAS:
            acc = tl.load(bias_ptr + output_idx, mask=valid_out, other=0.0).to(tl.float32)
        else:
            acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

        for k_start in tl.range(0, input_dim, BLOCK_K, num_stages=2, loop_unroll_factor=2):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            x_ptrs = x_ptr + row * stride_x_row + offs_k
            x = tl.load(x_ptrs, mask=offs_k < input_dim, other=0.0)

            w_ptrs = weight_ptr + output_idx[:, None] * stride_w_row + offs_k[None, :] * stride_w_col
            w_mask = valid_out[:, None] & (offs_k[None, :] < input_dim)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)

        out_ptrs = out_ptr + row * stride_out_row + output_idx
        tl.store(out_ptrs, acc, mask=valid_out)


    @triton.autotune(configs=_INPUT_4BIT_TILE_CONFIGS, key=["out_features", "block_size", "top_k"])
    @triton.jit
    def _sparse_input_linear_4bit_tiled_kernel(
        x_ptr,
        active_idx_ptr,
        packed_weight_ptr,
        absmax_ptr,
        code_ptr,
        bias_ptr,
        out_ptr,
        rows,
        out_features,
        in_features,
        top_k,
        block_size,
        quant_block_size,
        stride_x_row,
        stride_idx_row,
        stride_idx_col,
        stride_out_row,
        TOPK_MAX: tl.constexpr,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        row = tl.program_id(0)
        pid_n = tl.program_id(1)
        if row >= rows:
            return

        offs_out = pid_n * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        offs_in = tl.arange(0, BLOCK_IN)
        out_mask = offs_out < out_features

        if HAS_BIAS:
            acc = tl.load(bias_ptr + offs_out, mask=out_mask, other=0.0).to(tl.float32)
        else:
            acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

        out_idx = offs_out[:, None]
        for slot in range(TOPK_MAX):
            block_idx = tl.load(
                active_idx_ptr + row * stride_idx_row + slot * stride_idx_col,
                mask=slot < top_k,
                other=-1,
            )
            if block_idx >= 0:
                block_base = block_idx * block_size
                in_idx = block_base + offs_in
                in_valid = offs_in < block_size

                x_ptrs = x_ptr + row * stride_x_row + in_idx
                x = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)

                elem_idx = out_idx * in_features + in_idx[None, :]
                pair_idx = elem_idx >> 1
                valid = out_mask[:, None] & in_valid[None, :]

                packed = tl.load(packed_weight_ptr + pair_idx, mask=valid, other=0)
                low_nibble = packed & 0x0F
                high_nibble = packed >> 4
                nibble_idx = tl.where((elem_idx & 1) == 0, high_nibble, low_nibble).to(tl.int32)
                quant = tl.load(code_ptr + nibble_idx, mask=valid, other=0.0)

                abs_idx = elem_idx // quant_block_size
                scale = tl.load(absmax_ptr + abs_idx, mask=valid, other=0.0)
                w = quant.to(tl.float32) * scale.to(tl.float32)
                acc += tl.sum(w * x[None, :], axis=1)

        out_ptrs = out_ptr + row * stride_out_row + offs_out
        tl.store(out_ptrs, acc, mask=out_mask)


    @triton.jit
    def _sparse_input_linear_4bit_atomic_kernel(
        x_ptr,
        active_idx_ptr,
        packed_weight_ptr,
        absmax_ptr,
        code_ptr,
        out_ptr,
        rows,
        out_features,
        in_features,
        top_k,
        block_size,
        quant_block_size,
        stride_x_row,
        stride_idx_row,
        stride_idx_col,
        stride_out_row,
        BLOCK_IN: tl.constexpr,
        BLOCK_OUT: tl.constexpr,
    ):
        pid_pair = tl.program_id(0)
        pid_tile = tl.program_id(1)

        row = pid_pair // top_k
        if row >= rows:
            return

        offs_in = tl.arange(0, BLOCK_IN)
        offs_out = pid_tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        slot = pid_pair % top_k
        block_idx = tl.load(active_idx_ptr + row * stride_idx_row + slot * stride_idx_col)
        if block_idx < 0:
            return

        block_base = block_idx * block_size
        in_idx = block_base + offs_in
        in_valid = offs_in < block_size

        x_ptrs = x_ptr + row * stride_x_row + in_idx
        x = tl.load(x_ptrs, mask=in_valid, other=0.0).to(tl.float32)

        out_idx = offs_out[:, None]
        in_idx_2d = in_idx[None, :]
        elem_idx = out_idx * in_features + in_idx_2d
        pair_idx = elem_idx >> 1
        valid = (offs_out[:, None] < out_features) & in_valid[None, :]

        packed = tl.load(packed_weight_ptr + pair_idx, mask=valid, other=0)
        low_nibble = packed & 0x0F
        high_nibble = packed >> 4
        nibble_idx = tl.where((elem_idx & 1) == 0, high_nibble, low_nibble).to(tl.int32)

        quant = tl.load(code_ptr + nibble_idx, mask=valid, other=0.0)
        abs_idx = elem_idx // quant_block_size
        scale = tl.load(absmax_ptr + abs_idx, mask=valid, other=0.0)
        w = quant.to(tl.float32) * scale.to(tl.float32)
        acc = tl.sum(w * x[None, :], axis=1)
        out_ptrs = out_ptr + row * stride_out_row + offs_out
        tl.atomic_add(out_ptrs, acc, mask=offs_out < out_features)


    @triton.autotune(configs=_OUTPUT_4BIT_TILE_CONFIGS, key=["input_dim", "block_size", "quant_block_size"])
    @triton.jit
    def _sparse_output_linear_4bit_kernel(
        x_ptr,
        active_idx_ptr,
        packed_weight_ptr,
        absmax_ptr,
        code_ptr,
        bias_ptr,
        out_ptr,
        rows,
        input_dim,
        top_k,
        block_size,
        quant_block_size,
        stride_x_row,
        stride_idx_row,
        stride_idx_col,
        stride_out_row,
        BLOCK_OUT: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        pid_pair = tl.program_id(0)
        pid_tile = tl.program_id(1)

        row = pid_pair // top_k
        if row >= rows:
            return

        slot = pid_pair % top_k
        block_idx = tl.load(active_idx_ptr + row * stride_idx_row + slot * stride_idx_col)
        if block_idx < 0:
            return

        offs_out = pid_tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
        output_idx = block_idx * block_size + offs_out
        valid_out = offs_out < block_size

        if HAS_BIAS:
            acc = tl.load(bias_ptr + output_idx, mask=valid_out, other=0.0).to(tl.float32)
        else:
            acc = tl.zeros((BLOCK_OUT,), dtype=tl.float32)

        out_idx = output_idx[:, None]
        for k_start in tl.range(0, input_dim, BLOCK_K, num_stages=2, loop_unroll_factor=2):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            valid_k = offs_k < input_dim

            x_ptrs = x_ptr + row * stride_x_row + offs_k
            x = tl.load(x_ptrs, mask=valid_k, other=0.0).to(tl.float32)

            elem_idx = out_idx * input_dim + offs_k[None, :]
            pair_idx = elem_idx >> 1
            valid = valid_out[:, None] & valid_k[None, :]

            packed = tl.load(packed_weight_ptr + pair_idx, mask=valid, other=0)
            low_nibble = packed & 0x0F
            high_nibble = packed >> 4
            nibble_idx = tl.where((elem_idx & 1) == 0, high_nibble, low_nibble).to(tl.int32)
            quant = tl.load(code_ptr + nibble_idx, mask=valid, other=0.0)

            abs_idx = elem_idx // quant_block_size
            scale = tl.load(absmax_ptr + abs_idx, mask=valid, other=0.0)
            w = quant.to(tl.float32) * scale.to(tl.float32)
            acc += tl.sum(w * x[None, :], axis=1)

        out_ptrs = out_ptr + row * stride_out_row + output_idx
        tl.store(out_ptrs, acc, mask=valid_out)


class _SparseInputLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_flat: torch.Tensor,
        active_idx: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        block_size: int,
    ) -> torch.Tensor:
        if (not _TRITON_AVAILABLE) or x_flat.device.type != "cuda":
            raise RuntimeError("Triton sparse input kernel requires Triton on CUDA")

        rows = int(x_flat.shape[0])
        out_features = int(weight.shape[0])
        top_k = int(active_idx.shape[1])
        out = torch.empty((rows, out_features), device=x_flat.device, dtype=x_flat.dtype)

        if rows == 0 or out_features == 0:
            out.zero_()
            ctx.save_for_backward(weight)
            return out

        block_in = _next_power_of_two(block_size)
        has_bias = bias is not None
        use_fused = _should_use_fused_input_kernel(
            rows=rows,
            out_features=out_features,
            top_k=top_k,
            block_size=block_size,
            device=x_flat.device,
        )

        if top_k == 0:
            if has_bias:
                out.copy_(bias.to(device=out.device, dtype=out.dtype).unsqueeze(0).expand_as(out))
            else:
                out.zero_()
        elif use_fused:
            topk_max = _next_power_of_two(top_k)
            def grid(META):
                return (rows, triton.cdiv(out_features, META['BLOCK_OUT']))
            _sparse_input_linear_tiled_kernel[grid](
                x_flat,
                active_idx,
                weight,
                bias if has_bias else out,
                out,
                rows,
                out_features,
                top_k,
                block_size,
                x_flat.stride(0),
                active_idx.stride(0),
                active_idx.stride(1),
                weight.stride(0),
                weight.stride(1),
                out.stride(0),
                TOPK_MAX=topk_max,
                BLOCK_IN=block_in,
                HAS_BIAS=has_bias,
            )
        else:
            out.zero_()
            if has_bias:
                out += bias.to(device=out.device, dtype=out.dtype).unsqueeze(0)
            grid = (rows * top_k, triton.cdiv(out_features, 128))
            _sparse_input_linear_atomic_kernel[grid](
                x_flat,
                active_idx,
                weight,
                out,
                rows,
                out_features,
                top_k,
                block_size,
                x_flat.stride(0),
                active_idx.stride(0),
                active_idx.stride(1),
                weight.stride(0),
                weight.stride(1),
                out.stride(0),
                BLOCK_IN=block_in,
                BLOCK_OUT=128,
                num_warps=4,
                num_stages=1,
            )

        ctx.save_for_backward(weight)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight,) = ctx.saved_tensors
        grad_input = torch.matmul(grad_output.to(dtype=weight.dtype), weight).to(dtype=grad_output.dtype)
        return grad_input, None, None, None, None


class _SparseOutputLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_flat: torch.Tensor,
        active_idx: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        flat_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        if (not _TRITON_AVAILABLE) or x_flat.device.type != "cuda":
            raise RuntimeError("Triton sparse output kernel requires Triton on CUDA")

        rows = int(x_flat.shape[0])
        hidden_size = int(weight.shape[0])
        top_k = int(active_idx.shape[1])
        out = torch.zeros((rows, hidden_size), device=x_flat.device, dtype=x_flat.dtype)

        if rows > 0 and top_k > 0:
            def grid(META):
                return (rows * top_k, triton.cdiv(block_size, META['BLOCK_OUT']))
            _sparse_output_linear_kernel[grid](
                x_flat,
                active_idx,
                weight,
                bias if bias is not None else out,
                out,
                rows,
                int(x_flat.shape[1]),
                top_k,
                block_size,
                x_flat.stride(0),
                active_idx.stride(0),
                active_idx.stride(1),
                weight.stride(0),
                weight.stride(1),
                out.stride(0),
                HAS_BIAS=bias is not None,
            )

        ctx.save_for_backward(weight, flat_mask, out)
        out_masked = out * flat_mask.to(device=out.device, dtype=out.dtype)
        return out_masked

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight, flat_mask, out_before_mask = ctx.saved_tensors
        masked_grad = grad_output.to(dtype=weight.dtype) * flat_mask.to(device=grad_output.device, dtype=weight.dtype)
        grad_input = torch.matmul(masked_grad, weight).to(dtype=grad_output.dtype)

        grad_flat_mask = (grad_output.to(dtype=out_before_mask.dtype) * out_before_mask).to(dtype=grad_output.dtype)
        return grad_input, None, None, None, grad_flat_mask, None


def triton_sparse_input_linear(
    x_flat: torch.Tensor,
    active_idx: torch.Tensor,
    linear: torch.nn.Module | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    block_size: int,
) -> torch.Tensor:
    x_flat = _prepare_activation_tensor(x_flat)
    active_idx = _prepare_active_idx(active_idx, device=x_flat.device)

    if weight is None:
        if linear is None:
            raise ValueError("Either linear or weight must be provided to triton_sparse_input_linear")
        weight = materialize_linear_weight(
            linear,
            device=x_flat.device,
            dtype=_resolve_weight_dtype(x_flat),
        )
    else:
        weight = weight.to(device=x_flat.device, dtype=_resolve_weight_dtype(x_flat)).contiguous()
    if bias is None and linear is not None:
        bias = materialize_linear_bias(linear, device=x_flat.device, dtype=x_flat.dtype)
    elif bias is not None:
        bias = bias.to(device=x_flat.device, dtype=x_flat.dtype).contiguous()
    return _SparseInputLinearFn.apply(x_flat, active_idx, weight, bias, int(block_size))


def triton_sparse_output_linear(
    x_flat: torch.Tensor,
    active_idx: torch.Tensor,
    flat_mask: torch.Tensor,
    linear: torch.nn.Module | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    *,
    block_size: int,
) -> torch.Tensor:
    x_flat = _prepare_activation_tensor(x_flat)
    active_idx = _prepare_active_idx(active_idx, device=x_flat.device)
    flat_mask = _prepare_mask(flat_mask, device=x_flat.device)

    if weight is None:
        if linear is None:
            raise ValueError("Either linear or weight must be provided to triton_sparse_output_linear")
        weight = materialize_linear_weight(
            linear,
            device=x_flat.device,
            dtype=_resolve_weight_dtype(x_flat),
        )
    else:
        weight = weight.to(device=x_flat.device, dtype=_resolve_weight_dtype(x_flat)).contiguous()
    if bias is None and linear is not None:
        bias = materialize_linear_bias(linear, device=x_flat.device, dtype=x_flat.dtype)
    elif bias is not None:
        bias = bias.to(device=x_flat.device, dtype=x_flat.dtype).contiguous()
    return _SparseOutputLinearFn.apply(x_flat, active_idx, weight, bias, flat_mask, int(block_size))


class _SparseInputLinear4bitFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_flat: torch.Tensor,
        active_idx: torch.Tensor,
        packed_weight: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        out_features: int,
        in_features: int,
        quant_block_size: int,
        bias: torch.Tensor | None,
        block_size: int,
        quant_weight_ref: torch.Tensor | None,
    ) -> torch.Tensor:
        if (not _TRITON_AVAILABLE) or x_flat.device.type != "cuda":
            raise RuntimeError("4-bit Triton sparse input kernel requires Triton on CUDA")

        rows = int(x_flat.shape[0])
        top_k = int(active_idx.shape[1])
        out = torch.empty((rows, int(out_features)), device=x_flat.device, dtype=x_flat.dtype)

        if rows == 0 or int(out_features) == 0:
            out.zero_()
            ctx.quant_weight_ref = quant_weight_ref
            return out

        block_in = _next_power_of_two(block_size)
        has_bias = bias is not None
        use_fused = _should_use_fused_input_kernel(
            rows=rows,
            out_features=int(out_features),
            top_k=top_k,
            block_size=int(block_size),
            device=x_flat.device,
        )

        if top_k == 0:
            if has_bias:
                out.copy_(bias.unsqueeze(0).expand_as(out))
            else:
                out.zero_()
        elif use_fused:
            topk_max = _next_power_of_two(top_k)
            def grid(META):
                return (rows, triton.cdiv(int(out_features), META['BLOCK_OUT']))
            _sparse_input_linear_4bit_tiled_kernel[grid](
                x_flat,
                active_idx,
                packed_weight,
                absmax,
                code,
                bias if has_bias else out,
                out,
                rows,
                int(out_features),
                int(in_features),
                top_k,
                int(block_size),
                int(quant_block_size),
                x_flat.stride(0),
                active_idx.stride(0),
                active_idx.stride(1),
                out.stride(0),
                TOPK_MAX=topk_max,
                BLOCK_IN=block_in,
                HAS_BIAS=has_bias,
            )
        else:
            out.zero_()
            if has_bias:
                out += bias.unsqueeze(0)
            grid = (rows * top_k, triton.cdiv(int(out_features), 64))
            _sparse_input_linear_4bit_atomic_kernel[grid](
                x_flat,
                active_idx,
                packed_weight,
                absmax,
                code,
                out,
                rows,
                int(out_features),
                int(in_features),
                top_k,
                int(block_size),
                int(quant_block_size),
                x_flat.stride(0),
                active_idx.stride(0),
                active_idx.stride(1),
                out.stride(0),
                BLOCK_IN=block_in,
                BLOCK_OUT=64,
                num_warps=4,
                num_stages=1,
            )

        ctx.quant_weight_ref = quant_weight_ref
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        quant_weight_ref = getattr(ctx, "quant_weight_ref", None)
        grad_input = None
        if grad_output is not None:
            weight = _dequantize_4bit_weight_for_backward(
                quant_weight_ref,
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            grad_input = torch.matmul(grad_output, weight)
        return grad_input, None, None, None, None, None, None, None, None, None, None


class _SparseOutputLinear4bitFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_flat: torch.Tensor,
        active_idx: torch.Tensor,
        flat_mask: torch.Tensor,
        packed_weight: torch.Tensor,
        absmax: torch.Tensor,
        code: torch.Tensor,
        input_dim: int,
        quant_block_size: int,
        bias: torch.Tensor | None,
        block_size: int,
        quant_weight_ref: torch.Tensor | None,
    ) -> torch.Tensor:
        if (not _TRITON_AVAILABLE) or x_flat.device.type != "cuda":
            raise RuntimeError("4-bit Triton sparse output kernel requires Triton on CUDA")

        rows = int(x_flat.shape[0])
        hidden_size = int(flat_mask.shape[1])
        top_k = int(active_idx.shape[1])
        out = torch.zeros((rows, hidden_size), device=x_flat.device, dtype=x_flat.dtype)

        if rows > 0 and top_k > 0:
            def grid(META):
                return (rows * top_k, triton.cdiv(int(block_size), META['BLOCK_OUT']))
            _sparse_output_linear_4bit_kernel[grid](
                x_flat,
                active_idx,
                packed_weight,
                absmax,
                code,
                bias if bias is not None else out,
                out,
                rows,
                int(input_dim),
                top_k,
                int(block_size),
                int(quant_block_size),
                x_flat.stride(0),
                active_idx.stride(0),
                active_idx.stride(1),
                out.stride(0),
                HAS_BIAS=bias is not None,
            )

        ctx.quant_weight_ref = quant_weight_ref
        ctx.save_for_backward(flat_mask, out)
        return out * flat_mask.to(device=out.device, dtype=out.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        flat_mask, out_before_mask = ctx.saved_tensors
        quant_weight_ref = getattr(ctx, "quant_weight_ref", None)
        masked_grad = grad_output * flat_mask.to(device=grad_output.device, dtype=grad_output.dtype)
        weight = _dequantize_4bit_weight_for_backward(
            quant_weight_ref,
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input = torch.matmul(masked_grad, weight)
        grad_flat_mask = (grad_output.to(dtype=out_before_mask.dtype) * out_before_mask).to(dtype=grad_output.dtype)
        return grad_input, None, grad_flat_mask, None, None, None, None, None, None, None, None


def triton_sparse_input_linear_4bit(
    x_flat: torch.Tensor,
    active_idx: torch.Tensor,
    *,
    packed_weight: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    out_features: int,
    in_features: int,
    quant_block_size: int,
    bias: torch.Tensor | None,
    block_size: int,
    quant_weight_ref: torch.Tensor | None = None,
) -> torch.Tensor:
    x_flat = _prepare_activation_tensor(x_flat)
    active_idx = _prepare_active_idx(active_idx, device=x_flat.device)
    packed_weight = packed_weight.to(device=x_flat.device, dtype=torch.uint8).contiguous()
    absmax = absmax.to(device=x_flat.device, dtype=torch.float32).contiguous()
    code = code.to(device=x_flat.device, dtype=torch.float32).contiguous()
    if bias is not None:
        bias = bias.to(device=x_flat.device, dtype=x_flat.dtype).contiguous()

    return _SparseInputLinear4bitFn.apply(
        x_flat,
        active_idx,
        packed_weight,
        absmax,
        code,
        int(out_features),
        int(in_features),
        int(quant_block_size),
        bias,
        int(block_size),
        quant_weight_ref,
    )


def triton_sparse_output_linear_4bit(
    x_flat: torch.Tensor,
    active_idx: torch.Tensor,
    flat_mask: torch.Tensor,
    *,
    packed_weight: torch.Tensor,
    absmax: torch.Tensor,
    code: torch.Tensor,
    input_dim: int,
    quant_block_size: int,
    bias: torch.Tensor | None,
    block_size: int,
    quant_weight_ref: torch.Tensor | None = None,
) -> torch.Tensor:
    x_flat = _prepare_activation_tensor(x_flat)
    active_idx = _prepare_active_idx(active_idx, device=x_flat.device)
    flat_mask = _prepare_mask(flat_mask, device=x_flat.device)
    packed_weight = packed_weight.to(device=x_flat.device, dtype=torch.uint8).contiguous()
    absmax = absmax.to(device=x_flat.device, dtype=torch.float32).contiguous()
    code = code.to(device=x_flat.device, dtype=torch.float32).contiguous()
    if bias is not None:
        bias = bias.to(device=x_flat.device, dtype=x_flat.dtype).contiguous()

    return _SparseOutputLinear4bitFn.apply(
        x_flat,
        active_idx,
        flat_mask,
        packed_weight,
        absmax,
        code,
        int(input_dim),
        int(quant_block_size),
        bias,
        int(block_size),
        quant_weight_ref,
    )


def triton_fused_sparse_mlp_decode_4bit(
    x_flat: torch.Tensor,
    active_idx: torch.Tensor,
    flat_mask: torch.Tensor,
    *,
    gate_packed_weight: torch.Tensor,
    gate_absmax: torch.Tensor,
    gate_code: torch.Tensor,
    gate_input_dim: int,
    gate_quant_block_size: int,
    gate_bias: torch.Tensor | None,
    up_packed_weight: torch.Tensor,
    up_absmax: torch.Tensor,
    up_code: torch.Tensor,
    up_input_dim: int,
    up_quant_block_size: int,
    up_bias: torch.Tensor | None,
    down_packed_weight: torch.Tensor,
    down_absmax: torch.Tensor,
    down_code: torch.Tensor,
    down_out_features: int,
    down_in_features: int,
    down_quant_block_size: int,
    down_bias: torch.Tensor | None,
    block_size: int,
) -> torch.Tensor:
    gate = triton_sparse_output_linear_4bit(
        x_flat,
        active_idx,
        flat_mask,
        packed_weight=gate_packed_weight,
        absmax=gate_absmax,
        code=gate_code,
        input_dim=int(gate_input_dim),
        quant_block_size=int(gate_quant_block_size),
        bias=gate_bias,
        block_size=int(block_size),
        quant_weight_ref=None,
    )
    up = triton_sparse_output_linear_4bit(
        x_flat,
        active_idx,
        flat_mask,
        packed_weight=up_packed_weight,
        absmax=up_absmax,
        code=up_code,
        input_dim=int(up_input_dim),
        quant_block_size=int(up_quant_block_size),
        bias=up_bias,
        block_size=int(block_size),
        quant_weight_ref=None,
    )
    activated = F.silu(gate)
    activated.mul_(up)
    return triton_sparse_input_linear_4bit(
        activated,
        active_idx,
        packed_weight=down_packed_weight,
        absmax=down_absmax,
        code=down_code,
        out_features=int(down_out_features),
        in_features=int(down_in_features),
        quant_block_size=int(down_quant_block_size),
        bias=down_bias,
        block_size=int(block_size),
        quant_weight_ref=None,
    )
