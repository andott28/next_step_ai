from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class GQATaylorLayout:
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def hidden_size(self) -> int:
        return int(self.num_attention_heads * self.head_dim)

    @property
    def query_heads_per_group(self) -> int:
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"(got {self.num_attention_heads} and {self.num_key_value_heads})"
            )
        return int(self.num_attention_heads // self.num_key_value_heads)

    def validate(self) -> None:
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0 or self.head_dim <= 0:
            raise ValueError("All layout dimensions must be positive")
        _ = self.query_heads_per_group


@dataclass
class TaylorSSDLayerCache:
    state_S: torch.Tensor
    state_z: torch.Tensor
    local_k: torch.Tensor
    local_v: torch.Tensor
    local_valid: torch.Tensor
    cache_pos: int
    seen_tokens: int

    @property
    def S(self) -> torch.Tensor:
        return self.state_S

    @property
    def z(self) -> torch.Tensor:
        return self.state_z


@dataclass(frozen=True)
class TaylorSSDConfig:
    order: int = 2
    symmetric_quadratic: bool = True
    taylor_temperature: float = 1.0
    eps: float = 1e-6
    feature_map: str = "hybrid_performer"
    state_decay: float = 1.0
    local_window: int = 64
    feature_dim: int = 64

    def validate(self) -> None:
        if self.feature_map not in {"elu", "taylor", "hybrid_performer"}:
            raise ValueError(
                "TaylorSSDConfig.feature_map must be 'elu', 'taylor', or 'hybrid_performer', "
                f"got {self.feature_map!r}"
            )
        if self.feature_map == "taylor" and int(self.order) != 2:
            raise ValueError(f"TaylorSSDConfig: order=2 is the only supported Taylor order, got order={self.order}")
        if not (0.0 < float(self.state_decay) <= 1.0):
            raise ValueError("TaylorSSDConfig.state_decay must be in (0, 1]")
        if float(self.eps) <= 0.0:
            raise ValueError("TaylorSSDConfig.eps must be > 0")
        if int(self.local_window) <= 0:
            raise ValueError("TaylorSSDConfig.local_window must be > 0")
        if int(self.feature_dim) <= 0:
            raise ValueError("TaylorSSDConfig.feature_dim must be > 0")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _normalize_rope_tensor(
    tensor: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    out = tensor.to(device=device, dtype=dtype)
    if out.ndim == 2:
        out = out.unsqueeze(0).unsqueeze(1)
    elif out.ndim == 3:
        if out.shape[0] in {1, batch_size} and out.shape[1] == seq_len:
            out = out.unsqueeze(1)
        elif out.shape[0] == seq_len and out.shape[1] in {1, batch_size}:
            out = out.permute(1, 0, 2).unsqueeze(1)
        else:
            out = out.unsqueeze(0)
    elif out.ndim == 4:
        if out.shape[1] != 1 and out.shape[2] == 1:
            out = out.transpose(1, 2)
    else:
        raise RuntimeError(f"Unsupported RoPE tensor rank: {out.ndim}")

    if out.shape[0] == 1 and batch_size > 1:
        out = out.expand(batch_size, -1, -1, -1)
    if out.shape[0] != batch_size:
        raise RuntimeError(f"RoPE batch mismatch: expected {batch_size}, got {out.shape[0]}")
    if out.shape[-2] < seq_len:
        raise RuntimeError(f"RoPE sequence length mismatch: expected >= {seq_len}, got {out.shape[-2]}")
    if out.shape[-1] < head_dim:
        raise RuntimeError(f"RoPE head dim mismatch: expected >= {head_dim}, got {out.shape[-1]}")
    return out[..., :seq_len, :head_dim]


def _apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.permute(0, 2, 1, 3), k_embed.permute(0, 2, 1, 3)


class HybridSoftmaxLinearAttention(nn.Module):
    def __init__(self, *, layout: GQATaylorLayout, config: TaylorSSDConfig) -> None:
        super().__init__()
        self.layout = layout
        self.config = config
        self.local_window = int(config.local_window)
        self._state_decay_scalar = float(config.state_decay)
        head_map = torch.arange(layout.num_attention_heads, dtype=torch.long) // layout.query_heads_per_group
        self.register_buffer("query_to_kv", head_map, persistent=False)
        projection = self._build_projection_matrix(int(config.feature_dim), int(layout.head_dim))
        self.register_buffer("performer_projection", projection, persistent=False)

    @property
    def feature_dim(self) -> int:
        if self.config.feature_map == "hybrid_performer":
            return int(self.config.feature_dim)
        if self.config.feature_map == "elu":
            return int(self.layout.head_dim)
        d = int(self.layout.head_dim)
        return 1 + d + int(d * d)

    @staticmethod
    def _build_projection_matrix(feature_dim: int, head_dim: int) -> torch.Tensor:
        rows = max(int(feature_dim), 1)
        cols = max(int(head_dim), 1)
        blocks = []
        remaining = rows
        while remaining > 0:
            block_rows = min(remaining, cols)
            q, _ = torch.linalg.qr(torch.randn(cols, cols), mode="reduced")
            blocks.append(q[:, :block_rows].transpose(0, 1).contiguous())
            remaining -= block_rows
        matrix = torch.cat(blocks, dim=0)
        return F.normalize(matrix.float(), p=2.0, dim=-1, eps=1e-6)

    def _phi_elu(self, x: torch.Tensor) -> torch.Tensor:
        scale = math.sqrt(float(self.layout.head_dim))
        return F.relu(x.float() / scale) + float(self.config.eps)

    def _phi_performer(self, x: torch.Tensor) -> torch.Tensor:
        scaled = x.float() / (float(self.layout.head_dim) ** 0.25)
        projection = self.performer_projection.to(device=scaled.device, dtype=scaled.dtype)
        projected = torch.einsum("...d,fd->...f", scaled, projection)
        squared_norm = 0.5 * (scaled * scaled).sum(dim=-1, keepdim=True)
        logits = torch.clamp(projected - squared_norm, min=-20.0, max=20.0)
        return torch.exp(logits) / math.sqrt(float(self.config.feature_dim))

    def phi_q(self, query: torch.Tensor) -> torch.Tensor:
        if self.config.feature_map == "hybrid_performer":
            return self._phi_performer(query)
        if self.config.feature_map == "elu":
            return self._phi_elu(query)
        q = query.float()
        bsz, num_heads, head_dim = q.shape
        temp = float(self.config.taylor_temperature)
        ones = torch.ones((bsz, num_heads, 1), device=q.device, dtype=q.dtype)
        quadratic = torch.einsum("bhd,bhe->bhde", q, q).reshape(bsz, num_heads, head_dim * head_dim)

        if self.config.symmetric_quadratic:
            linear = q / (math.sqrt(temp) * (float(head_dim) ** 0.25))
            quadratic = quadratic / (temp * math.sqrt(2.0 * float(head_dim)))
            return torch.cat([ones, linear, quadratic], dim=-1)

        linear = q / (temp * math.sqrt(float(head_dim)))
        quadratic = quadratic / (temp * temp * math.sqrt(2.0) * float(head_dim))
        return torch.cat([ones, linear, quadratic], dim=-1)

    def phi_k(self, key: torch.Tensor) -> torch.Tensor:
        if self.config.feature_map == "hybrid_performer":
            return self._phi_performer(key)
        if self.config.feature_map == "elu":
            return self._phi_elu(key)
        k = key.float()
        bsz, num_heads, head_dim = k.shape
        temp = float(self.config.taylor_temperature)
        ones = torch.ones((bsz, num_heads, 1), device=k.device, dtype=k.dtype)
        quadratic = torch.einsum("bhd,bhe->bhde", k, k).reshape(bsz, num_heads, head_dim * head_dim)

        if self.config.symmetric_quadratic:
            linear = k / (math.sqrt(temp) * (float(head_dim) ** 0.25))
            quadratic = quadratic / (temp * math.sqrt(2.0 * float(head_dim)))
            return torch.cat([ones, linear, quadratic], dim=-1)

        linear = k
        quadratic = quadratic / math.sqrt(2.0)
        return torch.cat([ones, linear, quadratic], dim=-1)

    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> TaylorSSDLayerCache:
        feature_dim = int(self.feature_dim)
        state_shape = (batch_size, self.layout.num_key_value_heads, feature_dim, self.layout.head_dim)
        denom_shape = (batch_size, self.layout.num_key_value_heads, feature_dim)
        local_shape = (batch_size, self.layout.num_key_value_heads, self.local_window, self.layout.head_dim)
        local_valid_shape = (batch_size, self.layout.num_key_value_heads, self.local_window)
        return TaylorSSDLayerCache(
            state_S=torch.zeros(state_shape, device=device, dtype=torch.float32),
            state_z=torch.zeros(denom_shape, device=device, dtype=torch.float32),
            local_k=torch.zeros(local_shape, device=device, dtype=dtype),
            local_v=torch.zeros(local_shape, device=device, dtype=dtype),
            local_valid=torch.zeros(local_valid_shape, device=device, dtype=torch.bool),
            cache_pos=0,
            seen_tokens=0,
        )

    def prepare_cache(
        self,
        cache: TaylorSSDLayerCache | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> TaylorSSDLayerCache:
        if cache is None:
            return self.init_cache(batch_size=batch_size, device=device, dtype=dtype)

        state_S = getattr(cache, "state_S", getattr(cache, "S", None))
        state_z = getattr(cache, "state_z", getattr(cache, "z", None))
        if state_S is None or state_z is None:
            raise RuntimeError("Taylor cache missing recurrent state tensors")
        if state_S.shape[0] != batch_size:
            raise RuntimeError(
                f"Taylor cache batch mismatch: cache batch={state_S.shape[0]}, input batch={batch_size}"
            )

        expected_state_shape = (batch_size, self.layout.num_key_value_heads, int(self.feature_dim), self.layout.head_dim)
        expected_denom_shape = (batch_size, self.layout.num_key_value_heads, int(self.feature_dim))
        expected_local_shape = (batch_size, self.layout.num_key_value_heads, self.local_window, self.layout.head_dim)
        expected_valid_shape = (batch_size, self.layout.num_key_value_heads, self.local_window)
        if tuple(state_S.shape) != expected_state_shape:
            raise RuntimeError(
                f"Taylor recurrent state shape mismatch: expected {expected_state_shape}, got {tuple(state_S.shape)}"
            )
        if tuple(state_z.shape) != expected_denom_shape:
            raise RuntimeError(
                f"Taylor recurrent denom shape mismatch: expected {expected_denom_shape}, got {tuple(state_z.shape)}"
            )

        local_k = getattr(cache, "local_k", None)
        local_v = getattr(cache, "local_v", None)
        local_valid = getattr(cache, "local_valid", None)
        if local_k is None or local_v is None or local_valid is None:
            local_k = torch.zeros(expected_local_shape, device=device, dtype=dtype)
            local_v = torch.zeros(expected_local_shape, device=device, dtype=dtype)
            local_valid = torch.zeros(expected_valid_shape, device=device, dtype=torch.bool)
        if tuple(local_k.shape) != expected_local_shape or tuple(local_v.shape) != expected_local_shape:
            raise RuntimeError(
                f"Taylor local cache shape mismatch: expected {expected_local_shape}, "
                f"got k={tuple(local_k.shape)} v={tuple(local_v.shape)}"
            )
        if tuple(local_valid.shape) != expected_valid_shape:
            raise RuntimeError(
                f"Taylor local-valid shape mismatch: expected {expected_valid_shape}, got {tuple(local_valid.shape)}"
            )

        return TaylorSSDLayerCache(
            state_S=state_S.to(device=device, dtype=torch.float32),
            state_z=state_z.to(device=device, dtype=torch.float32),
            local_k=local_k.to(device=device, dtype=dtype),
            local_v=local_v.to(device=device, dtype=dtype),
            local_valid=local_valid.to(device=device, dtype=torch.bool),
            cache_pos=int(getattr(cache, "cache_pos", 0)) % self.local_window,
            seen_tokens=max(int(getattr(cache, "seen_tokens", 0)), 0),
        )

    def _apply_tail_update(
        self,
        state: TaylorSSDLayerCache,
        *,
        evicted_k: torch.Tensor | None,
        evicted_v: torch.Tensor | None,
    ) -> None:
        if self._state_decay_scalar < 1.0:
            state.state_S *= self._state_decay_scalar
            state.state_z *= self._state_decay_scalar
        if evicted_k is None or evicted_v is None:
            return
        phi_k = self.phi_k(evicted_k)
        state.state_S = state.state_S + torch.einsum("bhf,bhd->bhfd", phi_k, evicted_v.float())
        state.state_z = state.state_z + phi_k

    def _tail_attention(
        self,
        q: torch.Tensor,
        state: TaylorSSDLayerCache,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phi_q = self.phi_q(q)
        q_to_kv = self.query_to_kv.to(device=q.device)
        state_S = state.state_S.index_select(1, q_to_kv)
        state_z = state.state_z.index_select(1, q_to_kv)
        numer = torch.einsum("bhf,bhfd->bhd", phi_q, state_S)
        denom = torch.einsum("bhf,bhf->bh", phi_q, state_z)
        return numer, denom

    def _local_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: TaylorSSDLayerCache,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, num_heads, head_dim = q.shape
        del num_heads
        q_to_kv = self.query_to_kv.to(device=q.device)
        history_k = state.local_k.index_select(1, q_to_kv)
        history_v = state.local_v.index_select(1, q_to_kv)
        history_valid = state.local_valid.index_select(1, q_to_kv)
        if state.seen_tokens >= self.local_window:
            history_valid = history_valid.clone()
            history_valid[..., int(state.cache_pos)] = False

        current_k = k.index_select(1, q_to_kv).unsqueeze(-2)
        current_v = v.index_select(1, q_to_kv).unsqueeze(-2)
        current_valid = torch.ones((bsz, self.layout.num_attention_heads, 1), device=q.device, dtype=torch.bool)

        local_k = torch.cat([history_k, current_k], dim=-2)
        local_v = torch.cat([history_v, current_v], dim=-2)
        local_valid = torch.cat([history_valid, current_valid], dim=-1)
        scores = torch.einsum("bhd,bhld->bhl", q.float(), local_k.float()) / math.sqrt(float(head_dim))
        masked_scores = scores.masked_fill(~local_valid, float("-inf"))
        max_score = masked_scores.amax(dim=-1, keepdim=True)
        max_score = torch.where(torch.isfinite(max_score), max_score, torch.zeros_like(max_score))
        unnormalized = torch.exp(masked_scores - max_score) * local_valid.to(dtype=torch.float32)
        numer = torch.einsum("bhl,bhld->bhd", unnormalized, local_v.float())
        denom = unnormalized.sum(dim=-1)
        tail_scale = torch.exp(-max_score.squeeze(-1))
        return numer, denom, tail_scale

    def _step_with_state(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: TaylorSSDLayerCache,
    ) -> tuple[torch.Tensor, TaylorSSDLayerCache]:
        slot = int(state.cache_pos)
        evicted_k: torch.Tensor | None = None
        evicted_v: torch.Tensor | None = None
        if state.seen_tokens >= self.local_window:
            evicted_k = state.local_k[:, :, slot, :]
            evicted_v = state.local_v[:, :, slot, :]

        self._apply_tail_update(state, evicted_k=evicted_k, evicted_v=evicted_v)
        local_numer, local_denom, tail_scale = self._local_attention(q, k, v, state)
        tail_numer, tail_denom = self._tail_attention(q, state)
        numer = local_numer + (tail_numer * tail_scale.unsqueeze(-1))
        denom = (local_denom + (tail_denom * tail_scale)).clamp_min(float(self.config.eps))
        out = numer / denom.unsqueeze(-1)

        state.local_k[:, :, slot, :] = k.to(dtype=state.local_k.dtype)
        state.local_v[:, :, slot, :] = v.to(dtype=state.local_v.dtype)
        state.local_valid[:, :, slot] = True
        state.cache_pos = int((slot + 1) % self.local_window)
        state.seen_tokens = int(state.seen_tokens + 1)

        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite hybrid softmax-linear attention output")
        if not torch.isfinite(state.state_S).all() or not torch.isfinite(state.state_z).all():
            raise RuntimeError("Non-finite hybrid softmax-linear recurrent state")
        return out.to(dtype=q.dtype), state

    def step(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cache: TaylorSSDLayerCache | None,
    ) -> tuple[torch.Tensor, TaylorSSDLayerCache]:
        state = self.prepare_cache(cache, batch_size=q.shape[0], device=q.device, dtype=q.dtype)
        return self._step_with_state(q, k, v, state)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cache: TaylorSSDLayerCache | None,
    ) -> tuple[torch.Tensor, TaylorSSDLayerCache]:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "HybridSoftmaxLinearAttention expects q/k/v shaped [B,T,H,D], "
                f"got q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}"
            )
        bsz, seq_len, _num_heads, _head_dim = q.shape
        state = self.prepare_cache(cache, batch_size=bsz, device=q.device, dtype=q.dtype)
        if seq_len <= 0:
            return q.new_zeros(q.shape), state

        outputs = []
        for step_idx in range(seq_len):
            out_t, state = self._step_with_state(q[:, step_idx, :, :], k[:, step_idx, :, :], v[:, step_idx, :, :], state)
            outputs.append(out_t)
        return torch.stack(outputs, dim=1), state


class GQATaylorSSDSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        o_proj: nn.Module,
        rotary_emb: nn.Module | None,
        layout: GQATaylorLayout,
        config: TaylorSSDConfig,
        layer_idx: int,
    ) -> None:
        super().__init__()
        layout.validate()
        config.validate()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb
        self.layout = layout
        self.config = config
        self.layer_idx = int(layer_idx)
        head_map = torch.arange(layout.num_attention_heads, dtype=torch.long) // layout.query_heads_per_group
        self.register_buffer("query_to_kv", head_map, persistent=False)
        self.backend = HybridSoftmaxLinearAttention(layout=layout, config=config)
        self._runtime_cache: TaylorSSDLayerCache | None = None

    @classmethod
    def from_llama_attention(
        cls,
        source_attn: nn.Module,
        *,
        layer_idx: int,
        order: int = 2,
        symmetric_quadratic: bool = True,
        taylor_temperature: float = 1.0,
        eps: float = 1e-6,
        feature_map: str = "hybrid_performer",
        state_decay: float = 1.0,
        local_window: int = 64,
        feature_dim: int = 64,
    ) -> GQATaylorSSDSelfAttention:
        q_out = int(getattr(getattr(source_attn, "q_proj", None), "out_features", 0))
        k_out = int(getattr(getattr(source_attn, "k_proj", None), "out_features", 0))
        configured_attention_heads = int(
            getattr(source_attn, "num_heads", getattr(source_attn, "num_attention_heads", 1))
        )
        configured_key_value_heads = int(
            getattr(
                source_attn,
                "num_key_value_heads",
                getattr(getattr(source_attn, "config", None), "num_key_value_heads", configured_attention_heads),
            )
        )
        hidden_size = int(q_out)
        configured_head_dim = int(
            getattr(
                source_attn,
                "head_dim",
                hidden_size // max(configured_attention_heads, 1),
            )
        )
        if configured_head_dim <= 0:
            configured_head_dim = 1
        if q_out > 0 and k_out > 0 and q_out % configured_head_dim == 0 and k_out % configured_head_dim == 0:
            num_attention_heads = int(q_out // configured_head_dim)
            num_key_value_heads = int(k_out // configured_head_dim)
            head_dim = int(configured_head_dim)
        else:
            num_attention_heads = int(configured_attention_heads)
            num_key_value_heads = int(configured_key_value_heads)
            head_dim = int(hidden_size // max(num_attention_heads, 1))
        layout = GQATaylorLayout(
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        config = TaylorSSDConfig(
            order=int(order),
            symmetric_quadratic=bool(symmetric_quadratic),
            taylor_temperature=float(taylor_temperature),
            eps=float(eps),
            feature_map=str(feature_map),
            state_decay=float(state_decay),
            local_window=int(local_window),
            feature_dim=int(feature_dim),
        )
        return cls(
            q_proj=source_attn.q_proj,
            k_proj=source_attn.k_proj,
            v_proj=source_attn.v_proj,
            o_proj=source_attn.o_proj,
            rotary_emb=getattr(source_attn, "rotary_emb", None),
            layout=layout,
            config=config,
            layer_idx=int(layer_idx),
        )

    def reset_runtime_cache(self) -> None:
        self._runtime_cache = None

    @property
    def feature_dim(self) -> int:
        return int(self.backend.feature_dim)

    def _phi_q(self, query: torch.Tensor) -> torch.Tensor:
        return self.backend.phi_q(query)

    def _phi_k(self, key: torch.Tensor) -> torch.Tensor:
        return self.backend.phi_k(key)

    def _resolve_rope(
        self,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        position_ids: torch.LongTensor | None,
        value_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            isinstance(position_embeddings, tuple)
            and len(position_embeddings) == 2
            and torch.is_tensor(position_embeddings[0])
            and torch.is_tensor(position_embeddings[1])
        ):
            return position_embeddings
        if self.rotary_emb is None:
            return None
        try:
            rope = self.rotary_emb(value_states.transpose(1, 2), position_ids)
            if isinstance(rope, tuple) and len(rope) == 2:
                return rope
        except Exception:
            pass
        try:
            rope = self.rotary_emb(value_states.transpose(1, 2), seq_len=value_states.shape[1])
            if isinstance(rope, tuple) and len(rope) == 2:
                return rope
        except Exception:
            pass
        return None

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        position_ids: torch.LongTensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError(f"Expected hidden_states [B,T,H], got {tuple(hidden_states.shape)}")
        bsz, seq_len, _hidden = hidden_states.shape
        q_states = self.q_proj(hidden_states).view(bsz, seq_len, self.layout.num_attention_heads, self.layout.head_dim)
        k_states = self.k_proj(hidden_states).view(bsz, seq_len, self.layout.num_key_value_heads, self.layout.head_dim)
        v_states = self.v_proj(hidden_states).view(bsz, seq_len, self.layout.num_key_value_heads, self.layout.head_dim)
        rope_pair = self._resolve_rope(
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            value_states=v_states,
        )
        if rope_pair is None:
            return q_states, k_states, v_states

        cos_raw, sin_raw = rope_pair
        cos = _normalize_rope_tensor(
            cos_raw,
            batch_size=bsz,
            seq_len=seq_len,
            head_dim=self.layout.head_dim,
            device=q_states.device,
            dtype=q_states.dtype,
        )
        sin = _normalize_rope_tensor(
            sin_raw,
            batch_size=bsz,
            seq_len=seq_len,
            head_dim=self.layout.head_dim,
            device=q_states.device,
            dtype=q_states.dtype,
        )
        return _apply_rotary_pos_emb(q_states, k_states, cos=cos, sin=sin) + (v_states,)

    def _recurrent_attention(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        position_ids: torch.LongTensor | None,
        cache: TaylorSSDLayerCache | None,
    ) -> tuple[torch.Tensor, TaylorSSDLayerCache]:
        q_states, k_states, v_states = self._project_qkv(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
        context, state = self.backend.forward(q_states, k_states, v_states, cache=cache)
        attn_out = self.o_proj(context.reshape(hidden_states.shape[0], hidden_states.shape[1], self.layout.hidden_size))
        if not torch.isfinite(attn_out).all():
            raise RuntimeError(f"Non-finite Taylor attention output at layer {self.layer_idx}")
        return attn_out, state

    def _decode_step(
        self,
        hidden_states: torch.Tensor,
        *,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
        position_ids: torch.LongTensor | None,
        cache: TaylorSSDLayerCache | None,
    ) -> tuple[torch.Tensor, TaylorSSDLayerCache]:
        q_states, k_states, v_states = self._project_qkv(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
        )
        out_step, state = self.backend.step(
            q_states[:, 0, :, :],
            k_states[:, 0, :, :],
            v_states[:, 0, :, :],
            cache=cache,
        )
        attn_out = self.o_proj(out_step.unsqueeze(1).reshape(hidden_states.shape[0], 1, self.layout.hidden_size))
        if not torch.isfinite(attn_out).all():
            raise RuntimeError(f"Non-finite Taylor decode output at layer {self.layer_idx}")
        return attn_out, state

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> Any:
        del attention_mask, cache_position
        if past_key_value is None and "past_key_values" in kwargs:
            past_key_value = kwargs.pop("past_key_values")
        explicit_cache = past_key_value if isinstance(past_key_value, TaylorSSDLayerCache) else None
        model_managed_cache = past_key_value is not None and explicit_cache is None
        cache_enabled = bool(use_cache or past_key_value is not None)

        if not cache_enabled:
            self._runtime_cache = None
            attn_out, _ = self._recurrent_attention(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=None,
            )
            if output_attentions:
                return attn_out, None
            return attn_out, None

        cache_seed = explicit_cache if explicit_cache is not None else self._runtime_cache

        if hidden_states.shape[1] == 1:
            attn_out, new_cache = self._decode_step(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=cache_seed,
            )
        else:
            attn_out, new_cache = self._recurrent_attention(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache=cache_seed,
            )

        self._runtime_cache = new_cache
        if model_managed_cache:
            if output_attentions:
                return attn_out, None
            return attn_out, None

        present: Any = new_cache
        if output_attentions:
            return attn_out, None, present
        return attn_out, None, present
