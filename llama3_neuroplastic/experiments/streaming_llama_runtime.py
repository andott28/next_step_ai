from __future__ import annotations

import concurrent.futures
import copy
import os
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import bitsandbytes.functional as bnb_functional
import torch
import torch.nn as nn
import torch.nn.functional as F
from bitsandbytes.backends.cuda.ops import _dequantize_4bit_impl as _bnb_dequant_impl
from transformers import AutoConfig

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None

from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

try:
    from ..token_posting_archive import TokenPostingArchive
except ImportError:
    try:
        from llama3_neuroplastic.token_posting_archive import TokenPostingArchive
    except ImportError:
        TokenPostingArchive = None

try:
    from ..gqa_taylor_ssd import GQATaylorSSDSelfAttention, TaylorSSDLayerCache
except ImportError:
    from gqa_taylor_ssd import GQATaylorSSDSelfAttention, TaylorSSDLayerCache

try:
    from ..triton_sparse_mlp import (
        triton_fused_sparse_mlp_decode_4bit,
        triton_sparse_mlp_decode_4bit_single_kernel_sm75,
        triton_sparse_input_linear_4bit,
        triton_sparse_mlp_available,
        triton_sparse_output_linear_4bit,
    )
except Exception:
    try:
        from triton_sparse_mlp import (
            triton_fused_sparse_mlp_decode_4bit,
            triton_sparse_mlp_decode_4bit_single_kernel_sm75,
            triton_sparse_input_linear_4bit,
            triton_sparse_mlp_available,
            triton_sparse_output_linear_4bit,
        )
    except Exception:
        triton_fused_sparse_mlp_decode_4bit = None
        triton_sparse_mlp_decode_4bit_single_kernel_sm75 = None
        triton_sparse_input_linear_4bit = None
        triton_sparse_output_linear_4bit = None
        import warnings as _warnings
        _warnings.warn(
            "[StreamingLlamaRuntime] Triton sparse MLP kernels are unavailable on this platform. "
            "All sparse MLP layers will fall back to the PyTorch path, which is 3-5x slower. "
            "On Windows, ensure triton-windows is installed and CUDA compute capability >= 7.5 (Turing+).",
            RuntimeWarning,
            stacklevel=1,
        )

        def triton_sparse_mlp_available() -> bool:
            return False



import contextlib

from .runtime._helpers import (
    _DEFAULT_SPARSE_BASIS_TOP_K,
    _configure_llama_mlp_shape_only,
    _is_cuda_oom_error,
    _is_windows_pre_ampere_cuda,
    _normalize_sparse_mlp_execution,
    _resolve_background_prefetch_default,
    _resolve_gpu_lm_head_default,
    _resolve_show_progress_default,
    _resolve_snapshot_dir,
    _resolve_windows_batch_preload_default,
    _unpermute_headwise_tensor,
    _unpermute_o_factor_cols,
    _unpermute_q_factor_rows,
)
from .runtime.block_bank import (
    MlpBlockBankLayout,
    build_intermediate_mlp_block_bank_layout,
    validate_intermediate_mlp_block_bank_params,
)
from .runtime.lm_head import RuntimeLmHeadMixin
from .runtime.safetensor_loader import ShardedSafetensorLoader
from .runtime.session import RuntimeSessionMixin

_EXACT_BLOCKWISE_SPARSE = "exact_blockwise_sparse"


def _parse_layer_index_set(raw_value: str | None, total_layers: int, *, name: str) -> set[int]:
    raw = str(raw_value or "").strip().lower()
    if not raw or raw in {"none", "off", "false", "0"}:
        return set()
    if raw == "all":
        return set(range(max(0, int(total_layers))))

    selected: set[int] = set()
    for piece in raw.replace(";", ",").split(","):
        token = piece.strip()
        if not token:
            continue
        if "-" in token:
            bounds = token.split("-", 1)
            if len(bounds) != 2 or not bounds[0].strip() or not bounds[1].strip():
                raise RuntimeError(f"{name} contains invalid layer range {piece!r}")
            start = int(bounds[0].strip())
            end = int(bounds[1].strip())
            if end < start:
                start, end = end, start
            for layer_idx in range(start, end + 1):
                if 0 <= layer_idx < int(total_layers):
                    selected.add(int(layer_idx))
            continue
        layer_idx = int(token)
        if 0 <= layer_idx < int(total_layers):
            selected.add(int(layer_idx))
    return selected


class StreamingLlamaRuntime(RuntimeSessionMixin, RuntimeLmHeadMixin):
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float16,
        taylor_layers: list[int] | None = None,
        taylor_feature_map: str = "hybrid_performer",
        taylor_local_window: int = 64,
        taylor_feature_dim: int = 64,
        taylor_state_decay: float = 1.0,
        local_files_only: bool = True,
        ram_cache: bool = True,
        ram_cache_pinned: bool | None = None,
        materialize_lm_head: bool = True,
        sparse_basis_path: str | None = None,
        sparse_top_k: int | None = None,
        sparse_basis_top_k: int | None = None,
        sparse_mlp_execution: str | None = None,
        sparse_mlp_prefill_mode: str | None = None,
        sparse_mlp_prefill_top_k: int | None = None,
        vram_hot_cache_gb: float | None = None,
        hot_block_threshold: float = 0.80,
        attn_head_importance_path: str | None = None,
        mlp_skip_mask_path: str | None = None,
        attn_share_path: str | None = None,
        attn_active_heads: int | None = None,
        attn_head_activity_threshold: float = 0.10,
        attn_min_active_heads: int = 16,
        attn_max_active_heads: int | None = None,
        sparse_attn_prefill_mode: str | None = None,
        sparse_kv_prefill_mode: str | None = None,
        attn_share_prefill_mode: str | None = None,
        enable_triton_fused_sparse_mlp: bool = True,
        enable_cuda_h2d_overlap: bool = True,
        kv_basis_path: str | None = None,
        kv_sparse_top_k: int | None = None,

        attn_token_posting_path: str | None = None,
        attn_retrieval_ring_size: int = 256,
        attn_retrieval_num_sinks: int = 16,
        attn_retrieval_candidates: int = 64,
        attn_retrieval_r_query: int = 6,
        attn_retrieval_token_topk: int = 8,
        attn_retrieval_archive_capacity: int = 16384,
        hard_cuda_cache_flush: bool = False,
    ) -> None:
        self.snapshot_dir = _resolve_snapshot_dir(model_name_or_path, local_files_only=bool(local_files_only))
        self.config = AutoConfig.from_pretrained(str(self.snapshot_dir), local_files_only=bool(local_files_only))
        if str(getattr(self.config, "model_type", "")) != "llama":
            raise RuntimeError(f"Streaming runtime only supports llama models, got {self.config.model_type!r}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self._hard_cuda_cache_flush = bool(hard_cuda_cache_flush)
        self._runtime_events: list[dict[str, Any]] = []
        self._runtime_events_max = 128





        if getattr(self.config, "_attn_implementation", None) is None:
            self.config._attn_implementation = "sdpa"
        self._debug_steps = os.getenv("STREAMING_DEBUG_STEPS", "").strip().lower() in {"1", "true", "yes", "on"}
        self._debug_sync_cuda = os.getenv("STREAMING_DEBUG_SYNC_CUDA", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._debug_attn_row_cache = os.getenv("STREAMING_DEBUG_ATTN_ROW_CACHE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._sparse_basis_bias_mode = os.getenv("STREAMING_SPARSE_BIAS_MODE", "selected").strip().lower() or "selected"
        if self._sparse_basis_bias_mode not in {"selected", "none"}:
            raise RuntimeError(
                "STREAMING_SPARSE_BIAS_MODE must be one of: selected, none"
            )
        _sparse_mlp_execution_raw = (
            sparse_mlp_execution
            if sparse_mlp_execution is not None
            else os.getenv("STREAMING_SPARSE_BASIS_EXECUTION", "auto")
        )
        self._sparse_mlp_execution_request = _normalize_sparse_mlp_execution(_sparse_mlp_execution_raw)


        self._sparse_basis_execution = self._sparse_mlp_execution_request
        _sparse_mlp_prefill_mode_raw = (
            sparse_mlp_prefill_mode
            if sparse_mlp_prefill_mode is not None
            else os.getenv("STREAMING_SPARSE_MLP_PREFILL_MODE", "dense")
        )
        self._sparse_mlp_prefill_mode = str(_sparse_mlp_prefill_mode_raw).strip().lower() or "dense"
        if self._sparse_mlp_prefill_mode not in {"dense", "sparse", "hot_cache"}:
            raise RuntimeError("Sparse MLP prefill mode must be one of: dense, sparse, hot_cache")
        self._sparse_mlp_prefill_top_k = (
            int(sparse_mlp_prefill_top_k)
            if sparse_mlp_prefill_top_k is not None and int(sparse_mlp_prefill_top_k) > 0
            else None
        )
        _sparse_attn_prefill_mode_raw = (
            sparse_attn_prefill_mode
            if sparse_attn_prefill_mode is not None
            else os.getenv("STREAMING_SPARSE_ATTN_PREFILL_MODE", "dense")
        )
        self._sparse_attn_prefill_mode = str(_sparse_attn_prefill_mode_raw).strip().lower() or "dense"
        if self._sparse_attn_prefill_mode not in {"dense", "sparse"}:
            raise RuntimeError(
                "STREAMING_SPARSE_ATTN_PREFILL_MODE must be one of: dense, sparse"
            )
        _sparse_kv_prefill_mode_raw = (
            sparse_kv_prefill_mode
            if sparse_kv_prefill_mode is not None
            else os.getenv("STREAMING_SPARSE_KV_PREFILL_MODE", "dense")
        )
        self._sparse_kv_prefill_mode = str(_sparse_kv_prefill_mode_raw).strip().lower() or "dense"
        if self._sparse_kv_prefill_mode not in {"dense", "sparse"}:
            raise RuntimeError(
                "STREAMING_SPARSE_KV_PREFILL_MODE must be one of: dense, sparse"
            )
        _attn_head_selection_mode_raw = os.getenv("STREAMING_SPARSE_ATTN_HEAD_SELECTION", "balanced_gqa")
        self._attn_head_selection_mode = str(_attn_head_selection_mode_raw).strip().lower() or "topk"
        _attn_head_selection_aliases = {
            "topk": "topk",
            "top_k": "topk",
            "balanced_gqa": "balanced_gqa",
            "gqa_balanced": "balanced_gqa",
            "per_gqa": "balanced_gqa",
        }
        if self._attn_head_selection_mode not in _attn_head_selection_aliases:
            raise RuntimeError(
                "STREAMING_SPARSE_ATTN_HEAD_SELECTION must be one of: topk, balanced_gqa"
            )
        self._attn_head_selection_mode = _attn_head_selection_aliases[self._attn_head_selection_mode]
        _attn_share_prefill_mode_raw = (
            attn_share_prefill_mode
            if attn_share_prefill_mode is not None
            else os.getenv("STREAMING_ATTN_SHARE_PREFILL_MODE", "dense")
        )
        self._attn_share_prefill_mode = str(_attn_share_prefill_mode_raw).strip().lower() or "dense"
        if self._attn_share_prefill_mode not in {"dense", "shared"}:
            raise RuntimeError(
                "STREAMING_ATTN_SHARE_PREFILL_MODE must be one of: dense, shared"
            )
        _guard_chunk_blocks_raw = os.getenv("STREAMING_GUARD_MLP_CHUNK_BLOCKS", "").strip()
        self._guard_mlp_chunk_blocks = int(_guard_chunk_blocks_raw) if _guard_chunk_blocks_raw else 64
        self._guard_mlp_chunk_blocks = max(1, int(self._guard_mlp_chunk_blocks))
        self._show_progress = _resolve_show_progress_default()
        self._progress_line_width = 140
        self._prefer_gpu_lm_head = _resolve_gpu_lm_head_default()
        _gpu_lm_head_raw = os.getenv("STREAMING_GPU_LM_HEAD", "").strip().lower()
        self._explicit_gpu_lm_head = _gpu_lm_head_raw in {"1", "true", "yes", "on", "force"}
        _gpu_lm_head_dense_raw = os.getenv("STREAMING_GPU_LM_HEAD_DENSE", "").strip().lower()
        self._explicit_dense_gpu_lm_head = _gpu_lm_head_dense_raw in {"1", "true", "yes", "on", "force"}
        self._allow_dense_gpu_lm_head = True
        if _is_windows_pre_ampere_cuda(self.device) and not self._explicit_dense_gpu_lm_head:
            self._allow_dense_gpu_lm_head = False
        self.loader = ShardedSafetensorLoader(self.snapshot_dir, pin_ram_cache=ram_cache_pinned)
        self.loader._ram_cache_enabled = bool(ram_cache)
        self._enable_background_prefetch = bool(ram_cache) and _resolve_background_prefetch_default()
        self._enable_windows_batch_preload = bool(
            os.name == "nt"
            and ram_cache
            and _resolve_windows_batch_preload_default()
        )
        self._enable_cuda_h2d_overlap = bool(enable_cuda_h2d_overlap and torch.cuda.is_available())
        self._h2d_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device) if self._enable_cuda_h2d_overlap else None
        )
        self._lm_head_gpu_attempted = False
        self._lm_head_nf4_meta_gpu: dict[str, Any] | None = None
        self._prefer_gpu_quant_lm_head = self.device.type == "cuda"
        self._compact_sparse_attn_decode = self.device.type == "cuda"
        self._compact_attn_cache: dict[int, dict[str, torch.Tensor | None]] = {}
        self._compact_attn_decode_steps = 0
        self.num_layers = int(getattr(self.config, "num_hidden_layers", 0))
        if self.num_layers <= 0:
            raise RuntimeError("Invalid llama config: num_hidden_layers must be > 0")
        self._attn_force_dense_layers = _parse_layer_index_set(
            os.getenv("STREAMING_SPARSE_ATTN_DENSE_LAYERS", ""),
            self.num_layers,
            name="STREAMING_SPARSE_ATTN_DENSE_LAYERS",
        )

        self._allow_taylor_with_sparse_attn = os.getenv(
            "STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._taylor_requested_layers = (
            set(range(self.num_layers))
            if taylor_layers is None
            else {int(idx) for idx in taylor_layers if 0 <= int(idx) < self.num_layers}
        )
        self._taylor_auto_disabled_for_sparse_attn = bool(
            attn_head_importance_path
            and str(attn_head_importance_path).strip()
            and int(len(self._taylor_requested_layers)) > 0
            and not self._allow_taylor_with_sparse_attn
        )
        self.taylor_layer_set = (
            set()
            if self._taylor_auto_disabled_for_sparse_attn
            else set(self._taylor_requested_layers)
        )
        self.taylor_feature_map = str(taylor_feature_map)
        self.taylor_local_window = int(taylor_local_window)
        self.taylor_feature_dim = int(taylor_feature_dim)
        self.taylor_state_decay = float(taylor_state_decay)
        self._triton_sparse_mlp_requested = bool(enable_triton_fused_sparse_mlp)
        self._triton_fused_sparse_mlp = bool(
            enable_triton_fused_sparse_mlp
            and triton_sparse_mlp_available()
            and torch.cuda.is_available()
            and self.device.type == "cuda"
        )
        _disable_single_kernel = os.getenv("SCA_TRITON_DISABLE_SINGLE_KERNEL_MLP", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._single_kernel_decode_enabled = bool(self._triton_fused_sparse_mlp) and not _disable_single_kernel
        self._single_kernel_mlp_out_accum: torch.Tensor | None = None
        self._single_kernel_mlp_block_out: int = 64
        self._decode_backend_name: str = (
            "fused_sparse_decode_v1" if bool(self._triton_fused_sparse_mlp) else "dequant_sparse_fallback"
        )
        if self._triton_fused_sparse_mlp and _is_windows_pre_ampere_cuda(self.device):
            print(
                "[sparse] Triton fused 4-bit sparse MLP enabled on Windows pre-Ampere.",
                flush=True,
            )
        if vram_hot_cache_gb is None:
            raw_hot_gb = os.getenv("STREAMING_VRAM_HOT_CACHE_GB", "").strip()
            if raw_hot_gb:
                try:
                    vram_hot_cache_gb = float(raw_hot_gb)
                except ValueError:
                    vram_hot_cache_gb = None
        self._vram_hot_cache_limit_bytes: int | None = (
            int(float(vram_hot_cache_gb) * (1024 ** 3))
            if vram_hot_cache_gb is not None and float(vram_hot_cache_gb) > 0.0
            else None
        )
        self._vram_hot_cache_enabled: bool = self._vram_hot_cache_limit_bytes is not None
        self._vram_hot_cache_used_bytes: int = 0
        self._vram_nf4_cache: dict[str, dict[str, Any]] = {}
        self._vram_hot_cache_pressure_warned: bool = False
        self._vram_hot_cache_oom_warned: bool = False
        self._vram_hot_cache_disable_reason: str | None = None
        self._vram_hot_cache_margin_bytes: int = int(1.0 * (1024 ** 3))
        if self.device.type == "cuda":
            try:
                _, _vram_total = torch.cuda.mem_get_info(self.device)

                self._vram_hot_cache_margin_bytes = max(
                    self._vram_hot_cache_margin_bytes,
                    int(float(_vram_total) * 0.05),
                )
            except Exception:
                pass
        self._hot_block_threshold = float(max(0.0, min(1.0, hot_block_threshold)))
        self._mlp_hot_blocks_by_layer: dict[int, torch.Tensor] = {}
        self._hot_cache_calibration_active: bool = False
        self._hot_cache_calibration_hits: dict[int, torch.Tensor] = {}
        self._hot_cache_calibration_recency_power: float = float(
            max(1.0, float(os.getenv("STREAMING_HOT_CACHE_RECENCY_POWER", "2.0") or "2.0"))
        )
        self._vram_hot_cache_live_calibrated: bool = False
        self._last_hot_cache_calibration: dict[str, Any] | None = None
        self._traffic_current_phase: str = "idle"
        self._traffic_bytes_by_phase: dict[str, int] = defaultdict(int)
        self._traffic_layer_visits_by_phase: dict[str, int] = defaultdict(int)
        self._traffic_bytes_by_phase_layer: dict[tuple[str, int], int] = defaultdict(int)
        self._traffic_layer_visits_by_phase_layer: dict[tuple[str, int], int] = defaultdict(int)
        self._traffic_bytes_by_phase_tag: dict[tuple[str, str], int] = defaultdict(int)
        self._last_traffic_report: dict[str, Any] | None = None
        self._first_decode_t: float | None = None
        self._decode_done_t: float | None = None
        self._decode_profile_enabled: bool = False
        self._decode_profile_max_steps: int = 0
        self._decode_profile_steps: list[dict[str, Any]] = []
        self._last_decode_profile_report: dict[str, Any] | None = None
        self._session_token_ids_cpu: torch.LongTensor | None = None
        self._session_last_logits_cpu: torch.Tensor | None = None
        self._h2d_stage_slots: dict[str, int] = defaultdict(int)
        self._sparse_block_transfer_cache: dict[tuple[str, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = {}
        self._downproj_transfer_cache: dict[tuple[str, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]] = {}
        # CPU-side column cache for down_proj cold blocks.  Keyed by (full_name, block_id).
        # Stores (packed_cols_cpu, absmax_cols_cpu) extracted from the weight, so SSD-miss layers
        # only pay the full-weight SSD read ONCE; subsequent tokens hit this cheap CPU buffer.
        self._down_proj_col_cache: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        # Phase 4: pre-routed MLP block indices (keyed by layer_idx) for H2D prefetch overlap
        self._mlp_prefetch_active_blocks: dict[int, torch.Tensor] = {}
        _target_layer_mb_raw = os.getenv("STREAMING_TARGET_LAYER_MB", "").strip()
        try:
            self._target_layer_traffic_mb = float(_target_layer_mb_raw) if _target_layer_mb_raw else 30.0
        except ValueError:
            self._target_layer_traffic_mb = 30.0

        self._embed_weight_name = "model.embed_tokens.weight"
        self._embed_weight_cpu: torch.Tensor | None = None
        self._embed_row_cache: dict[int, torch.Tensor] = {}
        self._embed_row_cache_lock = threading.Lock()
        self._materialize_lm_head = bool(materialize_lm_head)

        self.norm = LlamaRMSNorm(int(self.config.hidden_size), eps=float(self.config.rms_norm_eps)).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.norm.weight.data.copy_(
            self.loader.load_parameter("model.norm.weight").to(device=self.device, dtype=self.dtype)
        )
        self.norm.requires_grad_(False)


        self._lm_head_weight_name = (
            "lm_head.weight"
            if "lm_head.weight" in self.loader.weight_map
            else self._embed_weight_name
        )
        self._lm_head_weight_cpu: torch.Tensor | None = None
        self._lm_head_weight_gpu: torch.Tensor | None = None
        self._lm_head_gpu_attempted = False
        self._lm_head_gpu_last_failure: str | None = None

        self.rotary_emb = LlamaRotaryEmbedding(self.config, device=self.device)
        self._taylor_caches: list[TaylorSSDLayerCache | None] = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None


        self._token_archive: TokenPostingArchive | None = None
        self._retrieval_layers: set[int] = set()
        self._retrieval_candidates: int = int(attn_retrieval_candidates)
        if attn_token_posting_path is not None and TokenPostingArchive is not None:
            _tp_path = Path(attn_token_posting_path)
            if _tp_path.exists():
                _tp_ckpt = torch.load(str(_tp_path), map_location="cpu", weights_only=False)
                _tp_cfg = _tp_ckpt.get("config", {})
                _tp_layers = list(_tp_cfg.get("retrieval_layers", []))
                _tp_rank = int(_tp_cfg.get("basis_rank", 32))
                _tp_G = int(_tp_cfg.get("num_kv_groups", getattr(self.config, "num_key_value_heads", 8)))
                _tp_D = int(_tp_cfg.get("head_dim", getattr(self.config, "head_dim", 128)))
                self._retrieval_layers = set(_tp_layers)
                self._token_archive = TokenPostingArchive(
                    retrieval_layers=_tp_layers,
                    num_kv_groups=_tp_G,
                    head_dim=_tp_D,
                    basis_rank=_tp_rank,
                    ring_size=int(attn_retrieval_ring_size),
                    num_sinks=int(attn_retrieval_num_sinks),
                    archive_capacity=int(attn_retrieval_archive_capacity),
                    token_topk=int(attn_retrieval_token_topk),
                    r_query=int(attn_retrieval_r_query),
                    candidates=int(attn_retrieval_candidates),
                    device=self.device,
                    dtype=self.dtype,
                )

                for _ls_key, _ls_val in _tp_ckpt.get("layer_states", {}).items():
                    _ls_idx = int(_ls_key)
                    _grp_bases = _ls_val.get("group_bases", [])
                    _idf_weights = _ls_val.get("idf_weights", [])
                    _key_means = _ls_val.get("key_means", [])
                    import numpy as _np
                    for _g_idx in range(len(_grp_bases)):
                        self._token_archive.load_basis(
                            _ls_idx,
                            _g_idx,
                            _np.asarray(_grp_bases[_g_idx], dtype=_np.float32),
                            _np.asarray(_idf_weights[_g_idx], dtype=_np.float32),
                            _np.asarray(_key_means[_g_idx], dtype=_np.float32)
                            if _g_idx < len(_key_means) else None,
                        )
                print(
                    f"[token_posting] loaded basis for {len(_tp_layers)} retrieval layers "
                    f"| rank={_tp_rank} G={_tp_G} D={_tp_D}",
                    flush=True,
                )
            else:
                print(f"[token_posting] path not found: {_tp_path}", flush=True)
        skeleton_config = copy.deepcopy(self.config)
        skeleton_config.intermediate_size = 1
        self._layer_skeleton = LlamaDecoderLayer(skeleton_config, layer_idx=0).to(device=self.device, dtype=self.dtype)
        _configure_llama_mlp_shape_only(
            self._layer_skeleton.mlp,
            hidden_size=int(self.config.hidden_size),
            intermediate_size=int(self.config.intermediate_size),
            bias=bool(getattr(self.config, "mlp_bias", False)),
            dtype=self.dtype,
        )
        for p in self._layer_skeleton.parameters():
            p.requires_grad = False
        self._layer_skeleton.eval()



        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._shared_taylor_attn = None
        if self.taylor_layer_set:
            self._shared_taylor_attn = GQATaylorSSDSelfAttention.from_llama_attention(
                source_attn=self._layer_skeleton.self_attn,
                layer_idx=0,
                feature_map=self.taylor_feature_map,
                local_window=self.taylor_local_window,
                feature_dim=self.taylor_feature_dim,
                state_decay=self.taylor_state_decay,
            ).to(device=self.device)
            self._shared_taylor_attn.eval()
            for p in self._shared_taylor_attn.parameters():
                p.requires_grad = False
        elif self._taylor_auto_disabled_for_sparse_attn:
            import warnings as _tw
            _tw.warn(
                "[taylor] Taylor-SSD attention was requested but is INCOMPATIBLE with sparse "
                "head-importance attention and has been automatically disabled.\n"
                "Why: Taylor-SSD maintains a per-layer recurrent state (S matrix + z normalization) "
                "across tokens. Sparse head routing changes which heads are active each token, "
                "breaking the recurrent state invariant and causing silent numeric divergence.\n"
                "To enable anyway (experimental, may produce garbage outputs): "
                "set STREAMING_ALLOW_TAYLOR_WITH_SPARSE_ATTN=1.\n"
                "To silence this warning: pass taylor_layers=[] to disable Taylor-SSD explicitly.",
                RuntimeWarning,
                stacklevel=2,
            )



        self._prefetch_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="layer_prefetch")
            if self._enable_background_prefetch
            else None
        )
        self._prefetch_lock = threading.Lock()
        self._prefetch_pending_layers: set[int] = set()

        _layer_state = self._layer_skeleton.state_dict(keep_vars=True)
        self._layer_state_items: list[tuple[str, torch.Tensor, bool]] = [
            (str(k), v, bool(v.is_floating_point()))
            for k, v in _layer_state.items()
        ]
        self._layer_param_keys: list[str] = [
            k for k, _v, _is_fp in self._layer_state_items if bool(_is_fp)
        ]
        if self._prefetch_executor is not None and self._layer_param_keys:
            self._schedule_prefetch_layer(0)







        _h = int(self.config.hidden_size)



        _max_nf4_bytes = _h * _h // 2


        _max_absmax_numel = _max_nf4_bytes // 32
        if torch.cuda.is_available():
            self._nf4_staging: torch.Tensor | None = torch.empty(
                _max_nf4_bytes, dtype=torch.uint8, device=self.device
            )




            self._absmax_staging: torch.Tensor | None = torch.empty(
                _max_absmax_numel, dtype=torch.float32, device=self.device
            )







            self._nested_absmax_staging: torch.Tensor | None = torch.empty(
                _max_absmax_numel, dtype=torch.uint8, device=self.device
            )
            _max_s2_absmax = max(_max_absmax_numel // 64, 1024)
            self._state2_absmax_staging: torch.Tensor | None = torch.empty(
                _max_s2_absmax, dtype=torch.float32, device=self.device
            )
            self._code_staging: torch.Tensor | None = torch.empty(
                256, dtype=torch.float32, device=self.device
            )
        else:
            self._nf4_staging = None
            self._absmax_staging = None
            self._nested_absmax_staging = None
            self._state2_absmax_staging = None
            self._code_staging = None



        _ffn = int(getattr(self.config, "intermediate_size", _h * 4))
        self._mlp_proj_staging: torch.Tensor | None = None
        self._mlp_proj_staging_numel: int = int(_ffn * _h)
        self._dense_mlp_staging_warned: bool = False



        _skip_mlp_staging = bool(sparse_basis_path and str(sparse_basis_path).strip())
        if torch.cuda.is_available() and not _skip_mlp_staging:
            try:
                self._mlp_proj_staging = torch.empty(
                    self._mlp_proj_staging_numel, dtype=self.dtype, device=self.device
                )
            except Exception as _e:
                if "out of memory" in str(_e).lower() or "alloc" in str(_e).lower():
                    import warnings
                    warnings.warn(
                        f"[StreamingLlamaRuntime] Insufficient VRAM for MLP staging "
                        f"({self._mlp_proj_staging_numel * 2 // 1024 // 1024} MB); "
                        "dense calibration paths may degrade to zero passthrough.", stacklevel=2
                    )
                    self._dense_mlp_staging_warned = True
                else:
                    raise


















        self._mlp_static_skip_mask: torch.Tensor | None = None
        self._mlp_static_skip_set: set[int] = set()
        _hidden = int(getattr(self.config, "hidden_size", 16384))
        _smc_raw = os.getenv("STREAMING_ENABLE_SMC", "").strip().lower()
        self._enable_smc = _smc_raw in {"1", "true", "yes", "on"}
        self._smc_x_in = torch.zeros(self.num_layers, _hidden, device=self.device, dtype=self.dtype)
        self._smc_out = torch.zeros(self.num_layers, _hidden, device=self.device, dtype=self.dtype)
        self._smc_valid = torch.zeros(self.num_layers, dtype=torch.bool, device=self.device)
        self._smc_threshold = float(os.getenv("STREAMING_SMC_THRESHOLD", "0.04"))
        # SMC for attention: separate state (attention is more sensitive than MLP)
        self._smc_attn_x_in = torch.zeros(self.num_layers, _hidden, device=self.device, dtype=self.dtype)
        self._smc_attn_out = torch.zeros(self.num_layers, _hidden, device=self.device, dtype=self.dtype)
        self._smc_attn_valid = torch.zeros(self.num_layers, dtype=torch.bool, device=self.device)
        self._smc_attn_threshold = float(os.getenv("STREAMING_SMC_ATTN_THRESHOLD", "0.03"))
        # Boundary layers that are never skipped (first 8 and last 8 of 126)
        _n = int(self.num_layers)
        self._smc_attn_protected: frozenset[int] = frozenset({*range(min(8, _n)), *range(max(0, _n - 8), _n)})
        self._sparse_routing: dict[int, dict[str, Any]] = {}
        self._sparse_top_k: int = 0
        self._sparse_runtime_top_k: int = 0
        self._sparse_block_size: int = 32
        self._sparse_num_blocks: int = 0
        self._sparse_top_k_by_layer: dict[int, int] = {}
        self._sparse_basis_top_k_by_layer: dict[int, int] = {}
        self._sparse_semantic_block_score_normalized: bool = False
        self._sparse_param_cache: dict[str, dict[str, Any]] = {}
        self._sparse_explicit_layer_selection: set[int] = set()
        self._sparse_checkpoint_basis_rank: int = 64
        self._upper_decode_guard_layers: set[int] = set()
        self._session_sparse_route_layers: set[int] = set()

        if sparse_basis_path and str(sparse_basis_path).strip():
            _payload = torch.load(str(sparse_basis_path), map_location="cpu", weights_only=False)
            _cfg = _payload.get("config", {})
            _raw_selection = _payload.get("layer_selection", [])
            if isinstance(_raw_selection, (list, tuple)):
                self._sparse_explicit_layer_selection = {
                    int(_idx) for _idx in _raw_selection if 0 <= int(_idx) < self.num_layers
                }
            self._sparse_block_size = int(_cfg.get("block_size", 32))
            _artifact_target_default = str(_cfg.get("artifact_target", "output_reconstruction")).strip().lower()
            _block_domain_default = str(
                _cfg.get(
                    "block_domain",
                    "intermediate" if _artifact_target_default == "intermediate_block_scores" else "output",
                )
            ).strip().lower()
            _default_num_blocks = (
                int(self.config.hidden_size) // max(int(self._sparse_block_size), 1)
                if _block_domain_default == "output"
                else int(getattr(self.config, "intermediate_size", 0)) // max(int(self._sparse_block_size), 1)
            )
            self._sparse_num_blocks = int(_cfg.get("num_blocks", _default_num_blocks))
            self._sparse_checkpoint_basis_rank = int(_cfg.get("basis_rank", self._sparse_checkpoint_basis_rank))

            _num_blocks = max(1, int(self._sparse_num_blocks))
            _default_top_k = max(1, int(round(_num_blocks * 0.02)))
            self._sparse_top_k = int(sparse_top_k) if sparse_top_k is not None else _default_top_k
            self._sparse_runtime_top_k = int(self._sparse_top_k)
            _layer_states = _payload.get("layer_states", {})
            _stats = _payload.get("stats", {})
            _block_top_k_by_layer = _cfg.get("top_k_by_layer", {}) or {}
            _basis_top_k_by_layer = _cfg.get("basis_top_k_by_layer", {}) or {}
            _default_basis_top_k = (
                int(sparse_basis_top_k)
                if sparse_basis_top_k is not None
                else int(_cfg.get("basis_top_k", _DEFAULT_SPARSE_BASIS_TOP_K))
            )
            self._sparse_semantic_block_score_normalized = bool(
                _cfg.get("semantic_block_score_normalized", False)
            )
            _block_importance_by_layer = {}
            if isinstance(_stats, dict):
                _bib = _stats.get("block_importance_by_layer")
                if isinstance(_bib, dict):
                    _block_importance_by_layer = _bib
            _build_mlp_hot_block_map = bool(
                getattr(self, "_vram_hot_cache_enabled", False)
                or str(getattr(self, "_sparse_mlp_prefill_mode", "dense")) == "hot_cache"
            )
            _basis_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            _evrs: list[float] = []
            _low_evr_count = 0
            _MIN_EXPLAINED_VARIANCE = 0.30
            for _lidx_s, _state in _layer_states.items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                _layer_stats = _stats.get(str(_lidx), {}) if isinstance(_stats, dict) else {}
                _artifact_target = str(
                    _state.get("artifact_target", _cfg.get("artifact_target", _artifact_target_default))
                ).strip().lower()
                _block_domain = str(
                    _state.get(
                        "block_domain",
                        _cfg.get(
                            "block_domain",
                            "intermediate" if _artifact_target == "intermediate_block_scores" else _block_domain_default,
                        ),
                    )
                ).strip().lower()
                if _block_domain == "intermediate":
                    _enc_w = _state["encoder_weight"].to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous()
                    _enc_b = _state["encoder_bias"].to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous()
                    _basis_rank = int(_enc_w.shape[0])
                    _expected_hidden = int(self.config.hidden_size)
                    if int(_enc_w.shape[1]) != _expected_hidden:
                        raise RuntimeError(
                            f"Sparse intermediate-router layer {_lidx}: encoder_weight has input dim "
                            f"{int(_enc_w.shape[1])}, expected hidden_size={_expected_hidden}."
                        )
                    _rank_effective = int(
                        self._coerce_sparse_layer_stat(_state, _layer_stats, "rank_effective", _basis_rank)
                    )
                    _rank_effective = int(max(1, min(_rank_effective, _basis_rank)))
                    _evr = float(
                        self._coerce_sparse_layer_stat(_state, _layer_stats, "explained_variance_ratio", 1.0)
                    )
                    _evrs.append(_evr)
                    if _evr < _MIN_EXPLAINED_VARIANCE:
                        _low_evr_count += 1
                        import warnings as _w
                        _w.warn(
                            f"[sparse_basis] Layer {_lidx} explained_variance_ratio={_evr:.3f} < "
                            f"{_MIN_EXPLAINED_VARIANCE:.2f}. The intermediate router may have poor recall.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    _score_weight_raw = _state.get("score_weight")
                    if not torch.is_tensor(_score_weight_raw):
                        raise RuntimeError(f"Sparse intermediate-router layer {_lidx} is missing score_weight.")
                    _route_score_weight = _score_weight_raw.to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous()
                    if int(_route_score_weight.ndim) != 2 or int(_route_score_weight.shape[1]) != int(_basis_rank):
                        raise RuntimeError(
                            f"Sparse intermediate-router layer {_lidx}: score_weight shape "
                            f"{tuple(_route_score_weight.shape)} does not match basis_rank={_basis_rank}."
                        )
                    _layer_num_blocks = int(_route_score_weight.shape[0])
                    _expected_intermediate_blocks = int(getattr(self.config, "intermediate_size", 0)) // int(
                        self._sparse_block_size
                    )
                    if _layer_num_blocks != _expected_intermediate_blocks:
                        raise RuntimeError(
                            f"Sparse intermediate-router layer {_lidx} stores {_layer_num_blocks} FFN blocks, "
                            f"expected {_expected_intermediate_blocks}."
                        )
                    _block_bank_layout = build_intermediate_mlp_block_bank_layout(
                        layer_idx=int(_lidx),
                        hidden_size=int(self.config.hidden_size),
                        intermediate_size=int(getattr(self.config, "intermediate_size", 0)),
                        block_size=int(self._sparse_block_size),
                        num_blocks=int(_layer_num_blocks),
                    )
                    _score_bias_raw = _state.get("score_bias")
                    _route_score_bias = None
                    if torch.is_tensor(_score_bias_raw):
                        _route_score_bias = _score_bias_raw.to(
                            device=_basis_device,
                            dtype=self.dtype,
                            non_blocking=False,
                        ).contiguous().view(-1)
                        if int(_route_score_bias.numel()) != _layer_num_blocks:
                            raise RuntimeError(
                                f"Sparse intermediate-router layer {_lidx}: score_bias has "
                                f"{int(_route_score_bias.numel())} values, expected {_layer_num_blocks}."
                            )
                    _down_block_norm_raw = _state.get("down_block_norm")
                    _down_block_norm = None
                    if torch.is_tensor(_down_block_norm_raw):
                        _down_block_norm = _down_block_norm_raw.to(
                            device=torch.device("cpu"),
                            dtype=torch.float32,
                            non_blocking=False,
                        ).contiguous().view(-1)
                        if int(_down_block_norm.numel()) != _layer_num_blocks:
                            raise RuntimeError(
                                f"Sparse intermediate-router layer {_lidx}: down_block_norm has "
                                f"{int(_down_block_norm.numel())} values, expected {_layer_num_blocks}."
                            )
                    _layer_top_k = int(
                        _block_top_k_by_layer.get(
                            str(_lidx),
                            _block_top_k_by_layer.get(_lidx, self._sparse_top_k),
                        )
                    )
                    _layer_top_k = int(max(1, min(_layer_top_k, _layer_num_blocks)))
                    _basis_top_k = int(
                        sparse_basis_top_k
                        if sparse_basis_top_k is not None
                        else _basis_top_k_by_layer.get(
                            str(_lidx),
                            _basis_top_k_by_layer.get(_lidx, _default_basis_top_k),
                        )
                    )
                    _basis_top_k = int(max(1, min(_basis_top_k, _rank_effective)))
                    _scale = _state.get("scale")
                    self._sparse_routing[_lidx] = {
                        "enc_w": _enc_w,
                        "enc_b": _enc_b,
                        "dec": None,
                        "dec_bias": None,
                        "dec_norm_t": _route_score_weight.transpose(0, 1).abs().contiguous(),
                        "route_score_weight": _route_score_weight,
                        "route_score_bias": _route_score_bias,
                        "down_block_norm": _down_block_norm,
                        "down_block_norm_ready": bool(_down_block_norm is not None),
                        "block_bank_layout": _block_bank_layout,
                        "scale": float(_scale.item()) if torch.is_tensor(_scale) and int(_scale.numel()) == 1 else 1.0,
                        "top_k": _layer_top_k,
                        "basis_top_k": _basis_top_k,
                        "basis_rank": int(_basis_rank),
                        "rank_effective": int(_rank_effective),
                        "artifact_target": _artifact_target,
                        "block_domain": _block_domain,
                        "block_size": int(self._sparse_block_size),
                        "num_blocks": int(_layer_num_blocks),
                        "explained_variance_ratio": float(_evr),
                        "samples": float(self._coerce_sparse_layer_stat(_state, _layer_stats, 'samples', -1.0)),
                        "pca_method": str(self._coerce_sparse_layer_stat(_state, _layer_stats, 'pca_method', 'unknown')),
                    }
                    self._sparse_top_k_by_layer[_lidx] = int(_layer_top_k)
                    self._sparse_basis_top_k_by_layer[_lidx] = int(_basis_top_k)
                    if _build_mlp_hot_block_map:
                        hot_blocks = self._derive_hot_blocks_for_layer(
                            layer_state=_state,
                            layer_stats=_layer_stats,
                            block_importance_by_layer=_block_importance_by_layer.get(str(_lidx), None),
                            layer_num_blocks=int(_layer_num_blocks),
                        )
                        if hot_blocks is not None and int(hot_blocks.numel()) > 0:
                            self._mlp_hot_blocks_by_layer[_lidx] = hot_blocks
                    continue
                _layer_num_blocks = int(_state["decoder_blocks"].shape[0])
                _expected_output_blocks = int(self.config.hidden_size) // int(self._sparse_block_size)
                if _layer_num_blocks != _expected_output_blocks:
                    raise RuntimeError(
                        f"Sparse basis layer {_lidx} stores {_layer_num_blocks} output blocks, "
                        f"expected {_expected_output_blocks}. The learned-basis artifact is defined "
                        "over hidden-size output blocks and cannot be mixed with FFN intermediate blocks."
                    )

                _enc_w_shape = _state["encoder_weight"].shape
                _expected_hidden = int(self.config.hidden_size)
                if int(_enc_w_shape[1]) != _expected_hidden:
                    raise RuntimeError(
                        f"Sparse basis layer {_lidx}: encoder_weight has input dim {int(_enc_w_shape[1])}, "
                        f"but model hidden_size={_expected_hidden}. "
                        "The checkpoint was fitted for a different model or hidden size."
                    )

                _dec_shape = _state["decoder_blocks"].shape
                if int(_dec_shape[1]) != int(_enc_w_shape[0]):
                    raise RuntimeError(
                        f"Sparse basis layer {_lidx}: decoder_blocks basis_rank dim={int(_dec_shape[1])} "
                        f"does not match encoder_weight rank={int(_enc_w_shape[0])}."
                    )

                _evr = float(
                    self._coerce_sparse_layer_stat(_state, _layer_stats, "explained_variance_ratio", 1.0)
                )
                _evrs.append(_evr)
                if _evr < _MIN_EXPLAINED_VARIANCE:
                    _low_evr_count += 1
                    import warnings as _w
                    _w.warn(
                        f"[sparse_basis] Layer {_lidx} explained_variance_ratio={_evr:.3f} < "
                        f"{_MIN_EXPLAINED_VARIANCE:.2f}. The basis is capturing less than "
                        f"{_MIN_EXPLAINED_VARIANCE*100:.0f}% of output variance; output quality "
                        "for this layer may be significantly degraded. Consider re-fitting with "
                        "more calibration samples (--max-rows-per-layer) or a higher basis_rank.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                _basis_rank = int(_state["encoder_weight"].shape[0])


                _rank_effective = int(
                    self._coerce_sparse_layer_stat(_state, _layer_stats, "rank_effective", _basis_rank)
                )
                _rank_effective = int(max(1, min(_rank_effective, _basis_rank)))
                _layer_top_k = int(
                    _block_top_k_by_layer.get(
                        str(_lidx),
                        _block_top_k_by_layer.get(_lidx, self._sparse_top_k),
                    )
                )
                _layer_top_k = int(max(1, min(_layer_top_k, _layer_num_blocks)))
                _basis_top_k = int(
                    sparse_basis_top_k
                    if sparse_basis_top_k is not None
                    else _basis_top_k_by_layer.get(
                        str(_lidx),
                        _basis_top_k_by_layer.get(_lidx, _default_basis_top_k),
                    )
                )

                _basis_top_k = int(max(1, min(_basis_top_k, _rank_effective)))
                _dec = _state["decoder_blocks"].to(
                    device=_basis_device,
                    dtype=self.dtype,
                    non_blocking=False,
                ).contiguous()
                _dec_bias = _state.get("decoder_bias")
                if torch.is_tensor(_dec_bias):
                    _dec_bias = _dec_bias.to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous()
                _scale = _state.get("scale")
                _dec_norm_t = _dec.norm(dim=-1).transpose(0, 1).contiguous()
                if self._sparse_semantic_block_score_normalized:
                    _dec_norm_t = F.normalize(_dec_norm_t, p=2.0, dim=-1, eps=1e-6)
                self._sparse_routing[_lidx] = {
                    "enc_w": _state["encoder_weight"].to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous(),
                    "enc_b": _state["encoder_bias"].to(
                        device=_basis_device,
                        dtype=self.dtype,
                        non_blocking=False,
                    ).contiguous(),
                    "dec": _dec,
                    "dec_bias": _dec_bias,
                    "dec_norm_t": _dec_norm_t,
                    "scale": float(_scale.item()) if torch.is_tensor(_scale) and int(_scale.numel()) == 1 else 1.0,
                    "top_k": _layer_top_k,
                    "basis_top_k": _basis_top_k,
                    "basis_rank": int(_basis_rank),
                    "rank_effective": int(_rank_effective),
                    "artifact_target": _artifact_target,
                    "block_domain": _block_domain,
                    "block_size": int(self._sparse_block_size),
                    "num_blocks": int(_layer_num_blocks),
                    "explained_variance_ratio": float(_evr),
                    "samples": float(self._coerce_sparse_layer_stat(_state, _layer_stats, "samples", -1.0)),
                    "pca_method": str(self._coerce_sparse_layer_stat(_state, _layer_stats, "pca_method", "unknown")),
                }
                self._sparse_top_k_by_layer[_lidx] = int(_layer_top_k)
                self._sparse_basis_top_k_by_layer[_lidx] = int(_basis_top_k)
                if _build_mlp_hot_block_map:
                    hot_blocks = self._derive_hot_blocks_for_layer(
                        layer_state=_state,
                        layer_stats=_layer_stats,
                        block_importance_by_layer=_block_importance_by_layer.get(str(_lidx), None),
                        layer_num_blocks=int(_layer_num_blocks),
                    )
                    if hot_blocks is not None and int(hot_blocks.numel()) > 0:
                        self._mlp_hot_blocks_by_layer[_lidx] = hot_blocks
            _pct = int(round(self._sparse_top_k * 100 / max(_num_blocks, 1)))
            _hot_layers = len(self._mlp_hot_blocks_by_layer)
            _hot_blocks_total = sum(int(v.numel()) for v in self._mlp_hot_blocks_by_layer.values())
            _summary = self.get_sparse_mlp_summary()
            print(
                f"[sparse] loaded routing for {len(self._sparse_routing)}/{self.num_layers} layers "
                f"| top_k={self._sparse_top_k}/{_num_blocks} blocks ({_pct}%) "
                f"| block_size={self._sparse_block_size} "
                f"| exec_request={self._sparse_mlp_execution_request} "
                f"| prefill={self._sparse_mlp_prefill_mode} "
                f"| prefill_top_k={self._sparse_mlp_prefill_top_k or self._sparse_top_k}",
                flush=True,
            )
            print(
                f"[sparse] targets={_summary.get('artifact_target_counts', {})} "
                f"| domains={_summary.get('block_domain_counts', {})} "
                f"| execution={_summary.get('execution_counts', {})}",
                flush=True,
            )
            if _evrs:
                print(
                    f"[sparse] explained_variance min={min(_evrs):.3f} "
                    f"mean={sum(_evrs) / len(_evrs):.3f} max={max(_evrs):.3f} "
                    f"| low(<{_MIN_EXPLAINED_VARIANCE:.2f})={_low_evr_count}",
                    flush=True,
                )
            if self._sparse_explicit_layer_selection:
                _upper_guard_start = max(self._sparse_explicit_layer_selection) + 1
                self._upper_decode_guard_layers = {
                    int(_idx)
                    for _idx in range(_upper_guard_start, int(self.num_layers))
                    if int(_idx) not in self._sparse_explicit_layer_selection
                }
                print(
                    f"[sparse] explicit layer selection: {len(self._sparse_explicit_layer_selection)}/{self.num_layers} "
                    f"MLP layers; non-selected layers use exact streamed dense guard execution",
                    flush=True,
                )
            if _hot_layers > 0:
                print(
                    f"[sparse] hot-block map ready for {_hot_layers} layers "
                    f"({int(_hot_blocks_total)} total blocks, threshold={self._hot_block_threshold:.2f})",
                    flush=True,
                )
            if self._vram_hot_cache_limit_bytes is not None:
                print(
                    f"[sparse] VRAM hot-cache budget: {self._vram_hot_cache_limit_bytes / (1024 ** 3):.2f} GB",
                    flush=True,
                )
            if (
                str(getattr(self, "_sparse_mlp_prefill_mode", "dense")) == "sparse"
                and _summary.get("execution_counts", {}).get(_EXACT_BLOCKWISE_SPARSE, 0) > 0
                and os.getenv("STREAMING_ALLOW_UNSAFE_SPARSE_MLP_PREFILL", "").strip().lower()
                not in {"1", "true", "yes", "on"}
            ):
                self._sparse_mlp_prefill_mode = "dense"
                print(
                    "[sparse] exact blockwise sparse MLP prefill is disabled for coherence; "
                    "using dense MLP prefill. Set STREAMING_ALLOW_UNSAFE_SPARSE_MLP_PREFILL=1 to force it.",
                    flush=True,
                )














        self._attn_active_head_indices: dict[int, torch.Tensor] = {}
        self._attn_head_importance: dict[int, torch.Tensor] = {}
        self._attn_active_heads: int = 0
        self._attn_dynamic_threshold = float(max(0.0, min(1.0, attn_head_activity_threshold)))
        self._attn_min_active_heads = max(1, int(attn_min_active_heads))
        self._attn_max_active_heads = int(attn_max_active_heads) if attn_max_active_heads is not None else 0
        self._attn_runtime_head_counts: dict[int, int] = {}
        self._attn_zero_only_layers: set[int] = set()
        self._attn_sparse_disabled_reason: str | None = None


        self._attn_sparse_param_meta: dict[str, dict[str, Any]] = {}
        self._attn_hot_head_cache: dict[int, dict[str, Any]] = {}


        self._sparse_attn_cpu_row_cache: dict = {}
        self._cpu_row_cache_hits: int = 0
        self._cpu_row_cache_misses: int = 0
        self._attn_loaded_q_rows: torch.Tensor | None = None
        self._attn_loaded_o_cols: torch.Tensor | None = None
        self._attn_qo_state: str = "unknown"
        self._debug_assert_sparse_attn_qo_zero = (
            os.getenv("STREAMING_DEBUG_ASSERT_SPARSE_ATTN_QO_ZERO", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self._cpu_scratch: dict[str, torch.Tensor] = {}

        self._attn_q_head_staging: torch.Tensor | None = None
        self._attn_share_groups: dict[str, dict[str, Any]] = {}
        self._attn_share_layer_state: dict[int, dict[str, Any]] = {}
        self._attn_share_exact_layers: set[int] = set()


        self._kv_routing: dict[int, dict[str, Any]] = {}
        self._kv_sparse_top_k: int = 0
        self._kv_sparse_block_size: int = 32
        self._kv_num_col_blocks: int = 0
        self._kv_sparse_param_cache: dict[str, dict[str, Any]] = {}
        self._kv_hot_blocks_by_layer: dict[int, torch.Tensor] = {}
        self._attn_kv_hot_block_cache: dict[str, Any] = {}
        self._kv_block_usage_ema: dict[int, torch.Tensor] = {}
        self._kv_block_usage_votes: dict[int, torch.Tensor] = {}
        self._kv_block_banked_until: dict[int, torch.Tensor] = {}
        self._kv_bank_step: int = 0
        self._kv_loaded_cols: torch.Tensor | None = None
        self._gqa_kv_loaded_rows: torch.Tensor | None = None

        if attn_head_importance_path and str(attn_head_importance_path).strip():
            _attn_payload = torch.load(
                str(attn_head_importance_path), map_location="cpu", weights_only=False
            )
            _attn_cfg = _attn_payload.get("config", {})
            _H   = int(_attn_cfg.get("num_heads",    getattr(self.config, "num_attention_heads", 128)))
            _D   = int(_attn_cfg.get("head_dim",     getattr(self.config, "head_dim", 128)))
            _Hid = int(_attn_cfg.get("hidden_size",  getattr(self.config, "hidden_size", 16384)))
            requested_heads = int(attn_active_heads) if attn_active_heads is not None else max(1, min(self._attn_min_active_heads, _H))
            _num_kv = int(_attn_cfg.get("num_kv_heads", getattr(self.config, "num_key_value_heads", max(1, _H // 16))))
            _per_head_weight_bytes = (_Hid * _D // 2) + (_Hid * _D // 64 * 4)
            _per_head_qo_mb = float(2 * _per_head_weight_bytes) / float(1024 ** 2)
            _kv_weight_bytes = (_num_kv * _D * _Hid // 2) + (_num_kv * _D * _Hid // 64 * 4)
            _kv_total_mb = float(2 * _kv_weight_bytes) / float(1024 ** 2)
            _budget_heads = max(
                1,
                int((float(self._target_layer_traffic_mb) - float(_kv_total_mb)) // max(_per_head_qo_mb, 1e-6)),
            )
            explicit_pool = self._attn_max_active_heads > 0
            if not explicit_pool:
                if attn_active_heads is not None:





                    self._attn_max_active_heads = int(requested_heads)
                else:
                    self._attn_max_active_heads = int(_budget_heads)
            self._attn_max_active_heads = max(1, min(self._attn_max_active_heads, _H))
            self._attn_min_active_heads = max(1, min(self._attn_min_active_heads, self._attn_max_active_heads))
            if attn_active_heads is not None:
                K = max(
                    1,
                    min(
                        self._attn_max_active_heads,
                        max(
                            min(self._attn_min_active_heads, self._attn_max_active_heads),
                            min(requested_heads, _budget_heads),
                        ),
                    ),
                )
            else:
                K = max(1, min(requested_heads, self._attn_max_active_heads))
            self._attn_active_heads = K

            _skipped_attn_layers: list[int] = []
            _forced_dense_attn_layers: list[int] = []
            _measured_importance: list[torch.Tensor] = []
            for _lidx_s, _state in _attn_payload.get("layer_states", {}).items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                if _lidx in self._attn_force_dense_layers:
                    _forced_dense_attn_layers.append(_lidx)
                    continue
                imp = _state["importance"].float()
                _token_count = int(_state.get("token_count", 0) or 0)
                _finite = bool(torch.isfinite(imp).all().item())
                _span = float((imp.max() - imp.min()).item()) if int(imp.numel()) > 0 and _finite else 0.0
                if _token_count <= 0 or int(imp.numel()) != _H or not _finite or _span <= 1e-8:
                    _skipped_attn_layers.append(_lidx)
                    continue
                self._attn_head_importance[_lidx] = imp
                _measured_importance.append(imp.detach().to(device=torch.device("cpu"), dtype=torch.float32).contiguous())
                selected_heads = self._select_static_attention_heads(
                    imp,
                    max_heads=min(self._attn_max_active_heads, _H),
                    num_heads=_H,
                    num_kv_heads=_num_kv,
                )
                self._attn_active_head_indices[_lidx] = selected_heads

            _synthesized_attn_layers: list[int] = []
            _fallback_importance: torch.Tensor | None = None
            if _measured_importance:
                _fallback_importance = torch.stack(_measured_importance, dim=0).mean(dim=0).contiguous()
            if _fallback_importance is None or int(_fallback_importance.numel()) != _H:
                _fallback_importance = torch.linspace(float(_H), 1.0, steps=_H, dtype=torch.float32)
            if float((_fallback_importance.max() - _fallback_importance.min()).item()) <= 1e-8:
                _fallback_importance = _fallback_importance + torch.linspace(0.0, 1.0, steps=_H, dtype=torch.float32)

            for _lidx in range(self.num_layers):
                if _lidx in self._attn_force_dense_layers:
                    continue
                if _lidx in self._attn_active_head_indices:
                    continue
                _fallback_layer_imp = _fallback_importance.clone()
                self._attn_head_importance[_lidx] = _fallback_layer_imp
                self._attn_active_head_indices[_lidx] = self._select_static_attention_heads(
                    _fallback_layer_imp,
                    max_heads=min(self._attn_max_active_heads, _H),
                    num_heads=_H,
                    num_kv_heads=_num_kv,
                )
                _synthesized_attn_layers.append(int(_lidx))

            if torch.cuda.is_available():

                self._attn_q_head_staging = torch.empty(
                    self._attn_max_active_heads * _D * _Hid, dtype=self.dtype, device=self.device
                )

            _pct_a = int(round(K * 100 // max(_H, 1)))
            print(
                f"[sparse_attn] loaded importance for {len(self._attn_active_head_indices)}/{self.num_layers} layers "
                f"| active_heads={K}/{_H} ({_pct_a}%)"
                f" | min={self._attn_min_active_heads} max={self._attn_max_active_heads}"
                f" threshold={self._attn_dynamic_threshold:.2f}"
                f" | target_layer_mb={self._target_layer_traffic_mb:.1f}"
                f" | prefill_qo={self._sparse_attn_prefill_mode}"
                f" prefill_kv={self._sparse_kv_prefill_mode}"
                f" selection={self._attn_head_selection_mode}",
                flush=True,
            )
            if _skipped_attn_layers:
                _preview = ",".join(str(i) for i in sorted(_skipped_attn_layers)[:12])
                if len(_skipped_attn_layers) > 12:
                    _preview = f"{_preview},..."
                print(
                    f"[sparse_attn] synthesized sparse head maps for {len(_skipped_attn_layers)} non-measured/uniform layers "
                    f"(layers={_preview})",
                    flush=True,
                )
            if _synthesized_attn_layers and len(_synthesized_attn_layers) != len(_skipped_attn_layers):
                _preview = ",".join(str(i) for i in sorted(_synthesized_attn_layers)[:12])
                if len(_synthesized_attn_layers) > 12:
                    _preview = f"{_preview},..."
                print(
                    f"[sparse_attn] synthesized fallback coverage for {len(_synthesized_attn_layers)} layers "
                    f"(layers={_preview})",
                    flush=True,
                )
            if _forced_dense_attn_layers:
                _preview = ",".join(str(i) for i in sorted(_forced_dense_attn_layers)[:12])
                if len(_forced_dense_attn_layers) > 12:
                    _preview = f"{_preview},..."
                print(
                    f"[sparse_attn] forced {len(_forced_dense_attn_layers)} measured layers dense "
                    f"via STREAMING_SPARSE_ATTN_DENSE_LAYERS (layers={_preview})",
                    flush=True,
                )


        if mlp_skip_mask_path and str(mlp_skip_mask_path).strip():
            _skip_payload = torch.load(str(mlp_skip_mask_path), map_location="cpu", weights_only=False)
            _skip_mask = _skip_payload if isinstance(_skip_payload, torch.Tensor) else _skip_payload.get("skip_mask")
            if _skip_mask is not None:
                self._mlp_static_skip_mask = _skip_mask.bool()
                self._mlp_static_skip_set = {
                    int(i) for i in _skip_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
                }
                n_skip = int(_skip_mask.sum().item())
                print(
                    f"[mlp-skip] loaded static skip mask: {n_skip}/{self.num_layers} layers will skip MLP",
                    flush=True,
                )


        if attn_share_path and str(attn_share_path).strip():
            _share_payload = torch.load(str(attn_share_path), map_location="cpu", weights_only=False)
            _share_cfg = _share_payload.get("config", {})
            _hidden_size_cfg = int(_share_cfg.get("hidden_size", getattr(self.config, "hidden_size", 0)))
            _num_heads_cfg = int(_share_cfg.get("num_heads", getattr(self.config, "num_attention_heads", 0)))
            _num_kv_heads_cfg = int(_share_cfg.get("num_kv_heads", getattr(self.config, "num_key_value_heads", 0) or getattr(self.config, "num_attention_heads", 0)))
            _head_dim_cfg = int(_share_cfg.get("head_dim", getattr(self.config, "head_dim", 0)))
            _hidden_size_model = int(getattr(self.config, "hidden_size", 0))
            _num_heads_model = int(getattr(self.config, "num_attention_heads", 0))
            _num_kv_heads_model = int(getattr(self.config, "num_key_value_heads", 0) or _num_heads_model)
            _head_dim_model = int(getattr(self.config, "head_dim", 0))
            if (
                _hidden_size_cfg != _hidden_size_model
                or _num_heads_cfg != _num_heads_model
                or _num_kv_heads_cfg != _num_kv_heads_model
                or _head_dim_cfg != _head_dim_model
            ):
                raise RuntimeError(
                    "Attention-sharing checkpoint dimensions do not match the loaded model: "
                    f"artifact hidden/heads/kv_heads/head_dim={_hidden_size_cfg}/{_num_heads_cfg}/{_num_kv_heads_cfg}/{_head_dim_cfg}, "
                    f"model={_hidden_size_model}/{_num_heads_model}/{_num_kv_heads_model}/{_head_dim_model}."
                )
            _share_dtype = self.dtype
            for _gid_raw, _group_state in (_share_payload.get("group_states", {}) or {}).items():
                _gid = str(_gid_raw)
                _entry: dict[str, Any] = {
                    "layers": tuple(int(v) for v in _group_state.get("layers", [])),
                    "sharing_format": str(_group_state.get("sharing_format", "matrix_v1")),
                }
                for _name in (
                    "q_base_u",
                    "q_base_v",
                    "o_base_u",
                    "o_base_v",
                    "k_base_u",
                    "k_base_v",
                    "v_base_u",
                    "v_base_v",
                    "q_base_u_heads",
                    "q_base_v_heads",
                    "o_base_u_heads",
                    "o_base_v_heads",
                    "k_base_u_heads",
                    "k_base_v_heads",
                    "v_base_u_heads",
                    "v_base_v_heads",
                ):
                    _tensor = _group_state.get(_name)
                    if torch.is_tensor(_tensor):
                        _entry[_name] = self.loader.prepare_h2d_source(
                            _tensor.to(dtype=_share_dtype).contiguous(),
                            dtype=_share_dtype,
                        )
                self._attn_share_groups[_gid] = _entry
            for _exact_layer in (_share_payload.get("exact_layers", []) or []):
                _exact_layer_idx = int(_exact_layer)
                if 0 <= _exact_layer_idx < self.num_layers:
                    self._attn_share_exact_layers.add(_exact_layer_idx)
            for _lidx_raw, _layer_state in (_share_payload.get("layer_states", {}) or {}).items():
                _lidx = int(_lidx_raw)
                if not (0 <= _lidx < self.num_layers):
                    continue
                _gid = str(_layer_state.get("group_id", ""))
                if _gid == "" or _gid not in self._attn_share_groups:
                    self._attn_share_exact_layers.add(_lidx)
                    continue
                _head_perm = torch.as_tensor(
                    _layer_state.get("head_perm", list(range(_num_heads_model))),
                    dtype=torch.long,
                    device=torch.device("cpu"),
                ).contiguous()
                if int(_head_perm.numel()) != _num_heads_model:
                    raise RuntimeError(
                        f"Attention-sharing layer {_lidx} head_perm has {int(_head_perm.numel())} entries; "
                        f"expected {_num_heads_model}."
                    )
                _kv_head_perm = torch.as_tensor(
                    _layer_state.get("kv_head_perm", list(range(_num_kv_heads_model))),
                    dtype=torch.long,
                    device=torch.device("cpu"),
                ).contiguous()
                if int(_kv_head_perm.numel()) != _num_kv_heads_model:
                    raise RuntimeError(
                        f"Attention-sharing layer {_lidx} kv_head_perm has {int(_kv_head_perm.numel())} entries; "
                        f"expected {_num_kv_heads_model}."
                    )
                _q_resid_u = _layer_state.get("q_resid_u")
                _q_resid_v = _layer_state.get("q_resid_v")
                _o_resid_u = _layer_state.get("o_resid_u")
                _o_resid_v = _layer_state.get("o_resid_v")
                _k_resid_u = _layer_state.get("k_resid_u")
                _k_resid_v = _layer_state.get("k_resid_v")
                _v_resid_u = _layer_state.get("v_resid_u")
                _v_resid_v = _layer_state.get("v_resid_v")
                self._attn_share_layer_state[_lidx] = {
                    "group_id": _gid,
                    "head_perm": _head_perm,
                    "kv_head_perm": _kv_head_perm,
                    "q_resid_u": None if not torch.is_tensor(_q_resid_u) else self.loader.prepare_h2d_source(
                        _q_resid_u.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "q_resid_v": None if not torch.is_tensor(_q_resid_v) else self.loader.prepare_h2d_source(
                        _q_resid_v.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "o_resid_u": None if not torch.is_tensor(_o_resid_u) else self.loader.prepare_h2d_source(
                        _o_resid_u.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "o_resid_v": None if not torch.is_tensor(_o_resid_v) else self.loader.prepare_h2d_source(
                        _o_resid_v.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "k_resid_u": None if not torch.is_tensor(_k_resid_u) else self.loader.prepare_h2d_source(
                        _k_resid_u.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "k_resid_v": None if not torch.is_tensor(_k_resid_v) else self.loader.prepare_h2d_source(
                        _k_resid_v.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "v_resid_u": None if not torch.is_tensor(_v_resid_u) else self.loader.prepare_h2d_source(
                        _v_resid_u.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                    "v_resid_v": None if not torch.is_tensor(_v_resid_v) else self.loader.prepare_h2d_source(
                        _v_resid_v.to(dtype=_share_dtype).contiguous(),
                        dtype=_share_dtype,
                    ),
                }
                for _name in (
                    "q_resid_u_heads",
                    "q_resid_v_heads",
                    "o_resid_u_heads",
                    "o_resid_v_heads",
                    "k_resid_u_heads",
                    "k_resid_v_heads",
                    "v_resid_u_heads",
                    "v_resid_v_heads",
                ):
                    _tensor = _layer_state.get(_name)
                    self._attn_share_layer_state[_lidx][_name] = (
                        None
                        if not torch.is_tensor(_tensor)
                        else self.loader.prepare_h2d_source(
                            _tensor.to(dtype=_share_dtype).contiguous(),
                            dtype=_share_dtype,
                        )
                    )
            if self._attn_share_layer_state:
                print(
                    f"[attn_share] loaded q/o sharing for {len(self._attn_share_layer_state)}/{self.num_layers} layers "
                    f"| groups={len(self._attn_share_groups)} | exact={len(self._attn_share_exact_layers)} "
                    f"| prefill={self._attn_share_prefill_mode}",
                    flush=True,
                )

        if kv_basis_path and str(kv_basis_path).strip():
            _kv_payload = torch.load(str(kv_basis_path), map_location="cpu", weights_only=False)
            _kv_cfg = _kv_payload.get("config", {})
            self._kv_sparse_block_size = int(_kv_cfg.get("block_size", 32))
            _kv_hidden_size = int(_kv_cfg.get("hidden_size", getattr(self.config, "hidden_size", 16384)))
            self._kv_num_col_blocks = _kv_hidden_size // self._kv_sparse_block_size
            _kv_default_top_k = max(1, int(round(self._kv_num_col_blocks * 0.10)))
            self._kv_sparse_top_k = int(kv_sparse_top_k) if kv_sparse_top_k is not None else _kv_default_top_k
            _kv_basis_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            for _lidx_s, _lstate in _kv_payload.get("layer_states", {}).items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                _dec = _lstate["decoder_blocks"].to(device=_kv_basis_device, dtype=self.dtype).contiguous()
                _dec_norm_t = _dec.norm(dim=-1).transpose(0, 1).contiguous()
                self._kv_routing[_lidx] = {
                    "enc_w": _lstate["encoder_weight"].to(device=_kv_basis_device, dtype=self.dtype).contiguous(),
                    "enc_b": _lstate["encoder_bias"].to(device=_kv_basis_device, dtype=self.dtype).contiguous(),
                    "dec": _dec,
                    "dec_norm_t": _dec_norm_t,
                    "top_k": self._kv_sparse_top_k,
                }
                _blk_imp = _lstate.get("block_importance")
                if torch.is_tensor(_blk_imp):
                    _sorted_imp, _sorted_idx = _blk_imp.float().sort(descending=True)
                    _hot_k = max(1, int(round(self._kv_num_col_blocks * 0.10)))
                    self._kv_hot_blocks_by_layer[_lidx] = _sorted_idx[:_hot_k].contiguous()
                    self._kv_block_usage_ema[_lidx] = _blk_imp.float().clone()
                else:
                    self._kv_block_usage_ema[_lidx] = torch.zeros(self._kv_num_col_blocks, dtype=torch.float32)
                self._kv_block_usage_votes[_lidx] = torch.zeros(self._kv_num_col_blocks, dtype=torch.int32)
                self._kv_block_banked_until[_lidx] = torch.zeros(self._kv_num_col_blocks, dtype=torch.int32)
            if self._show_progress:
                print(
                    f"[kv_sparse] Loaded KV routing for {len(self._kv_routing)} layers, "
                    f"top_k={self._kv_sparse_top_k}, num_col_blocks={self._kv_num_col_blocks}",
                    flush=True,
                )

    @staticmethod
    def _coerce_sparse_layer_stat(layer_state: dict[str, Any], layer_stats: Any, key: str, default: Any) -> Any:
        if key in layer_state:
            value = layer_state.get(key)
            if torch.is_tensor(value):
                if int(value.numel()) == 1:
                    return value.detach().to(device=torch.device("cpu")).view(-1)[0].item()
                return value
            return value
        if isinstance(layer_stats, dict) and key in layer_stats:
            value = layer_stats.get(key)
            if torch.is_tensor(value):
                if int(value.numel()) == 1:
                    return value.detach().to(device=torch.device("cpu")).view(-1)[0].item()
                return value
            return value
        return default

    def _resolve_sparse_mlp_execution_for_routing(
        self,
        layer_idx: int,
        routing: dict[str, Any] | None = None,
    ) -> str:
        routing = self._sparse_routing.get(int(layer_idx)) if routing is None else routing
        if routing is None:
            return "dense_guard"
        requested = _normalize_sparse_mlp_execution(getattr(self, "_sparse_mlp_execution_request", "auto"))
        if requested != "auto":
            return requested
        block_domain = str(routing.get("block_domain", "intermediate")).strip().lower()
        artifact_target = str(routing.get("artifact_target", "")).strip().lower()
        if block_domain == "intermediate" or artifact_target == "intermediate_block_scores":
            return _EXACT_BLOCKWISE_SPARSE
        raise RuntimeError(
            f"Sparse routing for this layer uses block_domain={block_domain!r} which is no longer supported. "
            f"Only 'intermediate' block_domain ({_EXACT_BLOCKWISE_SPARSE}) is supported."
        )

    def get_sparse_mlp_layer_info(self, layer_idx: int) -> dict[str, Any] | None:
        routing = self._sparse_routing.get(int(layer_idx))
        if routing is None:
            return None
        top_k = int(
            routing.get(
                "top_k",
                self._sparse_runtime_top_k if int(self._sparse_runtime_top_k) > 0 else self._sparse_top_k,
            )
        )
        basis_top_k = int(
            routing.get(
                "basis_top_k",
                self._sparse_basis_top_k_by_layer.get(int(layer_idx), _DEFAULT_SPARSE_BASIS_TOP_K),
            )
        )
        return {
            "layer_idx": int(layer_idx),
            "execution": self._resolve_sparse_mlp_execution_for_routing(int(layer_idx), routing=routing),
            "artifact_target": str(routing.get("artifact_target", "intermediate_block_scores")),
            "block_domain": str(routing.get("block_domain", "intermediate")),
            "block_size": int(routing.get("block_size", self._sparse_block_size)),
            "num_blocks": int(routing.get("num_blocks", self._sparse_num_blocks)),
            "top_k": int(top_k),
            "basis_top_k": int(basis_top_k),
            "rank_effective": int(routing.get("rank_effective", routing.get("basis_rank", 0))),
            "explained_variance_ratio": float(routing.get("explained_variance_ratio", 1.0)),
            "prefill_mode": str(getattr(self, "_sparse_mlp_prefill_mode", "dense")),
            "block_bank_layout": (
                routing["block_bank_layout"].as_dict()
                if isinstance(routing.get("block_bank_layout"), MlpBlockBankLayout)
                else None
            ),
        }

    def get_sparse_mlp_summary(self) -> dict[str, Any]:
        execution_counts: dict[str, int] = defaultdict(int)
        domain_counts: dict[str, int] = defaultdict(int)
        target_counts: dict[str, int] = defaultdict(int)
        evrs: list[float] = []
        for layer_idx, routing in self._sparse_routing.items():
            execution_counts[self._resolve_sparse_mlp_execution_for_routing(int(layer_idx), routing=routing)] += 1
            domain_counts[str(routing.get("block_domain", "intermediate"))] += 1
            target_counts[str(routing.get("artifact_target", "intermediate_block_scores"))] += 1
            evr = float(routing.get("explained_variance_ratio", 1.0))
            evrs.append(evr)
        summary: dict[str, Any] = {
            "layers": int(len(self._sparse_routing)),
            "execution_request": str(getattr(self, "_sparse_mlp_execution_request", "auto")),
            "prefill_mode": str(getattr(self, "_sparse_mlp_prefill_mode", "dense")),
            "execution_counts": dict(execution_counts),
            "block_domain_counts": dict(domain_counts),
            "artifact_target_counts": dict(target_counts),
            "top_k_default": int(self._sparse_top_k),
        }
        if evrs:
            summary["explained_variance"] = {
                "min": float(min(evrs)),
                "max": float(max(evrs)),
                "mean": float(sum(evrs) / len(evrs)),
            }
        return summary

    def get_runtime_status(self) -> dict[str, Any]:
        sparse_summary = self.get_sparse_mlp_summary()
        lm_head_status = self.get_lm_head_status()
        decode_profile_report = self.get_decode_profile_report()
        static_attn_layers = int(len(self._attn_active_head_indices))
        runtime_head_counts = [int(v) for v in self._attn_runtime_head_counts.values() if int(v) > 0]
        kv_layers = int(len(getattr(self, "_kv_routing", {}) or {}))
        hot_layers = int(sum(1 for blocks in self._mlp_hot_blocks_by_layer.values() if int(blocks.numel()) > 0))
        kv_hot_layers = int(sum(1 for blocks in self._kv_hot_blocks_by_layer.values() if int(blocks.numel()) > 0))
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "num_layers": int(self.num_layers),
            "triton_sparse_mlp_requested": bool(self._triton_sparse_mlp_requested),
            "triton_sparse_mlp_available": bool(triton_sparse_mlp_available()),
            "triton_sparse_mlp_enabled": bool(self._triton_fused_sparse_mlp),
            "decode_backend": str(getattr(self, "_decode_backend_name", "dequant_sparse_fallback")),
            "lm_head_enabled": bool(lm_head_status.get("enabled", False)),
            "lm_head_mode": str(lm_head_status.get("mode", "disabled")),
            "lm_head_on_gpu": bool(lm_head_status.get("on_gpu", False)),
            "lm_head_dtype": lm_head_status.get("dtype"),
            "lm_head_gpu_attempted": bool(lm_head_status.get("gpu_attempted", False)),
            "lm_head_gpu_preferred": bool(lm_head_status.get("gpu_preferred", False)),
            "lm_head_last_failure": (
                None
                if not str(lm_head_status.get("last_failure", "") or "").strip()
                else str(lm_head_status.get("last_failure"))
            ),
            "lm_head_weight_gb": float(lm_head_status.get("weight_gb", 0.0) or 0.0),
            "sparse_mlp_layers": int(sparse_summary.get("layers", 0)),
            "sparse_mlp_top_k": int(getattr(self, "_sparse_top_k", 0) or 0),
            "sparse_mlp_prefill_mode": str(getattr(self, "_sparse_mlp_prefill_mode", "dense")),
            "sparse_mlp_execution_request": str(getattr(self, "_sparse_mlp_execution_request", "auto")),
            "sparse_mlp_execution_counts": dict(sparse_summary.get("execution_counts", {})),
            "vram_hot_cache_enabled": bool(self._vram_hot_cache_enabled),
            "vram_hot_cache_live_calibrated": bool(self._vram_hot_cache_live_calibrated),
            "vram_hot_cache_used_gb": float(self._vram_hot_cache_used_bytes) / float(1024 ** 3),
            "vram_hot_cache_limit_gb": (
                float(self._vram_hot_cache_limit_bytes) / float(1024 ** 3)
                if self._vram_hot_cache_limit_bytes is not None
                else 0.0
            ),
            "hot_cache_mlp_layers": int(hot_layers),
            "hot_cache_kv_layers": int(kv_hot_layers),
            "last_hot_cache_calibration": dict(self._last_hot_cache_calibration or {}),
            "decode_mlp_hot_blocks_hit": int(getattr(self, "_decode_mlp_hot_blocks_hit", 0)),
            "decode_mlp_cold_blocks_streamed": int(getattr(self, "_decode_mlp_cold_blocks_streamed", 0)),
            "decode_down_hot_blocks_hit": int(getattr(self, "_decode_down_hot_blocks_hit", 0)),
            "decode_down_cold_blocks_streamed": int(getattr(self, "_decode_down_cold_blocks_streamed", 0)),
            "sparse_attention_layers": int(static_attn_layers),
            "compact_sparse_attention_decode": bool(getattr(self, "_compact_sparse_attn_decode", False)),
            "compact_sparse_attention_steps": int(getattr(self, "_compact_attn_decode_steps", 0)),
            "attn_backend_decode": (
                "compact_sparse_v1"
                if bool(getattr(self, "_compact_sparse_attn_decode", False)) and static_attn_layers > 0
                else "dense"
            ),
            "attn_active_heads_static": int(getattr(self, "_attn_active_heads", 0) or 0),
            "attn_runtime_head_stats": (
                {
                    "min": int(min(runtime_head_counts)),
                    "max": int(max(runtime_head_counts)),
                    "mean": float(sum(runtime_head_counts) / len(runtime_head_counts)),
                }
                if runtime_head_counts
                else {}
            ),
            "sparse_kv_layers": int(kv_layers),
            "sparse_kv_enabled_for_decode": bool(kv_layers > 0),
            "sparse_kv_prefill_mode": str(getattr(self, "_sparse_kv_prefill_mode", "dense")),
            "decode_profile_enabled": bool(self._decode_profile_enabled),
            "decode_profile_steps_recorded": (
                int(decode_profile_report.get("steps_recorded", 0))
                if decode_profile_report is not None
                else 0
            ),
            "decode_profile_summary": (
                dict(decode_profile_report.get("summary", {}))
                if decode_profile_report is not None
                else {}
            ),
            "runtime_events": self.get_runtime_events(),
        }

    def validate_throughput_probe(self) -> list[str]:
        status = self.get_runtime_status()
        errors: list[str] = []
        if int(status.get("sparse_mlp_layers", 0)) != int(self.num_layers):
            errors.append(
                f"sparse MLP routing covers {int(status.get('sparse_mlp_layers', 0))}/{int(self.num_layers)} layers"
            )
        if int(status.get("sparse_mlp_top_k", 0)) <= 0:
            errors.append("sparse-top-k is not active")
        if not bool(status.get("triton_sparse_mlp_enabled", False)):
            errors.append("Triton sparse MLP fast path is disabled")
        if str(status.get("decode_backend", "")) != "single_kernel_sparse_decode_sm75":
            errors.append(
                f"decode backend is {status.get('decode_backend')!r}, expected 'single_kernel_sparse_decode_sm75'"
            )
        if not bool(status.get("vram_hot_cache_enabled", False)):
            errors.append("VRAM hot cache is disabled")
        if str(status.get("sparse_mlp_prefill_mode", "dense")) != "hot_cache":
            errors.append(f"sparse MLP prefill mode is {status.get('sparse_mlp_prefill_mode')!r}, expected 'hot_cache'")
        if str(status.get("lm_head_mode", "")) != "gpu_nf4":
            errors.append(f"lm_head_mode is {status.get('lm_head_mode')!r}, expected 'gpu_nf4'")
        if not bool(status.get("lm_head_on_gpu", False)):
            errors.append("LM head is not on GPU")
        if not bool(status.get("sparse_kv_enabled_for_decode", False)):
            errors.append("sparse K/V routing is disabled")
        if str(status.get("sparse_kv_prefill_mode", "dense")) != "sparse":
            errors.append(f"sparse K/V prefill mode is {status.get('sparse_kv_prefill_mode')!r}, expected 'sparse'")
        if int(status.get("sparse_attention_layers", 0)) != int(self.num_layers):
            errors.append(
                f"sparse attention covers {int(status.get('sparse_attention_layers', 0))}/{int(self.num_layers)} layers"
            )
        if int(status.get("attn_active_heads_static", 0)) <= 0:
            errors.append("attn-active-heads is not active")
        if str(status.get("attn_backend_decode", "")) != "compact_sparse_v1":
            errors.append(
                f"attn backend is {status.get('attn_backend_decode')!r}, expected 'compact_sparse_v1'"
            )
        if int(status.get("compact_sparse_attention_steps", 0)) <= 0:
            errors.append("compact sparse attention decode steps did not advance")
        return errors

    def _schedule_prefetch_layer(self, layer_idx: int) -> None:
        if self._prefetch_executor is None:
            return
        if not self.loader._ram_cache_enabled:
            return
        if not self._layer_param_keys:
            return
        idx = int(layer_idx)
        if idx < 0 or idx >= self.num_layers:
            return
        with self._prefetch_lock:
            if idx in self._prefetch_pending_layers:
                return
            self._prefetch_pending_layers.add(idx)
        self._prefetch_executor.submit(self._prefetch_layer, idx)

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Warm the RAM cache for *layer_idx* from disk on a background thread.

        Only loads parameters that are not yet cached — safe to call concurrently
        with the GPU forward pass because it touches only CPU memory and the
        thread-safe RAM cache dict (protected by _ram_cache_lock).
        """
        try:
            if not self.loader._ram_cache_enabled:
                return
            if not self._layer_param_keys:
                return
            prefix = f"model.layers.{int(layer_idx)}."
            for k in self._layer_param_keys:
                full_name = f"{prefix}{k}"
                with self.loader._ram_cache_lock:
                    already = full_name in self.loader._ram_cache
                if not already:
                    with contextlib.suppress(Exception):
                        self.loader._load_raw_for_param(full_name)
        finally:
            with self._prefetch_lock:
                self._prefetch_pending_layers.discard(int(layer_idx))

    @staticmethod
    def _coerce_block_score_vector(value: Any, *, num_blocks: int) -> torch.Tensor | None:
        if value is None:
            return None
        vec: torch.Tensor | None = None
        if torch.is_tensor(value):
            vec = value.detach().flatten().to(dtype=torch.float32, device="cpu")
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return None
            vec = torch.as_tensor(value, dtype=torch.float32).flatten().to(device="cpu")
        elif isinstance(value, dict):
            for key in (
                "block_importance",
                "block_importance_probs",
                "_block_importance_probs",
                "block_scores",
                "importance",
                "scores",
            ):
                if key in value:
                    nested = StreamingLlamaRuntime._coerce_block_score_vector(value[key], num_blocks=num_blocks)
                    if nested is not None:
                        return nested
            tmp = torch.zeros((num_blocks,), dtype=torch.float32)
            found = False
            for key, raw_val in value.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                if not (0 <= idx < num_blocks):
                    continue
                try:
                    tmp[idx] = float(raw_val)
                    found = True
                except Exception:
                    continue
            if found:
                vec = tmp
        if vec is None:
            return None
        if vec.numel() >= num_blocks:
            return vec[:num_blocks].contiguous()
        padded = torch.zeros((num_blocks,), dtype=torch.float32)
        padded[: int(vec.numel())] = vec
        return padded

    def _derive_hot_blocks_for_layer(
        self,
        *,
        layer_state: dict[str, Any],
        layer_stats: Any,
        block_importance_by_layer: Any,
        layer_num_blocks: int,
    ) -> torch.Tensor | None:
        candidates = [
            layer_state.get("block_importance"),
            layer_state.get("block_importance_probs"),
            layer_state.get("_block_importance_probs"),
            layer_stats,
            block_importance_by_layer,
        ]
        scores: torch.Tensor | None = None
        for candidate in candidates:
            vec = self._coerce_block_score_vector(candidate, num_blocks=layer_num_blocks)
            if vec is not None and int(vec.numel()) == int(layer_num_blocks):
                scores = vec
                break

        if scores is None:
            decoder_bias = layer_state.get("decoder_bias")
            if torch.is_tensor(decoder_bias) and decoder_bias.ndim == 2:
                scores = decoder_bias.detach().abs().mean(dim=-1).to(dtype=torch.float32, device="cpu")
            else:
                decoder_blocks = layer_state.get("decoder_blocks")
                if torch.is_tensor(decoder_blocks) and decoder_blocks.ndim >= 2:
                    reduce_dims = tuple(range(1, decoder_blocks.ndim))
                    scores = decoder_blocks.detach().abs().mean(dim=reduce_dims).to(dtype=torch.float32, device="cpu")

        if scores is None or int(scores.numel()) == 0:
            return None

        scores = scores[:layer_num_blocks].to(dtype=torch.float32, device="cpu").contiguous()
        s_max = float(scores.max().item()) if scores.numel() > 0 else 0.0
        layer_idx = int(layer_state.get("layer_idx", -1))
        runtime_top_k = int(
            layer_state.get(
                "top_k",
                self._sparse_runtime_top_k if int(self._sparse_runtime_top_k) > 0 else self._sparse_top_k,
            )
        )
        if s_max <= 0.0:
            hot_count = self._target_hot_blocks_for_layer(
                layer_idx=int(layer_idx),
                runtime_top_k=int(runtime_top_k),
                count_slots=int(layer_num_blocks),
                previous_hot_count=0,
            )
            return torch.arange(hot_count, dtype=torch.long)

        norm_scores = scores / max(s_max, 1e-8)
        target_hot_count = self._target_hot_blocks_for_layer(
            layer_idx=int(layer_idx),
            runtime_top_k=int(runtime_top_k),
            count_slots=int(layer_num_blocks),
            previous_hot_count=0,
        )
        hot_idx = torch.nonzero(norm_scores >= self._hot_block_threshold, as_tuple=False).flatten().to(dtype=torch.long)
        if int(hot_idx.numel()) != int(target_hot_count):
            hot_idx = torch.topk(norm_scores, k=target_hot_count, largest=True).indices.to(dtype=torch.long)
        return hot_idx.sort().values.contiguous()

    def _order_blocks_for_layer_hot_cache(self, layer_idx: int, cpu_blocks: torch.Tensor) -> torch.Tensor:
        hot = self._mlp_hot_blocks_by_layer.get(int(layer_idx))
        if hot is None or int(hot.numel()) == 0 or int(cpu_blocks.numel()) <= 1:
            return cpu_blocks
        is_hot = torch.isin(cpu_blocks, hot.to(dtype=cpu_blocks.dtype, device=cpu_blocks.device))
        return torch.cat([cpu_blocks[is_hot], cpu_blocks[~is_hot]])

    def _disable_vram_hot_cache(self, reason: str) -> None:
        self._vram_hot_cache_enabled = False
        self._vram_hot_cache_disable_reason = str(reason)
        self._vram_hot_cache_limit_bytes = 0
        self._vram_hot_cache_used_bytes = 0
        self._vram_nf4_cache.clear()
        self._attn_hot_head_cache.clear()
        for _param in self._sparse_param_cache.values():
            if isinstance(_param, dict):
                _param.pop("vram_hot", None)
                _param.pop("vram_hot_down", None)
        if self.device.type == "cuda":
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()
        if not self._vram_hot_cache_oom_warned:
            self._vram_hot_cache_oom_warned = True
            print(
                f"[sparse] VRAM hot-cache disabled ({self._vram_hot_cache_disable_reason}); "
                "continuing with direct RAM->GPU sparse block streaming.",
                flush=True,
            )

    def pre_warm_vram_hot_cache(self) -> None:
        """Load top-importance MLP blocks, sparse K/V hot-blocks, and sparse Q/O head slices
        into VRAM before inference.  Paid once at startup; all three tiers use the same budget.
        """
        if not self._vram_hot_cache_enabled:
            print("[pre-warm] VRAM hot-cache is disabled; skipping.", flush=True)
            return
        if not self._mlp_hot_blocks_by_layer:
            print("[pre-warm] No hot-block index loaded (sparse basis not set?); skipping.", flush=True)
            return

        prev_phase = self._traffic_current_phase
        self._traffic_current_phase = "decode"
        cached_layers = 0
        try:
            # --- Tier 1: MLP gate / up / down hot blocks ---
            for layer_idx in range(self.num_layers):
                prefix = f"model.layers.{layer_idx}.mlp."
                new_this_layer = False
                for proj in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
                    full_name = f"{prefix}{proj}"
                    if full_name not in self._vram_nf4_cache:
                        self._get_sparse_4bit_param(full_name, store_raw_in_ram_cache=True)
                        if full_name in self._vram_nf4_cache:
                            new_this_layer = True
                if new_this_layer:
                    cached_layers += 1
                print(
                    f"\r[pre-warm] {layer_idx + 1}/{self.num_layers} layers "
                    f"| VRAM used: {self._vram_hot_cache_used_bytes / (1024 ** 3):.2f} GB",
                    end="",
                    flush=True,
                )
            print(
                f"\n[pre-warm] complete — {cached_layers} MLP layers cached, "
                f"VRAM hot-cache: {self._vram_hot_cache_used_bytes / (1024 ** 3):.2f} GB",
                flush=True,
            )

            # --- Tier 2: sparse K/V projection hot column-blocks ---
            if self._kv_hot_blocks_by_layer and self._vram_hot_cache_enabled:
                kv_cached = 0
                for layer_idx in range(self.num_layers):
                    attn_prefix = f"model.layers.{layer_idx}.self_attn."
                    for proj in ("k_proj.weight", "v_proj.weight"):
                        full_name = f"{attn_prefix}{proj}"
                        if full_name not in self._vram_nf4_cache:
                            with contextlib.suppress(Exception):
                                self._get_sparse_4bit_kv_param(full_name)
                                if full_name in self._vram_nf4_cache:
                                    kv_cached += 1
                if kv_cached > 0:
                    print(
                        f"[pre-warm] cached hot K/V blocks for {kv_cached} projections "
                        f"| VRAM used: {self._vram_hot_cache_used_bytes / (1024 ** 3):.2f} GB",
                        flush=True,
                    )

            # --- Tier 3: sparse Q/O head slices ---
            if self._attn_active_head_indices and self._vram_hot_cache_enabled:
                head_dim = int(
                    getattr(self.config, "head_dim", None)
                    or (
                        int(getattr(self.config, "hidden_size", 4096))
                        // int(getattr(self.config, "num_attention_heads", 32))
                    )
                )
                qo_cached = 0
                for layer_idx in range(self.num_layers):
                    if int(layer_idx) in self._attn_hot_head_cache:
                        continue
                    attn_prefix = f"model.layers.{layer_idx}.self_attn."
                    q_name = f"{attn_prefix}q_proj.weight"
                    o_name = f"{attn_prefix}o_proj.weight"
                    with contextlib.suppress(Exception):
                        meta_q = self._get_sparse_4bit_attn_meta(q_name, head_dim=head_dim)
                        meta_o = self._get_sparse_4bit_attn_meta(o_name, head_dim=head_dim)
                        result = self._maybe_cache_sparse_attn_hot_heads(
                            layer_idx=layer_idx,
                            q_name=q_name,
                            o_name=o_name,
                            meta_q=meta_q,
                            meta_o=meta_o,
                        )
                        if result is not None:
                            qo_cached += 1
                if qo_cached > 0:
                    print(
                        f"[pre-warm] cached sparse Q/O heads for {qo_cached} layers "
                        f"| VRAM used: {self._vram_hot_cache_used_bytes / (1024 ** 3):.2f} GB",
                        flush=True,
                    )
        finally:
            self._traffic_current_phase = prev_phase

    def _clear_vram_hot_cache_entries(self) -> None:
        self._vram_hot_cache_used_bytes = 0
        self._vram_hot_cache_pressure_warned = False
        self._vram_nf4_cache.clear()
        self._attn_hot_head_cache.clear()
        for param in self._sparse_param_cache.values():
            if isinstance(param, dict):
                param.pop("vram_hot", None)
                param.pop("vram_hot_down", None)
        for param in self._kv_sparse_param_cache.values():
            if isinstance(param, dict):
                param.pop("vram_hot_kv", None)
        if self.device.type == "cuda":
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    def _record_hot_cache_calibration_blocks(self, layer_idx: int, active_blocks: torch.Tensor) -> None:
        if not self._hot_cache_calibration_active:
            return
        routing = self._sparse_routing.get(int(layer_idx))
        if routing is None:
            return
        num_blocks = int(routing.get("num_blocks", self._sparse_num_blocks))
        if num_blocks <= 0:
            return
        blocks_cpu = active_blocks.detach().to(device=torch.device("cpu"), dtype=torch.long)
        if int(blocks_cpu.numel()) == 0:
            return
        counts = self._hot_cache_calibration_hits.get(int(layer_idx))
        if counts is None or int(counts.numel()) != int(num_blocks):
            counts = torch.zeros(int(num_blocks), dtype=torch.float32)
            self._hot_cache_calibration_hits[int(layer_idx)] = counts
        rows = blocks_cpu.reshape(-1, int(blocks_cpu.shape[-1]) if int(blocks_cpu.ndim) > 0 else 1)
        num_rows = int(rows.shape[0])
        if num_rows <= 0:
            return
        row_positions = torch.arange(1, num_rows + 1, dtype=torch.float32)
        row_weights = torch.pow(
            row_positions / float(max(num_rows, 1)),
            float(self._hot_cache_calibration_recency_power),
        )
        for row_idx in range(num_rows):
            row_blocks = rows[row_idx]
            row_blocks = row_blocks[(row_blocks >= 0) & (row_blocks < int(num_blocks))]
            if int(row_blocks.numel()) <= 0:
                continue
            counts.index_add_(
                0,
                row_blocks,
                torch.full(
                    (int(row_blocks.numel()),),
                    float(row_weights[row_idx].item()),
                    dtype=torch.float32,
                ),
            )

    def _estimate_mlp_hot_cache_bytes_per_block(self, layer_idx: int) -> int:
        # Read shape metadata from already-loaded cache entries ONLY — never call
        # _get_sparse_4bit_param here, which would trigger _maybe_cache_sparse_param_hot_blocks
        # on a param that is currently mid-load, causing infinite recursion and massive VRAM
        # over-allocation (each stack frame allocates a new ~16 MB GPU buffer).
        prefix = f"model.layers.{int(layer_idx)}.mlp."
        block_size = int(max(1, self._sparse_block_size))
        total = 0
        try:
            for proj in ("gate_proj.weight", "up_proj.weight"):
                meta = self._sparse_param_cache.get(f"{prefix}{proj}")
                if meta is not None:
                    total += int(meta.get("bytes_per_block", 0))
                    total += int(meta.get("absmax_per_block", 0)) * 4  # float32 absmax
                else:
                    hidden = int(getattr(self.config, "hidden_size", 4096))
                    total += (hidden // 2) * block_size
                    total += (hidden // 64) * block_size * 4

            down_meta = self._sparse_param_cache.get(f"{prefix}down_proj.weight")
            if down_meta is not None:
                out_features = int(down_meta.get("out_features", 0))
            else:
                out_features = int(getattr(self.config, "hidden_size", 4096))
            bytes_per_cblk = max(1, block_size // 2)
            total += out_features * bytes_per_cblk + out_features * 4
        except Exception:
            total = 0
        return int(total if total > 0 else 1_048_576)

    def _target_hot_blocks_for_layer(
        self,
        *,
        layer_idx: int,
        runtime_top_k: int,
        count_slots: int,
        previous_hot_count: int,
    ) -> int:
        env_target = os.getenv("STREAMING_HOT_CACHE_TARGET_BLOCKS", "").strip()
        if env_target:
            try:
                requested = int(env_target)
                if requested > 0:
                    return max(1, min(int(requested), int(count_slots)))
            except ValueError:
                pass

        fallback_count = int(previous_hot_count)
        if fallback_count <= 0:
            fallback_count = int(max(1, runtime_top_k))

        limit_bytes = getattr(self, "_vram_hot_cache_limit_bytes", None)
        if limit_bytes is None or int(limit_bytes) <= 0:
            return max(1, min(int(fallback_count), int(count_slots)))

        sparse_layers = max(1, len(self._sparse_routing) or int(self.num_layers))
        bytes_per_block = max(1, self._estimate_mlp_hot_cache_bytes_per_block(int(layer_idx)))
        remaining_budget = max(int(limit_bytes) - int(self._vram_hot_cache_used_bytes), 0)
        remaining_layers = max(1, sparse_layers - max(0, int(layer_idx)))
        budget_per_layer = int((float(remaining_budget) * 0.97) / float(remaining_layers))
        budget_count = int(budget_per_layer // bytes_per_block)
        target = max(int(fallback_count), int(budget_count))
        env_cap = os.getenv("STREAMING_HOT_CACHE_MAX_BLOCKS", "").strip()
        if env_cap:
            with contextlib.suppress(ValueError):
                target = min(int(target), int(env_cap))
        return max(1, min(int(target), int(count_slots)))

    def profile_mlp_contributions(
        self,
        token_ids_list: list[torch.LongTensor],
        *,
        threshold: float = 0.005,
        save_path: str | None = None,
    ) -> torch.Tensor:
        """Run sparse MLP forward on sample prompts; mark low-norm layers for static skipping.

        Returns a bool tensor of shape [num_layers] (True = skip that layer's MLP).
        Optionally saves to `save_path` as {"skip_mask": tensor}.
        """
        scores = torch.zeros(self.num_layers, dtype=torch.float32)
        counts = torch.zeros(self.num_layers, dtype=torch.int32)

        import types as _types

        _prev_phase = self._traffic_current_phase
        _prev_prefill_mode = self._sparse_mlp_prefill_mode
        _orig_dispatch = self._mlp_forward_dispatch.__func__

        def _recording_dispatch(self_inner, layer_idx, layer, mlp_input):
            out = _orig_dispatch(self_inner, layer_idx, layer, mlp_input)
            lidx = int(layer_idx)
            in_norm  = float(mlp_input.float().norm().item())
            out_norm = float(out.float().norm().item())
            scores[lidx] += out_norm / max(in_norm, 1e-8)
            counts[lidx] += 1
            return out

        self._mlp_forward_dispatch = _types.MethodType(_recording_dispatch, self)
        try:
            for token_ids in token_ids_list:
                ids = token_ids.to(device=torch.device("cpu"), dtype=torch.long)
                if ids.ndim == 1:
                    ids = ids.unsqueeze(0)
                self.reset_caches()
                self._set_traffic_phase("prefill")
                self._sparse_mlp_prefill_mode = "sparse"
                with torch.no_grad():
                    self._forward_prefill(ids.to(device=self.device), compute_logits=False)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
        finally:
            self._mlp_forward_dispatch = _types.MethodType(_orig_dispatch, self)
            self._sparse_mlp_prefill_mode = _prev_prefill_mode
            self._set_traffic_phase(_prev_phase or "idle")
            self.reset_caches()

        mean_scores = scores / counts.float().clamp(min=1)
        skip_mask = mean_scores < threshold
        n_skip = int(skip_mask.sum().item())
        print(
            f"[mlp-profile] threshold={threshold}: {n_skip}/{self.num_layers} layers qualify for static skip",
            flush=True,
        )
        if save_path:
            torch.save({"skip_mask": skip_mask, "scores": mean_scores}, save_path)
            print(f"[mlp-profile] saved to {save_path}", flush=True)
        return skip_mask

    def calibrate_vram_hot_cache(
        self,
        token_ids: torch.LongTensor,
        *,
        max_tokens: int = 64,
        rebuild_cache: bool = True,
        generate_decode_tokens: int = 0,
    ) -> dict[str, Any]:
        if not self._vram_hot_cache_enabled:
            print("[hot-cache-calibration] VRAM hot-cache is disabled; skipping.", flush=True)
            return {"updated_layers": 0, "tokens": 0}
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
            raise RuntimeError("Hot-cache calibration currently supports batch_size=1 only")
        if not self._sparse_routing:
            print("[hot-cache-calibration] No sparse routing is loaded; skipping.", flush=True)
            return {"updated_layers": 0, "tokens": 0}

        token_count = min(max(1, int(max_tokens)), int(token_ids.shape[1]))
        calibration_ids = token_ids[:, -token_count:].detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous()
        n_decode = max(0, int(generate_decode_tokens))
        print(
            f"[hot-cache-calibration] collecting live MLP routes from {token_count} prompt token(s)"
            + (f" + {n_decode} decode token(s)" if n_decode > 0 else ""),
            flush=True,
        )

        prev_phase = self._traffic_current_phase
        prev_prefill_mode = self._sparse_mlp_prefill_mode
        prev_active = self._hot_cache_calibration_active
        self._hot_cache_calibration_hits = {}
        self._hot_cache_calibration_active = True
        try:
            self.reset_caches()
            # Build KV cache token-by-token using the decode path (seq_len=1 per call).
            # _forward_prefill with _hot_cache_calibration_active=True falls through to
            # _sparse_mlp_forward_fast with seq_len=42 which crashes the Triton kernel
            # (only supports seq_len=1). Dense mode also fails — buffer allocated for sparse
            # blocks, not full weights. Token-by-token decode is safe and produces identical
            # KV cache state via causal attention. Hits from this phase are discarded below.
            self._set_traffic_phase("decode")
            calib_ids_gpu = calibration_ids.to(self.device)
            logits_calib = None
            with torch.no_grad():
                for _tok_pos in range(int(calib_ids_gpu.shape[1])):
                    _tok = calib_ids_gpu[:, _tok_pos : _tok_pos + 1]
                    logits_calib, _ = self.forward_token(_tok, position_index=_tok_pos)
            if n_decode > 0 and logits_calib is not None:
                self._set_traffic_phase("decode")
                # Discard prefill activations so the rebuild is based purely on decode routing.
                # Prefill uses a batched context (different block patterns from decode) and
                # previously dominated the hit counts (74% prefill vs 26% decode).
                self._hot_cache_calibration_hits = {}
                self._decode_mlp_hot_blocks_hit = getattr(self, "_decode_mlp_hot_blocks_hit", 0)
                self._decode_mlp_cold_blocks_streamed = getattr(self, "_decode_mlp_cold_blocks_streamed", 0)
                self._decode_down_hot_blocks_hit = getattr(self, "_decode_down_hot_blocks_hit", 0)
                self._decode_down_cold_blocks_streamed = getattr(self, "_decode_down_cold_blocks_streamed", 0)
                seq_gpu = calibration_ids.to(self.device)
                with torch.no_grad():
                    for _d in range(n_decode):
                        next_tok = self._sample_next_token(
                            logits_calib[:, -1, :],
                            do_sample=False, temperature=1.0, top_k=None, top_p=1.0,
                        ).view(1, 1).to(device=self.device)
                        pos = int(seq_gpu.shape[1])
                        seq_gpu = torch.cat([seq_gpu, next_tok], dim=-1)
                        logits_calib, _ = self.forward_token(next_tok, position_index=pos)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        finally:
            self._sparse_mlp_prefill_mode = prev_prefill_mode
            self._hot_cache_calibration_active = prev_active
            self.reset_caches()
            self._set_traffic_phase(prev_phase)

        updated_layers = 0
        total_hits = 0.0
        total_selected_blocks = 0
        for layer_idx, counts in sorted(self._hot_cache_calibration_hits.items()):
            routing = self._sparse_routing.get(int(layer_idx))
            if routing is None:
                continue
            hit_count = float(counts.sum().item())
            if hit_count <= 0.0:
                continue
            runtime_top_k = int(
                routing.get(
                    "top_k",
                    self._sparse_runtime_top_k if int(self._sparse_runtime_top_k) > 0 else self._sparse_top_k,
                )
            )
            previous_hot_blocks = self._mlp_hot_blocks_by_layer.get(int(layer_idx))
            previous_hot_count = int(previous_hot_blocks.numel()) if previous_hot_blocks is not None else 0
            target_hot_count = self._target_hot_blocks_for_layer(
                layer_idx=int(layer_idx),
                runtime_top_k=int(runtime_top_k),
                count_slots=int(counts.numel()),
                previous_hot_count=int(previous_hot_count),
            )
            hot_blocks = torch.topk(counts, k=target_hot_count, largest=True).indices.to(
                dtype=torch.long,
                device=torch.device("cpu"),
            ).contiguous()
            self._mlp_hot_blocks_by_layer[int(layer_idx)] = hot_blocks
            updated_layers += 1
            total_hits += float(hit_count)
            total_selected_blocks += int(hot_blocks.numel())

        print(
            "[hot-cache-calibration] updated "
            f"{updated_layers}/{len(self._sparse_routing)} layers "
            f"| hits={total_hits:.1f} | cached_blocks={total_selected_blocks}",
            flush=True,
        )

        if bool(rebuild_cache) and updated_layers > 0:
            print("[hot-cache-calibration] rebuilding VRAM hot-cache from live routes", flush=True)
            self._clear_vram_hot_cache_entries()
            self.pre_warm_vram_hot_cache()
        if updated_layers > 0:
            self._vram_hot_cache_live_calibrated = True
        result = {
            "tokens": int(token_count),
            "updated_layers": int(updated_layers),
            "hits": float(total_hits),
            "cached_blocks": int(total_selected_blocks),
            "vram_hot_cache_gb": float(self._vram_hot_cache_used_bytes) / float(1024 ** 3),
        }
        self._last_hot_cache_calibration = dict(result)
        return result

    def _can_reserve_vram_hot_cache(self, required_bytes: int) -> bool:
        req = int(max(required_bytes, 0))
        if req <= 0:
            return True

        hard_limit = self._vram_hot_cache_limit_bytes
        if hard_limit is not None and self._vram_hot_cache_used_bytes + req > hard_limit:
            return False

        if self.device.type != "cuda":
            return True

        try:
            sys_free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
            # On Windows WDDM, mem_get_info() can report stale "used" VRAM for
            # several seconds after a large allocation is freed (async WDDM dealloc).
            # Use PyTorch's own allocator bookkeeping as a more accurate estimate:
            # total - reserved gives what PyTorch hasn't claimed, which is immediately
            # updated after empty_cache() whereas WDDM lags behind.
            pytorch_reserved = torch.cuda.memory_reserved(self.device)
            allocator_free_bytes = max(0, int(total_bytes) - int(pytorch_reserved))
            free_bytes = max(int(sys_free_bytes), int(allocator_free_bytes))
        except Exception:
            return True

        dynamic_limit = self._vram_hot_cache_used_bytes + max(
            int(free_bytes) - int(self._vram_hot_cache_margin_bytes), 0
        )
        effective_limit = dynamic_limit
        if hard_limit is not None:
            effective_limit = min(int(hard_limit), int(dynamic_limit))

        if self._vram_hot_cache_used_bytes + req <= effective_limit:
            return True

        if not self._vram_hot_cache_pressure_warned:
            self._vram_hot_cache_pressure_warned = True
            print(
                "[sparse] VRAM hot-cache auto-clamp engaged: "
                f"used={self._vram_hot_cache_used_bytes / (1024 ** 3):.2f} GB, "
                f"sys_free={int(sys_free_bytes) / (1024 ** 3):.2f} GB, "
                f"allocator_free={int(allocator_free_bytes) / (1024 ** 3):.2f} GB, "
                f"margin={self._vram_hot_cache_margin_bytes / (1024 ** 3):.2f} GB.",
                flush=True,
            )
        return False

    @staticmethod
    def _plan_fixed_capacity_bank_update(
        *,
        existing_block_ids_cpu: torch.Tensor,
        existing_scores_cpu: torch.Tensor,
        active_count: int,
        capacity: int,
        novel_blocks_cpu: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        capacity_i = max(0, int(capacity))
        active_i = max(0, min(int(active_count), int(existing_block_ids_cpu.numel()), capacity_i))
        if capacity_i <= 0:
            return (
                torch.empty((0,), dtype=torch.long),
                existing_block_ids_cpu.clone(),
                existing_scores_cpu.clone(),
                active_i,
            )
        block_ids = existing_block_ids_cpu.clone()
        scores = existing_scores_cpu.clone()
        novel = novel_blocks_cpu.detach().to(device=torch.device("cpu"), dtype=torch.long).reshape(-1)
        if int(novel.numel()) <= 0:
            return torch.empty((0,), dtype=torch.long), block_ids, scores, active_i
        if int(novel.numel()) > capacity_i:
            novel = novel[:capacity_i].contiguous()

        slots: list[int] = []
        current_scores = scores.clone()
        base_score = float(current_scores[:active_i].max().item()) + 1.0 if active_i > 0 else 1.0
        next_free_slot = active_i
        for idx in range(int(novel.numel())):
            if next_free_slot < capacity_i:
                slot = int(next_free_slot)
                next_free_slot += 1
                active_i = max(active_i, slot + 1)
            else:
                slot = int(torch.argmin(current_scores[:capacity_i]).item())
            block_ids[slot] = int(novel[idx].item())
            current_scores[slot] = base_score + float(idx + 1) * 1e-3
            slots.append(slot)
        scores.copy_(current_scores)
        return torch.tensor(slots, dtype=torch.long), block_ids, scores, active_i

    def _maybe_cache_down_proj_hot_columns(
        self,
        *,
        full_name: str,
        param: dict[str, Any],
        hot_blocks: torch.Tensor,
    ) -> bool:
        block_size = int(self._sparse_block_size)
        in_features = int(param["in_features"])
        out_features = int(param["out_features"])
        quant_block_size = int(param["quant_block_size"])
        total_col_blocks = in_features // max(block_size, 1)
        hot_blocks = hot_blocks[(hot_blocks >= 0) & (hot_blocks < total_col_blocks)].to(dtype=torch.long, device="cpu")
        hot_blocks = hot_blocks.unique(sorted=True)
        if int(hot_blocks.numel()) == 0:
            return False

        layer_idx = -1
        parts = str(full_name).split(".")
        if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
            with contextlib.suppress(Exception):
                layer_idx = int(parts[2])
        routing = self._sparse_routing.get(int(layer_idx))
        runtime_top_k = int(
            routing.get(
                "top_k",
                self._sparse_runtime_top_k if int(self._sparse_runtime_top_k) > 0 else self._sparse_top_k,
            )
        ) if isinstance(routing, dict) else int(self._sparse_runtime_top_k if int(self._sparse_runtime_top_k) > 0 else self._sparse_top_k)
        target_capacity = self._target_hot_blocks_for_layer(
            layer_idx=int(layer_idx),
            runtime_top_k=int(runtime_top_k),
            count_slots=int(total_col_blocks),
            previous_hot_count=int(hot_blocks.numel()),
        )
        if int(hot_blocks.numel()) > int(target_capacity):
            hot_blocks = hot_blocks[: int(target_capacity)].contiguous()

        bytes_per_row = in_features // 2
        bytes_per_cblk = block_size // 2
        absmax_per_row = in_features // max(quant_block_size, 1)
        required_bytes = int(out_features * int(target_capacity) * bytes_per_cblk)
        required_bytes += int(out_features * int(target_capacity) * 4)
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return False

        raw_2d = param["packed_weight"].view(out_features, bytes_per_row)
        col_starts = hot_blocks * bytes_per_cblk
        col_range = torch.arange(bytes_per_cblk, dtype=torch.long)
        col_idx = (col_starts.unsqueeze(-1) + col_range.unsqueeze(0)).reshape(-1)
        packed_cols_cpu = raw_2d[:, col_idx].reshape(out_features, int(hot_blocks.numel()), bytes_per_cblk).contiguous()

        absmax_2d = param["absmax"].view(out_features, absmax_per_row)
        abs_idx = (hot_blocks * block_size) // quant_block_size
        absmax_cols_cpu = absmax_2d[:, abs_idx].contiguous()

        try:
            packed_cols_seed_gpu = self._copy_cpu_to_gpu(packed_cols_cpu, dtype=torch.uint8)
            absmax_cols_seed_gpu = self._copy_cpu_to_gpu(absmax_cols_cpu, dtype=torch.float32)
            packed_cols_gpu = torch.empty(
                (out_features, int(target_capacity), bytes_per_cblk),
                dtype=torch.uint8,
                device=self.device,
            )
            absmax_cols_gpu = torch.empty(
                (out_features, int(target_capacity)),
                dtype=torch.float32,
                device=self.device,
            )
            packed_cols_gpu[:, : int(hot_blocks.numel()), :].copy_(packed_cols_seed_gpu)
            absmax_cols_gpu[:, : int(hot_blocks.numel())].copy_(absmax_cols_seed_gpu)
        except Exception as _e:
            if _is_cuda_oom_error(_e):
                self._disable_vram_hot_cache("cuda_oom_during_down_hot_cache")
            return False

        lookup = torch.full((total_col_blocks,), -1, dtype=torch.int32)
        lookup[hot_blocks] = torch.arange(int(hot_blocks.numel()), dtype=torch.int32)
        block_ids_cpu = torch.full((int(target_capacity),), -1, dtype=torch.long)
        block_ids_cpu[: int(hot_blocks.numel())] = hot_blocks
        scores_cpu = torch.zeros((int(target_capacity),), dtype=torch.float32)
        if int(hot_blocks.numel()) > 0:
            scores_cpu[: int(hot_blocks.numel())] = torch.linspace(
                float(int(hot_blocks.numel())),
                1.0,
                steps=int(hot_blocks.numel()),
                dtype=torch.float32,
            )
        hot_cache = {
            "block_ids_cpu": block_ids_cpu,
            "lookup_cpu": lookup,
            "packed_cols_gpu": packed_cols_gpu,
            "absmax_cols_gpu": absmax_cols_gpu,
            "active_count": int(hot_blocks.numel()),
            "capacity_blocks": int(target_capacity),
            "scores_cpu": scores_cpu,
            "h2d_ready": not bool(self._enable_cuda_h2d_overlap and self._h2d_stream is not None),
        }
        param["vram_hot_down"] = hot_cache
        self._vram_nf4_cache[full_name] = hot_cache
        self._vram_hot_cache_used_bytes += required_bytes
        return True

    def _maybe_cache_sparse_param_hot_blocks(self, full_name: str, param: dict[str, Any]) -> None:
        if full_name in self._vram_nf4_cache:
            return
        traffic_phase = str(self._traffic_current_phase or "idle")
        if traffic_phase not in {"decode", "prefill"}:
            return
        if not self._vram_hot_cache_enabled:
            return
        if self.device.type != "cuda":
            return
        parts = str(full_name).split(".")
        if len(parts) < 4 or parts[0] != "model" or parts[1] != "layers":
            return
        try:
            layer_idx = int(parts[2])
        except Exception:
            return
        hot_blocks = self._mlp_hot_blocks_by_layer.get(layer_idx)


        if hot_blocks is not None:
            max_hot = int(max(1, self._sparse_top_k))
            if int(hot_blocks.numel()) > max_hot:
                hot_blocks = hot_blocks[:max_hot]
        if hot_blocks is None or int(hot_blocks.numel()) == 0:
            return
        if str(full_name).endswith(".mlp.down_proj.weight"):
            self._maybe_cache_down_proj_hot_columns(full_name=full_name, param=param, hot_blocks=hot_blocks)
            return

        total_blocks = int(param["packed_blocks"].shape[0])
        hot_blocks = hot_blocks[(hot_blocks >= 0) & (hot_blocks < total_blocks)].to(dtype=torch.long, device="cpu")
        hot_blocks = hot_blocks.unique(sorted=True)
        if int(hot_blocks.numel()) == 0:
            return

        packed_block_bytes = int(param["packed_blocks"].shape[1])
        absmax_block_bytes = int(param["absmax_blocks"].shape[1]) * 4
        required_bytes = int(hot_blocks.numel()) * (packed_block_bytes + absmax_block_bytes)
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return

        try:
            packed_hot_gpu = self._copy_cpu_to_gpu(
                param["packed_blocks"][hot_blocks].contiguous(), dtype=torch.uint8
            )
            absmax_hot_gpu = self._copy_cpu_to_gpu(
                param["absmax_blocks"][hot_blocks].contiguous(), dtype=torch.float32
            )
        except Exception as _e:
            if _is_cuda_oom_error(_e):
                self._disable_vram_hot_cache("cuda_oom_during_hot_block_cache")
            return
        lookup = torch.full((total_blocks,), -1, dtype=torch.int32)
        lookup[hot_blocks] = torch.arange(int(hot_blocks.numel()), dtype=torch.int32)
        hot_cache = {
            "block_ids_cpu": hot_blocks,
            "lookup_cpu": lookup,
            "packed_blocks_gpu": packed_hot_gpu,
            "absmax_blocks_gpu": absmax_hot_gpu,
            "h2d_ready": not bool(self._enable_cuda_h2d_overlap and self._h2d_stream is not None),
        }
        param["vram_hot"] = hot_cache
        self._vram_nf4_cache[full_name] = hot_cache
        self._vram_hot_cache_used_bytes += required_bytes

    def _merge_layer_hot_blocks(self, layer_idx: int | None, blocks: torch.Tensor) -> None:
        if layer_idx is None or int(blocks.numel()) <= 0:
            return
        layer_key = int(layer_idx)
        promoted = blocks.detach().to(device=torch.device("cpu"), dtype=torch.long).reshape(-1)
        promoted = promoted[promoted >= 0]
        if int(promoted.numel()) <= 0:
            return
        current = self._mlp_hot_blocks_by_layer.get(layer_key)
        if current is None or int(current.numel()) <= 0:
            self._mlp_hot_blocks_by_layer[layer_key] = promoted.unique(sorted=True).contiguous()
            return
        merged = torch.cat(
            [
                current.detach().to(device=torch.device("cpu"), dtype=torch.long).reshape(-1),
                promoted,
            ],
            dim=0,
        )
        self._mlp_hot_blocks_by_layer[layer_key] = merged.unique(sorted=True).contiguous()

    def _promote_sparse_param_hot_blocks_from_gpu(
        self,
        param: dict[str, Any],
        *,
        cold_blocks: torch.Tensor,
        cold_packed_gpu: torch.Tensor | None,
        cold_absmax_gpu: torch.Tensor | None,
        layer_idx: int | None,
    ) -> None:
        if str(self._traffic_current_phase or "idle") != "decode":
            return
        if not self._vram_hot_cache_enabled or self.device.type != "cuda":
            return
        if cold_packed_gpu is None or cold_absmax_gpu is None or int(cold_blocks.numel()) <= 0:
            return
        full_name = str(param.get("full_name", "")).strip()
        if not full_name or full_name.endswith(".mlp.down_proj.weight"):
            return
        total_blocks = int(param.get("blocks_per_col", 0))
        if total_blocks <= 0:
            return

        candidate_blocks = cold_blocks.detach().to(device=torch.device("cpu"), dtype=torch.long).reshape(-1)
        valid_mask = (candidate_blocks >= 0) & (candidate_blocks < total_blocks)
        if not bool(valid_mask.any()):
            return
        candidate_blocks = candidate_blocks[valid_mask]

        hot_cache = param.get("vram_hot")
        if hot_cache is not None:
            lookup = hot_cache["lookup_cpu"]
            novel_mask = lookup.index_select(0, candidate_blocks) < 0
            if not bool(novel_mask.any()):
                return
            novel_positions_cpu = torch.nonzero(valid_mask, as_tuple=False).flatten().index_select(
                0, torch.nonzero(novel_mask, as_tuple=False).flatten()
            )
            novel_blocks = candidate_blocks[novel_mask]
        else:
            novel_positions_cpu = torch.nonzero(valid_mask, as_tuple=False).flatten()
            novel_blocks = candidate_blocks

        if int(novel_blocks.numel()) <= 0:
            return
        novel_positions_gpu = novel_positions_cpu.to(device=self.device, dtype=torch.long)
        if int(novel_positions_cpu.numel()) == int(cold_blocks.numel()) and bool(valid_mask.all()):
            new_packed_gpu = cold_packed_gpu
            new_absmax_gpu = cold_absmax_gpu
        else:
            if self._h2d_stream is not None:
                self._wait_for_h2d_stream()
            new_packed_gpu = cold_packed_gpu.index_select(0, novel_positions_gpu).contiguous()
            new_absmax_gpu = cold_absmax_gpu.index_select(0, novel_positions_gpu).contiguous()

        required_bytes = int(new_packed_gpu.numel() * new_packed_gpu.element_size())
        required_bytes += int(new_absmax_gpu.numel() * new_absmax_gpu.element_size())
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return

        try:
            if hot_cache is not None:
                if not bool(hot_cache.get("h2d_ready", True)):
                    self._wait_for_h2d_stream()
                elif self._h2d_stream is not None:
                    self._wait_for_h2d_stream()
                old_blocks = hot_cache["block_ids_cpu"].to(dtype=torch.long, device=torch.device("cpu"))
                old_packed_gpu = hot_cache["packed_blocks_gpu"]
                old_absmax_gpu = hot_cache["absmax_blocks_gpu"]
                block_ids_cpu = torch.cat([old_blocks, novel_blocks], dim=0).contiguous()
                packed_blocks_gpu = torch.cat([old_packed_gpu, new_packed_gpu], dim=0).contiguous()
                absmax_blocks_gpu = torch.cat([old_absmax_gpu, new_absmax_gpu], dim=0).contiguous()
                h2d_ready = True
            else:
                block_ids_cpu = novel_blocks.contiguous()
                packed_blocks_gpu = new_packed_gpu.contiguous()
                absmax_blocks_gpu = new_absmax_gpu.contiguous()
                h2d_ready = not bool(self._enable_cuda_h2d_overlap and self._h2d_stream is not None)
        except Exception as exc:
            if _is_cuda_oom_error(exc):
                return
            raise

        lookup = torch.full((total_blocks,), -1, dtype=torch.int32)
        lookup[block_ids_cpu] = torch.arange(int(block_ids_cpu.numel()), dtype=torch.int32)
        updated_cache = {
            "block_ids_cpu": block_ids_cpu,
            "lookup_cpu": lookup,
            "packed_blocks_gpu": packed_blocks_gpu,
            "absmax_blocks_gpu": absmax_blocks_gpu,
            "h2d_ready": h2d_ready,
        }
        param["vram_hot"] = updated_cache
        self._vram_nf4_cache[full_name] = updated_cache
        cached_param = self._sparse_param_cache.get(full_name)
        if cached_param is not None:
            cached_param["vram_hot"] = updated_cache
        self._vram_hot_cache_used_bytes += required_bytes
        self._merge_layer_hot_blocks(layer_idx, novel_blocks)

    def _promote_down_proj_hot_columns_from_gpu(
        self,
        param: dict[str, Any],
        *,
        cold_blocks: torch.Tensor,
        cold_cols_packed_gpu: torch.Tensor | None,
        cold_cols_absmax_gpu: torch.Tensor | None,
        layer_idx: int | None,
    ) -> None:
        if str(self._traffic_current_phase or "idle") != "decode":
            return
        if not self._vram_hot_cache_enabled or self.device.type != "cuda":
            return
        if cold_cols_packed_gpu is None or cold_cols_absmax_gpu is None or int(cold_blocks.numel()) <= 0:
            return
        full_name = str(param.get("full_name", "")).strip()
        if not full_name:
            return
        block_size = int(self._sparse_block_size)
        total_col_blocks = int(param["in_features"]) // max(block_size, 1)
        if total_col_blocks <= 0:
            return

        candidate_blocks = cold_blocks.detach().to(device=torch.device("cpu"), dtype=torch.long).reshape(-1)
        valid_mask = (candidate_blocks >= 0) & (candidate_blocks < total_col_blocks)
        if not bool(valid_mask.any()):
            return
        candidate_blocks = candidate_blocks[valid_mask]

        hot_cache = param.get("vram_hot_down")
        if hot_cache is not None:
            lookup = hot_cache["lookup_cpu"]
            novel_mask = lookup.index_select(0, candidate_blocks) < 0
            if not bool(novel_mask.any()):
                return
            novel_positions_cpu = torch.nonzero(valid_mask, as_tuple=False).flatten().index_select(
                0, torch.nonzero(novel_mask, as_tuple=False).flatten()
            )
            novel_blocks = candidate_blocks[novel_mask]
        else:
            novel_positions_cpu = torch.nonzero(valid_mask, as_tuple=False).flatten()
            novel_blocks = candidate_blocks

        if int(novel_blocks.numel()) <= 0:
            return
        capacity = int(hot_cache.get("capacity_blocks", 0) if hot_cache is not None else 0)
        if capacity <= 0:
            return
        active_count = int(hot_cache.get("active_count", 0) if hot_cache is not None else 0)
        block_ids_cpu = hot_cache.get("block_ids_cpu")
        scores_cpu = hot_cache.get("scores_cpu")
        if not torch.is_tensor(block_ids_cpu) or int(block_ids_cpu.numel()) != capacity:
            block_ids_cpu = torch.full((capacity,), -1, dtype=torch.long)
        else:
            block_ids_cpu = block_ids_cpu.to(device=torch.device("cpu"), dtype=torch.long).contiguous()
        if not torch.is_tensor(scores_cpu) or int(scores_cpu.numel()) != capacity:
            scores_cpu = torch.zeros((capacity,), dtype=torch.float32)
        else:
            scores_cpu = scores_cpu.to(device=torch.device("cpu"), dtype=torch.float32).contiguous()
        slot_plan_cpu, updated_block_ids_cpu, updated_scores_cpu, updated_active_count = self._plan_fixed_capacity_bank_update(
            existing_block_ids_cpu=block_ids_cpu,
            existing_scores_cpu=scores_cpu,
            active_count=int(active_count),
            capacity=int(capacity),
            novel_blocks_cpu=novel_blocks,
        )
        if int(slot_plan_cpu.numel()) <= 0:
            return
        selected_novel_count = int(slot_plan_cpu.numel())
        selected_positions_cpu = novel_positions_cpu[:selected_novel_count].contiguous()
        selected_blocks_cpu = novel_blocks[:selected_novel_count].contiguous()
        selected_positions_gpu = selected_positions_cpu.to(device=self.device, dtype=torch.long)
        slot_plan_gpu = slot_plan_cpu.to(device=self.device, dtype=torch.long)
        if int(selected_positions_cpu.numel()) == int(cold_blocks.numel()) and bool(valid_mask.all()):
            new_cols_packed_gpu = cold_cols_packed_gpu.index_select(1, selected_positions_gpu).contiguous() if int(selected_positions_cpu.numel()) != int(cold_blocks.numel()) else cold_cols_packed_gpu
            new_cols_absmax_gpu = cold_cols_absmax_gpu.index_select(1, selected_positions_gpu).contiguous() if int(selected_positions_cpu.numel()) != int(cold_blocks.numel()) else cold_cols_absmax_gpu
        else:
            if self._h2d_stream is not None:
                self._wait_for_h2d_stream()
            new_cols_packed_gpu = cold_cols_packed_gpu.index_select(1, selected_positions_gpu).contiguous()
            new_cols_absmax_gpu = cold_cols_absmax_gpu.index_select(1, selected_positions_gpu).contiguous()

        try:
            if not bool(hot_cache.get("h2d_ready", True)):
                self._wait_for_h2d_stream()
            elif self._h2d_stream is not None:
                self._wait_for_h2d_stream()
            hot_cache["packed_cols_gpu"].index_copy_(1, slot_plan_gpu, new_cols_packed_gpu)
            hot_cache["absmax_cols_gpu"].index_copy_(1, slot_plan_gpu, new_cols_absmax_gpu)
            hot_cache["block_ids_cpu"] = updated_block_ids_cpu
            hot_cache["scores_cpu"] = updated_scores_cpu
            hot_cache["active_count"] = int(updated_active_count)
            hot_cache["h2d_ready"] = True
        except Exception as exc:
            if _is_cuda_oom_error(exc):
                return
            raise

        lookup = torch.full((total_col_blocks,), -1, dtype=torch.int32)
        active_block_ids_cpu = updated_block_ids_cpu[: int(updated_active_count)]
        valid_active = active_block_ids_cpu >= 0
        active_block_ids_cpu = active_block_ids_cpu[valid_active]
        if int(active_block_ids_cpu.numel()) > 0:
            lookup[active_block_ids_cpu] = torch.arange(int(active_block_ids_cpu.numel()), dtype=torch.int32)
        hot_cache["lookup_cpu"] = lookup
        param["vram_hot_down"] = hot_cache
        self._vram_nf4_cache[full_name] = hot_cache
        cached_param = self._sparse_param_cache.get(full_name)
        if cached_param is not None:
            cached_param["vram_hot_down"] = hot_cache
        self._merge_layer_hot_blocks(layer_idx, selected_blocks_cpu)



    def _get_sparse_4bit_kv_param(self, full_name: str) -> dict[str, Any]:
        """Preprocess a K or V projection NF4 weight into column-block layout for sparse loading."""
        cached = self._kv_sparse_param_cache.get(full_name)
        if cached is not None:
            return cached

        raw_weight, quant_aux = self.loader._load_raw_for_param(full_name)
        if not quant_aux:
            raise RuntimeError(f"Sparse 4-bit KV path expected quantized weights for {full_name}")

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        absmax = quant_state.absmax
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax + quant_state.offset

        kv_hidden = int(quant_state.shape[0])
        hidden_size = int(quant_state.shape[1])
        block_size = self._kv_sparse_block_size
        num_col_blocks = hidden_size // block_size
        quant_block_size = int(quant_state.blocksize)


        bytes_per_col_step = block_size // 2
        raw_2d = raw_weight.view(kv_hidden, hidden_size // 2)

        col_blocks_packed = (
            raw_2d.view(kv_hidden, num_col_blocks, bytes_per_col_step)
            .permute(1, 0, 2)
            .reshape(num_col_blocks, kv_hidden * bytes_per_col_step)
            .contiguous()
        )









        abs_per_row = hidden_size // quant_block_size
        absmax_2d = absmax.to(dtype=torch.float32).view(kv_hidden, abs_per_row)
        if block_size < quant_block_size:



            _qb_per_cb = quant_block_size // block_size

            qblock_absmax = absmax_2d.t().contiguous()


            col_blocks_absmax = qblock_absmax.repeat_interleave(_qb_per_cb, dim=0)
            abs_per_col_block = 1
            dequant_block_size = quant_block_size
            sub_per_quant = _qb_per_cb
        else:
            abs_per_col_block = block_size // quant_block_size
            dequant_block_size = quant_block_size
            sub_per_quant = 1
            qblock_absmax = None
            col_blocks_absmax = (
                absmax_2d.view(kv_hidden, num_col_blocks, abs_per_col_block)
                .permute(1, 0, 2)
                .reshape(num_col_blocks, kv_hidden * abs_per_col_block)
                .contiguous()
            )


        parts = str(full_name).split(".")
        layer_idx = int(parts[2]) if len(parts) >= 4 and parts[1] == "layers" else -1

        result: dict[str, Any] = {
            "packed_cols":      col_blocks_packed,
            "absmax_cols":      col_blocks_absmax,
            "qblock_absmax":    qblock_absmax,
            "code":             quant_state.code.to(dtype=torch.float32).contiguous(),
            "code_gpu":         quant_state.code.to(device=self.device, dtype=torch.float32).contiguous()
                                if self.device.type == "cuda" else None,
            "kv_hidden":        kv_hidden,
            "hidden_size":      hidden_size,
            "block_size":       block_size,
            "num_col_blocks":   num_col_blocks,
            "quant_block_size":  quant_block_size,
            "dequant_block_size": dequant_block_size,
            "sub_per_quant":    sub_per_quant,
            "quant_type":       str(quant_state.quant_type),
            "dtype":            quant_state.dtype,
            "layer_idx":        layer_idx,
        }

        self._maybe_cache_kv_hot_blocks(full_name, result)
        self._kv_sparse_param_cache[full_name] = result
        return result

    def _maybe_cache_kv_hot_blocks(self, full_name: str, param: dict[str, Any]) -> None:
        """Pin hot K/V column-blocks to VRAM, using same budget as MLP/attn hot cache."""
        if not getattr(self, "_vram_hot_cache_enabled", False):
            return
        if self.device.type != "cuda":
            return
        if int(param.get("sub_per_quant", 1)) > 1:
            return
        layer_idx = int(param.get("layer_idx", -1))
        if layer_idx < 0:
            return
        if full_name in getattr(self, "_vram_nf4_cache", {}):
            return
        hot_blocks = self._kv_hot_blocks_by_layer.get(layer_idx)
        if hot_blocks is None or int(hot_blocks.numel()) == 0:
            return
        num_col_blocks = int(param["num_col_blocks"])
        hot_blocks = hot_blocks[(hot_blocks >= 0) & (hot_blocks < num_col_blocks)]
        hot_blocks = hot_blocks.unique(sorted=True)
        if int(hot_blocks.numel()) == 0:
            return
        packed_block_bytes = int(param["packed_cols"].shape[1])
        absmax_block_bytes = int(param["absmax_cols"].shape[1]) * 4
        required_bytes = int(hot_blocks.numel()) * (packed_block_bytes + absmax_block_bytes)
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return
        try:
            packed_hot_gpu = self._copy_cpu_to_gpu(
                param["packed_cols"][hot_blocks].contiguous(), dtype=torch.uint8
            )
            absmax_hot_gpu = self._copy_cpu_to_gpu(
                param["absmax_cols"][hot_blocks].contiguous(), dtype=torch.float32
            )
        except Exception as _e:
            if _is_cuda_oom_error(_e):
                self._disable_vram_hot_cache("cuda_oom_kv_hot_blocks")
            return
        lookup = torch.full((num_col_blocks,), -1, dtype=torch.int32)
        lookup[hot_blocks] = torch.arange(int(hot_blocks.numel()), dtype=torch.int32)
        hot_cache = {
            "block_ids_cpu":   hot_blocks,
            "lookup_cpu":      lookup,
            "packed_cols_gpu": packed_hot_gpu,
            "absmax_cols_gpu": absmax_hot_gpu,
        }
        param["vram_hot_kv"] = hot_cache
        self._vram_nf4_cache[full_name] = hot_cache
        self._vram_hot_cache_used_bytes += required_bytes

    def _route_kv_blocks(
        self,
        hidden_norm: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor | None:
        """Return active K/V column-block indices for this layer, or None."""
        if int(layer_idx) in self._retrieval_layers and self._token_archive is not None:

            return None
        if (
            str(self._traffic_current_phase or "idle") == "prefill"
            and self._sparse_kv_prefill_mode == "dense"
        ):
            return None
        if self._should_use_attn_share_for_layer(layer_idx):
            return None
        routing = self._kv_routing.get(layer_idx)
        if routing is None:
            return None
        enc_w = routing["enc_w"]
        enc_b = routing["enc_b"]
        dec_norm_t = routing["dec_norm_t"]
        top_k = int(routing.get("top_k", self._kv_sparse_top_k))
        N = hidden_norm.shape[0] * hidden_norm.shape[1]
        h = hidden_norm.reshape(N, -1).to(device=enc_w.device, dtype=enc_w.dtype)
        latent = F.silu(F.linear(h, enc_w, enc_b))
        scores = torch.matmul(latent.abs(), dec_norm_t)

        banked_until = self._kv_block_banked_until.get(layer_idx)
        if banked_until is not None:
            banked_mask = banked_until > self._kv_bank_step
            if banked_mask.any():
                scores[:, banked_mask.to(device=scores.device)] = float("-inf")
        top_k = max(1, min(top_k, int(scores.shape[-1])))
        return scores.topk(top_k, dim=-1).indices

    def _load_sparse_kv(
        self,
        layer_idx: int,
        active_col_blocks: torch.Tensor,
        *,
        kv_hidden: int,
        hidden_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load K and V partial weight matrices for active column blocks only.

        Returns (k_partial, v_partial), each [kv_hidden, K_blocks * block_size]
        float16 on self.device.
        """
        prefix = f"model.layers.{int(layer_idx)}.self_attn."
        block_size = self._kv_sparse_block_size

        def _gather_proj(proj_name: str) -> torch.Tensor:
            full_name = f"{prefix}{proj_name}.weight"
            param = self._get_sparse_4bit_kv_param(full_name)
            sub_per_quant = int(param.get("sub_per_quant", 1))

            code_gpu = param.get("code_gpu")
            if code_gpu is None:
                code_gpu = param["code"].to(device=self.device)

            if sub_per_quant > 1:


                quant_block_size = int(param["quant_block_size"])
                num_col_blocks = int(param["num_col_blocks"])
                bytes_per_col_step = block_size // 2

                quant_blocks, qb_inverse = torch.unique(
                    active_col_blocks // sub_per_quant, sorted=True, return_inverse=True
                )
                K_qb = int(quant_blocks.numel())


                cb0_idx = quant_blocks * sub_per_quant
                cb1_idx = (quant_blocks * sub_per_quant + 1).clamp(max=num_col_blocks - 1)
                packed_cb0 = param["packed_cols"][cb0_idx]
                packed_cb1 = param["packed_cols"][cb1_idx]
                packed_pairs = torch.cat(
                    [packed_cb0.view(K_qb, kv_hidden, bytes_per_col_step),
                     packed_cb1.view(K_qb, kv_hidden, bytes_per_col_step)],
                    dim=2,
                ).contiguous()


                qblock_absmax_cpu = param["qblock_absmax"][quant_blocks].contiguous()

                packed_gpu = self._copy_cpu_to_gpu(packed_pairs.reshape(-1), dtype=torch.uint8)
                absmax_gpu = self._copy_cpu_to_gpu(qblock_absmax_cpu.reshape(-1), dtype=torch.float32)
                self._wait_for_h2d_stream()

                out_size = K_qb * kv_hidden * quant_block_size
                out_fp16 = torch.empty(out_size, dtype=self.dtype, device=self.device)
                _bnb_dequant_impl(
                    packed_gpu.reshape(-1),
                    absmax_gpu.reshape(-1),
                    quant_block_size,
                    str(param["quant_type"]),
                    self.dtype,
                    out=out_fp16,
                )


                dequanted = out_fp16.view(K_qb, kv_hidden, quant_block_size)



                qb_inv_gpu = qb_inverse.to(device=self.device, dtype=torch.long)
                half0_mask = ((active_col_blocks % sub_per_quant) == 0).to(device=self.device)
                sel = dequanted[qb_inv_gpu]
                first_half  = sel[:, :, :block_size]
                second_half = sel[:, :, block_size:block_size * 2]
                mask_exp = half0_mask[:, None, None].expand(-1, kv_hidden, block_size)
                result_3d = torch.where(mask_exp, first_half, second_half)
                return result_3d.permute(1, 0, 2).reshape(kv_hidden, -1).contiguous()


            hot_cache = param.get("vram_hot_kv")
            parts_packed: list[torch.Tensor] = []
            parts_absmax: list[torch.Tensor] = []
            cold_blocks = active_col_blocks

            if hot_cache is not None:
                lookup = hot_cache["lookup_cpu"]
                clamped = active_col_blocks.clamp(0, int(lookup.shape[0]) - 1)
                slots = lookup[clamped]
                hot_mask_cpu = slots >= 0
                if hot_mask_cpu.any():
                    hot_slots = slots[hot_mask_cpu].to(device=self.device, dtype=torch.long)
                    parts_packed.append(hot_cache["packed_cols_gpu"].index_select(0, hot_slots))
                    parts_absmax.append(hot_cache["absmax_cols_gpu"].index_select(0, hot_slots))
                cold_blocks = active_col_blocks[~hot_mask_cpu]

            if int(cold_blocks.numel()) > 0:
                cold_packed_cpu = param["packed_cols"][cold_blocks].contiguous()
                cold_absmax_cpu = param["absmax_cols"][cold_blocks].contiguous()
                cold_packed_gpu = self._copy_cpu_to_gpu(cold_packed_cpu, dtype=torch.uint8)
                cold_absmax_gpu = self._copy_cpu_to_gpu(cold_absmax_cpu, dtype=torch.float32)
                self._wait_for_h2d_stream()
                parts_packed.append(cold_packed_gpu)
                parts_absmax.append(cold_absmax_gpu)

            if not parts_packed:
                raise RuntimeError(f"No blocks loaded for {full_name}")

            packed_gpu = parts_packed[0] if len(parts_packed) == 1 else torch.cat(parts_packed, dim=0).contiguous()
            absmax_gpu = parts_absmax[0] if len(parts_absmax) == 1 else torch.cat(parts_absmax, dim=0).contiguous()

            K_blocks = int(packed_gpu.shape[0])
            out_size = K_blocks * kv_hidden * block_size
            out_fp16 = torch.empty(out_size, dtype=self.dtype, device=self.device)
            _bnb_dequant_impl(
                packed_gpu.reshape(-1),
                absmax_gpu.reshape(-1),
                int(param["dequant_block_size"]),
                str(param["quant_type"]),
                self.dtype,
                out=out_fp16,
            )
            return out_fp16.view(K_blocks, kv_hidden, block_size).permute(1, 0, 2).reshape(kv_hidden, -1).contiguous()

        k_partial = _gather_proj("k_proj")
        v_partial = _gather_proj("v_proj")
        return k_partial, v_partial

    def _clear_kv_skeleton(
        self,
        k_weight: torch.Tensor,
        v_weight: torch.Tensor,
    ) -> None:
        """Zero out previously-written column blocks in K/V skeleton weights."""
        if self._kv_loaded_cols is not None and int(self._kv_loaded_cols.numel()) > 0:
            cols = self._kv_loaded_cols
            k_weight.index_fill_(1, cols, 0.0)
            v_weight.index_fill_(1, cols, 0.0)
        else:
            k_weight.zero_()
            v_weight.zero_()
        self._kv_loaded_cols = None

    def _populate_sparse_kv_skeleton(
        self,
        layer_idx: int,
        active_col_blocks: torch.Tensor,
        layer: LlamaDecoderLayer,
    ) -> None:
        """Populate K/V skeleton weights with only the active column blocks.

        For prefill (N > 1): union of all tokens' active blocks.
        For decode (N == 1): that token's active blocks.
        The attention forward uses standard F.linear on the sparse skeleton;
        zero columns contribute zero to K/V output.
        """
        k_weight = layer.self_attn.k_proj.weight
        v_weight = layer.self_attn.v_proj.weight


        if active_col_blocks.shape[0] == 1:
            blocks_cpu = active_col_blocks[0].cpu()
        else:
            blocks_cpu = active_col_blocks.reshape(-1).cpu().unique(sorted=True)
        blocks_cpu = blocks_cpu.sort().values

        kv_hidden = int(k_weight.shape[0])
        hidden_size = int(k_weight.shape[1])
        block_size = self._kv_sparse_block_size


        self._clear_kv_skeleton(k_weight, v_weight)


        k_partial, v_partial = self._load_sparse_kv(
            layer_idx, blocks_cpu, kv_hidden=kv_hidden, hidden_size=hidden_size
        )


        col_offsets = (
            blocks_cpu.unsqueeze(-1) * block_size
            + torch.arange(block_size, dtype=torch.long)
        ).reshape(-1).to(device=self.device, dtype=torch.long)


        k_weight.index_copy_(1, col_offsets, k_partial)
        v_weight.index_copy_(1, col_offsets, v_partial)
        self._kv_loaded_cols = col_offsets

    def _update_kv_block_banking(
        self,
        layer_idx: int,
        active_col_blocks: torch.Tensor,
    ) -> None:
        """Update block usage EMA and banking state for a layer."""
        ema = self._kv_block_usage_ema.get(layer_idx)
        if ema is None:
            return
        decay = 0.95
        num_blocks = int(ema.numel())
        used = active_col_blocks.reshape(-1).cpu()
        used = used[(used >= 0) & (used < num_blocks)]
        current_usage = torch.zeros(num_blocks, dtype=torch.float32)
        if int(used.numel()) > 0:
            current_usage.scatter_add_(0, used.long(), torch.ones(int(used.numel()), dtype=torch.float32))
            current_usage = (current_usage > 0).float()
        ema.mul_(decay)
        ema.add_(current_usage * (1.0 - decay))
        votes = self._kv_block_usage_votes.get(layer_idx)
        banked_until = self._kv_block_banked_until.get(layer_idx)
        if votes is None or banked_until is None:
            return
        low_threshold = 0.001
        vote_threshold = 3
        cooldown = 64
        step = self._kv_bank_step
        low_mask = ema < low_threshold
        in_cooldown = banked_until > step
        vote_mask = low_mask & ~in_cooldown
        votes[vote_mask] += 1
        votes[~vote_mask] = 0
        can_bank = votes >= vote_threshold
        if can_bank.any():
            banked_until[can_bank] = step + cooldown
            votes[can_bank] = 0

    def _next_h2d_scratch_key(self, tag: str, dtype: torch.dtype) -> str:
        key = f"{str(tag)}:{str(dtype)}"
        slot = int(self._h2d_stage_slots[key] % 2)
        self._h2d_stage_slots[key] += 1
        return f"{key}:{slot}"

    def _copy_cpu_to_gpu(
        self,
        tensor: torch.Tensor,
        *,
        dtype: torch.dtype,
        layer_idx: int | None = None,
        tag: str = "h2d",
    ) -> torch.Tensor:
        scratch_key = self._next_h2d_scratch_key(tag, dtype)
        prepared = self.loader._stage_h2d_source_via_scratch(
            tensor,
            dtype=dtype,
            scratch_key=scratch_key,
        )
        self._record_h2d_bytes(
            int(prepared.numel() * prepared.element_size()),
            layer_idx=layer_idx,
            tag=tag,
        )
        if self.device.type != "cuda":
            return prepared.to(device=self.device, dtype=dtype)
        try:
            out = torch.empty(tuple(prepared.shape), dtype=dtype, device=self.device)
            if self._h2d_stream is not None:
                with torch.cuda.stream(self._h2d_stream):
                    out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                    self.loader._record_h2d_scratch_use(scratch_key, device=out.device)
            else:
                out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                self.loader._record_h2d_scratch_use(scratch_key, device=out.device)
            return out
        except Exception as _e:
            if not _is_cuda_oom_error(_e):
                raise


            self._disable_vram_hot_cache("cuda_oom_during_h2d_copy")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            out = torch.empty(tuple(prepared.shape), dtype=dtype, device=self.device)
            if self._h2d_stream is not None:
                with torch.cuda.stream(self._h2d_stream):
                    out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                    self.loader._record_h2d_scratch_use(scratch_key, device=out.device)
            else:
                out.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                self.loader._record_h2d_scratch_use(scratch_key, device=out.device)
            return out

    def _copy_cpu_to_existing_gpu(
        self,
        dest: torch.Tensor,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        scratch_key = self._next_h2d_scratch_key("h2d_existing", dest.dtype)
        prepared = self.loader._stage_h2d_source_via_scratch(
            tensor,
            dtype=dest.dtype,
            scratch_key=scratch_key,
        )
        if self.device.type != "cuda":
            dest.copy_(prepared)
            return prepared
        if self._h2d_stream is not None:
            with torch.cuda.stream(self._h2d_stream):
                dest.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
                self.loader._record_h2d_scratch_use(scratch_key, device=dest.device)
        else:
            dest.copy_(prepared, non_blocking=bool(prepared.is_pinned()))
            self.loader._record_h2d_scratch_use(scratch_key, device=dest.device)
        return prepared

    def _ensure_cpu_scratch(
        self,
        name: str,
        *,
        numel: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        buf = self._cpu_scratch.get(name)
        if buf is None or buf.dtype != dtype or int(buf.numel()) < int(numel):
            pin = bool(torch.cuda.is_available() and self.loader._pin_ram_cache)
            buf = torch.empty(int(numel), dtype=dtype, pin_memory=pin)
            self._cpu_scratch[name] = buf
        return buf[: int(numel)]

    def _wait_for_h2d_stream(self) -> None:
        if self.device.type != "cuda":
            return
        if self._h2d_stream is not None:
            torch.cuda.current_stream(self.device).wait_stream(self._h2d_stream)

    @staticmethod
    def _blocks_cache_key(blocks: torch.Tensor) -> tuple[int, ...]:
        cpu_blocks = blocks.detach().to(device=torch.device("cpu"), dtype=torch.long).reshape(-1)
        return tuple(int(v) for v in cpu_blocks.tolist())

    def _get_sparse_transfer_cache(
        self,
        cache: dict[tuple[str, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]],
        *,
        full_name: str,
        ordered_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if str(self._traffic_current_phase or "idle") != "decode":
            return None
        if self.device.type != "cuda":
            return None
        key = (str(full_name), self._blocks_cache_key(ordered_blocks))
        cached = cache.get(key)
        if cached is None:
            return None
        packed_gpu, absmax_gpu = cached
        if packed_gpu.device != self.device or absmax_gpu.device != self.device:
            return None
        return packed_gpu, absmax_gpu

    @staticmethod
    def _partition_cached_block_lookup(
        lookup: torch.Tensor,
        *,
        ordered_blocks: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
        hot_mask_cpu = lookup >= 0
        if bool(hot_mask_cpu.any()):
            hot_positions_cpu = torch.nonzero(hot_mask_cpu, as_tuple=False).flatten()
            hot_slots_gpu = lookup.index_select(0, hot_positions_cpu).to(device=device, dtype=torch.long)
        else:
            hot_positions_cpu = None
            hot_slots_gpu = None
        cold_mask_cpu = ~hot_mask_cpu
        cold_blocks = ordered_blocks[cold_mask_cpu]
        return hot_positions_cpu, hot_slots_gpu, cold_blocks, cold_mask_cpu

    @staticmethod
    def _merge_blockwise_parts(
        *,
        total_blocks: int,
        hot_positions_cpu: torch.Tensor | None,
        hot_packed: torch.Tensor | None,
        cold_mask_cpu: torch.Tensor | None,
        cold_packed: torch.Tensor | None,
        hot_absmax: torch.Tensor | None,
        cold_absmax: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed_template = hot_packed if hot_packed is not None else cold_packed
        absmax_template = hot_absmax if hot_absmax is not None else cold_absmax
        if packed_template is None or absmax_template is None:
            raise RuntimeError("Sparse block gather produced no data")
        packed_shape = (int(total_blocks),) + tuple(packed_template.shape[1:])
        absmax_shape = (int(total_blocks),) + tuple(absmax_template.shape[1:])
        packed = torch.empty(packed_shape, device=packed_template.device, dtype=packed_template.dtype)
        absmax = torch.empty(absmax_shape, device=absmax_template.device, dtype=absmax_template.dtype)
        if hot_packed is not None and hot_absmax is not None and hot_positions_cpu is not None and int(hot_positions_cpu.numel()) > 0:
            hot_positions_gpu = hot_positions_cpu.to(device=packed.device, dtype=torch.long)
            packed.index_copy_(0, hot_positions_gpu, hot_packed)
            absmax.index_copy_(0, hot_positions_gpu, hot_absmax)
        if cold_packed is not None and cold_absmax is not None and cold_mask_cpu is not None and bool(cold_mask_cpu.any()):
            cold_positions_cpu = torch.nonzero(cold_mask_cpu, as_tuple=False).flatten()
            cold_positions_gpu = cold_positions_cpu.to(device=packed.device, dtype=torch.long)
            packed.index_copy_(0, cold_positions_gpu, cold_packed)
            absmax.index_copy_(0, cold_positions_gpu, cold_absmax)
        return packed.contiguous(), absmax.contiguous()

    @staticmethod
    def _merge_columnwise_parts(
        *,
        total_blocks: int,
        hot_positions_cpu: torch.Tensor | None,
        hot_packed: torch.Tensor | None,
        cold_mask_cpu: torch.Tensor | None,
        cold_packed: torch.Tensor | None,
        hot_absmax: torch.Tensor | None,
        cold_absmax: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        packed_template = hot_packed if hot_packed is not None else cold_packed
        absmax_template = hot_absmax if hot_absmax is not None else cold_absmax
        if packed_template is None or absmax_template is None:
            raise RuntimeError("Sparse block gather produced no data")
        packed_shape = (int(packed_template.shape[0]), int(total_blocks)) + tuple(packed_template.shape[2:])
        absmax_shape = (int(absmax_template.shape[0]), int(total_blocks)) + tuple(absmax_template.shape[2:])
        packed = torch.empty(packed_shape, device=packed_template.device, dtype=packed_template.dtype)
        absmax = torch.empty(absmax_shape, device=absmax_template.device, dtype=absmax_template.dtype)
        if hot_packed is not None and hot_absmax is not None and hot_positions_cpu is not None and int(hot_positions_cpu.numel()) > 0:
            hot_positions_gpu = hot_positions_cpu.to(device=packed.device, dtype=torch.long)
            packed.index_copy_(1, hot_positions_gpu, hot_packed)
            absmax.index_copy_(1, hot_positions_gpu, hot_absmax)
        if cold_packed is not None and cold_absmax is not None and cold_mask_cpu is not None and bool(cold_mask_cpu.any()):
            cold_positions_cpu = torch.nonzero(cold_mask_cpu, as_tuple=False).flatten()
            cold_positions_gpu = cold_positions_cpu.to(device=packed.device, dtype=torch.long)
            packed.index_copy_(1, cold_positions_gpu, cold_packed)
            absmax.index_copy_(1, cold_positions_gpu, cold_absmax)
        return packed.contiguous(), absmax.contiguous()

    def _put_sparse_transfer_cache(
        self,
        cache: dict[tuple[str, tuple[int, ...]], tuple[torch.Tensor, torch.Tensor]],
        *,
        full_name: str,
        ordered_blocks: torch.Tensor,
        packed_gpu: torch.Tensor,
        absmax_gpu: torch.Tensor,
    ) -> None:
        if str(self._traffic_current_phase or "idle") != "decode":
            return
        if self.device.type != "cuda":
            return
        stale_keys = [key for key in cache.keys() if key[0] == str(full_name)]
        for stale_key in stale_keys:
            cache.pop(stale_key, None)
        key = (str(full_name), self._blocks_cache_key(ordered_blocks))
        cache[key] = (packed_gpu, absmax_gpu)

    def _wait_h2d_stream_for_current(self) -> None:
        if self.device.type != "cuda":
            return
        if self._h2d_stream is not None:
            self._h2d_stream.wait_stream(torch.cuda.current_stream(self.device))

    def _clear_sparse_attn_qo_buffers(
        self,
        *,
        q_skel: torch.Tensor,
        o_skel: torch.Tensor,
        force_full: bool = False,
    ) -> None:
        need_full_clear = bool(force_full or self._attn_qo_state not in {"sparse", "zero"})
        if need_full_clear:
            q_skel.zero_()
            o_skel.zero_()
        else:
            if self._attn_loaded_q_rows is not None and int(self._attn_loaded_q_rows.numel()) > 0:
                q_skel.index_fill_(0, self._attn_loaded_q_rows, 0)
            if self._attn_loaded_o_cols is not None and int(self._attn_loaded_o_cols.numel()) > 0:
                o_skel.index_fill_(1, self._attn_loaded_o_cols, 0)
        self._attn_loaded_q_rows = None
        self._attn_loaded_o_cols = None
        self._attn_qo_state = "zero"

    def _maybe_cache_sparse_attn_hot_heads(
        self,
        *,
        layer_idx: int,
        q_name: str,
        o_name: str,
        meta_q: dict[str, Any],
        meta_o: dict[str, Any],
    ) -> dict[str, Any] | None:
        cached = self._attn_hot_head_cache.get(int(layer_idx))
        if cached is not None:
            return cached
        if str(self._traffic_current_phase or "idle") not in {"decode", "prefill"}:
            return None
        if not self._vram_hot_cache_enabled:
            return None
        if self.device.type != "cuda":
            return None

        static_heads = self._attn_active_head_indices.get(int(layer_idx))
        if static_heads is None or int(static_heads.numel()) <= 0:
            return None
        pool_heads = static_heads.to(dtype=torch.long, device="cpu").unique(sorted=True)
        if int(pool_heads.numel()) <= 0:
            return None

        head_dim = int(meta_q["head_dim"])
        num_heads_total = int(meta_q["num_heads_total"])
        in_features_q = int(meta_q["in_features"])
        q_block_size = int(meta_q["quant_block_size"])
        bytes_per_head_q = in_features_q // 2 * head_dim
        absmax_per_head_q = head_dim * in_features_q // q_block_size

        out_features_o = int(meta_o["out_features"])
        in_features_o = int(meta_o["in_features"])
        o_block_size = int(meta_o["quant_block_size"])
        bytes_per_row_o = in_features_o // 2
        bytes_per_head_col = head_dim // 2
        absmax_per_row_o = in_features_o // o_block_size
        absmax_per_head_col = head_dim // o_block_size
        pool_size = int(pool_heads.numel())

        required_bytes = 0
        required_bytes += int(pool_size * bytes_per_head_q)
        required_bytes += int(pool_size * absmax_per_head_q * 4)
        required_bytes += int(out_features_o * pool_size * bytes_per_head_col)
        required_bytes += int(out_features_o * pool_size * absmax_per_head_col * 4)
        if not self._can_reserve_vram_hot_cache(required_bytes):
            return None

        q_raw, _ = self.loader._load_raw_for_param(q_name)
        q_packed_cpu = q_raw.view(num_heads_total, bytes_per_head_q).index_select(0, pool_heads).contiguous()
        q_absmax_cpu = (
            meta_q["absmax_flat"]
            .view(num_heads_total, absmax_per_head_q)
            .index_select(0, pool_heads)
            .contiguous()
        )

        o_raw, _ = self.loader._load_raw_for_param(o_name)
        o_packed_2d = o_raw.view(out_features_o, bytes_per_row_o)
        o_col_offsets = (
            pool_heads.unsqueeze(-1) * bytes_per_head_col
            + torch.arange(bytes_per_head_col, dtype=torch.long)
        ).reshape(-1)
        o_packed_cpu = (
            o_packed_2d.index_select(1, o_col_offsets)
            .view(out_features_o, pool_size, bytes_per_head_col)
            .contiguous()
        )
        o_absmax_2d = meta_o["absmax_flat"].view(out_features_o, absmax_per_row_o)
        o_abs_offsets = (
            pool_heads.unsqueeze(-1) * absmax_per_head_col
            + torch.arange(absmax_per_head_col, dtype=torch.long)
        ).reshape(-1)
        o_absmax_cpu = (
            o_absmax_2d.index_select(1, o_abs_offsets)
            .view(out_features_o, pool_size, absmax_per_head_col)
            .contiguous()
        )

        try:
            q_packed_gpu = self._copy_cpu_to_gpu(
                q_packed_cpu,
                dtype=torch.uint8,
                layer_idx=layer_idx,
                tag="attn_hotcache_q_packed",
            )
            q_absmax_gpu = self._copy_cpu_to_gpu(
                q_absmax_cpu,
                dtype=torch.float32,
                layer_idx=layer_idx,
                tag="attn_hotcache_q_absmax",
            )
            o_packed_gpu = self._copy_cpu_to_gpu(
                o_packed_cpu,
                dtype=torch.uint8,
                layer_idx=layer_idx,
                tag="attn_hotcache_o_packed",
            )
            o_absmax_gpu = self._copy_cpu_to_gpu(
                o_absmax_cpu,
                dtype=torch.float32,
                layer_idx=layer_idx,
                tag="attn_hotcache_o_absmax",
            )
        except Exception as _e:
            if _is_cuda_oom_error(_e):
                self._disable_vram_hot_cache("cuda_oom_during_attn_head_hot_cache")
                return None
            raise

        if not self._vram_hot_cache_enabled:
            return None





        lookup = torch.full((num_heads_total,), -1, dtype=torch.int32)
        lookup[pool_heads] = torch.arange(pool_size, dtype=torch.int32)
        cached = {
            "pool_heads_cpu": pool_heads,
            "lookup_cpu": lookup,
            "q_packed_gpu": q_packed_gpu,
            "q_absmax_gpu": q_absmax_gpu,
            "o_packed_gpu": o_packed_gpu,
            "o_absmax_gpu": o_absmax_gpu,
            "h2d_ready": not bool(self._enable_cuda_h2d_overlap and self._h2d_stream is not None),
        }
        self._attn_hot_head_cache[int(layer_idx)] = cached
        self._vram_hot_cache_used_bytes += required_bytes
        return cached

    def _prepare_sparse_blocks_for_param(
        self,
        param: dict[str, Any],
        *,
        ordered_blocks: torch.Tensor,
        layer_idx: int | None = None,
        tag_prefix: str = "mlp_sparse",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        full_name = str(param.get("full_name", "")).strip()
        cached_transfer = self._get_sparse_transfer_cache(
            self._sparse_block_transfer_cache,
            full_name=full_name,
            ordered_blocks=ordered_blocks,
        )
        if cached_transfer is not None:
            return cached_transfer

        def _gather(*, allow_hot_cache: bool) -> tuple[torch.Tensor, torch.Tensor]:
            traffic_phase = str(self._traffic_current_phase or "idle")
            prefill_dense = bool(
                traffic_phase == "prefill"
                and str(getattr(self, "_sparse_mlp_prefill_mode", "dense")) == "dense"
            )
            hot_cache = param.get("vram_hot") if allow_hot_cache and not prefill_dense else None
            hot_packed = None
            hot_absmax = None
            hot_positions_cpu = None
            cold_mask_cpu = None
            cold_blocks = ordered_blocks
            if hot_cache is not None:
                if not bool(hot_cache.get("h2d_ready", True)):
                    self._wait_for_h2d_stream()
                    hot_cache["h2d_ready"] = True
                lookup = hot_cache["lookup_cpu"].index_select(0, ordered_blocks.to(dtype=torch.long))
                hot_positions_cpu, hot_slots, cold_blocks, cold_mask_cpu = self._partition_cached_block_lookup(
                    lookup,
                    ordered_blocks=ordered_blocks,
                    device=self.device,
                )
                if hot_slots is not None:
                    if str(self._traffic_current_phase or "idle") == "decode":
                        self._decode_mlp_hot_blocks_hit += int(hot_slots.numel())
                    hot_packed = hot_cache["packed_blocks_gpu"].index_select(0, hot_slots)
                    hot_absmax = hot_cache["absmax_blocks_gpu"].index_select(0, hot_slots)

            cold_packed = None
            cold_absmax = None
            if int(cold_blocks.numel()) > 0:
                if str(self._traffic_current_phase or "idle") == "decode":
                    self._decode_mlp_cold_blocks_streamed += int(cold_blocks.numel())
                cold_packed_cpu, cold_absmax_cpu = self._load_cold_blocks_direct(param, cold_blocks)
                cold_packed = self._copy_cpu_to_gpu(
                    cold_packed_cpu,
                    dtype=torch.uint8,
                    layer_idx=layer_idx,
                    tag=f"{tag_prefix}_packed",
                )
                cold_absmax = self._copy_cpu_to_gpu(
                    cold_absmax_cpu,
                    dtype=torch.float32,
                    layer_idx=layer_idx,
                    tag=f"{tag_prefix}_absmax",
                )
                self._promote_sparse_param_hot_blocks_from_gpu(
                    param,
                    cold_blocks=cold_blocks,
                    cold_packed_gpu=cold_packed,
                    cold_absmax_gpu=cold_absmax,
                    layer_idx=layer_idx,
                )

            return self._merge_blockwise_parts(
                total_blocks=int(ordered_blocks.numel()),
                hot_positions_cpu=hot_positions_cpu,
                hot_packed=hot_packed,
                cold_mask_cpu=cold_mask_cpu,
                cold_packed=cold_packed,
                hot_absmax=hot_absmax,
                cold_absmax=cold_absmax,
            )

        try:
            packed_gpu, absmax_gpu = _gather(allow_hot_cache=True)
        except Exception as _e:
            if not _is_cuda_oom_error(_e):
                raise
            if torch.cuda.is_available():
                try:
                    _free, _total = torch.cuda.mem_get_info(self.device)
                    _alloc = torch.cuda.memory_allocated(self.device)
                    _reserved = torch.cuda.memory_reserved(self.device)
                    print(
                        f"[oom_diag] VRAM at OOM — total={_total/(1024**3):.2f}GB "
                        f"free(driver)={_free/(1024**3):.2f}GB "
                        f"allocated={_alloc/(1024**3):.2f}GB "
                        f"reserved={_reserved/(1024**3):.2f}GB "
                        f"hot_cache_used={self._vram_hot_cache_used_bytes/(1024**3):.2f}GB "
                        f"exc_type={type(_e).__name__!r} exc={str(_e)[:200]!r}",
                        flush=True,
                    )
                except Exception:
                    pass
            self._disable_vram_hot_cache("cuda_oom_during_sparse_block_prepare")
            try:
                packed_gpu, absmax_gpu = _gather(allow_hot_cache=False)
            except Exception as _e2:
                if not _is_cuda_oom_error(_e2):
                    raise
                torch.cuda.empty_cache()
                packed_gpu, absmax_gpu = _gather(allow_hot_cache=False)
        self._put_sparse_transfer_cache(
            self._sparse_block_transfer_cache,
            full_name=full_name,
            ordered_blocks=ordered_blocks,
            packed_gpu=packed_gpu,
            absmax_gpu=absmax_gpu,
        )
        return packed_gpu, absmax_gpu

    def _prepare_sparse_blocks_for_param_pair(
        self,
        param_a: dict[str, Any],
        param_b: dict[str, Any],
        *,
        ordered_blocks: torch.Tensor,
        layer_idx: int | None = None,
        tag_prefix: str = "mlp_sparse_pair",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Like _prepare_sparse_blocks_for_param called twice, but batches the
        cold-block CPU→pinned→GPU copies so that param_a and param_b cold packed
        tensors are staged in a single contiguous pinned buffer (one H2D call).
        Same for absmax.  Hot-cache paths are handled per-param as normal.
        Returns (a_packed_gpu, a_absmax_gpu, b_packed_gpu, b_absmax_gpu).
        """





        full_name_a = str(param_a.get("full_name", "")).strip()
        full_name_b = str(param_b.get("full_name", "")).strip()
        cached_a = self._get_sparse_transfer_cache(
            self._sparse_block_transfer_cache,
            full_name=full_name_a,
            ordered_blocks=ordered_blocks,
        )
        cached_b = self._get_sparse_transfer_cache(
            self._sparse_block_transfer_cache,
            full_name=full_name_b,
            ordered_blocks=ordered_blocks,
        )
        if cached_a is not None and cached_b is not None:
            return cached_a[0], cached_a[1], cached_b[0], cached_b[1]

        traffic_phase = str(self._traffic_current_phase or "idle")
        prefill_dense = bool(
            traffic_phase == "prefill"
            and str(getattr(self, "_sparse_mlp_prefill_mode", "dense")) == "dense"
        )

        def _hot_and_cold(param):
            hot_cache = param.get("vram_hot") if not prefill_dense else None
            hot_packed = None
            hot_absmax = None
            hot_positions_cpu = None
            cold_mask_cpu = None
            cold_blocks = ordered_blocks
            if hot_cache is not None:
                if not bool(hot_cache.get("h2d_ready", True)):
                    self._wait_for_h2d_stream()
                    hot_cache["h2d_ready"] = True
                lookup = hot_cache["lookup_cpu"].index_select(0, ordered_blocks.to(dtype=torch.long))
                hot_positions_cpu, hot_slots, cold_blocks, cold_mask_cpu = self._partition_cached_block_lookup(
                    lookup,
                    ordered_blocks=ordered_blocks,
                    device=self.device,
                )
                if hot_slots is not None:
                    if str(self._traffic_current_phase or "idle") == "decode":
                        self._decode_mlp_hot_blocks_hit += int(hot_slots.numel())
                    hot_packed = hot_cache["packed_blocks_gpu"].index_select(0, hot_slots)
                    hot_absmax = hot_cache["absmax_blocks_gpu"].index_select(0, hot_slots)
            return hot_packed, hot_absmax, hot_positions_cpu, cold_blocks, cold_mask_cpu

        if cached_a is not None:
            hot_a_packed, hot_a_absmax = cached_a[0], cached_a[1]
            hot_a_positions = None
            cold_a = ordered_blocks[:0]
            cold_a_mask = torch.zeros(int(ordered_blocks.numel()), dtype=torch.bool)
        else:
            hot_a_packed, hot_a_absmax, hot_a_positions, cold_a, cold_a_mask = _hot_and_cold(param_a)
        if cached_b is not None:
            hot_b_packed, hot_b_absmax = cached_b[0], cached_b[1]
            hot_b_positions = None
            cold_b = ordered_blocks[:0]
            cold_b_mask = torch.zeros(int(ordered_blocks.numel()), dtype=torch.bool)
        else:
            hot_b_packed, hot_b_absmax, hot_b_positions, cold_b, cold_b_mask = _hot_and_cold(param_b)


        cold_a_packed_gpu = None
        cold_a_absmax_gpu = None
        cold_b_packed_gpu = None
        cold_b_absmax_gpu = None

        has_cold_a = int(cold_a.numel()) > 0
        has_cold_b = int(cold_b.numel()) > 0

        if has_cold_a and has_cold_b:

            cold_a_packed_cpu, cold_a_absmax_cpu = self._load_cold_blocks_direct(param_a, cold_a)
            cold_b_packed_cpu, cold_b_absmax_cpu = self._load_cold_blocks_direct(param_b, cold_b)
            combined_packed_cpu = torch.cat([cold_a_packed_cpu, cold_b_packed_cpu], dim=0)
            combined_packed_gpu = self._copy_cpu_to_gpu(
                combined_packed_cpu,
                dtype=torch.uint8,
                layer_idx=layer_idx,
                tag=f"{tag_prefix}_packed",
            )
            na = int(cold_a.numel())
            cold_a_packed_gpu = combined_packed_gpu[:na]
            cold_b_packed_gpu = combined_packed_gpu[na:]

            combined_absmax_cpu = torch.cat([cold_a_absmax_cpu, cold_b_absmax_cpu], dim=0)
            combined_absmax_gpu = self._copy_cpu_to_gpu(
                combined_absmax_cpu,
                dtype=torch.float32,
                layer_idx=layer_idx,
                tag=f"{tag_prefix}_absmax",
            )
            na_abs = int(cold_a_absmax_cpu.shape[0])
            cold_a_absmax_gpu = combined_absmax_gpu[:na_abs]
            cold_b_absmax_gpu = combined_absmax_gpu[na_abs:]
            self._promote_sparse_param_hot_blocks_from_gpu(
                param_a,
                cold_blocks=cold_a,
                cold_packed_gpu=cold_a_packed_gpu,
                cold_absmax_gpu=cold_a_absmax_gpu,
                layer_idx=layer_idx,
            )
            self._promote_sparse_param_hot_blocks_from_gpu(
                param_b,
                cold_blocks=cold_b,
                cold_packed_gpu=cold_b_packed_gpu,
                cold_absmax_gpu=cold_b_absmax_gpu,
                layer_idx=layer_idx,
            )
        elif has_cold_a:
            cold_a_packed_cpu, cold_a_absmax_cpu = self._load_cold_blocks_direct(param_a, cold_a)
            cold_a_packed_gpu = self._copy_cpu_to_gpu(
                cold_a_packed_cpu,
                dtype=torch.uint8,
                layer_idx=layer_idx,
                tag=f"{tag_prefix}_a_packed",
            )
            cold_a_absmax_gpu = self._copy_cpu_to_gpu(
                cold_a_absmax_cpu,
                dtype=torch.float32,
                layer_idx=layer_idx,
                tag=f"{tag_prefix}_a_absmax",
            )
            self._promote_sparse_param_hot_blocks_from_gpu(
                param_a,
                cold_blocks=cold_a,
                cold_packed_gpu=cold_a_packed_gpu,
                cold_absmax_gpu=cold_a_absmax_gpu,
                layer_idx=layer_idx,
            )
        elif has_cold_b:
            cold_b_packed_cpu, cold_b_absmax_cpu = self._load_cold_blocks_direct(param_b, cold_b)
            cold_b_packed_gpu = self._copy_cpu_to_gpu(
                cold_b_packed_cpu,
                dtype=torch.uint8,
                layer_idx=layer_idx,
                tag=f"{tag_prefix}_b_packed",
            )
            cold_b_absmax_gpu = self._copy_cpu_to_gpu(
                cold_b_absmax_cpu,
                dtype=torch.float32,
                layer_idx=layer_idx,
                tag=f"{tag_prefix}_b_absmax",
            )
            self._promote_sparse_param_hot_blocks_from_gpu(
                param_b,
                cold_blocks=cold_b,
                cold_packed_gpu=cold_b_packed_gpu,
                cold_absmax_gpu=cold_b_absmax_gpu,
                layer_idx=layer_idx,
            )

        a_packed, a_absmax = self._merge_blockwise_parts(
            total_blocks=int(ordered_blocks.numel()),
            hot_positions_cpu=hot_a_positions,
            hot_packed=hot_a_packed,
            cold_mask_cpu=cold_a_mask,
            cold_packed=cold_a_packed_gpu,
            hot_absmax=hot_a_absmax,
            cold_absmax=cold_a_absmax_gpu,
        )
        b_packed, b_absmax = self._merge_blockwise_parts(
            total_blocks=int(ordered_blocks.numel()),
            hot_positions_cpu=hot_b_positions,
            hot_packed=hot_b_packed,
            cold_mask_cpu=cold_b_mask,
            cold_packed=cold_b_packed_gpu,
            hot_absmax=hot_b_absmax,
            cold_absmax=cold_b_absmax_gpu,
        )
        self._put_sparse_transfer_cache(
            self._sparse_block_transfer_cache,
            full_name=full_name_a,
            ordered_blocks=ordered_blocks,
            packed_gpu=a_packed,
            absmax_gpu=a_absmax,
        )
        self._put_sparse_transfer_cache(
            self._sparse_block_transfer_cache,
            full_name=full_name_b,
            ordered_blocks=ordered_blocks,
            packed_gpu=b_packed,
            absmax_gpu=b_absmax,
        )
        return a_packed, a_absmax, b_packed, b_absmax

    def _compute_sparse_basis_latent(
        self,
        flat_hidden: torch.Tensor,
        layer_idx: int,
        routing: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        routing = self._sparse_routing.get(layer_idx) if routing is None else routing
        if routing is None:
            raise RuntimeError(f"No sparse basis routing state loaded for layer {int(layer_idx)}")

        enc_w = routing["enc_w"]
        enc_b = routing["enc_b"]
        hidden_proj = flat_hidden.to(device=enc_w.device, dtype=enc_w.dtype)
        latent_dense = F.linear(hidden_proj, enc_w, enc_b)
        latent_dim = int(latent_dense.shape[-1])
        if latent_dim <= 0:
            return latent_dense
        basis_top_k = int(
            routing.get(
                "basis_top_k",
                self._sparse_basis_top_k_by_layer.get(int(layer_idx), _DEFAULT_SPARSE_BASIS_TOP_K),
            )
        )
        basis_top_k = max(1, min(basis_top_k, latent_dim))
        if basis_top_k >= latent_dim:
            return latent_dense
        topk_idx = torch.topk(latent_dense.abs(), k=basis_top_k, dim=-1).indices
        latent_mask = torch.zeros_like(latent_dense)
        latent_mask.scatter_(1, topk_idx, 1.0)
        return latent_dense * latent_mask

    def _get_or_build_routing_down_block_norm(
        self,
        *,
        layer_idx: int,
        routing: dict[str, Any],
    ) -> torch.Tensor | None:
        cached = routing.get("down_block_norm")
        if torch.is_tensor(cached):
            return cached.to(device=torch.device("cpu"), dtype=torch.float32).contiguous()
        routing["down_block_norm_ready"] = True
        return None

    def _register_runtime_sparse_basis_layer(
        self,
        layer_idx: int,
        *,
        encoder_weight: torch.Tensor,
        encoder_bias: torch.Tensor,
        decoder_blocks: torch.Tensor,
        decoder_bias: torch.Tensor | None,
        scale: float,
        top_k: int,
        basis_top_k: int,
        session_local: bool,
    ) -> None:
        _basis_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        _dec = decoder_blocks.to(device=_basis_device, dtype=self.dtype, non_blocking=False).contiguous()
        _layer_num_blocks = int(_dec.shape[0])
        _expected_output_blocks = int(self.config.hidden_size) // max(int(self._sparse_block_size), 1)
        if _layer_num_blocks != _expected_output_blocks:
            raise RuntimeError(
                f"Sparse basis layer {int(layer_idx)} stores {_layer_num_blocks} output blocks, "
                f"expected {_expected_output_blocks}. The learned-basis artifact is defined "
                "over hidden-size output blocks and cannot be mixed with FFN intermediate blocks."
            )
        _enc_w = encoder_weight.to(device=_basis_device, dtype=self.dtype, non_blocking=False).contiguous()
        _enc_b = encoder_bias.to(device=_basis_device, dtype=self.dtype, non_blocking=False).contiguous()
        _basis_rank = int(_enc_w.shape[0])
        _layer_top_k = int(max(1, min(int(top_k), _layer_num_blocks)))
        _basis_top_k = int(max(1, min(int(basis_top_k), _basis_rank)))
        _dec_bias = None
        if torch.is_tensor(decoder_bias):
            _dec_bias = decoder_bias.to(device=_basis_device, dtype=self.dtype, non_blocking=False).contiguous()
        _dec_norm_t = _dec.norm(dim=-1).transpose(0, 1).contiguous()
        if self._sparse_semantic_block_score_normalized:
            _dec_norm_t = F.normalize(_dec_norm_t, p=2.0, dim=-1, eps=1e-6)
        self._sparse_routing[int(layer_idx)] = {
            "enc_w": _enc_w,
            "enc_b": _enc_b,
            "dec": _dec,
            "dec_bias": _dec_bias,
            "dec_norm_t": _dec_norm_t,
            "scale": float(scale),
            "top_k": _layer_top_k,
            "basis_top_k": _basis_top_k,
            "basis_rank": int(_basis_rank),
            "rank_effective": int(_basis_rank),
            "artifact_target": "output_reconstruction",
            "block_domain": "output",
            "block_size": int(self._sparse_block_size),
            "num_blocks": int(_layer_num_blocks),
            "explained_variance_ratio": 1.0,
            "samples": -1.0,
            "pca_method": "runtime_local_fit",
        }
        self._sparse_top_k_by_layer[int(layer_idx)] = int(_layer_top_k)
        self._sparse_basis_top_k_by_layer[int(layer_idx)] = int(_basis_top_k)
        if session_local:
            self._session_sparse_route_layers.add(int(layer_idx))
            self._mlp_hot_blocks_by_layer.pop(int(layer_idx), None)

    def _route_sparse_mlp(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor | None:
        """Return active block indices [N, top_k] for this layer, or None (→ dense).

        Uses the learned encoder/decoder from the basis checkpoint to predict
        which MLP output blocks will have the largest norm for this hidden state,
        then returns the top-K block indices.  All tensors moved to GPU lazily.
        """
        routing = self._sparse_routing.get(layer_idx)
        if routing is None:
            return None

        flat_hidden = hidden.view(hidden.shape[0] * hidden.shape[1], -1)
        latent = self._compute_sparse_basis_latent(flat_hidden, layer_idx, routing=routing)
        route_score_weight = routing.get("route_score_weight")
        route_score_bias = routing.get("route_score_bias")
        if torch.is_tensor(route_score_weight):
            score_weight = route_score_weight.to(device=latent.device, dtype=latent.dtype)
            score_bias = None
            if torch.is_tensor(route_score_bias):
                score_bias = route_score_bias.to(device=latent.device, dtype=latent.dtype)
            scores = F.linear(latent, score_weight, score_bias).clamp_min(0.0)
        else:
            dec_norm_t = routing["dec_norm_t"]
            scores = torch.matmul(latent.abs(), dec_norm_t)
        down_block_norm = self._get_or_build_routing_down_block_norm(layer_idx=int(layer_idx), routing=routing)
        if torch.is_tensor(down_block_norm) and int(down_block_norm.numel()) == int(scores.shape[-1]):
            scores = scores * down_block_norm.to(device=scores.device, dtype=scores.dtype).unsqueeze(0)
        k_runtime = int(
            routing.get(
                "top_k",
                self._sparse_runtime_top_k if int(self._sparse_runtime_top_k) > 0 else self._sparse_top_k,
            )
        )
        if (
            str(self._traffic_current_phase or "idle") == "prefill"
            and str(getattr(self, "_sparse_mlp_prefill_mode", "dense")) == "sparse"
            and getattr(self, "_sparse_mlp_prefill_top_k", None) is not None
        ):
            k_runtime = int(self._sparse_mlp_prefill_top_k)
        k_runtime = max(1, min(k_runtime, int(scores.shape[-1])))
        active_blocks = scores.topk(k_runtime, dim=-1).indices
        return active_blocks

    def _ensure_single_kernel_mlp_scratch(self, *, hidden_size: int, block_out: int) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("Single-kernel sparse MLP scratch requires CUDA")
        hidden_size = int(hidden_size)
        block_out = int(block_out)
        if hidden_size <= 0 or block_out <= 0:
            raise RuntimeError("Invalid single-kernel scratch dimensions")
        if (
            self._single_kernel_mlp_out_accum is None
            or int(self._single_kernel_mlp_out_accum.numel()) < hidden_size
        ):
            self._single_kernel_mlp_out_accum = torch.zeros(
                hidden_size,
                device=self.device,
                dtype=torch.float32,
            )
        self._single_kernel_mlp_block_out = int(block_out)

    def _debug_assert_sparse_attn_qo_zero_inactive(
        self,
        *,
        q_skel: torch.Tensor,
        o_skel: torch.Tensor,
    ) -> None:
        if not bool(getattr(self, "_debug_assert_sparse_attn_qo_zero", False)):
            return
        loaded_q = getattr(self, "_attn_loaded_q_rows", None)
        loaded_o = getattr(self, "_attn_loaded_o_cols", None)
        with torch.no_grad():
            q_row_energy = q_skel.abs().sum(dim=1)
            if loaded_q is not None and int(loaded_q.numel()) > 0:
                loaded_q_u = loaded_q.to(device=q_skel.device, dtype=torch.long).unique(sorted=True)
                q_row_energy.index_fill_(0, loaded_q_u, 0)
            bad_q = torch.nonzero(q_row_energy > 0, as_tuple=False).reshape(-1)
            if int(bad_q.numel()) > 0:
                raise RuntimeError(
                    f"Sparse attention q_proj skeleton has non-zero inactive rows "
                    f"(first offending row={int(bad_q[0].item())})."
                )

            o_col_energy = o_skel.abs().sum(dim=0)
            if loaded_o is not None and int(loaded_o.numel()) > 0:
                loaded_o_u = loaded_o.to(device=o_skel.device, dtype=torch.long).unique(sorted=True)
                o_col_energy.index_fill_(0, loaded_o_u, 0)
            bad_o = torch.nonzero(o_col_energy > 0, as_tuple=False).reshape(-1)
            if int(bad_o.numel()) > 0:
                raise RuntimeError(
                    f"Sparse attention o_proj skeleton has non-zero inactive columns "
                    f"(first offending col={int(bad_o[0].item())})."
                )

    def _ensure_mlp_proj_staging(self) -> bool:
        if self._mlp_proj_staging is not None:
            return True
        if self.device.type != "cuda":
            return False
        try:
            self._mlp_proj_staging = torch.empty(
                int(self._mlp_proj_staging_numel),
                dtype=self.dtype,
                device=self.device,
            )
            return True
        except Exception as _e:
            if "out of memory" in str(_e).lower() or "alloc" in str(_e).lower():
                if not self._dense_mlp_staging_warned:
                    self._dense_mlp_staging_warned = True
                    print(
                        "[dense_mlp] Unable to allocate GPU staging for dense fallback; "
                        "dense calibration paths may use zero passthrough.",
                        flush=True,
                    )
                return False
            raise

    def _materialize_sparse_4bit_param_raw_views(
        self,
        param: dict[str, Any],
        *,
        store_raw_in_ram_cache: bool = True,
    ) -> dict[str, Any]:
        if "packed_weight" in param and "packed_blocks" in param and "absmax_blocks" in param:
            return param
        full_name = str(param.get("full_name", "")).strip()
        if not full_name:
            raise RuntimeError("Sparse 4-bit parameter metadata is missing full_name; cannot load cold blocks.")
        raw_weight, _quant_aux = self.loader._load_raw_for_param(
            full_name,
            store_in_ram_cache=bool(store_raw_in_ram_cache),
        )
        raw_flat = raw_weight.reshape(-1)
        blocks_per_col = int(param["blocks_per_col"])
        bytes_per_block = int(param["bytes_per_block"])
        absmax_per_block = int(param["absmax_per_block"])
        absmax_cpu = param["absmax_cpu"]
        param["packed_weight"] = raw_flat
        param["packed_blocks"] = raw_flat.view(blocks_per_col, bytes_per_block)
        param["absmax"] = absmax_cpu
        param["absmax_blocks"] = absmax_cpu.view(blocks_per_col, absmax_per_block)
        return param

    def _load_down_proj_cold_cols(
        self,
        down_param: dict,
        cold_blocks: torch.Tensor,
        *,
        H_out: int,
        bytes_per_row: int,
        bytes_per_cblk: int,
        absmax_per_row: int,
        block_size: int,
        qbs: int,
        down_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cold_packed_cpu, cold_absmax_cpu) for down_proj cold column blocks.

        Checks _down_proj_col_cache first so that layers whose column blocks were
        already extracted (during a previous token or pre_warm) never trigger a
        full 436 MB weight reload from SSD.  Only blocks absent from the cache go
        through _materialize_sparse_4bit_param_raw_views; their slices are then
        stored in the cache for all future tokens (~320 KB per block entry).

        After the first decode token the cache covers all commonly-active blocks,
        eliminating the 17×436 MB/token SSD reads that caused 208 s/token.
        """
        cold_list = cold_blocks.tolist()
        layer_cache = self._down_proj_col_cache.get(down_name)

        # Fast path: every cold block already in CPU cache
        if layer_cache is not None and all(b in layer_cache for b in cold_list):
            packed_parts = [layer_cache[b][0] for b in cold_list]
            absmax_parts = [layer_cache[b][1] for b in cold_list]
            cold_packed_cpu = torch.stack(packed_parts, dim=1).contiguous()
            cold_absmax_cpu = torch.stack(absmax_parts, dim=1).contiguous()
            return cold_packed_cpu, cold_absmax_cpu

        # Need to load the raw weight for the cache-miss blocks
        miss_blocks = [b for b in cold_list if layer_cache is None or b not in layer_cache]

        down_param_raw = self._materialize_sparse_4bit_param_raw_views(down_param)
        raw_2d = down_param_raw["packed_weight"].view(H_out, bytes_per_row)
        absmax_2d = down_param_raw["absmax"].view(H_out, absmax_per_row)

        if layer_cache is None:
            layer_cache = {}
            self._down_proj_col_cache[down_name] = layer_cache

        # Extract and cache each missing block individually (~256 KB + 64 KB each)
        for b in miss_blocks:
            b_start = int(b) * bytes_per_cblk
            col_slice = raw_2d[:, b_start : b_start + bytes_per_cblk].contiguous()
            abs_idx = (int(b) * block_size) // qbs
            abs_slice = absmax_2d[:, abs_idx : abs_idx + 1].contiguous()
            layer_cache[b] = (col_slice, abs_slice)

        # Assemble result for ALL cold blocks (hits + newly loaded)
        packed_parts = [layer_cache[b][0] for b in cold_list]
        absmax_parts = [layer_cache[b][1] for b in cold_list]
        cold_packed_cpu = torch.stack(packed_parts, dim=1).contiguous()
        cold_absmax_cpu = torch.stack(absmax_parts, dim=1).contiguous()
        return cold_packed_cpu, cold_absmax_cpu

    def _load_cold_blocks_direct(
        self,
        param: dict[str, Any],
        cold_blocks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load only the requested cold blocks for an NF4 sparse param.

        Uses load_nf4_packed_blocks to read just the needed byte ranges from disk
        (~13 MB per projection) instead of materializing the full 436 MB weight tensor.
        absmax is already in _sparse_param_cache (loaded during pre_warm), so no absmax I/O.

        Returns (packed_blocks_cpu, absmax_blocks_cpu_f32).
        """
        bytes_per_block = int(param["bytes_per_block"])
        absmax_per_block = int(param["absmax_per_block"])
        full_name = str(param.get("full_name", "")).strip()
        cold_idx = cold_blocks.to(dtype=torch.long)

        absmax_cpu = param.get("absmax_cpu")
        if absmax_cpu is not None:
            absmax_blocks = absmax_cpu.reshape(-1, absmax_per_block)
            cold_absmax_cpu = absmax_blocks[cold_idx].contiguous().to(dtype=torch.float32)
        else:
            raw_param = self._materialize_sparse_4bit_param_raw_views(param)
            return (
                raw_param["packed_blocks"][cold_idx].contiguous(),
                raw_param["absmax_blocks"][cold_idx].contiguous().to(dtype=torch.float32),
            )

        if not full_name:
            raw_param = self._materialize_sparse_4bit_param_raw_views(param)
            return (
                raw_param["packed_blocks"][cold_idx].contiguous(),
                cold_absmax_cpu,
            )

        try:
            cold_packed_cpu = self.loader.load_nf4_packed_blocks(
                full_name,
                cold_idx,
                bytes_per_block=bytes_per_block,
            )
            return cold_packed_cpu, cold_absmax_cpu
        except Exception:
            raw_param = self._materialize_sparse_4bit_param_raw_views(param)
            return (
                raw_param["packed_blocks"][cold_idx].contiguous(),
                cold_absmax_cpu,
            )

    def _get_sparse_4bit_param(
        self,
        full_name: str,
        *,
        store_raw_in_ram_cache: bool = True,
        require_raw: bool = True,
    ) -> dict[str, Any]:
        cached = self._sparse_param_cache.get(full_name)
        if cached is not None and not bool(require_raw):
            param = dict(cached)
            param.setdefault("full_name", str(full_name))
            return param




        if (
            cached is not None
            and str(self._traffic_current_phase or "idle") == "prefill"
            and str(getattr(self, "_sparse_mlp_prefill_mode", "dense")) == "hot_cache"
            and ("vram_hot" in cached or "vram_hot_down" in cached)
        ):
            blocks_per_col = int(cached["blocks_per_col"])
            absmax_per_block = int(cached["absmax_per_block"])
            bytes_per_block = int(cached["bytes_per_block"])
            param = dict(cached)
            param.setdefault("full_name", str(full_name))
            param["packed_weight"] = torch.empty(0, dtype=torch.uint8)
            param["packed_blocks"] = torch.empty((blocks_per_col, bytes_per_block), dtype=torch.uint8)
            param["absmax"] = cached["absmax_cpu"]
            param["absmax_blocks"] = cached["absmax_cpu"].view(blocks_per_col, absmax_per_block)
            return param
        raw_weight: torch.Tensor | None = None
        if cached is None:
            raw_weight, quant_aux = self.loader._load_raw_for_param(
                full_name,
                store_in_ram_cache=bool(store_raw_in_ram_cache),
            )
            if not quant_aux:
                raise RuntimeError(f"Sparse 4-bit path expected quantized weights for {full_name}")

            quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
            absmax = quant_state.absmax
            if quant_state.nested:
                absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
                absmax = absmax + quant_state.offset

            in_features = int(quant_state.shape[1])
            out_features = int(quant_state.shape[0])
            block_size = int(self._sparse_block_size)
            bytes_per_row = in_features // 2
            bytes_per_block = bytes_per_row * block_size
            blocks_per_col = out_features // max(block_size, 1)
            absmax_cpu = absmax.to(dtype=torch.float32).contiguous()
            absmax_per_row = in_features // int(quant_state.blocksize)
            absmax_per_block = absmax_per_row * block_size

            cached = {
                "full_name": str(full_name),
                "absmax_cpu": absmax_cpu,
                "code": quant_state.code.to(dtype=torch.float32).contiguous(),
                "code_gpu": quant_state.code.to(device=self.device, dtype=torch.float32, non_blocking=True).contiguous()
                if self.device.type == "cuda"
                else None,
                "out_features": out_features,
                "in_features": in_features,
                "bytes_per_row": bytes_per_row,
                "bytes_per_block": bytes_per_block,
                "blocks_per_col": blocks_per_col,
                "absmax_per_row": absmax_per_row,
                "absmax_per_block": absmax_per_block,
                "quant_block_size": int(quant_state.blocksize),
                "quant_type": str(quant_state.quant_type),
                "dtype": quant_state.dtype,
            }
            param_views = dict(cached)
            raw_flat = raw_weight.reshape(-1)
            param_views["packed_weight"] = raw_flat
            param_views["packed_blocks"] = raw_flat.view(blocks_per_col, bytes_per_block)
            param_views["absmax"] = absmax_cpu
            param_views["absmax_blocks"] = absmax_cpu.view(blocks_per_col, absmax_per_block)
            self._maybe_cache_sparse_param_hot_blocks(full_name, param_views)
            if "vram_hot" in param_views:
                cached["vram_hot"] = param_views["vram_hot"]
            if "vram_hot_down" in param_views:
                cached["vram_hot_down"] = param_views["vram_hot_down"]
            self._sparse_param_cache[full_name] = cached
            return param_views

        param = dict(cached)
        param.setdefault("full_name", str(full_name))
        self._materialize_sparse_4bit_param_raw_views(
            param,
            store_raw_in_ram_cache=bool(store_raw_in_ram_cache),
        )
        self._maybe_cache_sparse_param_hot_blocks(full_name, param)
        if "vram_hot" in param:
            cached["vram_hot"] = param["vram_hot"]
        if "vram_hot_down" in param:
            cached["vram_hot_down"] = param["vram_hot_down"]
        cached.setdefault("full_name", str(full_name))
        self._sparse_param_cache[full_name] = cached
        return param

    def _load_optional_bias(
        self,
        full_name: str,
        bias_ref: torch.Tensor | None,
        *,
        device: torch.device,
        dtype: torch.dtype,
        layer_idx: int | None = None,
        tag: str = "bias",
    ) -> torch.Tensor | None:
        if bias_ref is None:
            return None
        bias_name = full_name[:-7] + ".bias" if full_name.endswith(".weight") else ""
        if not bias_name:
            self._record_h2d_bytes(
                int(bias_ref.numel() * bias_ref.element_size()),
                layer_idx=layer_idx,
                tag=tag,
            )
            return bias_ref.to(device=device, dtype=dtype, non_blocking=True)
        bias = self.loader.load_parameter(bias_name)
        self._record_h2d_bytes(
            int(bias.numel() * bias.element_size()),
            layer_idx=layer_idx,
            tag=tag,
        )
        return bias.to(device=device, dtype=dtype, non_blocking=True)

    def _sparse_mlp_forward(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        active_blocks: torch.Tensor,
    ) -> torch.Tensor:
        """Run LlamaMLP on only the active neuron blocks; fall back if needed.

        active_blocks: [N, top_k] — block indices selected by _route_sparse_mlp.

        For each token we gather the rows of gate_proj/up_proj and the columns of
        down_proj that correspond to active neurons, run the SiLU-gated MLP on
        that small slice (~2% of weights for top_k=2%), and return the output.

        Falls back to dense MLP if the module doesn't expose the expected Linear
        sub-layers.
        """
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj   = getattr(mlp, "up_proj",   None)
        down_proj = getattr(mlp, "down_proj",  None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return mlp(hidden)
        if not (hasattr(gate_proj, "weight") and gate_proj.weight is not None):
            return mlp(hidden)

        block_size = self._sparse_block_size
        N, H = hidden.shape[0] * hidden.shape[1], hidden.shape[-1]
        h = hidden.view(N, H)



        neuron_offsets = torch.arange(block_size, device=active_blocks.device)
        active_neurons = (
            active_blocks.unsqueeze(-1) * block_size + neuron_offsets
        ).reshape(N, -1)



        unique_neurons = active_neurons.squeeze(0) if N == 1 else active_neurons.unique()

        def _is_4bit_linear(linear: nn.Module) -> bool:
            weight = getattr(linear, "weight", None)
            return bool(torch.is_tensor(weight) and getattr(weight, "quant_state", None) is not None)

        def _load_quantized_weight_cpu(param_name: str) -> torch.Tensor | None:
            try:
                raw_weight, quant_aux = self.loader._load_raw_for_param(param_name)
            except Exception:
                return None
            if not quant_aux:
                return None
            quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
            return bnb_functional.dequantize_4bit(raw_weight, quant_state=quant_state)

        def _project_active_inputs(linear: nn.Module, neurons: torch.Tensor) -> torch.Tensor:
            bias = getattr(linear, "bias", None)
            weight = getattr(linear, "weight", None)
            proj_weight = None
            linear_name = getattr(linear, "_sparse_param_name", "")
            if linear_name:
                dense_weight_cpu = _load_quantized_weight_cpu(linear_name)
                if dense_weight_cpu is not None:
                    proj_weight = dense_weight_cpu.index_select(0, neurons.cpu()).to(
                        device=h.device,
                        dtype=h.dtype,
                        non_blocking=True,
                    )
                    del dense_weight_cpu
            if proj_weight is None and _is_4bit_linear(linear):
                quant_state = weight.quant_state
                dense_weight = bnb_functional.dequantize_4bit(weight.t(), quant_state).t()
                proj_weight = dense_weight.index_select(0, neurons)
                del dense_weight
            if proj_weight is None:
                proj_weight = weight.index_select(0, neurons)
            if bias is not None:
                bias = bias.index_select(0, neurons)
            return F.linear(h, proj_weight, bias)

        def _project_active_outputs(x: torch.Tensor, linear: nn.Module, neurons: torch.Tensor) -> torch.Tensor:
            bias = getattr(linear, "bias", None)
            weight = getattr(linear, "weight", None)
            proj_weight = None
            linear_name = getattr(linear, "_sparse_param_name", "")
            if linear_name:
                dense_weight_cpu = _load_quantized_weight_cpu(linear_name)
                if dense_weight_cpu is not None:
                    proj_weight = dense_weight_cpu.index_select(1, neurons.cpu()).to(
                        device=x.device,
                        dtype=x.dtype,
                        non_blocking=True,
                    )
                    del dense_weight_cpu
            if proj_weight is None and _is_4bit_linear(linear):
                quant_state = weight.quant_state
                dense_weight = bnb_functional.dequantize_4bit(weight.t(), quant_state).t()
                proj_weight = dense_weight.index_select(1, neurons)
                del dense_weight
            if proj_weight is None:
                proj_weight = weight.index_select(1, neurons)
            return F.linear(x, proj_weight, bias)

        prefix = f"model.layers.{int(layer_idx)}.mlp."
        gate_proj._sparse_param_name = f"{prefix}gate_proj.weight"
        up_proj._sparse_param_name = f"{prefix}up_proj.weight"
        down_proj._sparse_param_name = f"{prefix}down_proj.weight"
        gate = _project_active_inputs(gate_proj, unique_neurons)
        up   = _project_active_inputs(up_proj, unique_neurons)
        act  = F.silu(gate) * up
        del gate, up
        out  = _project_active_outputs(act, down_proj, unique_neurons)
        return out.view_as(hidden)

    def _dense_mlp_forward_streaming(
        self,
        mlp: nn.Module,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return mlp(hidden)

        flat_hidden = hidden.view(-1, hidden.shape[-1])

        def _linear_stream(x: torch.Tensor, linear: nn.Module) -> torch.Tensor:
            weight = getattr(linear, "weight", None)
            bias = getattr(linear, "bias", None)
            if weight is None:
                raise RuntimeError("Dense MLP streaming path requires linear.weight")
            weight_gpu = weight.to(device=x.device, dtype=x.dtype, non_blocking=True)
            bias_gpu = None if bias is None else bias.to(device=x.device, dtype=x.dtype, non_blocking=True)
            y = F.linear(x, weight_gpu, bias_gpu)
            del weight_gpu, bias_gpu
            return y

        gate = _linear_stream(flat_hidden, gate_proj)
        up = _linear_stream(flat_hidden, up_proj)
        act = F.silu(gate) * up
        del gate, up
        out = _linear_stream(act, down_proj)
        return out.view_as(hidden)

    def _build_sparse_active_layout(
        self,
        *,
        layer_idx: int,
        active_blocks: torch.Tensor,
        rows: int,
        block_size: int,
        max_valid_blocks: int,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        active_blocks_cpu = active_blocks.detach().to(device=torch.device("cpu"), dtype=torch.long).view(rows, -1)
        union_blocks = active_blocks_cpu.reshape(-1)
        if rows > 1:
            union_blocks = union_blocks.unique(sorted=True)
        ordered_blocks = self._order_blocks_for_layer_hot_cache(layer_idx, union_blocks)
        ordered_blocks = ordered_blocks[(ordered_blocks >= 0) & (ordered_blocks < int(max_valid_blocks))]
        ordered_blocks = ordered_blocks.unique(sorted=False).to(dtype=torch.long, device=torch.device("cpu")).contiguous()
        ordered_blocks = self._order_blocks_for_layer_hot_cache(layer_idx, ordered_blocks).contiguous()
        num_active_blocks = int(ordered_blocks.numel())
        if num_active_blocks <= 0:
            return (
                ordered_blocks,
                torch.empty((rows, active_blocks_cpu.shape[1]), dtype=torch.int32, device=self.device),
                torch.empty((rows, 0), dtype=dtype, device=self.device),
                torch.empty((0,), dtype=torch.long, device=torch.device("cpu")),
            )

        union_lookup = torch.full((int(max_valid_blocks),), -1, dtype=torch.int32)
        union_lookup[ordered_blocks] = torch.arange(num_active_blocks, dtype=torch.int32)
        active_local_cpu = torch.full(active_blocks_cpu.shape, -1, dtype=torch.int32)
        valid_slots = (active_blocks_cpu >= 0) & (active_blocks_cpu < int(max_valid_blocks))
        if bool(valid_slots.any()):
            active_local_cpu[valid_slots] = union_lookup[active_blocks_cpu[valid_slots]]

        active_local = active_local_cpu.to(device=self.device, dtype=torch.int32).contiguous()
        active_dim = num_active_blocks * int(block_size)
        flat_mask = torch.zeros((rows, active_dim), device=self.device, dtype=dtype)



        neuron_offsets = torch.arange(int(block_size), device=self.device, dtype=torch.long)

        al_long = active_local.long()
        valid_mask = al_long >= 0
        if bool(valid_mask.any()):
            row_ids = torch.arange(rows, device=self.device, dtype=torch.long).unsqueeze(1).expand_as(al_long)

            slot_safe = al_long.clamp(min=0)

            col_starts = slot_safe.unsqueeze(-1) * int(block_size)
            col_idx = (col_starts + neuron_offsets.view(1, 1, -1))
            row_idx_exp = row_ids.unsqueeze(-1).expand_as(col_idx)
            valid_exp = valid_mask.unsqueeze(-1).expand_as(col_idx)
            flat_mask.scatter_(
                1,
                col_idx.reshape(rows, -1),
                valid_exp.to(dtype=flat_mask.dtype).reshape(rows, -1),
            )
            del row_idx_exp, col_idx, col_starts, slot_safe, row_ids, valid_exp
        del al_long, valid_mask

        active_neurons = (
            ordered_blocks.unsqueeze(-1) * int(block_size)
            + torch.arange(int(block_size), device=torch.device("cpu"), dtype=torch.long)
        ).reshape(-1)
        return ordered_blocks, active_local, flat_mask.contiguous(), active_neurons.contiguous()

    def _prefetch_mlp_blocks_for_layer(
        self,
        layer_idx: int,
        active_blocks: torch.Tensor,
    ) -> None:
        """Start H2D transfer for MLP gate+up blocks on _h2d_stream before attention runs.

        Results are stored in _sparse_block_transfer_cache so the call inside
        _sparse_mlp_forward_fast gets a cache hit — the transfer already finished.
        Only has effect if _h2d_stream is available (H2D overlap enabled).
        """
        if self._h2d_stream is None:
            return
        prefix = f"model.layers.{int(layer_idx)}.mlp."
        try:
            gate_param = self._get_sparse_4bit_param(f"{prefix}gate_proj.weight", require_raw=False)
            up_param = self._get_sparse_4bit_param(f"{prefix}up_proj.weight", require_raw=False)
        except Exception:
            return
        if gate_param is None or up_param is None:
            return
        max_valid_blocks = min(
            int(gate_param.get("blocks_per_col", 0)),
            int(up_param.get("blocks_per_col", 0)),
        )
        if max_valid_blocks <= 0:
            return

        # Build ordered_blocks (CPU tensor ops — same logic as _build_sparse_active_layout)
        active_blocks_cpu = active_blocks.detach().to(device=torch.device("cpu"), dtype=torch.long).reshape(-1)
        ordered_blocks = self._order_blocks_for_layer_hot_cache(layer_idx, active_blocks_cpu)
        ordered_blocks = ordered_blocks[(ordered_blocks >= 0) & (ordered_blocks < int(max_valid_blocks))]
        ordered_blocks = ordered_blocks.unique(sorted=False).to(dtype=torch.long, device=torch.device("cpu")).contiguous()
        ordered_blocks = self._order_blocks_for_layer_hot_cache(layer_idx, ordered_blocks).contiguous()
        if int(ordered_blocks.numel()) == 0:
            return

        try:
            self._prepare_sparse_blocks_for_param_pair(
                gate_param,
                up_param,
                ordered_blocks=ordered_blocks,
                layer_idx=int(layer_idx),
                tag_prefix="mlp_sparse_gate_up",
            )
        except Exception:
            pass  # Non-fatal: actual call in _sparse_mlp_forward_fast will re-try

    def _sparse_mlp_forward_fast(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        active_blocks: torch.Tensor,
        _oom_retry_depth: int = 0,
    ) -> torch.Tensor:
        if (
            layer_idx in self._sparse_routing
            and str(self._sparse_routing[int(layer_idx)].get("block_domain", "intermediate")) != "intermediate"
        ):
            raise RuntimeError(
                f"Layer {int(layer_idx)} does not provide intermediate FFN routing data required for "
                f"{_EXACT_BLOCKWISE_SPARSE} execution."
            )

        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return self._sparse_mlp_forward(layer_idx, mlp, hidden, active_blocks)

        block_size = self._sparse_block_size
        flat_hidden = hidden.view(-1, hidden.shape[-1])
        prefix = f"model.layers.{int(layer_idx)}.mlp."

        gate_param = self._get_sparse_4bit_param(f"{prefix}gate_proj.weight", require_raw=False)
        up_param = self._get_sparse_4bit_param(f"{prefix}up_proj.weight", require_raw=False)
        down_param = self._get_sparse_4bit_param(f"{prefix}down_proj.weight", require_raw=False)
        routing = self._sparse_routing.get(int(layer_idx))
        block_bank_layout = routing.get("block_bank_layout") if isinstance(routing, dict) else None
        if isinstance(block_bank_layout, MlpBlockBankLayout):
            validate_intermediate_mlp_block_bank_params(
                layout=block_bank_layout,
                gate_in_features=int(gate_param["in_features"]),
                gate_out_features=int(gate_param["out_features"]),
                up_in_features=int(up_param["in_features"]),
                up_out_features=int(up_param["out_features"]),
                down_in_features=int(down_param["in_features"]),
                down_out_features=int(down_param["out_features"]),
            )
        max_valid_blocks = min(
            int(gate_param["blocks_per_col"]),
            int(up_param["blocks_per_col"]),
            int(down_param["in_features"]) // int(block_size),
        )
        ordered_blocks, active_local, flat_mask, active_neurons = self._build_sparse_active_layout(
            layer_idx=layer_idx,
            active_blocks=active_blocks,
            rows=int(flat_hidden.shape[0]),
            block_size=int(block_size),
            max_valid_blocks=int(max_valid_blocks),
            dtype=flat_hidden.dtype,
        )
        num_active_blocks = int(ordered_blocks.shape[0])
        if num_active_blocks <= 0:
            return torch.zeros_like(hidden)

        K_S = num_active_blocks * block_size
        H_in = int(gate_param["in_features"])

        gate_packed_gpu, gate_absmax_gpu, up_packed_gpu, up_absmax_gpu = self._prepare_sparse_blocks_for_param_pair(
            gate_param,
            up_param,
            ordered_blocks=ordered_blocks,
            layer_idx=int(layer_idx),
            tag_prefix="mlp_sparse_gate_up",
        )


        self._wait_for_h2d_stream()

        H_out = int(down_param["out_features"])
        I_in = int(down_param["in_features"])
        qbs = int(down_param["quant_block_size"])
        bytes_per_row = I_in // 2
        bytes_per_cblk = block_size // 2
        absmax_per_row = I_in // qbs

        _down_prefill_mode = str(getattr(self, "_sparse_mlp_prefill_mode", "dense"))
        _prefill_dense_for_down = bool(
            str(self._traffic_current_phase or "idle") == "prefill"
            and _down_prefill_mode == "dense"
        )
        _down_hot_cache_prefill = bool(
            str(self._traffic_current_phase or "idle") == "prefill"
            and _down_prefill_mode == "hot_cache"
        )

        _use_down_vram_only = _down_hot_cache_prefill and down_param.get("vram_hot_down") is not None



        gathered_down_packed_gpu = None
        gathered_down_absmax_gpu = None
        cached_down = self._get_sparse_transfer_cache(
            self._downproj_transfer_cache,
            full_name=str(down_param.get("full_name", "")).strip(),
            ordered_blocks=ordered_blocks,
        )
        if cached_down is not None:
            gathered_down_packed_gpu, gathered_down_absmax_gpu = cached_down

        down_hot_cache = None if _prefill_dense_for_down else down_param.get("vram_hot_down")
        if gathered_down_packed_gpu is None and gathered_down_absmax_gpu is None and down_hot_cache is not None:
            try:
                if not bool(down_hot_cache.get("h2d_ready", True)):
                    self._wait_for_h2d_stream()
                    down_hot_cache["h2d_ready"] = True
                lookup = down_hot_cache["lookup_cpu"].index_select(0, ordered_blocks.to(dtype=torch.long))
                hot_positions_cpu, hot_slots, cold_blocks, cold_mask_cpu = self._partition_cached_block_lookup(
                    lookup,
                    ordered_blocks=ordered_blocks,
                    device=self.device,
                )

                hot_cols_packed = None
                hot_cols_absmax = None
                if hot_slots is not None:
                    if str(self._traffic_current_phase or "idle") == "decode":
                        self._decode_down_hot_blocks_hit += int(hot_slots.numel())
                        scores_cpu = down_hot_cache.get("scores_cpu")
                        if torch.is_tensor(scores_cpu) and int(scores_cpu.numel()) >= int(hot_slots.numel()):
                            hot_slots_cpu = hot_slots.detach().to(device=torch.device("cpu"), dtype=torch.long)
                            scores_cpu.index_add_(
                                0,
                                hot_slots_cpu,
                                torch.ones(int(hot_slots_cpu.numel()), dtype=torch.float32),
                            )
                    hot_cols_packed = down_hot_cache["packed_cols_gpu"].index_select(1, hot_slots)
                    hot_cols_absmax = down_hot_cache["absmax_cols_gpu"].index_select(1, hot_slots)

                cold_cols_packed = None
                cold_cols_absmax = None
                if int(cold_blocks.numel()) > 0 and not _use_down_vram_only:
                    if str(self._traffic_current_phase or "idle") == "decode":
                        self._decode_down_cold_blocks_streamed += int(cold_blocks.numel())
                    down_name = str(down_param.get("full_name", "")).strip()
                    cold_packed_cpu, cold_absmax_cpu = self._load_down_proj_cold_cols(
                        down_param, cold_blocks,
                        H_out=H_out, bytes_per_row=bytes_per_row,
                        bytes_per_cblk=bytes_per_cblk, absmax_per_row=absmax_per_row,
                        block_size=block_size, qbs=qbs,
                        down_name=down_name,
                    )
                    cold_cols_packed = self._copy_cpu_to_gpu(
                        cold_packed_cpu,
                        dtype=torch.uint8,
                        layer_idx=int(layer_idx),
                        tag="mlp_sparse_down_packed",
                    )
                    cold_cols_absmax = self._copy_cpu_to_gpu(
                        cold_absmax_cpu,
                        dtype=torch.float32,
                        layer_idx=int(layer_idx),
                        tag="mlp_sparse_down_absmax",
                    )
                    self._promote_down_proj_hot_columns_from_gpu(
                        down_param,
                        cold_blocks=cold_blocks,
                        cold_cols_packed_gpu=cold_cols_packed,
                        cold_cols_absmax_gpu=cold_cols_absmax,
                        layer_idx=int(layer_idx),
                    )

                if (hot_cols_packed is not None and hot_cols_absmax is not None) or (
                    cold_cols_packed is not None and cold_cols_absmax is not None
                ):
                    gathered_down_packed_gpu, gathered_down_absmax_gpu = self._merge_columnwise_parts(
                        total_blocks=int(ordered_blocks.numel()),
                        hot_positions_cpu=hot_positions_cpu,
                        hot_packed=hot_cols_packed,
                        cold_mask_cpu=cold_mask_cpu,
                        cold_packed=cold_cols_packed,
                        hot_absmax=hot_cols_absmax,
                        cold_absmax=cold_cols_absmax,
                    )
            except Exception as _e:
                if not _is_cuda_oom_error(_e):
                    raise
                self._disable_vram_hot_cache("cuda_oom_during_down_hot_prepare")
                gathered_down_packed_gpu = None
                gathered_down_absmax_gpu = None
        if (gathered_down_packed_gpu is None or gathered_down_absmax_gpu is None) and not _use_down_vram_only:
            if str(self._traffic_current_phase or "idle") == "decode":
                self._decode_down_cold_blocks_streamed += int(ordered_blocks.numel())

            down_name = str(down_param.get("full_name", "")).strip()
            gathered_down_packed_cpu, gathered_down_absmax_cpu = self._load_down_proj_cold_cols(
                down_param, ordered_blocks,
                H_out=H_out, bytes_per_row=bytes_per_row,
                bytes_per_cblk=bytes_per_cblk, absmax_per_row=absmax_per_row,
                block_size=block_size, qbs=qbs,
                down_name=down_name,
            )
            try:
                gathered_down_packed_gpu = self._copy_cpu_to_gpu(
                    gathered_down_packed_cpu,
                    dtype=torch.uint8,
                    layer_idx=int(layer_idx),
                    tag="mlp_sparse_down_packed",
                )
                gathered_down_absmax_gpu = self._copy_cpu_to_gpu(
                    gathered_down_absmax_cpu,
                    dtype=torch.float32,
                    layer_idx=int(layer_idx),
                    tag="mlp_sparse_down_absmax",
                )
                self._promote_down_proj_hot_columns_from_gpu(
                    down_param,
                    cold_blocks=ordered_blocks,
                    cold_cols_packed_gpu=gathered_down_packed_gpu,
                    cold_cols_absmax_gpu=gathered_down_absmax_gpu,
                    layer_idx=int(layer_idx),
                )
            except Exception as _e:
                if not _is_cuda_oom_error(_e):
                    raise
                self._disable_vram_hot_cache("cuda_oom_during_down_dma")
                gathered_down_packed_gpu = self._copy_cpu_to_gpu(
                    gathered_down_packed_cpu,
                    dtype=torch.uint8,
                    layer_idx=int(layer_idx),
                    tag="mlp_sparse_down_packed",
                )
                gathered_down_absmax_gpu = self._copy_cpu_to_gpu(
                    gathered_down_absmax_cpu,
                    dtype=torch.float32,
                    layer_idx=int(layer_idx),
                    tag="mlp_sparse_down_absmax",
                )
        if gathered_down_packed_gpu is not None and gathered_down_absmax_gpu is not None:
            self._put_sparse_transfer_cache(
                self._downproj_transfer_cache,
                full_name=str(down_param.get("full_name", "")).strip(),
                ordered_blocks=ordered_blocks,
                packed_gpu=gathered_down_packed_gpu,
                absmax_gpu=gathered_down_absmax_gpu,
            )
        _triton_was_enabled = bool(self._triton_fused_sparse_mlp)
        triton_out = self._sparse_mlp_forward_fast_triton(
            hidden=hidden,
            flat_hidden=flat_hidden,
            ordered_blocks=ordered_blocks,
            active_local=active_local,
            flat_mask=flat_mask,
            active_neurons=active_neurons,
            gate_bias=getattr(gate_proj, "bias", None),
            up_bias=getattr(up_proj, "bias", None),
            down_bias=getattr(down_proj, "bias", None),
            gate_param=gate_param,
            up_param=up_param,
            down_param=down_param,
            gate_packed_gpu=gate_packed_gpu,
            gate_absmax_gpu=gate_absmax_gpu,
            up_packed_gpu=up_packed_gpu,
            up_absmax_gpu=up_absmax_gpu,
            down_packed_gpu=gathered_down_packed_gpu,
            down_absmax_gpu=gathered_down_absmax_gpu,
        )
        if triton_out is not None:
            return triton_out
        if _triton_was_enabled and (not self._triton_fused_sparse_mlp) and int(_oom_retry_depth) < 1:
            if self._h2d_stream is not None:
                with contextlib.suppress(Exception):
                    self._wait_for_h2d_stream()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return self._sparse_mlp_forward_fast(
                layer_idx,
                mlp,
                hidden,
                active_blocks,
                _oom_retry_depth=int(_oom_retry_depth) + 1,
            )
        if self._h2d_stream is not None and (gathered_down_packed_gpu is not None or gathered_down_absmax_gpu is not None):
            try:
                self._wait_for_h2d_stream()
            except Exception as _e:
                if not _is_cuda_oom_error(_e):
                    raise
                self._disable_vram_hot_cache("cuda_oom_after_triton_fallback_wait")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        def _gather_and_dequant(
            param: dict[str, Any],
            packed_gpu: torch.Tensor,
            absmax_gpu: torch.Tensor,
            bias_param: torch.Tensor | None,
        ) -> torch.Tensor:
            out_gpu = torch.empty((K_S, H_in), dtype=self.dtype, device=self.device)
            _bnb_dequant_impl(
                packed_gpu.reshape(-1),
                absmax_gpu.reshape(-1),
                param["quant_block_size"],
                param["quant_type"],
                out_gpu.dtype,
                out=out_gpu,
            )

            bias_gpu = None
            if bias_param is not None:
                bias_gpu = bias_param[active_neurons].to(device=self.device, dtype=self.dtype)

            return F.linear(flat_hidden, out_gpu, bias_gpu) * flat_mask

        try:
            gate = _gather_and_dequant(gate_param, gate_packed_gpu, gate_absmax_gpu, getattr(gate_proj, "bias", None))
            up = _gather_and_dequant(up_param, up_packed_gpu, up_absmax_gpu, getattr(up_proj, "bias", None))

            activated = F.silu(gate) * up
            del gate, up

            if gathered_down_packed_gpu is None or gathered_down_absmax_gpu is None:
                raise RuntimeError("Sparse MLP down_proj GPU buffers were not prepared")

            code_gpu = down_param.get("code_gpu")
            if code_gpu is None:
                code_gpu = down_param["code"].to(device=self.device, dtype=torch.float32)
            code_gpu = code_gpu.to(device=self.device, dtype=self.dtype)
            packed_down = gathered_down_packed_gpu.view(H_out, num_active_blocks, bytes_per_cblk)
            down_weight_pairs = torch.empty(
                (H_out, num_active_blocks, bytes_per_cblk, 2),
                dtype=self.dtype,
                device=self.device,
            )
            hi_idx = ((packed_down >> 4) & 0x0F).to(dtype=torch.long)
            down_weight_pairs[..., 0] = code_gpu.index_select(0, hi_idx.reshape(-1)).view_as(hi_idx)
            del hi_idx
            lo_idx = (packed_down & 0x0F).to(dtype=torch.long)
            down_weight_pairs[..., 1] = code_gpu.index_select(0, lo_idx.reshape(-1)).view_as(lo_idx)
            del lo_idx
            down_weight_active = down_weight_pairs.reshape(H_out, num_active_blocks, block_size)
            down_weight_active.mul_(gathered_down_absmax_gpu.unsqueeze(-1).to(dtype=self.dtype))
            down_weight_active = down_weight_active.reshape(H_out, K_S).contiguous()

            down_bias = getattr(down_proj, "bias", None)
            bias_gpu = None if down_bias is None else down_bias.to(device=self.device, dtype=self.dtype)

            out = F.linear(activated, down_weight_active, bias_gpu)
            return out.view_as(hidden)
        except Exception as _e:
            if not _is_cuda_oom_error(_e):
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _k_dim = int(active_blocks.shape[-1]) if active_blocks.ndim > 0 else 1
            if int(_oom_retry_depth) < 2 and _k_dim > 1:
                _new_k = max(1, _k_dim // 2)
                if int(self._sparse_runtime_top_k) <= 0 or _new_k < int(self._sparse_runtime_top_k):
                    self._sparse_runtime_top_k = int(_new_k)
                    print(
                        f"[sparse] runtime top_k clamped to {int(self._sparse_runtime_top_k)} after CUDA OOM.",
                        flush=True,
                    )
                _reduced_blocks = active_blocks[..., :_new_k].contiguous()
                print(
                    f"[sparse] CUDA OOM in sparse MLP fallback; retrying with top_k={_new_k}.",
                    flush=True,
                )
                return self._sparse_mlp_forward_fast(
                    layer_idx,
                    mlp,
                    hidden,
                    _reduced_blocks,
                    _oom_retry_depth=int(_oom_retry_depth) + 1,
                )
            print(
                "[sparse] CUDA OOM in sparse MLP fallback at minimum retry budget; "
                "using zero MLP contribution for this layer.",
                flush=True,
            )
            return torch.zeros_like(hidden)

    def _sparse_mlp_forward_fast_triton(
        self,
        *,
        hidden: torch.Tensor,
        flat_hidden: torch.Tensor,
        ordered_blocks: torch.Tensor,
        active_local: torch.Tensor,
        flat_mask: torch.Tensor,
        active_neurons: torch.Tensor,
        gate_bias: torch.Tensor | None,
        up_bias: torch.Tensor | None,
        down_bias: torch.Tensor | None,
        gate_param: dict[str, Any],
        up_param: dict[str, Any],
        down_param: dict[str, Any],
        gate_packed_gpu: torch.Tensor,
        gate_absmax_gpu: torch.Tensor,
        up_packed_gpu: torch.Tensor,
        up_absmax_gpu: torch.Tensor,
        down_packed_gpu: torch.Tensor | None,
        down_absmax_gpu: torch.Tensor | None,
    ) -> torch.Tensor | None:
        self._decode_backend_name = "dequant_sparse_fallback"
        if not self._triton_fused_sparse_mlp:
            return None
        if triton_fused_sparse_mlp_decode_4bit is None:
            return None
        if down_packed_gpu is None or down_absmax_gpu is None:
            return None

        block_size = int(self._sparse_block_size)
        hidden_size = int(flat_hidden.shape[-1])
        top_k = int(ordered_blocks.numel())
        rows = int(flat_hidden.shape[0])
        gate_code_gpu = gate_param.get("code_gpu")
        if gate_code_gpu is None:
            gate_code_gpu = gate_param["code"].to(device=self.device, dtype=torch.float32)
        up_code_gpu = up_param.get("code_gpu")
        if up_code_gpu is None:
            up_code_gpu = up_param["code"].to(device=self.device, dtype=torch.float32)
        down_code_gpu = down_param.get("code_gpu")
        if down_code_gpu is None:
            down_code_gpu = down_param["code"].to(device=self.device, dtype=torch.float32)

        can_use_single_kernel = bool(
            self._single_kernel_decode_enabled
            and triton_sparse_mlp_decode_4bit_single_kernel_sm75 is not None
            and self.device.type == "cuda"
            and torch.cuda.get_device_capability(self.device) == (7, 5)
            and rows == 1
            and int(block_size) == 32
            and top_k > 0
            and int(active_local.numel()) > 0
            and bool((active_local >= 0).all().item())
        )
        if can_use_single_kernel:
            try:
                block_out = int(self._single_kernel_mlp_block_out or 64)
                if block_out not in {32, 64}:
                    block_out = 64
                self._ensure_single_kernel_mlp_scratch(hidden_size=hidden_size, block_out=block_out)
                if self._single_kernel_mlp_out_accum is None:
                    raise RuntimeError("single-kernel sparse MLP scratch allocation failed")
                self._wait_for_h2d_stream()
                out_buf = torch.empty((1, hidden_size), device=self.device, dtype=torch.float16)
                down_single = triton_sparse_mlp_decode_4bit_single_kernel_sm75(
                    flat_hidden,
                    ordered_blocks,
                    gate_packed_gpu,
                    gate_absmax_gpu,
                    gate_code_gpu,
                    up_packed_gpu,
                    up_absmax_gpu,
                    up_code_gpu,
                    down_packed_gpu,
                    down_absmax_gpu,
                    down_code_gpu,
                    out_buf,
                    self._single_kernel_mlp_out_accum,
                    hidden_size,
                    int(block_size),
                    int(gate_param["quant_block_size"]),
                    int(top_k),
                )
                if down_bias is not None:
                    down_bias_gpu = down_bias.to(device=self.device, dtype=down_single.dtype)
                    down_single = down_single + down_bias_gpu.view(1, -1)
                self._decode_backend_name = "single_kernel_sparse_decode_sm75"
                return down_single.view_as(hidden).to(dtype=hidden.dtype)
            except Exception as exc:
                if not hasattr(self, "_single_kernel_fail_count"):
                    self._single_kernel_fail_count = 0
                self._single_kernel_fail_count += 1
                print(
                    f"[sparse] single-kernel Triton decode failed (attempt {self._single_kernel_fail_count}): {exc}",
                    flush=True,
                )
                traceback.print_exc()
                if self._single_kernel_fail_count >= 3:
                    print("[sparse] disabling single-kernel path after 3 failures", flush=True)
                    self._single_kernel_decode_enabled = False

        try:
            active_dim = int(flat_mask.shape[1])

            gate_bias_gpu = None
            if gate_bias is not None:
                gate_bias_gpu = gate_bias[active_neurons].to(device=self.device, dtype=flat_hidden.dtype)
            up_bias_gpu = None
            if up_bias is not None:
                up_bias_gpu = up_bias[active_neurons].to(device=self.device, dtype=flat_hidden.dtype)
            down_bias_gpu = None
            if down_bias is not None:
                down_bias_gpu = down_bias.to(device=self.device, dtype=flat_hidden.dtype)

            self._wait_for_h2d_stream()
            down = triton_fused_sparse_mlp_decode_4bit(
                flat_hidden,
                active_local,
                flat_mask,
                gate_packed_weight=gate_packed_gpu.reshape(-1),
                gate_absmax=gate_absmax_gpu.reshape(-1),
                gate_code=gate_code_gpu,
                gate_input_dim=int(gate_param["in_features"]),
                gate_quant_block_size=int(gate_param["quant_block_size"]),
                gate_bias=gate_bias_gpu,
                up_packed_weight=up_packed_gpu.reshape(-1),
                up_absmax=up_absmax_gpu.reshape(-1),
                up_code=up_code_gpu,
                up_input_dim=int(up_param["in_features"]),
                up_quant_block_size=int(up_param["quant_block_size"]),
                up_bias=up_bias_gpu,
                down_packed_weight=down_packed_gpu.reshape(-1),
                down_absmax=down_absmax_gpu.reshape(-1),
                down_code=down_code_gpu,
                down_out_features=int(down_param["out_features"]),
                down_in_features=int(active_dim),
                down_quant_block_size=int(block_size),
                down_bias=down_bias_gpu,
                block_size=block_size,
            )
            self._decode_backend_name = "fused_sparse_decode_v1"
            return down.view_as(hidden).to(dtype=hidden.dtype)
        except Exception as exc:
            print(f"[sparse] Triton fused 4-bit path failed; falling back to dequant path: {exc}", flush=True)
            self._triton_fused_sparse_mlp = False
            self._decode_backend_name = "dequant_sparse_fallback"
            return None
    def _dense_mlp_forward_streaming_fast_details(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        *,
        capture_intermediate_block_scores: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return self._dense_mlp_forward_streaming(mlp, hidden), None
        if not self._ensure_mlp_proj_staging():
            return torch.zeros_like(hidden), None

        flat_hidden = hidden.view(-1, hidden.shape[-1])
        prefix = f"model.layers.{int(layer_idx)}.mlp."




        max_numel = max(
            gate_proj.weight.numel(),
            up_proj.weight.numel(),
            down_proj.weight.numel(),
        )
        if self._mlp_proj_staging is None or int(self._mlp_proj_staging.numel()) < int(max_numel):
            raise RuntimeError("MLP projection staging buffer is unavailable or undersized")
        staging = self._mlp_proj_staging[:max_numel]

        def _linear_stream(x: torch.Tensor, linear: nn.Module, param_name: str) -> torch.Tensor:
            weight = getattr(linear, "weight", None)
            if weight is None:
                raise RuntimeError("Dense MLP streaming path requires linear.weight")
            w0, w1 = weight.shape[0], weight.shape[1]
            weight_view = staging[: weight.numel()].view(w0, w1)
            raw = self.loader.load_parameter(param_name)
            self._record_h2d_bytes(
                int(raw.numel() * raw.element_size()),
                layer_idx=layer_idx,
                tag="mlp_dense_weight",
            )

            weight_view.copy_(raw.to(dtype=self.dtype))
            del raw
            bias_gpu = self._load_optional_bias(
                param_name,
                getattr(linear, "bias", None),
                device=x.device,
                dtype=x.dtype,
                layer_idx=layer_idx,
                tag="mlp_dense_bias",
            )
            y = F.linear(x, weight_view.to(dtype=x.dtype), bias_gpu)
            del bias_gpu
            return y

        gate = _linear_stream(flat_hidden, gate_proj, f"{prefix}gate_proj.weight")
        up = _linear_stream(flat_hidden, up_proj, f"{prefix}up_proj.weight")
        act = F.silu(gate) * up
        del gate, up
        block_scores = None
        if capture_intermediate_block_scores:
            intermediate_dim = int(act.shape[-1])
            block_size = int(self._sparse_block_size)
            if intermediate_dim % max(block_size, 1) != 0:
                raise RuntimeError(
                    f"Layer {int(layer_idx)} intermediate size {intermediate_dim} is not divisible by "
                    f"sparse block_size {block_size}."
                )
            block_scores = act.view(int(act.shape[0]), intermediate_dim // block_size, block_size).abs().mean(dim=-1)
        out = _linear_stream(act, down_proj, f"{prefix}down_proj.weight")
        return out.view_as(hidden), block_scores

    def _dense_mlp_forward_streaming_fast(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        out, _block_scores = self._dense_mlp_forward_streaming_fast_details(
            layer_idx,
            mlp,
            hidden,
            capture_intermediate_block_scores=False,
        )
        return out

    def _dense_guard_mlp_forward_exact_chunked_4bit(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        *,
        chunk_blocks: int | None = None,
        _oom_retry_depth: int = 0,
    ) -> torch.Tensor:
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return self._dense_mlp_forward_streaming(mlp, hidden)

        prefix = f"model.layers.{int(layer_idx)}.mlp."
        if self.device.type != "cuda":
            return self._dense_mlp_forward_streaming(mlp, hidden)

        flat_hidden = hidden.view(-1, hidden.shape[-1]).to(device=self.device, dtype=self.dtype)
        
        def _load_and_linear(param_name: str, proj: nn.Module, x: torch.Tensor, is_gate: bool = False) -> torch.Tensor:
            out_feat = getattr(proj, "out_features", None)
            in_feat = getattr(proj, "in_features", None)
            if out_feat is None or in_feat is None:
                raise RuntimeError("Dense guard requires shape-configured MLP projections")
                
            weight = torch.empty((out_feat, in_feat), device=self.device, dtype=self.dtype)
            
            self.loader.load_parameter_into(
                name=param_name,
                out=weight,
                dtype=self.dtype,
                staging=self._nf4_staging,
                absmax_staging=self._absmax_staging,
                nested_absmax_staging=self._nested_absmax_staging,
                state2_absmax_staging=self._state2_absmax_staging,
                code_staging=self._code_staging,
            )
            
            bias_gpu = self._load_optional_bias(
                param_name, getattr(proj, "bias", None), device=self.device, dtype=self.dtype, layer_idx=layer_idx, tag="mlp_guard_bias"
            )
            
            y = F.linear(x, weight, bias_gpu)
            del weight, bias_gpu
            if is_gate:
                y = F.silu(y)
            return y
            
        gate = _load_and_linear(f"{prefix}gate_proj.weight", gate_proj, flat_hidden, is_gate=True)
        up = _load_and_linear(f"{prefix}up_proj.weight", up_proj, flat_hidden, is_gate=False)
        act = gate * up
        del gate, up
        
        out = _load_and_linear(f"{prefix}down_proj.weight", down_proj, act, is_gate=False)
        return out.view_as(hidden)

    def _mlp_forward_dispatch(
        self,
        layer_idx: int,
        layer: LlamaDecoderLayer,
        mlp_input: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self._mlp_static_skip_mask is not None
            and int(layer_idx) < int(self._mlp_static_skip_mask.shape[0])
            and bool(self._mlp_static_skip_mask[int(layer_idx)].item())
        ):
            return torch.zeros_like(mlp_input)


        _lidx = int(layer_idx)
        if (
            bool(getattr(self, "_enable_smc", False))
            and str(self._traffic_current_phase or "idle") == "decode"
            and bool(self._smc_valid[_lidx].item())
            and _lidx not in self._mlp_static_skip_set
            and mlp_input.ndim >= 1
        ):
            _x_flat = mlp_input.view(-1)
            _ref = self._smc_x_in[_lidx]
            _delta = (_x_flat - _ref).norm()
            _ref_norm = _ref.norm().clamp(min=1e-6)
            if (_delta / _ref_norm).item() < self._smc_threshold:
                return self._smc_out[_lidx].view_as(mlp_input)

        _current_mlp_prefill_mode = str(getattr(self, "_sparse_mlp_prefill_mode", "dense"))
        if (
            str(self._traffic_current_phase or "idle") == "prefill"
            and _current_mlp_prefill_mode == "dense"
        ):
            return self._dense_guard_mlp_forward_exact_chunked_4bit(layer_idx, layer.mlp, mlp_input)
        if (
            str(self._traffic_current_phase or "idle") == "prefill"
            and _current_mlp_prefill_mode == "hot_cache"
            and not self._hot_cache_calibration_active
        ):
            hot_blocks = self._mlp_hot_blocks_by_layer.get(int(layer_idx))
            if hot_blocks is not None and int(hot_blocks.numel()) > 0:


                max_hot = int(max(1, self._sparse_top_k))
                if int(hot_blocks.numel()) > max_hot:
                    hot_blocks = hot_blocks[:max_hot]


                rows = int(mlp_input.view(-1, mlp_input.shape[-1]).shape[0])
                hot_blocks_1row = hot_blocks.unsqueeze(0)  # [1, K]
                if rows == 1:
                    return self._sparse_mlp_forward_fast(layer_idx, layer.mlp, mlp_input, hot_blocks_1row)
                # Multi-token prefill: Triton kernel only supports seq_len=1.
                # Process each row independently — attention is already computed for all
                # tokens together; only MLP needs the per-row workaround.
                flat_in = mlp_input.view(rows, -1)
                row_outs = []
                for _r in range(rows):
                    _row = flat_in[_r : _r + 1].unsqueeze(0)
                    row_outs.append(self._sparse_mlp_forward_fast(layer_idx, layer.mlp, _row, hot_blocks_1row))
                return torch.cat(row_outs, dim=1).view_as(mlp_input)

            return self._dense_guard_mlp_forward_exact_chunked_4bit(layer_idx, layer.mlp, mlp_input)
        if layer_idx in self._sparse_routing:
            if str(self._sparse_routing[int(layer_idx)].get("block_domain", "intermediate")) != "intermediate":
                raise RuntimeError(
                    f"Layer {int(layer_idx)} does not provide intermediate FFN routing data required for "
                    f"{_EXACT_BLOCKWISE_SPARSE} execution."
                )
            # Always route from mlp_input (post-attention) for correctness.
            # Pre-routed blocks (from hidden_norm pre-attention) are in
            # _sparse_block_transfer_cache as H2D cache hits when routing matches.
            self._mlp_prefetch_active_blocks.pop(int(layer_idx), None)
            active_blocks = self._route_sparse_mlp(mlp_input, layer_idx)
            if active_blocks is None:
                raise RuntimeError(f"Sparse basis routing disappeared for layer {int(layer_idx)}")
            self._record_hot_cache_calibration_blocks(layer_idx, active_blocks)
            _mlp_out = self._sparse_mlp_forward_fast(layer_idx, layer.mlp, mlp_input, active_blocks)
        else:
            _mlp_out = self._dense_guard_mlp_forward_exact_chunked_4bit(layer_idx, layer.mlp, mlp_input)


        if bool(getattr(self, "_enable_smc", False)) and str(self._traffic_current_phase or "idle") == "decode":
            self._smc_x_in[_lidx].copy_(mlp_input.view(-1), non_blocking=True)
            self._smc_out[_lidx].copy_(_mlp_out.view(-1), non_blocking=True)
            self._smc_valid[_lidx] = True

        return _mlp_out



    def _should_use_attn_share_for_layer(self, layer_idx: int) -> bool:
        if int(layer_idx) not in self._attn_share_layer_state:
            return False
        return not (str(self._traffic_current_phase or "idle") == "prefill" and self._attn_share_prefill_mode == "dense")

    def _load_shared_attn_qo(self, layer_idx: int) -> None:
        state = self._attn_share_layer_state.get(int(layer_idx))
        if state is None:
            raise RuntimeError(f"No attention-sharing state registered for layer {int(layer_idx)}")
        group = self._attn_share_groups.get(str(state["group_id"]))
        if group is None:
            raise RuntimeError(
                f"Layer {int(layer_idx)} references missing attention-sharing group {state['group_id']!r}"
            )

        head_dim = int(getattr(self.config, "head_dim", 0))
        q_skel = self._layer_skeleton.self_attn.q_proj.weight
        o_skel = self._layer_skeleton.self_attn.o_proj.weight
        head_perm = torch.as_tensor(state["head_perm"], dtype=torch.long, device=torch.device("cpu")).contiguous()

        if "q_base_u_heads" in group:
            q_base_u_cpu = _unpermute_headwise_tensor(group["q_base_u_heads"], head_perm)
            q_base_v_cpu = _unpermute_headwise_tensor(group["q_base_v_heads"], head_perm)
            q_base_u_gpu = self._copy_cpu_to_gpu(
                q_base_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_q_base_u_heads",
            )
            q_base_v_gpu = self._copy_cpu_to_gpu(
                q_base_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_q_base_v_heads",
            )
            self._wait_for_h2d_stream()
            q_recon = torch.bmm(q_base_u_gpu, q_base_v_gpu).reshape_as(q_skel)
            q_skel.copy_(q_recon)

            q_resid_u_heads_cpu = state.get("q_resid_u_heads")
            q_resid_v_heads_cpu = state.get("q_resid_v_heads")
            if torch.is_tensor(q_resid_u_heads_cpu) and torch.is_tensor(q_resid_v_heads_cpu):
                q_resid_u_heads_cpu = _unpermute_headwise_tensor(q_resid_u_heads_cpu, head_perm)
                q_resid_v_heads_cpu = _unpermute_headwise_tensor(q_resid_v_heads_cpu, head_perm)
                q_resid_u_heads_gpu = self._copy_cpu_to_gpu(
                    q_resid_u_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_q_resid_u_heads",
                )
                q_resid_v_heads_gpu = self._copy_cpu_to_gpu(
                    q_resid_v_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_q_resid_v_heads",
                )
                self._wait_for_h2d_stream()
                q_skel.add_(torch.bmm(q_resid_u_heads_gpu, q_resid_v_heads_gpu).reshape_as(q_skel))

            o_base_u_cpu = _unpermute_headwise_tensor(group["o_base_u_heads"], head_perm)
            o_base_v_cpu = _unpermute_headwise_tensor(group["o_base_v_heads"], head_perm)
            o_base_u_gpu = self._copy_cpu_to_gpu(
                o_base_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_o_base_u_heads",
            )
            o_base_v_gpu = self._copy_cpu_to_gpu(
                o_base_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_o_base_v_heads",
            )
            self._wait_for_h2d_stream()
            o_recon = torch.bmm(o_base_u_gpu, o_base_v_gpu).permute(1, 0, 2).reshape_as(o_skel)
            o_skel.copy_(o_recon)

            o_resid_u_heads_cpu = state.get("o_resid_u_heads")
            o_resid_v_heads_cpu = state.get("o_resid_v_heads")
            if torch.is_tensor(o_resid_u_heads_cpu) and torch.is_tensor(o_resid_v_heads_cpu):
                o_resid_u_heads_cpu = _unpermute_headwise_tensor(o_resid_u_heads_cpu, head_perm)
                o_resid_v_heads_cpu = _unpermute_headwise_tensor(o_resid_v_heads_cpu, head_perm)
                o_resid_u_heads_gpu = self._copy_cpu_to_gpu(
                    o_resid_u_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_o_resid_u_heads",
                )
                o_resid_v_heads_gpu = self._copy_cpu_to_gpu(
                    o_resid_v_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_o_resid_v_heads",
                )
                self._wait_for_h2d_stream()
                o_skel.add_(torch.bmm(o_resid_u_heads_gpu, o_resid_v_heads_gpu).permute(1, 0, 2).reshape_as(o_skel))

            self._attn_loaded_q_rows = None
            self._attn_loaded_o_cols = None
            self._attn_qo_state = "shared"
            return

        q_base_u_cpu = _unpermute_q_factor_rows(group["q_base_u"], head_perm, head_dim=head_dim)
        q_base_v_cpu = group["q_base_v"]
        q_base_u_gpu = self._copy_cpu_to_gpu(
            q_base_u_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_q_base_u",
        )
        q_base_v_gpu = self._copy_cpu_to_gpu(
            q_base_v_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_q_base_v",
        )
        self._wait_for_h2d_stream()
        torch.mm(q_base_u_gpu, q_base_v_gpu, out=q_skel)

        q_resid_u_cpu = state.get("q_resid_u")
        q_resid_v_cpu = state.get("q_resid_v")
        if torch.is_tensor(q_resid_u_cpu) and torch.is_tensor(q_resid_v_cpu):
            q_resid_u_cpu = _unpermute_q_factor_rows(q_resid_u_cpu, head_perm, head_dim=head_dim)
            q_resid_u_gpu = self._copy_cpu_to_gpu(
                q_resid_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_q_resid_u",
            )
            q_resid_v_gpu = self._copy_cpu_to_gpu(
                q_resid_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_q_resid_v",
            )
            self._wait_for_h2d_stream()
            q_skel.addmm_(q_resid_u_gpu, q_resid_v_gpu)

        o_base_u_cpu = group["o_base_u"]
        o_base_v_cpu = _unpermute_o_factor_cols(group["o_base_v"], head_perm, head_dim=head_dim)
        o_base_u_gpu = self._copy_cpu_to_gpu(
            o_base_u_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_o_base_u",
        )
        o_base_v_gpu = self._copy_cpu_to_gpu(
            o_base_v_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_o_base_v",
        )
        self._wait_for_h2d_stream()
        torch.mm(o_base_u_gpu, o_base_v_gpu, out=o_skel)

        o_resid_u_cpu = state.get("o_resid_u")
        o_resid_v_cpu = state.get("o_resid_v")
        if torch.is_tensor(o_resid_u_cpu) and torch.is_tensor(o_resid_v_cpu):
            o_resid_v_cpu = _unpermute_o_factor_cols(o_resid_v_cpu, head_perm, head_dim=head_dim)
            o_resid_u_gpu = self._copy_cpu_to_gpu(
                o_resid_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_o_resid_u",
            )
            o_resid_v_gpu = self._copy_cpu_to_gpu(
                o_resid_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_o_resid_v",
            )
            self._wait_for_h2d_stream()
            o_skel.addmm_(o_resid_u_gpu, o_resid_v_gpu)

        self._attn_loaded_q_rows = None
        self._attn_loaded_o_cols = None
        self._attn_qo_state = "shared"

    def _load_shared_attn_kv(self, layer_idx: int) -> None:
        state = self._attn_share_layer_state.get(int(layer_idx))
        if state is None:
            raise RuntimeError(f"No attention-sharing state registered for layer {int(layer_idx)}")
        group = self._attn_share_groups.get(str(state["group_id"]))
        if group is None:
            raise RuntimeError(
                f"Layer {int(layer_idx)} references missing attention-sharing group {state['group_id']!r}"
            )

        head_dim = int(getattr(self.config, "head_dim", 0))
        k_skel = self._layer_skeleton.self_attn.k_proj.weight
        v_skel = self._layer_skeleton.self_attn.v_proj.weight
        kv_head_perm = torch.as_tensor(state["kv_head_perm"], dtype=torch.long, device=torch.device("cpu")).contiguous()

        if "k_base_u_heads" in group:
            k_base_u_cpu = _unpermute_headwise_tensor(group["k_base_u_heads"], kv_head_perm)
            k_base_v_cpu = _unpermute_headwise_tensor(group["k_base_v_heads"], kv_head_perm)
            k_base_u_gpu = self._copy_cpu_to_gpu(
                k_base_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_k_base_u_heads",
            )
            k_base_v_gpu = self._copy_cpu_to_gpu(
                k_base_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_k_base_v_heads",
            )
            self._wait_for_h2d_stream()
            k_skel.copy_(torch.bmm(k_base_u_gpu, k_base_v_gpu).reshape_as(k_skel))

            k_resid_u_heads_cpu = state.get("k_resid_u_heads")
            k_resid_v_heads_cpu = state.get("k_resid_v_heads")
            if torch.is_tensor(k_resid_u_heads_cpu) and torch.is_tensor(k_resid_v_heads_cpu):
                k_resid_u_heads_cpu = _unpermute_headwise_tensor(k_resid_u_heads_cpu, kv_head_perm)
                k_resid_v_heads_cpu = _unpermute_headwise_tensor(k_resid_v_heads_cpu, kv_head_perm)
                k_resid_u_heads_gpu = self._copy_cpu_to_gpu(
                    k_resid_u_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_k_resid_u_heads",
                )
                k_resid_v_heads_gpu = self._copy_cpu_to_gpu(
                    k_resid_v_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_k_resid_v_heads",
                )
                self._wait_for_h2d_stream()
                k_skel.add_(torch.bmm(k_resid_u_heads_gpu, k_resid_v_heads_gpu).reshape_as(k_skel))

            v_base_u_cpu = _unpermute_headwise_tensor(group["v_base_u_heads"], kv_head_perm)
            v_base_v_cpu = _unpermute_headwise_tensor(group["v_base_v_heads"], kv_head_perm)
            v_base_u_gpu = self._copy_cpu_to_gpu(
                v_base_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_v_base_u_heads",
            )
            v_base_v_gpu = self._copy_cpu_to_gpu(
                v_base_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_v_base_v_heads",
            )
            self._wait_for_h2d_stream()
            v_skel.copy_(torch.bmm(v_base_u_gpu, v_base_v_gpu).reshape_as(v_skel))

            v_resid_u_heads_cpu = state.get("v_resid_u_heads")
            v_resid_v_heads_cpu = state.get("v_resid_v_heads")
            if torch.is_tensor(v_resid_u_heads_cpu) and torch.is_tensor(v_resid_v_heads_cpu):
                v_resid_u_heads_cpu = _unpermute_headwise_tensor(v_resid_u_heads_cpu, kv_head_perm)
                v_resid_v_heads_cpu = _unpermute_headwise_tensor(v_resid_v_heads_cpu, kv_head_perm)
                v_resid_u_heads_gpu = self._copy_cpu_to_gpu(
                    v_resid_u_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_v_resid_u_heads",
                )
                v_resid_v_heads_gpu = self._copy_cpu_to_gpu(
                    v_resid_v_heads_cpu,
                    dtype=self.dtype,
                    layer_idx=layer_idx,
                    tag="attn_share_v_resid_v_heads",
                )
                self._wait_for_h2d_stream()
                v_skel.add_(torch.bmm(v_resid_u_heads_gpu, v_resid_v_heads_gpu).reshape_as(v_skel))

            self._kv_loaded_cols = None
            return

        k_base_u_cpu = _unpermute_q_factor_rows(group["k_base_u"], kv_head_perm, head_dim=head_dim)
        k_base_v_cpu = group["k_base_v"]
        k_base_u_gpu = self._copy_cpu_to_gpu(
            k_base_u_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_k_base_u",
        )
        k_base_v_gpu = self._copy_cpu_to_gpu(
            k_base_v_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_k_base_v",
        )
        self._wait_for_h2d_stream()
        torch.mm(k_base_u_gpu, k_base_v_gpu, out=k_skel)

        k_resid_u_cpu = state.get("k_resid_u")
        k_resid_v_cpu = state.get("k_resid_v")
        if torch.is_tensor(k_resid_u_cpu) and torch.is_tensor(k_resid_v_cpu):
            k_resid_u_cpu = _unpermute_q_factor_rows(k_resid_u_cpu, kv_head_perm, head_dim=head_dim)
            k_resid_u_gpu = self._copy_cpu_to_gpu(
                k_resid_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_k_resid_u",
            )
            k_resid_v_gpu = self._copy_cpu_to_gpu(
                k_resid_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_k_resid_v",
            )
            self._wait_for_h2d_stream()
            k_skel.addmm_(k_resid_u_gpu, k_resid_v_gpu)

        v_base_u_cpu = _unpermute_q_factor_rows(group["v_base_u"], kv_head_perm, head_dim=head_dim)
        v_base_v_cpu = group["v_base_v"]
        v_base_u_gpu = self._copy_cpu_to_gpu(
            v_base_u_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_v_base_u",
        )
        v_base_v_gpu = self._copy_cpu_to_gpu(
            v_base_v_cpu,
            dtype=self.dtype,
            layer_idx=layer_idx,
            tag="attn_share_v_base_v",
        )
        self._wait_for_h2d_stream()
        torch.mm(v_base_u_gpu, v_base_v_gpu, out=v_skel)

        v_resid_u_cpu = state.get("v_resid_u")
        v_resid_v_cpu = state.get("v_resid_v")
        if torch.is_tensor(v_resid_u_cpu) and torch.is_tensor(v_resid_v_cpu):
            v_resid_u_cpu = _unpermute_q_factor_rows(v_resid_u_cpu, kv_head_perm, head_dim=head_dim)
            v_resid_u_gpu = self._copy_cpu_to_gpu(
                v_resid_u_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_v_resid_u",
            )
            v_resid_v_gpu = self._copy_cpu_to_gpu(
                v_resid_v_cpu,
                dtype=self.dtype,
                layer_idx=layer_idx,
                tag="attn_share_v_resid_v",
            )
            self._wait_for_h2d_stream()
            v_skel.addmm_(v_resid_u_gpu, v_resid_v_gpu)

        self._kv_loaded_cols = None

    def _select_static_attention_heads(
        self,
        importance: torch.Tensor,
        *,
        max_heads: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> torch.Tensor:
        scores = importance.detach().to(device=torch.device("cpu"), dtype=torch.float32).contiguous().view(-1)
        total_heads = min(int(num_heads), int(scores.numel()))
        if total_heads <= 0:
            return torch.empty((0,), dtype=torch.long)

        target_heads = max(1, min(int(max_heads), total_heads))
        selection_mode = str(getattr(self, "_attn_head_selection_mode", "topk"))
        kv_heads = max(1, min(int(num_kv_heads), total_heads))
        if selection_mode != "balanced_gqa" or kv_heads <= 1:
            return torch.topk(scores[:total_heads], k=target_heads, largest=True).indices.to(
                dtype=torch.long
            ).contiguous()

        ranked_heads_by_group: list[list[int]] = []
        for group_idx in range(kv_heads):
            group_start = int(group_idx * total_heads // kv_heads)
            group_end = int((group_idx + 1) * total_heads // kv_heads)
            if group_end <= group_start:
                continue
            group_heads = list(range(group_start, group_end))
            group_heads.sort(key=lambda head_idx: (-float(scores[head_idx].item()), int(head_idx)))
            ranked_heads_by_group.append(group_heads)

        if not ranked_heads_by_group:
            return torch.topk(scores[:total_heads], k=target_heads, largest=True).indices.to(
                dtype=torch.long
            ).contiguous()

        selected_heads: list[int] = []
        selected_set: set[int] = set()
        max_group_depth = max(len(group_heads) for group_heads in ranked_heads_by_group)
        for depth_idx in range(max_group_depth):
            round_candidates: list[tuple[float, int]] = []
            for group_heads in ranked_heads_by_group:
                if depth_idx < len(group_heads):
                    head_idx = int(group_heads[depth_idx])
                    round_candidates.append((float(scores[head_idx].item()), head_idx))
            round_candidates.sort(key=lambda item: (-item[0], item[1]))
            for _score, head_idx in round_candidates:
                if head_idx in selected_set:
                    continue
                selected_heads.append(head_idx)
                selected_set.add(head_idx)
                if len(selected_heads) >= target_heads:
                    return torch.tensor(selected_heads, dtype=torch.long).contiguous()

        global_heads = list(range(total_heads))
        global_heads.sort(key=lambda head_idx: (-float(scores[head_idx].item()), int(head_idx)))
        for head_idx in global_heads:
            if head_idx in selected_set:
                continue
            selected_heads.append(int(head_idx))
            selected_set.add(int(head_idx))
            if len(selected_heads) >= target_heads:
                break

        return torch.tensor(selected_heads[:target_heads], dtype=torch.long).contiguous()

    def _get_attn_active_heads(self, layer_idx: int) -> torch.Tensor | None:
        """Return sorted active head indices [K] for this layer, or None (dense).

        Blends static calibration importance with a live Taylor state_z signal.
        state_z[0, g, :].norm() reflects how much KV group g has accumulated
        context - groups that have not fired much are down-weighted.
        """
        if self._attn_sparse_disabled_reason is not None:
            return None
        if int(layer_idx) in getattr(self, "_attn_force_dense_layers", set()):
            return None

        static_indices = self._attn_active_head_indices.get(layer_idx)
        if static_indices is None:
            return None

        static_imp = self._attn_head_importance.get(layer_idx)
        if static_imp is None:
            k0 = max(1, min(int(self._attn_active_heads), int(static_indices.numel())))
            return static_indices[:k0].sort().values

        taylor_cache = self._taylor_caches[layer_idx]
        if taylor_cache is None:
            k0 = max(1, min(int(self._attn_active_heads), int(static_indices.numel())))
            return static_indices[:k0].sort().values

        state_z = taylor_cache.state_z
        num_kv_heads = int(state_z.shape[1])
        group_norms = state_z[0].norm(dim=-1).float().to(device=torch.device("cpu"))

        heads_per_group = max(1, int(static_imp.shape[0]) // max(num_kv_heads, 1))
        head_norms = group_norms.repeat_interleave(heads_per_group)
        if int(head_norms.numel()) < int(static_imp.shape[0]):
            pad = int(static_imp.shape[0]) - int(head_norms.numel())
            head_norms = torch.cat([head_norms, head_norms.new_zeros((pad,))], dim=0)
        elif int(head_norms.numel()) > int(static_imp.shape[0]):
            head_norms = head_norms[: int(static_imp.shape[0])]

        static_indices_cpu = static_indices.to(device=torch.device("cpu"), dtype=torch.long)
        static_imp_pool = static_imp.to(device=torch.device("cpu")).index_select(0, static_indices_cpu)
        head_norms_pool = head_norms.index_select(0, static_indices_cpu)
        norm_s = static_imp_pool / static_imp_pool.max().clamp_min(1e-8)
        norm_d = head_norms_pool / head_norms_pool.max().clamp_min(1e-8)
        combined = norm_s * norm_d

        total_heads = int(combined.shape[0])
        max_heads = int(self._attn_max_active_heads) if int(self._attn_max_active_heads) > 0 else total_heads
        max_heads = max(1, min(max_heads, total_heads))
        min_heads = max(1, min(int(self._attn_min_active_heads), max_heads))

        if self._attn_dynamic_threshold > 0.0:
            dynamic_floor = combined.max().clamp_min(1e-8) * float(self._attn_dynamic_threshold)
            dynamic_count = int((combined >= dynamic_floor).sum().item())
            target_k = max(min_heads, min(max_heads, dynamic_count))
        else:
            target_k = max(min_heads, min(max_heads, int(self._attn_active_heads)))

        self._attn_runtime_head_counts[layer_idx] = int(target_k)
        local_top = torch.topk(combined, k=target_k, largest=True).indices
        return static_indices_cpu.index_select(0, local_top).sort().values

    def _shrink_attn_sparse_budget_on_oom(self, *, head_dim: int, hidden_size: int) -> bool:
        """Reduce sparse-attention head budget to recover VRAM on CUDA OOM."""
        cur_max = int(self._attn_max_active_heads) if int(self._attn_max_active_heads) > 0 else int(self._attn_active_heads)
        min_heads = max(1, int(self._attn_min_active_heads))
        if cur_max <= min_heads:
            return False

        new_max = max(min_heads, cur_max // 2)
        if new_max >= cur_max:
            return False

        self._attn_max_active_heads = int(new_max)
        self._attn_active_heads = max(1, min(int(self._attn_active_heads), int(new_max)))
        for _lidx, _indices in list(self._attn_active_head_indices.items()):
            if int(_indices.numel()) > int(new_max):
                self._attn_active_head_indices[_lidx] = _indices[: int(new_max)].contiguous()

        if self.device.type == "cuda":
            try:
                self._attn_q_head_staging = torch.empty(
                    int(new_max) * int(head_dim) * int(hidden_size),
                    dtype=self.dtype,
                    device=self.device,
                )
            except Exception:
                return False
            torch.cuda.empty_cache()

        print(
            f"[sparse_attn] CUDA OOM recovery: reducing max active heads to {int(new_max)} and retrying.",
            flush=True,
        )
        return True
    def _get_sparse_4bit_attn_meta(self, full_name: str, *, head_dim: int) -> dict[str, Any]:
        """Cache NF4 metadata for an attention projection weight.

        Stores only the dequantised absmax (flat float32, ~16 MB) and the NF4
        codebook (16 floats).  Raw packed bytes are never stored here — they are
        fetched O(1) from loader._ram_cache on each call, keeping the per-entry
        overhead to ~64 MB instead of doubling the 128 MB byte buffer.
        """
        cached = self._attn_sparse_param_meta.get(full_name)
        if cached is not None:
            return cached

        raw_weight, quant_aux = self.loader._load_raw_for_param(full_name)
        if not quant_aux:
            raise RuntimeError(f"Sparse 4-bit attention path: expected NF4 weights for {full_name!r}")

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        absmax = quant_state.absmax
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax + quant_state.offset

        out_features = int(quant_state.shape[0])
        in_features  = int(quant_state.shape[1])

        cached = {
            "out_features":    out_features,
            "in_features":     in_features,
            "head_dim":        int(head_dim),
            "num_heads_total": out_features // int(head_dim),
            "quant_block_size": int(quant_state.blocksize),
            "quant_type":      str(quant_state.quant_type),
            "code":            self.loader.prepare_h2d_source(
                quant_state.code.to(dtype=torch.float32).contiguous(),
                dtype=torch.float32,
                pin_override=False,
            ),


            "absmax_flat":     self.loader.prepare_h2d_source(
                absmax.to(dtype=torch.float32).contiguous(),
                dtype=torch.float32,
                pin_override=False,
            ),
        }
        self._attn_sparse_param_meta[full_name] = cached
        return cached

    def _derive_gqa_kv_groups_from_q_heads(
        self,
        active_heads: torch.Tensor | None,
        *,
        kv_out_heads: int | None = None,
    ) -> torch.Tensor | None:
        """Map sparse Q heads to the GQA K/V groups required by those heads."""
        if active_heads is None or int(active_heads.numel()) <= 0:
            return None

        num_q_heads = int(getattr(self.config, "num_attention_heads", 0) or 0)
        num_kv_heads = int(getattr(self.config, "num_key_value_heads", 0) or 0)
        if num_q_heads <= 0:
            num_q_heads = int(getattr(self.config, "num_heads", 128))
        if num_kv_heads <= 0:
            num_kv_heads = int(kv_out_heads) if kv_out_heads is not None and int(kv_out_heads) > 0 else num_q_heads
        if num_q_heads <= 0 or num_kv_heads <= 0:
            return None

        heads_per_group = max(1, int(num_q_heads) // max(1, int(num_kv_heads)))
        heads_cpu = active_heads.to(device=torch.device("cpu"), dtype=torch.long).reshape(-1)
        heads_cpu = heads_cpu[(heads_cpu >= 0) & (heads_cpu < int(num_q_heads))]
        if int(heads_cpu.numel()) <= 0:
            return None

        groups = torch.div(heads_cpu, heads_per_group, rounding_mode="floor")
        groups = groups.clamp_(0, max(0, int(num_kv_heads) - 1)).unique(sorted=True).to(dtype=torch.long)
        if int(groups.numel()) <= 0:
            return None
        if int(groups.numel()) >= int(num_kv_heads):
            return None
        return groups.contiguous()

    def _load_gqa_sparse_kv_groups(
        self,
        layer_idx: int,
        kv_groups: torch.Tensor,
        layer: LlamaDecoderLayer,
        *,
        head_dim: int,
    ) -> None:
        """Load only GQA-derived K/V output-row groups into the zeroed layer skeleton."""
        if self._nf4_staging is None or self._attn_q_head_staging is None:
            return

        groups_cpu = kv_groups.to(device=torch.device("cpu"), dtype=torch.long).reshape(-1).unique(sorted=True)
        if int(groups_cpu.numel()) <= 0:
            return

        row_offsets = (
            groups_cpu.unsqueeze(-1) * int(head_dim)
            + torch.arange(int(head_dim), dtype=torch.long)
        ).reshape(-1).contiguous()

        def _load_one_projection(proj_name: str, weight: torch.Tensor) -> None:
            full_name = f"model.layers.{int(layer_idx)}.self_attn.{proj_name}.weight"
            meta = self._get_sparse_4bit_attn_meta(full_name, head_dim=head_dim)
            raw_weight, _ = self.loader._load_raw_for_param(full_name)

            out_features = int(meta["out_features"])
            in_features = int(meta["in_features"])
            qbs = int(meta["quant_block_size"])
            rows = row_offsets[(row_offsets >= 0) & (row_offsets < out_features)].contiguous()
            if int(rows.numel()) <= 0:
                return

            packed_row_bytes = in_features // 2
            packed_cpu = raw_weight.view(out_features, packed_row_bytes).index_select(0, rows).contiguous()

            absmax_per_row = max(1, in_features // max(1, qbs))
            absmax_cpu = (
                meta["absmax_flat"]
                .view(out_features, absmax_per_row)
                .index_select(0, rows)
                .contiguous()
            )

            packed_flat = packed_cpu.reshape(-1)
            n_packed = int(packed_flat.numel())
            if n_packed > int(self._nf4_staging.numel()):
                raise RuntimeError(
                    f"GQA sparse K/V staging buffer too small for {full_name}: "
                    f"need {n_packed} bytes, have {int(self._nf4_staging.numel())}"
                )

            gathered = self._copy_cpu_to_existing_gpu(self._nf4_staging[:n_packed], packed_flat)
            self._record_h2d_bytes(
                int(gathered.numel() * gathered.element_size()),
                layer_idx=layer_idx,
                tag=f"attn_gqa_sparse_{proj_name}_packed",
            )
            absmax_gpu = self._copy_cpu_to_gpu(
                absmax_cpu.reshape(-1),
                dtype=torch.float32,
                layer_idx=layer_idx,
                tag=f"attn_gqa_sparse_{proj_name}_absmax",
            )
            self._wait_for_h2d_stream()

            out_numel = int(rows.numel()) * int(in_features)
            if out_numel > int(self._attn_q_head_staging.numel()):
                partial = torch.empty((int(rows.numel()), int(in_features)), dtype=self.dtype, device=self.device)
            else:
                partial = self._attn_q_head_staging[:out_numel].view(int(rows.numel()), int(in_features))
            _bnb_dequant_impl(
                self._nf4_staging[:n_packed],
                absmax_gpu,
                qbs,
                str(meta["quant_type"]),
                self.dtype,
                out=partial,
            )

            row_idx_gpu = rows.to(device=self.device, dtype=torch.long)
            weight.index_copy_(0, row_idx_gpu, partial)

        self._kv_loaded_cols = None
        _load_one_projection("k_proj", layer.self_attn.k_proj.weight)
        self._wait_h2d_stream_for_current()
        _load_one_projection("v_proj", layer.self_attn.v_proj.weight)
        self._gqa_kv_loaded_rows = row_offsets.to(device=self.device, dtype=torch.long)

    def _load_sparse_attn_heads(
        self,
        layer_idx: int,
        active_heads: torch.Tensor,
        *,
        head_dim: int,
        hidden_size: int,
    ) -> None:
        """Load only the active K heads' NF4 bytes for q_proj and o_proj.

        q_proj (row gather → GPU dequant):
            Gathers K head-row slices of NF4 bytes on CPU, transfers to the
            pre-allocated _nf4_staging buffer (no cudaMalloc), dequantises with
            _bnb_dequant_impl into _attn_q_head_staging, then scatters into the
            zeroed skeleton q_proj.weight at the active row positions.

        o_proj (column gather → CPU decode → GPU scatter):
            Adapts the down_proj column-gather pattern from _sparse_mlp_forward_fast.
            Gathers K head-column slices on CPU, decodes NF4 nibbles via the code
            table, scales by absmax, transfers FP16 result, and scatters into the
            zeroed skeleton o_proj.weight.

        k_proj and v_proj are loaded normally by _load_layer() — they are small
        (8.4 MB each for 405B GQA with 8 KV heads) and always needed in full.
        """
        if self._nf4_staging is None or self._attn_q_head_staging is None:
            return

        prefix = f"model.layers.{int(layer_idx)}."
        q_name = f"{prefix}self_attn.q_proj.weight"
        o_name = f"{prefix}self_attn.o_proj.weight"

        meta_q = self._get_sparse_4bit_attn_meta(q_name, head_dim=head_dim)
        meta_o = self._get_sparse_4bit_attn_meta(o_name, head_dim=head_dim)

        active_cpu = active_heads.to(device=torch.device("cpu"), dtype=torch.long).unique(sorted=True)
        [int(v) for v in active_cpu.tolist()]
        K = int(active_cpu.shape[0])


        q_skel = self._layer_skeleton.self_attn.q_proj.weight
        o_skel = self._layer_skeleton.self_attn.o_proj.weight
        # Always do a full zero-fill on entry so no stale rows from a previous
        # token's hot-head cache short-circuit can corrupt the current token.
        self._clear_sparse_attn_qo_buffers(q_skel=q_skel, o_skel=o_skel, force_full=True)
        loaded_q_rows: list[torch.Tensor] = []
        loaded_o_cols: list[torch.Tensor] = []

        hot_cache = self._maybe_cache_sparse_attn_hot_heads(
            layer_idx=layer_idx,
            q_name=q_name,
            o_name=o_name,
            meta_q=meta_q,
            meta_o=meta_o,
        )
        if hot_cache is not None:
            if not bool(hot_cache.get("h2d_ready", True)):
                self._wait_for_h2d_stream()
                hot_cache["h2d_ready"] = True
            lookup = hot_cache["lookup_cpu"].index_select(0, active_cpu)
            hot_mask = lookup >= 0
            if bool(hot_mask.any().item()):
                hot_active_cpu = active_cpu[hot_mask].contiguous()
                hot_slots = lookup[hot_mask].to(device=self.device, dtype=torch.long)
                hot_active_gpu = hot_active_cpu.to(device=self.device, dtype=torch.long)
                K_hot = int(hot_active_cpu.shape[0])

                q_packed_gpu = hot_cache["q_packed_gpu"].index_select(0, hot_slots).reshape(-1)
                q_absmax_gpu = hot_cache["q_absmax_gpu"].index_select(0, hot_slots).reshape(-1)
                q_partial = self._attn_q_head_staging[: K_hot * head_dim * hidden_size].view(K_hot * head_dim, hidden_size)
                _bnb_dequant_impl(
                    q_packed_gpu,
                    q_absmax_gpu,
                    meta_q["quant_block_size"],
                    meta_q["quant_type"],
                    self.dtype,
                    out=q_partial,
                )
                row_idx = (
                    hot_active_gpu.unsqueeze(-1) * head_dim
                    + torch.arange(head_dim, device=self.device)
                ).reshape(-1)
                q_skel.index_copy_(0, row_idx, q_partial)
                loaded_q_rows.append(row_idx.detach().clone())

                o_packed_gpu = hot_cache["o_packed_gpu"].index_select(1, hot_slots).reshape(-1)
                o_absmax_gpu = hot_cache["o_absmax_gpu"].index_select(1, hot_slots).reshape(-1)
                o_partial = self._attn_q_head_staging[: hidden_size * K_hot * head_dim].view(hidden_size, K_hot * head_dim)
                _bnb_dequant_impl(
                    o_packed_gpu,
                    o_absmax_gpu,
                    meta_o["quant_block_size"],
                    meta_o["quant_type"],
                    self.dtype,
                    out=o_partial,
                )
                col_idx_gpu = (
                    hot_active_gpu.unsqueeze(-1) * head_dim
                    + torch.arange(head_dim, device=self.device)
                ).reshape(-1)
                o_skel.index_copy_(1, col_idx_gpu, o_partial)
                loaded_o_cols.append(col_idx_gpu.detach().clone())
            if bool(torch.all(lookup >= 0).item()):
                self._attn_loaded_q_rows = torch.cat(loaded_q_rows, dim=0).contiguous() if loaded_q_rows else None
                self._attn_loaded_o_cols = torch.cat(loaded_o_cols, dim=0).contiguous() if loaded_o_cols else None
                self._attn_qo_state = "sparse"
                self._debug_assert_sparse_attn_qo_zero_inactive(q_skel=q_skel, o_skel=o_skel)
                return
            active_cpu = active_cpu[lookup < 0].contiguous()
            active_heads = active_cpu.to(device=active_heads.device, dtype=torch.long)
            K = int(active_cpu.shape[0])
            if K <= 0:
                self._attn_loaded_q_rows = torch.cat(loaded_q_rows, dim=0).contiguous() if loaded_q_rows else None
                self._attn_loaded_o_cols = torch.cat(loaded_o_cols, dim=0).contiguous() if loaded_o_cols else None
                self._attn_qo_state = "sparse"
                self._debug_assert_sparse_attn_qo_zero_inactive(q_skel=q_skel, o_skel=o_skel)
                return


        H_out = int(meta_o["out_features"])
        H_in  = int(meta_o["in_features"])
        qbs   = int(meta_o["quant_block_size"])
        _head_key = tuple(active_cpu.tolist())
        _cpu_cache_key = (int(layer_idx), _head_key)
        _cpu_cached = self._sparse_attn_cpu_row_cache.get(_cpu_cache_key)
        if _cpu_cached is not None:
            self._cpu_row_cache_hits += 1
            if self._debug_attn_row_cache and (
                self._cpu_row_cache_hits <= 3 or (self._cpu_row_cache_hits % 126 == 0)
            ):
                _phase_dbg = str(self._traffic_current_phase or "?")
                print(f"[attn_row_cache] HIT  layer={layer_idx} phase={_phase_dbg} hits={self._cpu_row_cache_hits} misses={self._cpu_row_cache_misses}", flush=True)
            q_rows_cpu = _cpu_cached["q_rows"]
            q_abs_cpu  = _cpu_cached["q_abs"]
            o_cols_cpu = _cpu_cached["o_cols"]
            o_abs_cpu  = _cpu_cached["o_abs"]
        else:
            self._cpu_row_cache_misses += 1
            if self._debug_attn_row_cache and (
                self._cpu_row_cache_misses <= 5 or (self._cpu_row_cache_misses % 126 == 0)
            ):
                _phase_dbg = str(self._traffic_current_phase or "?")
                print(f"[attn_row_cache] MISS layer={layer_idx} phase={_phase_dbg} hits={self._cpu_row_cache_hits} misses={self._cpu_row_cache_misses}", flush=True)
            q_raw, _ = self.loader._load_raw_for_param(q_name)
            bytes_per_head_q = meta_q["in_features"] // 2 * head_dim
            packed_2d_q = q_raw.view(meta_q["num_heads_total"], bytes_per_head_q)
            q_rows_cpu = packed_2d_q.index_select(0, active_cpu).contiguous()

            absmax_per_head_q = head_dim * meta_q["in_features"] // meta_q["quant_block_size"]
            absmax_2d_q = meta_q["absmax_flat"].view(meta_q["num_heads_total"], absmax_per_head_q)
            q_abs_cpu = absmax_2d_q.index_select(0, active_cpu).contiguous()

            o_raw, _ = self.loader._load_raw_for_param(o_name)
            bytes_per_row_o    = H_in // 2
            bytes_per_head_col = head_dim // 2
            raw_2d_o = o_raw.view(H_out, bytes_per_row_o)
            o_col_offsets = (
                active_cpu.unsqueeze(-1) * bytes_per_head_col
                + torch.arange(bytes_per_head_col, dtype=torch.long)
            ).reshape(-1)
            o_cols_cpu = raw_2d_o.index_select(1, o_col_offsets).contiguous()

            absmax_per_head_col = head_dim // qbs
            absmax_per_row_o    = H_in // qbs
            absmax_2d_o = meta_o["absmax_flat"].view(H_out, absmax_per_row_o)
            o_abs_offsets = (
                active_cpu.unsqueeze(-1) * absmax_per_head_col
                + torch.arange(absmax_per_head_col, dtype=torch.long)
            ).reshape(-1)
            o_abs_cpu = absmax_2d_o.index_select(1, o_abs_offsets).contiguous()

            # Pin all tensors before caching: _stage_h2d_source_via_scratch
            # short-circuits for pinned tensors (no blocking event sync, no scratch
            # memcpy), so all subsequent tokens DMA directly from pinned → GPU.
            _pin = self.loader._maybe_pin_cpu_tensor
            self._sparse_attn_cpu_row_cache[_cpu_cache_key] = {
                "q_rows": _pin(q_rows_cpu.contiguous()),
                "q_abs":  _pin(q_abs_cpu.contiguous()),
                "o_cols": _pin(o_cols_cpu.contiguous()),
                "o_abs":  _pin(o_abs_cpu.contiguous()),
            }

        gathered_q = self._copy_cpu_to_existing_gpu(self._nf4_staging[: q_rows_cpu.numel()], q_rows_cpu.reshape(-1))
        n_q = gathered_q.numel()
        self._record_h2d_bytes(
            int(gathered_q.numel() * gathered_q.element_size()),
            layer_idx=layer_idx,
            tag="attn_sparse_q_packed",
        )
        absmax_q_gpu = self._copy_cpu_to_gpu(
            q_abs_cpu.reshape(-1),
            dtype=torch.float32,
            layer_idx=layer_idx,
            tag="attn_sparse_q_absmax",
        )

        self._wait_for_h2d_stream()
        q_partial = self._attn_q_head_staging[: K * head_dim * hidden_size].view(K * head_dim, hidden_size)
        _bnb_dequant_impl(
            self._nf4_staging[:n_q], absmax_q_gpu,
            meta_q["quant_block_size"], meta_q["quant_type"], self.dtype, out=q_partial,
        )
        active_gpu = active_heads.to(device=self.device, dtype=torch.long)
        row_idx = (
            active_gpu.unsqueeze(-1) * head_dim
            + torch.arange(head_dim, device=self.device)
        ).reshape(-1)
        q_skel.index_copy_(0, row_idx, q_partial)
        loaded_q_rows.append(row_idx.detach().clone())


        gathered_o = self._copy_cpu_to_existing_gpu(self._nf4_staging[: o_cols_cpu.numel()], o_cols_cpu.reshape(-1))
        n_o = gathered_o.numel()
        self._record_h2d_bytes(
            int(gathered_o.numel() * gathered_o.element_size()),
            layer_idx=layer_idx,
            tag="attn_sparse_o_packed",
        )
        absmax_o_gpu = self._copy_cpu_to_gpu(
            o_abs_cpu.reshape(-1),
            dtype=torch.float32,
            layer_idx=layer_idx,
            tag="attn_sparse_o_absmax",
        )
        self._wait_for_h2d_stream()

        o_partial = self._attn_q_head_staging[: H_out * K * head_dim].view(H_out, K * head_dim)
        _bnb_dequant_impl(
            self._nf4_staging[:n_o], absmax_o_gpu,
            qbs, meta_o["quant_type"], self.dtype, out=o_partial,
        )
        if self._debug_sync_cuda:
            torch.cuda.synchronize()
        col_idx_gpu = (
            active_gpu.unsqueeze(-1) * head_dim
            + torch.arange(head_dim, device=self.device)
        ).reshape(-1)
        o_skel.index_copy_(1, col_idx_gpu, o_partial)
        loaded_o_cols.append(col_idx_gpu.detach().clone())
        self._attn_loaded_q_rows = torch.cat(loaded_q_rows, dim=0).contiguous() if loaded_q_rows else None
        self._attn_loaded_o_cols = torch.cat(loaded_o_cols, dim=0).contiguous() if loaded_o_cols else None
        self._attn_qo_state = "sparse"
        self._debug_assert_sparse_attn_qo_zero_inactive(q_skel=q_skel, o_skel=o_skel)

    def _bootstrap_compact_attn_cache(self, layer_idx: int) -> dict[str, torch.Tensor | None]:
        cache = getattr(self, "_compact_attn_cache", None)
        if not isinstance(cache, dict):
            self._compact_attn_cache = {}
            cache = self._compact_attn_cache
        entry = cache.get(int(layer_idx))
        if isinstance(entry, dict):
            return entry

        key_tensor = None
        value_tensor = None
        dense_cache = getattr(self, "_dense_cache", None)
        key_cache = getattr(dense_cache, "key_cache", None)
        value_cache = getattr(dense_cache, "value_cache", None)
        if (
            isinstance(key_cache, list)
            and isinstance(value_cache, list)
            and 0 <= int(layer_idx) < len(key_cache)
            and 0 <= int(layer_idx) < len(value_cache)
            and torch.is_tensor(key_cache[int(layer_idx)])
            and torch.is_tensor(value_cache[int(layer_idx)])
        ):
            key_tensor = key_cache[int(layer_idx)].to(device=self.device, dtype=self.dtype).contiguous()
            value_tensor = value_cache[int(layer_idx)].to(device=self.device, dtype=self.dtype).contiguous()
        entry = {"k": key_tensor, "v": value_tensor}
        cache[int(layer_idx)] = entry
        return entry

    def _should_use_compact_sparse_attention(
        self,
        *,
        layer_idx: int,
        active_heads: torch.Tensor | None,
        use_attention_cache: bool,
        use_shared_attn: bool,
    ) -> bool:
        if not bool(getattr(self, "_compact_sparse_attn_decode", False)):
            return False
        if self.device.type != "cuda":
            return False
        if not bool(use_attention_cache):
            return False
        if str(self._traffic_current_phase or "idle") != "decode":
            return False
        if use_shared_attn:
            return False
        if int(layer_idx) in self.taylor_layer_set or int(layer_idx) in self._retrieval_layers:
            return False
        if active_heads is None or int(active_heads.numel()) <= 0:
            return False
        return True

    def _forward_compact_sparse_attn(
        self,
        *,
        layer_idx: int,
        layer: LlamaDecoderLayer,
        hidden_norm: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        active_heads: torch.Tensor,
        use_attention_cache: bool,
    ) -> torch.Tensor:
        cfg = self.config
        num_heads = int(getattr(cfg, "num_attention_heads", 0) or getattr(cfg, "num_heads", 0) or 0)
        num_kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or num_heads)
        hidden_size = int(getattr(cfg, "hidden_size", int(hidden_norm.shape[-1])))
        head_dim = int(getattr(cfg, "head_dim", 0) or max(1, hidden_size // max(num_heads, 1)))
        active_heads_gpu = (
            active_heads.to(device=self.device, dtype=torch.long).reshape(-1).unique(sorted=True).contiguous()
        )
        if int(active_heads_gpu.numel()) <= 0:
            return torch.zeros_like(hidden_norm)
        if int(active_heads_gpu[-1].item()) >= int(num_heads):
            raise RuntimeError(
                f"Layer {int(layer_idx)} sparse-attention head index {int(active_heads_gpu[-1].item())} "
                f"is out of range for num_heads={int(num_heads)}."
            )

        self._wait_for_h2d_stream()
        flat_hidden = hidden_norm.view(-1, hidden_norm.shape[-1]).to(device=self.device, dtype=self.dtype).contiguous()
        head_offsets = (
            active_heads_gpu.unsqueeze(-1) * int(head_dim)
            + torch.arange(int(head_dim), device=self.device, dtype=torch.long)
        ).reshape(-1)
        if int(head_offsets.numel()) != int(active_heads_gpu.numel()) * int(head_dim):
            raise RuntimeError("Sparse-attention head offsets are inconsistent with active head count")
        q_weight = layer.self_attn.q_proj.weight.index_select(0, head_offsets)
        q_bias_full = getattr(layer.self_attn.q_proj, "bias", None)
        q_bias = None
        if q_bias_full is not None:
            q_bias = q_bias_full.index_select(0, head_offsets).to(device=self.device, dtype=flat_hidden.dtype)
        q_raw = F.linear(flat_hidden, q_weight.to(dtype=flat_hidden.dtype), q_bias)
        q = q_raw.view(int(flat_hidden.shape[0]), int(active_heads_gpu.numel()), 1, int(head_dim))

        k_raw = layer.self_attn.k_proj(flat_hidden).view(int(flat_hidden.shape[0]), 1, int(num_kv_heads), int(head_dim)).transpose(1, 2)
        v_raw = layer.self_attn.v_proj(flat_hidden).view(int(flat_hidden.shape[0]), 1, int(num_kv_heads), int(head_dim)).transpose(1, 2)
        cos, sin = rope
        q, k_new = apply_rotary_pos_emb(q, k_raw, cos, sin)

        cache_entry = self._bootstrap_compact_attn_cache(int(layer_idx))
        past_k = cache_entry.get("k")
        past_v = cache_entry.get("v")
        if torch.is_tensor(past_k) and torch.is_tensor(past_v):
            k_all = torch.cat([past_k, k_new], dim=2)
            v_all = torch.cat([past_v, v_raw], dim=2)
        else:
            k_all = k_new
            v_all = v_raw
        if use_attention_cache:
            cache_entry["k"] = k_all.detach().contiguous()
            cache_entry["v"] = v_all.detach().contiguous()

        queries_per_group = max(1, int(num_heads) // max(1, int(num_kv_heads)))
        kv_group_idx = torch.div(active_heads_gpu, queries_per_group, rounding_mode="floor").clamp_(0, max(0, int(num_kv_heads) - 1))
        attn_out = torch.empty_like(q)
        if int(kv_group_idx.numel()) != int(active_heads_gpu.numel()):
            raise RuntimeError("Sparse-attention KV group mapping is inconsistent with active heads")
        for group_idx in kv_group_idx.unique(sorted=True).tolist():
            group_mask = kv_group_idx == int(group_idx)
            q_group = q[:, group_mask, :, :]
            k_group = k_all[:, int(group_idx): int(group_idx) + 1, :, :]
            v_group = v_all[:, int(group_idx): int(group_idx) + 1, :, :]
            out_group = F.scaled_dot_product_attention(
                q_group,
                k_group,
                v_group,
                scale=float(head_dim) ** -0.5,
            )
            attn_out[:, group_mask, :, :] = out_group

        compact_hidden = attn_out.transpose(1, 2).reshape(int(hidden_norm.shape[0]), int(hidden_norm.shape[1]), -1)
        o_weight = layer.self_attn.o_proj.weight.index_select(1, head_offsets)
        if int(o_weight.shape[1]) != int(head_offsets.numel()):
            raise RuntimeError("Sparse-attention o_proj gather shape does not match active-head ordering")
        o_bias = getattr(layer.self_attn.o_proj, "bias", None)
        if o_bias is not None:
            o_bias = o_bias.to(device=self.device, dtype=compact_hidden.dtype)
        out = F.linear(
            compact_hidden.view(-1, compact_hidden.shape[-1]).to(dtype=self.dtype),
            o_weight.to(dtype=self.dtype),
            o_bias,
        )
        self._compact_attn_decode_steps = int(getattr(self, "_compact_attn_decode_steps", 0)) + 1
        return out.view_as(hidden_norm).to(dtype=hidden_norm.dtype)

    def _batch_preload_layer(self, layer_idx: int) -> None:
        """Batch-load ALL tensors for a layer into RAM cache with a single safe_open per shard.

        On Windows, repeated safe_open calls on the same large shard crash after ~5 opens
        because Windows does not reliably reclaim the mmap VA region between opens of a
        5 GB file.  By grouping all reads from the same shard into one _load_exact_tensors
        call we reduce shard 1 opens from N (one per weight) to 1 — avoiding the crash.
        """
        if not self.loader._ram_cache_enabled:
            return
        prefix = f"model.layers.{int(layer_idx)}."

        phase = str(self._traffic_current_phase or "idle")
        allow_sparse_attn = not (
            phase == "prefill" and self._sparse_attn_prefill_mode == "dense"
        )
        use_shared_attn = self._should_use_attn_share_for_layer(layer_idx)
        active_attn_heads = (
            None
            if use_shared_attn
            else self._get_attn_active_heads(layer_idx) if allow_sparse_attn else None
        )
        allow_sparse_kv = not (
            phase == "prefill" and self._sparse_kv_prefill_mode == "dense"
        )

        skip_attn: set[str] = set()
        if active_attn_heads is not None:
            skip_attn.update({"self_attn.q_proj.weight", "self_attn.o_proj.weight"})
        if use_shared_attn:
            skip_attn.update(
                {
                    "self_attn.q_proj.weight",
                    "self_attn.o_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                }
            )
        elif (
            allow_sparse_kv
            and int(layer_idx) not in self._retrieval_layers
            and (int(layer_idx) in self._kv_routing or active_attn_heads is not None)
        ):
            skip_attn.update({"self_attn.k_proj.weight", "self_attn.v_proj.weight"})


        uncached_bases: list[str] = []
        for k, _dest, is_fp in self._layer_state_items:
            if not is_fp:
                continue
            if str(k).startswith("mlp."):
                continue
            if str(k) in skip_attn:
                continue
            full_name = f"{prefix}{k}"
            with self.loader._ram_cache_lock:
                already = full_name in self.loader._ram_cache
            if not already:
                uncached_bases.append(full_name)

        if not uncached_bases:
            return



        all_names: list[str] = []
        for base in uncached_bases:
            all_names.append(base)
            all_names.extend(self.loader._quant_aux_by_base.get(base, []))

        try:
            tensors = self.loader._load_exact_tensors(all_names)
        except Exception:
            return


        for base in uncached_bases:
            if base not in tensors:
                continue
            aux_names_for_base = self.loader._quant_aux_by_base.get(base, [])
            weight = tensors[base]
            quant_aux = {n: tensors[n] for n in aux_names_for_base if n in tensors}

            weight = self.loader._maybe_pin_cpu_tensor(weight.contiguous())
            quant_aux_pinned = {
                n: self.loader._maybe_pin_cpu_tensor(v.contiguous())
                for n, v in quant_aux.items()
            }

            with self.loader._ram_cache_lock:
                if base in self.loader._ram_cache:
                    continue
                self.loader._ram_cache[base] = (weight, quant_aux_pinned)
                if self.loader._ram_cache_limit_bytes is not None:
                    self.loader._ram_cache_lru[base] = None
                    self.loader._ram_cache_lru.move_to_end(base, last=True)
                    nbytes = ShardedSafetensorLoader._entry_nbytes(weight, quant_aux_pinned)
                    self.loader._ram_cache_entry_bytes[base] = nbytes
                    self.loader._ram_cache_current_bytes += nbytes
                    self.loader._evict_ram_cache_locked()

    def _load_layer(self, layer_idx: int) -> LlamaDecoderLayer:
        self._layer_skeleton.layer_idx = int(layer_idx)
        if hasattr(self._layer_skeleton, "self_attn"):
            self._layer_skeleton.self_attn.layer_idx = int(layer_idx)
        self._record_layer_visit(layer_idx)


        if self._enable_windows_batch_preload:
            self._batch_preload_layer(layer_idx)

        if self._show_progress and torch.cuda.is_available():
            try:
                _free_vram, _total_vram = torch.cuda.mem_get_info(self.device)
                _alloc_vram = torch.cuda.memory_allocated(self.device)
                _line = (
                    f"  [vram] layer {int(layer_idx) + 1:03d}/{self.num_layers}"
                    f"  free={_free_vram / 1e9:.2f} GB"
                    f"  alloc={_alloc_vram / 1e9:.2f} GB"
                    f"  total={_total_vram / 1e9:.2f} GB"
                    f"  phase={self._traffic_current_phase}"
                )
                print(f"\r{_line.ljust(int(self._progress_line_width))}", end="", flush=True)
            except Exception:
                pass

        layer_state_items = self._layer_state_items




        _allow_sparse_attn = not (
            str(self._traffic_current_phase or "idle") == "prefill"
            and self._sparse_attn_prefill_mode == "dense"
        )
        _use_shared_attn = self._should_use_attn_share_for_layer(layer_idx)
        _active_attn_heads = (
            None
            if _use_shared_attn
            else self._get_attn_active_heads(layer_idx) if _allow_sparse_attn else None
        )
        _head_dim = int(getattr(self.config, "head_dim", 128))
        _hidden   = int(getattr(self.config, "hidden_size", 16384))
        _allow_sparse_kv = not (
            str(self._traffic_current_phase or "idle") == "prefill"
            and self._sparse_kv_prefill_mode == "dense"
        )
        _gqa_kv_groups: torch.Tensor | None = None
        if (
            _active_attn_heads is not None
            and _allow_sparse_kv
            and not _use_shared_attn
            and layer_idx not in self._retrieval_layers
            and layer_idx not in self._kv_routing
        ):
            _gqa_kv_groups = self._derive_gqa_kv_groups_from_q_heads(_active_attn_heads)


        _skip_attn: set = set()
        if _active_attn_heads is not None:
            _skip_attn = _skip_attn | {"self_attn.q_proj.weight", "self_attn.o_proj.weight"}
        if _use_shared_attn:
            _skip_attn = _skip_attn | {
                "self_attn.q_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            }



        if layer_idx in self._kv_routing and _allow_sparse_kv and not _use_shared_attn and layer_idx not in self._retrieval_layers or _gqa_kv_groups is not None:
            _skip_attn = _skip_attn | {"self_attn.k_proj.weight", "self_attn.v_proj.weight"}


            self._layer_skeleton.self_attn.k_proj.weight.zero_()
            self._layer_skeleton.self_attn.v_proj.weight.zero_()
            self._kv_loaded_cols = None
            self._gqa_kv_loaded_rows = None
        else:
            self._gqa_kv_loaded_rows = None


        prefix = f"model.layers.{int(layer_idx)}."
        for k, dest, is_fp in layer_state_items:
            full_name = f"{prefix}{k}"
            if k.startswith("mlp.") and is_fp:
                continue
            if k in _skip_attn:
                continue
            if is_fp:
                try:
                    self.loader.load_parameter_into(
                        name=full_name,
                        out=dest,
                        dtype=self.dtype,
                        staging=self._nf4_staging,
                        absmax_staging=self._absmax_staging,
                        nested_absmax_staging=self._nested_absmax_staging,
                        state2_absmax_staging=self._state2_absmax_staging,
                        code_staging=self._code_staging,
                        byte_counter=lambda n, _layer_idx=layer_idx, _k=k: self._record_h2d_bytes(
                            int(n),
                            layer_idx=_layer_idx,
                            tag=f"layer_load:{_k}",
                        ),
                    )
                except Exception as _e:
                    if not _is_cuda_oom_error(_e):
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    try:
                        self.loader.load_parameter_into(
                            name=full_name,
                            out=dest,
                            dtype=self.dtype,
                            staging=self._nf4_staging,
                            absmax_staging=self._absmax_staging,
                            nested_absmax_staging=self._nested_absmax_staging,
                            state2_absmax_staging=self._state2_absmax_staging,
                            code_staging=self._code_staging,
                            byte_counter=lambda n, _layer_idx=layer_idx, _k=k: self._record_h2d_bytes(
                                int(n),
                                layer_idx=_layer_idx,
                                tag=f"layer_load:{_k}",
                            ),
                        )
                    except Exception as _e2:
                        if not _is_cuda_oom_error(_e2):
                            raise
                        raise RuntimeError(
                            f"CUDA OOM loading {full_name}; refusing to continue with zeroed layer weights."
                        ) from _e2
            else:
                raw = self.loader.load_parameter(full_name)
                dest.copy_(raw)

        if _active_attn_heads is None and not _use_shared_attn:
            self._attn_loaded_q_rows = None
            self._attn_loaded_o_cols = None
            self._attn_qo_state = "dense"

        if _use_shared_attn:
            self._load_shared_attn_qo(layer_idx)
            self._load_shared_attn_kv(layer_idx)

        if _gqa_kv_groups is not None and not _use_shared_attn:
            while True:
                try:
                    self._wait_h2d_stream_for_current()
                    self._load_gqa_sparse_kv_groups(
                        layer_idx,
                        _gqa_kv_groups,
                        layer=self._layer_skeleton,
                        head_dim=_head_dim,
                    )
                    break
                except Exception as _e:
                    if not _is_cuda_oom_error(_e):
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        f"[sparse_attn] CUDA OOM loading GQA sparse K/V for layer {layer_idx}; "
                        "falling back to full K/V load for this layer.",
                        flush=True,
                    )
                    self.loader.load_parameter_into(
                        name=f"{prefix}self_attn.k_proj.weight",
                        out=self._layer_skeleton.self_attn.k_proj.weight,
                        dtype=self.dtype,
                        staging=self._nf4_staging,
                        absmax_staging=self._absmax_staging,
                        nested_absmax_staging=self._nested_absmax_staging,
                        state2_absmax_staging=self._state2_absmax_staging,
                        code_staging=self._code_staging,
                        byte_counter=lambda n, _layer_idx=layer_idx: self._record_h2d_bytes(
                            int(n),
                            layer_idx=_layer_idx,
                            tag="layer_load:self_attn.k_proj.weight",
                        ),
                    )
                    self.loader.load_parameter_into(
                        name=f"{prefix}self_attn.v_proj.weight",
                        out=self._layer_skeleton.self_attn.v_proj.weight,
                        dtype=self.dtype,
                        staging=self._nf4_staging,
                        absmax_staging=self._absmax_staging,
                        nested_absmax_staging=self._nested_absmax_staging,
                        state2_absmax_staging=self._state2_absmax_staging,
                        code_staging=self._code_staging,
                        byte_counter=lambda n, _layer_idx=layer_idx: self._record_h2d_bytes(
                            int(n),
                            layer_idx=_layer_idx,
                            tag="layer_load:self_attn.v_proj.weight",
                        ),
                    )
                    self._gqa_kv_loaded_rows = None
                    break



        if _active_attn_heads is not None and not _use_shared_attn:
            if int(layer_idx) in self._attn_zero_only_layers:
                self._clear_sparse_attn_qo_buffers(
                    q_skel=self._layer_skeleton.self_attn.q_proj.weight,
                    o_skel=self._layer_skeleton.self_attn.o_proj.weight,
                )
            else:
                _attn_retry = _active_attn_heads
                while True:
                    try:
                        self._wait_h2d_stream_for_current()
                        self._load_sparse_attn_heads(
                            layer_idx, _attn_retry, head_dim=_head_dim, hidden_size=_hidden,
                        )
                        break
                    except Exception as _e:
                        if not _is_cuda_oom_error(_e):
                            raise
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if self._shrink_attn_sparse_budget_on_oom(head_dim=_head_dim, hidden_size=_hidden):
                            _attn_retry = self._get_attn_active_heads(layer_idx)
                            if _attn_retry is not None and int(_attn_retry.numel()) > 0:
                                continue

                        raise RuntimeError(
                            f"CUDA OOM loading sparse attention q/o for layer {int(layer_idx)} at the minimum head budget."
                        ) from _e


        next_idx = layer_idx + 1
        if next_idx < self.num_layers:
            self._schedule_prefetch_layer(next_idx)
        next2_idx = layer_idx + 2
        if next2_idx < self.num_layers:
            self._schedule_prefetch_layer(next2_idx)

        return self._layer_skeleton

    def _release_modules(self, *modules: nn.Module) -> None:

        pass

    def _embed_tokens_cpu(self, token_ids: torch.LongTensor) -> torch.Tensor:
        if getattr(self, "_show_progress", False) or getattr(self, "_debug_steps", False):
            print(f"[profile] CPU Warning: embed_tokens_cpu invoked for {token_ids.shape[1] if token_ids.ndim > 1 else 1} tokens!", flush=True)
        token_ids_cpu = token_ids.to(device=torch.device("cpu"), dtype=torch.long)
        if self._embed_weight_cpu is not None:
            return F.embedding(token_ids_cpu, self._embed_weight_cpu)

        flat_ids = token_ids_cpu.reshape(-1)
        unique_ids = flat_ids.unique(sorted=False)
        missing_ids = []
        with self._embed_row_cache_lock:
            for token_id in unique_ids.tolist():
                if int(token_id) not in self._embed_row_cache:
                    missing_ids.append(int(token_id))
        if missing_ids:
            fetched_rows = self.loader.load_rows(self._embed_weight_name, missing_ids).to(dtype=self.dtype)
            with self._embed_row_cache_lock:
                for idx, token_id in enumerate(missing_ids):
                    self._embed_row_cache[int(token_id)] = fetched_rows[idx].contiguous()

        row_tensors = []
        with self._embed_row_cache_lock:
            for token_id in flat_ids.tolist():
                row_tensors.append(self._embed_row_cache[int(token_id)])
        return torch.stack(row_tensors, dim=0).view(int(token_ids_cpu.shape[0]), int(token_ids_cpu.shape[1]), -1)

    def _lm_head_forward_cpu(self, hidden: torch.Tensor) -> torch.Tensor:
        if getattr(self, "_show_progress", False) or getattr(self, "_debug_steps", False):
            print(f"[profile] CPU Bottleneck: LM Head processing on CPU! (GPU underutilized)", flush=True)
        lm_head_weight_cpu = self._ensure_lm_head_weight_cpu()
        if lm_head_weight_cpu is None:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")
        hidden_cpu = hidden.to(device=torch.device("cpu"), dtype=lm_head_weight_cpu.dtype)
        return F.linear(hidden_cpu, lm_head_weight_cpu)

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: int | None,
        top_p: float,
    ) -> torch.Tensor:
        if (not do_sample) or temperature <= 0.0:
            return torch.argmax(logits, dim=-1)

        scores = logits / max(float(temperature), 1e-5)
        if top_k is not None and top_k > 0:
            top_k = min(int(top_k), scores.shape[-1])
            topk_vals = torch.topk(scores, top_k, dim=-1).values
            min_topk = topk_vals[:, -1].unsqueeze(-1)
            scores = scores.masked_fill(scores < min_topk, float("-inf"))

        if 0.0 < float(top_p) < 1.0:
            sorted_scores, sorted_idx = torch.sort(scores, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_scores, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative > float(top_p)
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_scores = sorted_scores.masked_fill(sorted_mask, float("-inf"))
            scores = torch.full_like(scores, float("-inf"))
            scores.scatter_(dim=-1, index=sorted_idx, src=sorted_scores)

        probs = torch.softmax(scores, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def forward_token(
        self,
        token_ids: torch.LongTensor,
        *,
        position_index: int,
        capture_layers: Sequence[int] | None = None,
        use_attention_cache: bool = True,
    ) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1 or int(token_ids.shape[1]) != 1:
            raise RuntimeError("StreamingLlamaRuntime.forward_token currently supports batch_size=1 and seq_len=1 only")

        capture_set = {int(idx) for idx in capture_layers or []}
        captures: dict[int, dict[str, torch.Tensor]] = {}
        printed_progress = False

        hidden = self._embed_tokens_cpu(token_ids).to(device=self.device, dtype=self.dtype)
        position_ids = torch.tensor([[int(position_index)]], device=self.device, dtype=torch.long)
        rope = self.rotary_emb(hidden, position_ids)
        profile_step = self._begin_decode_profile_step(position_index=int(position_index))

        for layer_idx in range(self.num_layers):
            layer_cpu_start = time.perf_counter() if profile_step is not None else 0.0
            ev_start: torch.cuda.Event | None = None
            ev_attn_end: torch.cuda.Event | None = None
            ev_mlp_end: torch.cuda.Event | None = None
            ev_end: torch.cuda.Event | None = None
            if profile_step is not None and self.device.type == "cuda":
                ev_start = torch.cuda.Event(enable_timing=True)
                ev_attn_end = torch.cuda.Event(enable_timing=True)
                ev_mlp_end = torch.cuda.Event(enable_timing=True)
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_start.record()
            if self._show_progress:
                printed_progress = True
            if self._show_progress and torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"  [layer {layer_idx + 1}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB (alloc/reserv)",
                    end="\r",
                    flush=True,
                )
            elif self._show_progress:
                print(f"  [layer {layer_idx + 1}/{self.num_layers}] loading...", end="\r", flush=True)
            if self._debug_steps:
                print(f"[debug] layer {layer_idx}: load", flush=True)
            layer = self._load_layer(layer_idx)
            taylor_attn: GQATaylorSSDSelfAttention | None = None
            residual = hidden
            hidden_norm = layer.input_layernorm(hidden)
            _use_shared_attn = self._should_use_attn_share_for_layer(layer_idx)
            _active_attn_heads = None if _use_shared_attn else self._get_attn_active_heads(layer_idx)



            _active_kv_blocks = None
            if layer_idx not in self._retrieval_layers:
                _active_kv_blocks = self._route_kv_blocks(hidden_norm, layer_idx)
            if _active_kv_blocks is not None:
                self._populate_sparse_kv_skeleton(layer_idx, _active_kv_blocks, layer)

            # Phase 4: pre-transfer cold MLP blocks to GPU while attention runs.
            # Route from hidden_norm (pre-attention) as a proxy to predict which blocks
            # will be needed; blocks go into _sparse_block_transfer_cache as H2D cache
            # hits.  Actual routing in _mlp_forward_dispatch always uses mlp_input
            # (post-attention) for correctness — the dict is discarded there.
            if (
                layer_idx in self._sparse_routing
                and str(self._traffic_current_phase or "idle") == "decode"
                and self._h2d_stream is not None
            ):
                _prefetch_blocks = self._route_sparse_mlp(hidden_norm, layer_idx)
                if _prefetch_blocks is not None:
                    self._mlp_prefetch_active_blocks[layer_idx] = _prefetch_blocks
                    self._prefetch_mlp_blocks_for_layer(layer_idx, _prefetch_blocks)

            # Phase 5.2: SMC for attention — skip dense attention if hidden state unchanged
            _lidx = int(layer_idx)
            _is_dense_attn_path = (
                not (layer_idx in self.taylor_layer_set and use_attention_cache)
                and not (layer_idx in self._retrieval_layers and self._token_archive is not None and use_attention_cache)
                and not self._should_use_compact_sparse_attention(
                    layer_idx=_lidx, active_heads=_active_attn_heads,
                    use_attention_cache=bool(use_attention_cache), use_shared_attn=bool(_use_shared_attn),
                )
            )
            _smc_attn_used = False
            if (
                bool(getattr(self, "_enable_smc", False))
                and str(self._traffic_current_phase or "idle") == "decode"
                and _lidx not in self._smc_attn_protected
                and bool(self._smc_attn_valid[_lidx].item())
                and _is_dense_attn_path
            ):
                _x_flat = hidden_norm.view(-1)
                _ref = self._smc_attn_x_in[_lidx]
                _delta = (_x_flat - _ref).norm()
                _ref_norm = _ref.norm().clamp(min=1e-6)
                if (_delta / _ref_norm).item() < self._smc_attn_threshold:
                    attn_out = self._smc_attn_out[_lidx].view_as(hidden_norm)
                    _smc_attn_used = True

            if not _smc_attn_used:
                if layer_idx in self.taylor_layer_set and use_attention_cache:
                    if self._debug_steps:
                        print(f"[debug] layer {layer_idx}: build_taylor", flush=True)
                    taylor_attn = self._shared_taylor_attn
                    taylor_attn.layer_idx = layer_idx
                    if self._debug_steps:
                        print(f"[debug] layer {layer_idx}: attn_forward_taylor", flush=True)
                    attn_out, _attn_weights, present = taylor_attn(
                        hidden_states=hidden_norm,
                        position_ids=position_ids,
                        position_embeddings=rope,
                        past_key_value=self._taylor_caches[layer_idx],
                        use_cache=True,
                    )
                    self._taylor_caches[layer_idx] = present
                elif (
                    layer_idx in self._retrieval_layers
                    and self._token_archive is not None
                    and use_attention_cache
                ):
                    if self._debug_steps:
                        print(f"[debug] layer {layer_idx}: attn_forward_retrieval", flush=True)
                    attn_out = self._forward_retrieval_attn(
                        layer_idx=layer_idx,
                        layer=layer,
                        hidden_norm=hidden_norm,
                        rope=rope,
                        position_index=int(position_ids.view(-1)[0].item()),
                        active_heads=_active_attn_heads,
                    )
                elif not _is_dense_attn_path:
                    if self._debug_steps:
                        print(f"[debug] layer {layer_idx}: attn_forward_compact_sparse", flush=True)
                    attn_out = self._forward_compact_sparse_attn(
                        layer_idx=_lidx,
                        layer=layer,
                        hidden_norm=hidden_norm,
                        rope=rope,
                        active_heads=_active_attn_heads,
                        use_attention_cache=bool(use_attention_cache),
                    )
                else:
                    if self._debug_steps:
                        print(f"[debug] layer {layer_idx}: attn_forward_dense", flush=True)
                    attn_out, _attn_weights = layer.self_attn(
                        hidden_states=hidden_norm,
                        position_embeddings=rope,
                        attention_mask=None,
                        past_key_values=self._dense_cache if use_attention_cache else None,
                        cache_position=position_ids.view(-1),
                    )
                # Update SMC attn state for non-protected dense layers
                if (
                    bool(getattr(self, "_enable_smc", False))
                    and str(self._traffic_current_phase or "idle") == "decode"
                    and _is_dense_attn_path
                    and _lidx not in self._smc_attn_protected
                ):
                    self._smc_attn_x_in[_lidx].copy_(hidden_norm.view(-1), non_blocking=True)
                    self._smc_attn_out[_lidx].copy_(attn_out.view(-1), non_blocking=True)
                    self._smc_attn_valid[_lidx] = True

            if ev_attn_end is not None:
                ev_attn_end.record()

            hidden = residual + attn_out
            residual = hidden
            mlp_input = layer.post_attention_layernorm(hidden)

            if self._debug_steps:
                print(f"[debug] layer {layer_idx}: route", flush=True)
            mlp_out = self._mlp_forward_dispatch(layer_idx, layer, mlp_input)
            if ev_mlp_end is not None:
                ev_mlp_end.record()

            if layer_idx in capture_set:
                captures[layer_idx] = {
                    "mlp_input": mlp_input.detach().cpu(),
                    "dense_mlp_out": mlp_out.detach().cpu(),
                }
            hidden = residual + mlp_out

            if _active_kv_blocks is not None:
                self._update_kv_block_banking(layer_idx, _active_kv_blocks)
            self._release_modules(layer, *( [taylor_attn] if taylor_attn is not None else [] ))
            if str(self._traffic_current_phase or "idle") == "decode":
                self.clear_sparse_transfer_caches()
            if ev_end is not None:
                ev_end.record()
            if profile_step is not None:
                self._record_decode_profile_layer(
                    profile_step,
                    layer_idx=int(layer_idx),
                    cpu_ms=(time.perf_counter() - layer_cpu_start) * 1000.0,
                    events=(
                        (ev_start, ev_attn_end, ev_mlp_end, ev_end)
                        if ev_start is not None
                        and ev_attn_end is not None
                        and ev_mlp_end is not None
                        and ev_end is not None
                        else None
                    ),
                )


        if self._kv_routing:
            self._kv_bank_step += 1

        if printed_progress:
            print("", flush=True)
        hidden = self.norm(hidden)
        if not self._materialize_lm_head:
            logits = torch.zeros((hidden.shape[0], hidden.shape[1], 1), dtype=torch.float32)
        else:
            logits = self._lm_head_forward(hidden).float()
        if profile_step is not None:
            self._finalize_decode_profile_step(profile_step)
        return logits, captures

    def _forward_prefill(
        self,
        token_ids: torch.LongTensor,
        *,
        position_offset: int = 0,
        compute_logits: bool = True,
    ) -> torch.Tensor:
        """Process all prompt tokens with each of the 126 layers loaded only once.

        Instead of the naive loop (load layer N for token 1, load layer N for
        token 2, …), we invert the loops: load layer N once, then run all prompt
        tokens through it sequentially before unloading.  For a P-token prompt
        this reduces layer loads from P×126 to 126.

        Attention and MLP are both processed over the whole prompt per layer.
        Sparse q/o heads are still loaded once per layer, and sparse K/V routing
        already unions active column blocks across prompt tokens, so batching the
        actual attention call collapses the hot-path Python/CUDA launch overhead
        without changing the streamed-weight footprint. Taylor-SSD still walks
        the sequence recurrently internally, but doing that inside one module
        call is much cheaper than re-entering the full attention stack token by
        token from Python.
        """
        seq_len = int(token_ids.shape[1])
        all_hidden = self._embed_tokens_cpu(token_ids).to(device=self.device, dtype=self.dtype)

        position_start = int(position_offset)
        position_ids_all = torch.arange(
            position_start,
            position_start + seq_len,
            device=self.device,
            dtype=torch.long,
        )
        position_ids_batch = position_ids_all.unsqueeze(0)










        if seq_len > 1:
            _min_val = torch.finfo(self.dtype).min
            total_kv_len = position_start + seq_len
            q_pos = torch.arange(position_start, position_start + seq_len, device=self.device)
            k_pos = torch.arange(0, total_kv_len, device=self.device)

            future = k_pos.unsqueeze(0) > q_pos.unsqueeze(1)
            _causal_mask = future.to(dtype=self.dtype) * _min_val
            _causal_mask = _causal_mask.unsqueeze(0).unsqueeze(0)
        else:
            _causal_mask = None

        import time
        for layer_idx in range(self.num_layers):
            t_start = time.perf_counter()
            ev_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            ev_load = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            ev_attn = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            ev_mlp = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            if ev_start: ev_start.record()
            if self._show_progress and torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"  [prefill layer {layer_idx + 1}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB",
                    end="\r", flush=True,
                )
            layer = self._load_layer(layer_idx)
            if ev_load: ev_load.record()


            use_taylor = layer_idx in self.taylor_layer_set
            if use_taylor:
                taylor_attn = self._shared_taylor_attn
                taylor_attn.layer_idx = layer_idx
            h_norm = layer.input_layernorm(all_hidden)
            _active_kv_blocks_prefill = self._route_kv_blocks(h_norm, layer_idx)
            _allow_sparse_kv_prefill = self._sparse_kv_prefill_mode != "dense"
            if _active_kv_blocks_prefill is not None and _allow_sparse_kv_prefill:
                self._populate_sparse_kv_skeleton(layer_idx, _active_kv_blocks_prefill, layer)
            rope_all = self.rotary_emb(all_hidden, position_ids_batch)

            if use_taylor:
                attn_out, _, present = taylor_attn(
                    hidden_states=h_norm,
                    position_embeddings=rope_all,
                    past_key_value=self._taylor_caches[layer_idx],
                    use_cache=True,
                )
                self._taylor_caches[layer_idx] = present
            else:
                attn_out, _ = layer.self_attn(
                    hidden_states=h_norm,
                    position_embeddings=rope_all,
                    attention_mask=_causal_mask,
                    past_key_values=self._dense_cache,
                    cache_position=position_ids_all.view(-1),
                )
            all_hidden = all_hidden + attn_out
            if ev_attn: ev_attn.record()









            if torch.cuda.is_available() and self._debug_sync_cuda:
                try:
                    torch.cuda.synchronize(self.device)
                except Exception as _sync_e:
                    if _is_cuda_oom_error(_sync_e):
                        print(
                            f"[prefill_sync] CUDA error after attention at layer {layer_idx}: "
                            f"{type(_sync_e).__name__!r}: {str(_sync_e)[:200]!r}",
                            flush=True,
                        )

                    else:
                        raise


            residual = all_hidden
            mlp_input = layer.post_attention_layernorm(all_hidden)
            mlp_out = self._mlp_forward_dispatch(layer_idx, layer, mlp_input)
            all_hidden = residual + mlp_out

            self._release_modules(layer)
            if ev_mlp: 
                ev_mlp.record()
                torch.cuda.synchronize()
                load_ms = ev_start.elapsed_time(ev_load) if ev_load else 0.0
                attn_ms = ev_load.elapsed_time(ev_attn) if ev_attn else 0.0
                mlp_ms = ev_attn.elapsed_time(ev_mlp) if ev_mlp else 0.0
                total_gpu_ms = ev_start.elapsed_time(ev_mlp) if ev_mlp else 0.0
                total_cpu_ms = (time.perf_counter() - t_start) * 1000.0
                
                if total_cpu_ms > total_gpu_ms + 10.0 and self._show_progress:
                    print(f"\n[profile] layer {layer_idx} CPU bottleneck detected! CPU: {total_cpu_ms:.1f}ms, GPU: {total_gpu_ms:.1f}ms (load={load_ms:.1f}ms, attn={attn_ms:.1f}ms, mlp={mlp_ms:.1f}ms)", flush=True)
                elif self._debug_steps:
                    print(f"\n[profile] layer {layer_idx} CPU: {total_cpu_ms:.1f}ms, GPU: {total_gpu_ms:.1f}ms", flush=True)
                
                if hasattr(self, "_record_runtime_event"):
                    self._record_runtime_event("prefill_layer_profile", layer_idx=layer_idx, cpu_ms=total_cpu_ms, gpu_ms=total_gpu_ms, load_ms=load_ms, attn_ms=attn_ms, mlp_ms=mlp_ms)

        if self._show_progress:
            print("", flush=True)




        self._wait_for_h2d_stream()



        if self._token_archive is not None and self._dense_cache is not None:
            self._token_archive.warm_up_from_dense_cache(
                self._dense_cache, seq_len=int(token_ids.shape[1])
            )
            self._release_dense_cache_for_retrieval_layers()

        if not bool(compute_logits):
            return all_hidden

        all_hidden = self.norm(all_hidden)

        logits = self._lm_head_forward(all_hidden[:, -1:, :]).float()
        return logits





    def _forward_retrieval_attn(
        self,
        layer_idx: int,
        layer: LlamaDecoderLayer,
        hidden_norm: torch.Tensor,
        rope: tuple,
        position_index: int,
        active_heads: torch.Tensor | None,
    ) -> torch.Tensor:
        """Exact attention on a sparse shortlist of archived tokens.

        Replaces the dense full-context ``layer.self_attn(past_key_values=...)``
        call for retrieval layers.  The shortlist is assembled by the
        ``TokenPostingArchive``:
          • sink tokens  — first num_sinks, always on GPU
          • archive candidates — top-M from posting-list probe
          • ring tokens  — most recent ring_size, always on GPU

        The new token's exact K/V is computed with the fully-loaded skeleton
        weights and appended to the archive before returning.
        """
        archive = self._token_archive
        cfg = self.config
        head_dim = int(getattr(cfg, "head_dim", 128))
        num_heads = int(getattr(cfg, "num_attention_heads", 128))
        num_kv = int(getattr(cfg, "num_key_value_heads", 8))
        queries_per_group = num_heads // num_kv


        cos, sin = rope
        q_raw = layer.self_attn.q_proj(hidden_norm)
        k_raw = layer.self_attn.k_proj(hidden_norm)
        v_raw = layer.self_attn.v_proj(hidden_norm)


        q = q_raw.view(1, 1, num_heads, head_dim).transpose(1, 2)
        k = k_raw.view(1, 1, num_kv, head_dim).transpose(1, 2)
        v = v_raw.view(1, 1, num_kv, head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)



        if active_heads is not None:
            active_h_list: list[int] = active_heads.tolist()
        else:
            active_h_list = list(range(num_heads))


        from collections import defaultdict as _defaultdict
        group_to_heads: dict[int, list[int]] = _defaultdict(list)
        for h in active_h_list:
            group_to_heads[h // queries_per_group].append(h)


        # Derive a unique dedup stamp from (position, layer) so it is always
        # monotonically increasing and aligned with the actual sequence position.
        # Multiplier 200 > max number of retrieval layers (126), guaranteeing
        # per-(position, layer) uniqueness without a mutable global counter.
        step = position_index * 200 + layer_idx

        out_by_head: dict[int, torch.Tensor] = {}

        for g, heads_in_group in group_to_heads.items():

            h_indices = torch.tensor(heads_in_group, device=self.device, dtype=torch.long)
            q_rep = q[0, h_indices, 0, :]


            k_ctx, v_ctx = archive.fetch_shortlist_kv(
                layer_idx, g, q_rep, step, M=self._retrieval_candidates,
            )


            k_new_g = k[0, g, 0:1, :]
            v_new_g = v[0, g, 0:1, :]

            if k_ctx.shape[0] > 0:
                k_all = torch.cat([k_ctx, k_new_g], dim=0)
                v_all = torch.cat([v_ctx, v_new_g], dim=0)
            else:
                k_all = k_new_g
                v_all = v_new_g


            k_4d = k_all.unsqueeze(0).unsqueeze(0)
            v_4d = v_all.unsqueeze(0).unsqueeze(0)


            for h in heads_in_group:
                q_h = q[0:1, h:h+1, :, :]
                out_h = F.scaled_dot_product_attention(
                    q_h, k_4d, v_4d,
                    scale=head_dim ** -0.5,
                )
                out_by_head[h] = out_h[0, 0, 0, :]




        out_flat = torch.zeros(
            1, 1, num_heads * head_dim,
            dtype=self.dtype, device=self.device,
        )
        for h, out_h in out_by_head.items():
            out_flat[0, 0, h * head_dim : (h + 1) * head_dim] = out_h

        attn_out = layer.self_attn.o_proj(out_flat)


        k_new_cpu = k[0, :, 0, :].detach().cpu()
        v_new_cpu = v[0, :, 0, :].detach().cpu()
        archive.append_token(layer_idx, position_index, k_new_cpu, v_new_cpu)

        return attn_out

    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = 50,
        top_p: float = 1.0,
        token_callback: Callable[[torch.Tensor, torch.LongTensor], None] | None = None,
        reuse_session_cache: bool = False,
    ) -> torch.LongTensor:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise RuntimeError("StreamingLlamaRuntime.generate currently supports batch_size=1 only")
        if not self._materialize_lm_head:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")

        generated = input_ids.to(device=self.device)
        prompt_ids_cpu = input_ids.detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous()
        processed_ids_cpu = prompt_ids_cpu
        self._reset_traffic_stats()
        logits: torch.Tensor | None = None
        reuse_prefix_len = 0
        prev_session_len = 0
        with torch.no_grad():
            prompt_len = int(generated.shape[1])
            if prompt_len <= 0:
                raise RuntimeError("No prompt tokens were provided")

            if bool(reuse_session_cache) and self._session_token_ids_cpu is not None:
                prev_session_len = int(self._session_token_ids_cpu.shape[-1])
                reuse_prefix_len = self._longest_common_prefix_len(self._session_token_ids_cpu, prompt_ids_cpu)
                if reuse_prefix_len > 0:
                    if self.taylor_layer_set and reuse_prefix_len < prev_session_len:
                        print("[session] Taylor cache cannot crop on divergence; falling back to cold prefill", flush=True)
                        reuse_prefix_len = 0
                    elif reuse_prefix_len < prev_session_len and not self._crop_attention_caches(reuse_prefix_len):
                        print("[session] cache crop unavailable; falling back to cold prefill", flush=True)
                        reuse_prefix_len = 0

            if reuse_prefix_len <= 0:
                self.reset_caches()
                _skip_calib = os.getenv("STREAMING_SKIP_LIVE_CALIBRATION", "").strip().lower() in {"1", "true", "yes"}
                if (
                    getattr(self, "_vram_hot_cache_enabled", False)
                    and prompt_len > 0
                    and not bool(getattr(self, "_vram_hot_cache_live_calibrated", False))
                    and not _skip_calib
                ):
                    self.calibrate_vram_hot_cache(
                        prompt_ids_cpu,
                        max_tokens=256,
                        rebuild_cache=True,
                        generate_decode_tokens=2,
                    )
                if _skip_calib and not bool(getattr(self, "_vram_hot_cache_live_calibrated", False)):
                    print("[hot-cache] skipping live calibration; pre-warming VRAM from static hot-block map", flush=True)
                    self.pre_warm_vram_hot_cache()
                    self._vram_hot_cache_live_calibrated = True
                self._set_traffic_phase("prefill")
                if prompt_len > 1:
                    print(f"[prompt] batched prefill: {prompt_len} tokens × 1 layer pass", flush=True)
                    logits = self._forward_prefill(generated)
                else:
                    logits, _ = self.forward_token(generated[:, 0:1], position_index=0)
            else:
                self._set_traffic_phase("prefill")
                suffix_len = int(prompt_len - reuse_prefix_len)
                if suffix_len > 0:
                    print(
                        f"[prompt] delta prefill: reused {reuse_prefix_len}/{prompt_len} tokens; "
                        f"streaming {suffix_len} new tokens",
                        flush=True,
                    )
                    suffix_ids = generated[:, reuse_prefix_len:]
                    if suffix_len > 1:
                        logits = self._forward_prefill(suffix_ids, position_offset=reuse_prefix_len)
                    else:
                        logits, _ = self.forward_token(suffix_ids[:, 0:1], position_index=reuse_prefix_len)
                elif prev_session_len == prompt_len and self._session_last_logits_cpu is not None:
                    print(f"[prompt] prefix cache hit: reused {reuse_prefix_len}/{prompt_len} tokens", flush=True)
                    logits = self._session_last_logits_cpu.to(device=self.device, dtype=torch.float32)
                else:
                    replay_prefix_len = max(0, int(prompt_len - 1))
                    if not self._crop_attention_caches(replay_prefix_len):
                        self.reset_caches()
                        self._set_traffic_phase("prefill")
                        print(f"[prompt] replay prefill: {prompt_len} tokens × 1 layer pass", flush=True)
                        logits = self._forward_prefill(generated)
                    else:
                        print(
                            f"[prompt] replay tail: reused {replay_prefix_len}/{prompt_len} tokens; "
                            f"replaying final prompt token",
                            flush=True,
                        )
                        logits, _ = self.forward_token(
                            generated[:, replay_prefix_len:replay_prefix_len + 1],
                            position_index=replay_prefix_len,
                        )
            if logits is None:
                raise RuntimeError("No prompt tokens were provided")





            self._materialize_lm_head_on_gpu()

            self._set_traffic_phase("decode")
            self._first_decode_t = time.perf_counter()
            for _step_idx in range(int(max_new_tokens)):
                next_token = self._sample_next_token(
                    logits[:, -1, :],
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=float(top_p),
                ).view(1, 1)
                generated = torch.cat([generated, next_token.to(device=self.device, dtype=generated.dtype)], dim=-1)
                if token_callback is not None:
                    token_callback(next_token, generated)
                processed_ids_cpu = torch.cat(
                    [
                        processed_ids_cpu,
                        next_token.detach().to(device=torch.device("cpu"), dtype=torch.long).contiguous(),
                    ],
                    dim=-1,
                )
                if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                    break
                if _step_idx + 1 >= int(max_new_tokens):
                    break
                logits, _ = self.forward_token(next_token, position_index=int(generated.shape[1]) - 1)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._decode_done_t = time.perf_counter()
            self._set_session_state(processed_ids_cpu, logits)
        self._set_traffic_phase("idle")
        self._finalize_traffic_report()
        return generated
