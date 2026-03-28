from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch.nn.functional as F
import bitsandbytes.functional as bnb_functional
from bitsandbytes.backends.cuda.ops import _dequantize_4bit_impl as _bnb_dequant_impl
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

try:
    from transformers.cache_utils import DynamicCache
except Exception:  # pragma: no cover
    DynamicCache = None

from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding

try:
    from ..gqa_taylor_ssd import GQATaylorSSDSelfAttention, TaylorSSDLayerCache
except ImportError:  # pragma: no cover
    from gqa_taylor_ssd import GQATaylorSSDSelfAttention, TaylorSSDLayerCache

def _resolve_ram_cache_limit_bytes() -> Optional[int]:
    raw = os.getenv("STREAMING_RAM_CACHE_MAX_GB", "").strip()
    if raw:
        try:
            limit_gb = float(raw)
        except ValueError:
            limit_gb = 0.0
        if limit_gb > 0:
            return int(limit_gb * (1024 ** 3))
        return None
    return None


def _resolve_snapshot_dir(model_name_or_path: str, *, local_files_only: bool) -> Path:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return candidate.resolve()
    snapshot_dir = snapshot_download(
        repo_id=str(model_name_or_path),
        local_files_only=bool(local_files_only),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "special_tokens_map.json"],
    )
    return Path(snapshot_dir).resolve()


class ShardedSafetensorLoader:
    def __init__(self, snapshot_dir: Path, *, cache_shard_handles: Optional[bool] = None) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        index_path = self.snapshot_dir / "model.safetensors.index.json"
        self.weight_map: Dict[str, str] = {}
        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            raw_map = payload.get("weight_map", {})
            self.weight_map = {str(k): str(v) for k, v in raw_map.items()}
        else:
            safetensor_files = sorted(self.snapshot_dir.glob("*.safetensors"))
            if not safetensor_files:
                raise RuntimeError(f"No safetensors shards found in {self.snapshot_dir}")
            with safe_open(str(safetensor_files[0]), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    self.weight_map[str(name)] = safetensor_files[0].name

        self._available_names = set(self.weight_map.keys())
        self._quant_aux_by_base: Dict[str, List[str]] = defaultdict(list)
        for name in self.weight_map:
            if ".weight." not in name:
                continue
            base = name.split(".weight.", 1)[0] + ".weight"
            self._quant_aux_by_base[base].append(name)
        # Windows + many concurrent safe_open mappings for 405B shards reliably
        # hard-crash once the stream crosses into later shards. Re-open on demand
        # there; keep the old cached-handle fast path elsewhere.
        if cache_shard_handles is None:
            cache_shard_handles = os.name != "nt"
        self._cache_shard_handles = bool(cache_shard_handles)
        self._shard_handles: Dict[str, Any] = {}
        # RAM weight cache: maps full parameter name → (weight_bytes, quant_aux_dict)
        # Both tensors are pinned (page-locked) for fast DMA to GPU.
        # After the first forward pass all weights live here; subsequent tokens
        # do RAM→GPU transfers instead of SSD→GPU (typically 5–10× faster).
        self._ram_cache: Dict[str, Tuple[torch.Tensor, Dict[str, torch.Tensor]]] = {}
        self._ram_cache_lru: "OrderedDict[str, None]" = OrderedDict()
        self._ram_cache_entry_bytes: Dict[str, int] = {}
        self._ram_cache_current_bytes: int = 0
        self._ram_cache_limit_bytes: Optional[int] = _resolve_ram_cache_limit_bytes()
        self._ram_cache_enabled: bool = True
        self._ram_cache_lock = threading.Lock()

    @staticmethod
    def _entry_nbytes(weight: torch.Tensor, quant_aux: Dict[str, torch.Tensor]) -> int:
        total = int(weight.numel() * weight.element_size())
        for tensor in quant_aux.values():
            total += int(tensor.numel() * tensor.element_size())
        return total

    def _evict_ram_cache_locked(self) -> None:
        if self._ram_cache_limit_bytes is None:
            return
        while self._ram_cache_current_bytes > self._ram_cache_limit_bytes and self._ram_cache_lru:
            victim, _ = self._ram_cache_lru.popitem(last=False)
            self._ram_cache.pop(victim, None)
            self._ram_cache_current_bytes -= self._ram_cache_entry_bytes.pop(victim, 0)

    def _load_exact_tensors(self, names: Sequence[str]) -> Dict[str, torch.Tensor]:
        requested = [str(name) for name in names if str(name) in self._available_names]
        by_shard: Dict[str, List[str]] = defaultdict(list)
        for name in requested:
            by_shard[self.weight_map[name]].append(name)

        out: Dict[str, torch.Tensor] = {}
        for shard_name, shard_keys in by_shard.items():
            shard_path = self.snapshot_dir / shard_name
            if self._cache_shard_handles:
                if shard_name not in self._shard_handles:
                    self._shard_handles[shard_name] = safe_open(str(shard_path), framework="pt", device="cpu")
                handle = self._shard_handles[shard_name]
                for key in shard_keys:
                    out[key] = handle.get_tensor(key)
                continue

            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                for key in shard_keys:
                    tensor = handle.get_tensor(key)
                    if os.name == "nt":
                        tensor = tensor.clone().contiguous()
                    out[key] = tensor
        return out

    def load_parameter(self, name: str) -> torch.Tensor:
        full_name = str(name)
        weight, quant_aux = self._load_raw_for_param(full_name)
        if not quant_aux:
            return weight

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        return bnb_functional.dequantize_4bit(weight, quant_state=quant_state).cpu()

    def _load_raw_for_param(self, full_name: str) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Return (weight_bytes, quant_aux) for a parameter, using RAM cache when warm.

        On cache miss the tensors are loaded from disk and stored in the RAM cache
        as pinned (page-locked) memory so subsequent tokens DMA directly to the GPU
        staging buffer without going through the SSD.
        """
        if self._ram_cache_enabled:
            with self._ram_cache_lock:
                cached = self._ram_cache.get(full_name)
                if cached is not None and self._ram_cache_limit_bytes is not None:
                    self._ram_cache_lru.move_to_end(full_name, last=True)
            if cached is not None:
                weight, quant_aux = cached
                return weight, dict(quant_aux)

        aux_names = self._quant_aux_by_base.get(full_name, [])
        tensors = self._load_exact_tensors([full_name, *aux_names])
        if full_name not in tensors:
            raise KeyError(f"Tensor '{full_name}' not found in safetensors index")

        weight = tensors[full_name]
        quant_aux = {k: v for k, v in tensors.items() if k != full_name}

        if self._ram_cache_enabled:
            if os.name == "nt" and self._cache_shard_handles:
                weight = weight.clone().contiguous()
                quant_aux = {k: v.clone().contiguous() for k, v in quant_aux.items()}
            # Store as plain CPU tensors — no pin_memory() to avoid CUDA driver
            # interactions from the background prefetch thread, which would race
            # with GPU kernels on the main thread and risk OOM on Windows.
            # Plain RAM→GPU DMA is still 5–10× faster than SSD→GPU.
            with self._ram_cache_lock:
                cached = self._ram_cache.get(full_name)
                if cached is not None:
                    if self._ram_cache_limit_bytes is not None:
                        self._ram_cache_lru.move_to_end(full_name, last=True)
                    weight_cached, quant_aux_cached = cached
                    return weight_cached, dict(quant_aux_cached)
                self._ram_cache[full_name] = (weight, quant_aux)
                if self._ram_cache_limit_bytes is not None:
                    self._ram_cache_lru[full_name] = None
                    self._ram_cache_lru.move_to_end(full_name, last=True)
                    self._ram_cache_entry_bytes[full_name] = self._entry_nbytes(weight, quant_aux)
                    self._ram_cache_current_bytes += self._ram_cache_entry_bytes[full_name]
                    self._evict_ram_cache_locked()
            return weight, dict(quant_aux)

        return weight, dict(quant_aux)

    def load_parameter_into(
        self,
        name: str,
        out: torch.Tensor,
        dtype: torch.dtype,
        staging: Optional[torch.Tensor] = None,
        absmax_staging: Optional[torch.Tensor] = None,
        nested_absmax_staging: Optional[torch.Tensor] = None,
        state2_absmax_staging: Optional[torch.Tensor] = None,
        code_staging: Optional[torch.Tensor] = None,
    ) -> None:
        """Dequantize NF4 weight directly into a pre-allocated GPU skeleton buffer.

        All GPU tensors are pre-allocated: `staging` (uint8, NF4 bytes),
        `absmax_staging` (fp32, dequantized absmax output),
        `nested_absmax_staging` (uint8, doubly-quantized absmax input),
        `state2_absmax_staging` (fp32, secondary quant scales),
        `code_staging` (fp32, dequant codebook).

        QuantState is created on CPU (fast dict parsing, zero GPU allocs).
        Small tensors are memcpy'd into the pre-allocated GPU buffers and all
        dequant kernels run on GPU — same speed as GPU QuantState, but zero
        cudaMalloc calls that would fragment the pool and crash at layer ~48.

        On the first forward pass weights are loaded from disk and stored in the
        RAM cache.  All subsequent tokens hit the RAM cache, eliminating SSD I/O.
        """
        full_name = str(name)
        weight, quant_aux = self._load_raw_for_param(full_name)
        if not quant_aux:
            out.copy_(weight.to(dtype=dtype))
            return

        # DMA NF4 bytes into the pre-allocated staging buffer — no cudaMalloc.
        n = weight.numel()
        if staging is not None:
            weight_gpu = staging[:n]
            weight_gpu.copy_(weight.reshape(-1))
        else:
            weight_gpu = weight.to(device=out.device)

        if absmax_staging is not None:
            # Fast path: keep QuantState parsing on CPU and use the pre-allocated
            # GPU staging buffers for the nested absmax decode as well.
            quant_state_cpu = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
            if quant_state_cpu.nested:
                if (
                    nested_absmax_staging is not None
                    and state2_absmax_staging is not None
                    and code_staging is not None
                ):
                    n_nested = quant_state_cpu.absmax.numel()
                    nested_gpu = nested_absmax_staging[:n_nested]
                    nested_gpu.copy_(quant_state_cpu.absmax)

                    n_s2 = quant_state_cpu.state2.absmax.numel()
                    s2_gpu = state2_absmax_staging[:n_s2]
                    s2_gpu.copy_(quant_state_cpu.state2.absmax)

                    code_staging[: quant_state_cpu.state2.code.numel()].copy_(quant_state_cpu.state2.code)
                    code_gpu = code_staging[: quant_state_cpu.state2.code.numel()]

                    absmax = absmax_staging[:n_nested]
                    bnb_functional.dequantize_blockwise(
                        nested_gpu,
                        absmax=s2_gpu,
                        code=code_gpu,
                        out=absmax,
                        blocksize=quant_state_cpu.state2.blocksize,
                    )
                    absmax.add_(quant_state_cpu.offset)
                else:
                    quant_state_cpu = None
            else:
                n_abs = quant_state_cpu.absmax.numel()
                absmax = absmax_staging[:n_abs]
                absmax.copy_(quant_state_cpu.absmax)
                if absmax.dtype != torch.float32:
                    absmax = absmax.float()

            if quant_state_cpu is not None:
                if n * 2 != out.numel():
                    raise RuntimeError(
                        f"Shape mismatch for '{full_name}': {n} NF4 bytes â†’ {n * 2} elements "
                        f"but out has {out.numel()} elements (shape {out.shape})"
                    )

                _bnb_dequant_impl(
                    weight_gpu,
                    absmax,
                    quant_state_cpu.blocksize,
                    quant_state_cpu.quant_type,
                    quant_state_cpu.dtype,
                    out=out,
                )
                del absmax, quant_state_cpu
                return

        # Create QuantState on GPU so absmax tensors are already on the right device.
        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=out.device)

        if n * 2 != out.numel():
            raise RuntimeError(
                f"Shape mismatch for '{full_name}': {n} NF4 bytes → {n * 2} elements "
                f"but out has {out.numel()} elements (shape {out.shape})"
            )

        # Use _bnb_dequant_impl directly rather than the high-level dequantize_4bit.
        # The high-level function routes through torch.ops custom-op dispatch which
        # converts the 'shape' argument from torch.Size → list; bitsandbytes then
        # checks `out.shape == shape` where out.shape is torch.Size and shape is a
        # list — Python considers them unequal and raises RuntimeError even when the
        # dimensions are identical.  _bnb_dequant_impl avoids that dispatch path.
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax.float() + quant_state.offset
        else:
            absmax = quant_state.absmax.float()
        _bnb_dequant_impl(weight_gpu, absmax, quant_state.blocksize, quant_state.quant_type, quant_state.dtype, out=out)
        del absmax, quant_state

    def load_module_state(
        self,
        *,
        prefix: str,
        expected_keys: Iterable[str],
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        state: Dict[str, torch.Tensor] = {}
        for key in expected_keys:
            full_name = f"{prefix}{key}"
            tensor = self.load_parameter(full_name)
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            state[str(key)] = tensor
        return state


class StreamingLlamaRuntime:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        taylor_layers: Optional[List[int]] = None,
        taylor_feature_map: str = "hybrid_performer",
        taylor_local_window: int = 64,
        taylor_feature_dim: int = 64,
        taylor_state_decay: float = 1.0,
        local_files_only: bool = True,
        ram_cache: bool = True,
        materialize_lm_head: bool = True,
        sparse_basis_path: Optional[str] = None,
        sparse_top_k: Optional[int] = None,
        attn_head_importance_path: Optional[str] = None,
        attn_active_heads: Optional[int] = None,
    ) -> None:
        self.snapshot_dir = _resolve_snapshot_dir(model_name_or_path, local_files_only=bool(local_files_only))
        self.config = AutoConfig.from_pretrained(str(self.snapshot_dir), local_files_only=bool(local_files_only))
        if str(getattr(self.config, "model_type", "")) != "llama":
            raise RuntimeError(f"Streaming runtime only supports llama models, got {self.config.model_type!r}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        # AutoConfig.from_pretrained does not set _attn_implementation (that happens inside
        # PreTrainedModel.from_pretrained).  Leaving it as None causes a KeyError in
        # ALL_ATTENTION_FUNCTIONS when layer.self_attn is called directly (e.g. in the
        # no-cache data-collection path).  'sdpa' is available on any PyTorch 2.x + CUDA
        # setup and is faster than 'eager' for the single-token forward we do here.
        if getattr(self.config, "_attn_implementation", None) is None:
            self.config._attn_implementation = "sdpa"
        self._debug_steps = os.getenv("STREAMING_DEBUG_STEPS", "").strip().lower() in {"1", "true", "yes", "on"}
        self.loader = ShardedSafetensorLoader(self.snapshot_dir)
        self.loader._ram_cache_enabled = bool(ram_cache)
        self._enable_background_prefetch = bool(ram_cache)  # pin_memory removed; safe on Windows
        self.num_layers = int(getattr(self.config, "num_hidden_layers", 0))
        if self.num_layers <= 0:
            raise RuntimeError("Invalid llama config: num_hidden_layers must be > 0")

        self.taylor_layer_set = (
            set(range(self.num_layers))
            if taylor_layers is None
            else {int(idx) for idx in taylor_layers if 0 <= int(idx) < self.num_layers}
        )
        self.taylor_feature_map = str(taylor_feature_map)
        self.taylor_local_window = int(taylor_local_window)
        self.taylor_feature_dim = int(taylor_feature_dim)
        self.taylor_state_decay = float(taylor_state_decay)

        # Keep embed_tokens on CPU — it's ~3.9 GiB for 405B and only one row is needed per token.
        self.embed_tokens = nn.Embedding(
            int(self.config.vocab_size),
            int(self.config.hidden_size),
            padding_idx=getattr(self.config, "pad_token_id", None),
        ).to(device=torch.device("cpu"), dtype=self.dtype)
        self.embed_tokens.weight.data.copy_(
            self.loader.load_parameter("model.embed_tokens.weight").to(dtype=self.dtype)
        )
        self.embed_tokens.requires_grad_(False)
        self._materialize_lm_head = bool(materialize_lm_head)

        self.norm = LlamaRMSNorm(int(self.config.hidden_size), eps=float(self.config.rms_norm_eps)).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.norm.weight.data.copy_(
            self.loader.load_parameter("model.norm.weight").to(device=self.device, dtype=self.dtype)
        )
        self.norm.requires_grad_(False)

        # Keep lm_head on CPU alongside embed_tokens — projection is done on CPU at end of forward.
        self.lm_head: Optional[nn.Linear] = None
        if self._materialize_lm_head:
            self.lm_head = nn.Linear(int(self.config.hidden_size), int(self.config.vocab_size), bias=False)
            lm_head_weight_name = (
                "lm_head.weight"
                if "lm_head.weight" in self.loader.weight_map
                else "model.embed_tokens.weight"
            )
            if lm_head_weight_name == "model.embed_tokens.weight":
                self.lm_head.weight = self.embed_tokens.weight
            else:
                self.lm_head.to(device=torch.device("cpu"), dtype=self.dtype)
                self.lm_head.weight.data.copy_(
                    self.loader.load_parameter(lm_head_weight_name).to(dtype=self.dtype)
                )
            self.lm_head.requires_grad_(False)

        self.rotary_emb = LlamaRotaryEmbedding(self.config, device=self.device)
        self._taylor_caches: List[Optional[TaylorSSDLayerCache]] = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None
        self._layer_skeleton = LlamaDecoderLayer(self.config, layer_idx=0).to(device=self.device, dtype=self.dtype)
        for p in self._layer_skeleton.parameters():
            p.requires_grad = False
        self._layer_skeleton.eval()
        # Keep the large MLP weights on CPU. The 8 GB target only has enough VRAM
        # for attention + activations + one streamed projection at a time.
        self._layer_skeleton.mlp.to(device=torch.device("cpu"), dtype=self.dtype)
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

        # Background thread pool (1 worker) that warms the RAM cache for the
        # next layer while the GPU processes the current one.
        self._prefetch_executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="layer_prefetch")
            if self._enable_background_prefetch
            else None
        )
        # Floating-point parameter keys of the skeleton (populated on first _load_layer).
        self._layer_param_keys: Optional[List[str]] = None

        # Pre-allocate one fixed staging buffer for the NF4 uint8 bytes.
        # Sized to the largest weight in any decoder layer (gate_proj / up_proj / down_proj).
        # By allocating this once at startup—before any layer is streamed—we guarantee
        # a single contiguous CUDA allocation and avoid the pool fragmentation that
        # accumulates over ~48 layers and causes STATUS_ACCESS_VIOLATION when a later
        # weight.to(device=...) forces a new cudaMalloc near the 8 GB VRAM ceiling.
        _h = int(self.config.hidden_size)
        # Attention-only staging: load_parameter_into is only used for attention weights.
        # MLP weights are loaded via load_parameter (CPU dequant) in _dense_mlp_forward_streaming_fast.
        # Largest attention weight is q_proj/o_proj [hidden, hidden].
        _max_nf4_bytes = _h * _h // 2  # NF4 = 2 values per byte
        # absmax has one fp32 value per block of 64 weight elements.
        # num_blocks = max_weight_numel / 64 = (_max_nf4_bytes * 2) / 64 = _max_nf4_bytes / 32
        _max_absmax_numel = _max_nf4_bytes // 32
        if torch.cuda.is_available():
            self._nf4_staging: Optional[torch.Tensor] = torch.empty(
                _max_nf4_bytes, dtype=torch.uint8, device=self.device
            )
            # Pre-allocated fp32 output buffer for dequantize_blockwise(absmax).
            # Without this, each large weight allocates ~54 MB from the pool for the
            # dequantized absmax. Repeated alloc/free fragments the 80 MB free pool
            # until no contiguous block remains and cudaMalloc crashes.
            self._absmax_staging: Optional[torch.Tensor] = torch.empty(
                _max_absmax_numel, dtype=torch.float32, device=self.device
            )
            # --- Zero-alloc QuantState staging ---
            # QuantState.from_dict(device=GPU) allocates ~15 MB of small GPU tensors
            # per weight (nested absmax uint8 + state2 float32 + code).  Over 7 weights
            # × 48 layers these fragment the CUDA pool → STATUS_ACCESS_VIOLATION.
            # Pre-allocating them here lets us create QuantState on CPU (fast dict
            # parsing, no GPU allocs) then memcpy the small tensors into these fixed
            # buffers before calling the GPU dequant kernels.
            self._nested_absmax_staging: Optional[torch.Tensor] = torch.empty(
                _max_absmax_numel, dtype=torch.uint8, device=self.device
            )  # ~13 MB for 405B — holds the doubly-quantized absmax bytes
            _max_s2_absmax = max(_max_absmax_numel // 64, 1024)
            self._state2_absmax_staging: Optional[torch.Tensor] = torch.empty(
                _max_s2_absmax, dtype=torch.float32, device=self.device
            )  # ~830 KB — holds secondary quantization scales
            self._code_staging: Optional[torch.Tensor] = torch.empty(
                256, dtype=torch.float32, device=self.device
            )  # 1 KB — dequantization codebook (same for all NF4 weights)
        else:
            self._nf4_staging = None
            self._absmax_staging = None
            self._nested_absmax_staging = None
            self._state2_absmax_staging = None
            self._code_staging = None
        # Try to pre-allocate MLP staging at init.  On constrained VRAM setups
        # (e.g. Windows where DWM reserves several GB of the 8 GB card), this
        # allocation may fail.  If it does, dense MLP falls back to zero-passthrough
        # which is acceptable for attention calibration; inference uses sparse MLP
        # exclusively (--sparse-basis-path) so the dense path is never reached there.
        _ffn = int(getattr(self.config, "intermediate_size", _h * 4))
        self._mlp_proj_staging: Optional[torch.Tensor] = None
        # Skip the ~1.6 GB MLP staging buffer when sparse routing is active —
        # sparse inference never uses _dense_mlp_forward_streaming_fast, and
        # skipping this allocation saves critical VRAM on 8 GB cards.
        _skip_mlp_staging = bool(sparse_basis_path and str(sparse_basis_path).strip())
        if torch.cuda.is_available() and not _skip_mlp_staging:
            try:
                self._mlp_proj_staging = torch.empty(
                    _ffn * _h, dtype=self.dtype, device=self.device
                )
            except Exception as _e:
                if "out of memory" in str(_e).lower() or "alloc" in str(_e).lower():
                    import warnings
                    warnings.warn(
                        f"[StreamingLlamaRuntime] Insufficient VRAM for MLP staging "
                        f"({_ffn * _h * 2 // 1024 // 1024} MB); dense MLP layers will "
                        f"use zero-passthrough (acceptable for attn calibration; "
                        f"inference should use --sparse-basis-path)."
                    )
                else:
                    raise

        # ── Sparse MLP routing weights ────────────────────────────────────────
        # Loaded from the learned-basis checkpoint produced by
        # init_learned_basis_from_dense_mlp.py.  Kept in CPU RAM (each layer's
        # routing tensors are ~3 MB — trivial compared to the 1.6 GB MLP weights)
        # and moved to GPU lazily when a layer is processed.
        #
        # For each sparse layer, the checkpoint stores:
        #   encoder_weight [basis_rank, hidden_size] — projects hidden → latent
        #   encoder_bias   [basis_rank]
        #   decoder_blocks [num_blocks, basis_rank, block_size] — predicts block outputs
        #
        # Routing:  latent = enc_w @ h + enc_b
        #           scores = ||einsum('nr,brs->nbs', latent, dec)||  per block
        #           active  = topk(scores) → active neuron indices
        #
        # Execution: only the rows of gate/up and columns of down corresponding
        #            to active neurons are gathered and matmul'd → ~sparse_top_k/
        #            num_blocks fraction of the MLP compute.
        self._sparse_routing: Dict[int, Dict[str, torch.Tensor]] = {}
        self._sparse_top_k: int = 0
        self._sparse_block_size: int = 32
        self._sparse_param_cache: Dict[str, Dict[str, Any]] = {}

        if sparse_basis_path and str(sparse_basis_path).strip():
            _payload = torch.load(str(sparse_basis_path), map_location="cpu", weights_only=False)
            _cfg = _payload.get("config", {})
            self._sparse_block_size = int(_cfg.get("block_size", 32))
            # top_k: honour explicit override, else use 2 % of blocks as default
            _num_blocks = int(getattr(self.config, "intermediate_size", 4 * int(self.config.hidden_size))) // self._sparse_block_size
            _default_top_k = max(1, int(round(_num_blocks * 0.02)))
            self._sparse_top_k = int(sparse_top_k) if sparse_top_k is not None else _default_top_k
            _layer_states = _payload.get("layer_states", {})
            for _lidx_s, _state in _layer_states.items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                self._sparse_routing[_lidx] = {
                    # Keep on CPU — each layer's routing tensors are ~12 MB total
                    # (enc_w [96,16384] + dec [1664,96,32]). Storing all 83 layers
                    # on GPU would consume ~1 GB of VRAM permanently. Transfer is
                    # done lazily per-layer in _route_sparse_mlp (~12 MB/layer).
                    "enc_w": _state["encoder_weight"].to(dtype=self.dtype),  # [R, H] CPU
                    "enc_b": _state["encoder_bias"].to(dtype=self.dtype),    # [R]    CPU
                    "dec":   _state["decoder_blocks"].to(dtype=self.dtype),  # [B, R, S] CPU
                }
            _pct = int(round(self._sparse_top_k * 100 / max(_num_blocks, 1)))
            print(f"[sparse] loaded routing for {len(self._sparse_routing)}/{self.num_layers} layers "
                  f"| top_k={self._sparse_top_k}/{_num_blocks} blocks ({_pct}%) "
                  f"| block_size={self._sparse_block_size}", flush=True)

        # ── Sparse Attention Head routing ─────────────────────────────────────
        # Loaded from the importance checkpoint produced by
        # init_learned_attn_head_importance.py.  At inference, only the top-K
        # heads' NF4 bytes for q_proj (row gather) and o_proj (column gather)
        # are transferred from CPU RAM to GPU, saving up to 75–87% of attention
        # PCIe bandwidth per token.
        #
        # _attn_active_head_indices[layer_idx] = Tensor[K] of sorted head indices
        # _attn_head_importance[layer_idx]     = Tensor[num_heads] mean norms (for
        #                                        dynamic re-ranking via Taylor state_z)
        self._attn_active_head_indices: Dict[int, torch.Tensor] = {}
        self._attn_head_importance: Dict[int, torch.Tensor] = {}
        self._attn_active_heads: int = 0
        # Metadata cache: code + dequantised absmax per weight (no packed bytes —
        # raw bytes are fetched O(1) from loader._ram_cache on every call).
        self._attn_sparse_param_meta: Dict[str, Dict[str, Any]] = {}
        # GPU FP16 staging buffer for dequantised partial q_proj rows: [K*head_dim, hidden].
        self._attn_q_head_staging: Optional[torch.Tensor] = None

        if attn_head_importance_path and str(attn_head_importance_path).strip():
            _attn_payload = torch.load(
                str(attn_head_importance_path), map_location="cpu", weights_only=False
            )
            _attn_cfg = _attn_payload.get("config", {})
            _H   = int(_attn_cfg.get("num_heads",    getattr(self.config, "num_attention_heads", 128)))
            _D   = int(_attn_cfg.get("head_dim",     getattr(self.config, "head_dim", 128)))
            _Hid = int(_attn_cfg.get("hidden_size",  getattr(self.config, "hidden_size", 16384)))
            K    = int(attn_active_heads) if attn_active_heads is not None else max(1, _H // 2)
            self._attn_active_heads = K

            for _lidx_s, _state in _attn_payload.get("layer_states", {}).items():
                _lidx = int(_lidx_s)
                if not (0 <= _lidx < self.num_layers):
                    continue
                imp = _state["importance"].float()          # [num_heads]
                self._attn_head_importance[_lidx] = imp
                top_k = torch.topk(imp, k=min(K, _H), largest=True).indices.sort().values
                self._attn_active_head_indices[_lidx] = top_k  # sorted for contiguous gather

            if torch.cuda.is_available():
                # Pre-allocate FP16 buffer for dequantised partial q_proj rows.
                self._attn_q_head_staging = torch.empty(
                    K * _D * _Hid, dtype=self.dtype, device=self.device
                )

            _pct_a = int(round(K * 100 // max(_H, 1)))
            print(
                f"[sparse_attn] loaded importance for {len(self._attn_active_head_indices)}/{self.num_layers} layers "
                f"| active_heads={K}/{_H} ({_pct_a}%)",
                flush=True,
            )

    def reset_caches(self) -> None:
        self._taylor_caches = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prefetch_layer(self, layer_idx: int) -> None:
        """Warm the RAM cache for *layer_idx* from disk on a background thread.

        Only loads parameters that are not yet cached — safe to call concurrently
        with the GPU forward pass because it touches only CPU memory and the
        thread-safe RAM cache dict (protected by _ram_cache_lock).
        """
        if not self.loader._ram_cache_enabled:
            return
        if self._layer_param_keys is None:
            return  # skeleton not yet inspected; skip this prefetch
        prefix = f"model.layers.{int(layer_idx)}."
        for k in self._layer_param_keys:
            full_name = f"{prefix}{k}"
            with self.loader._ram_cache_lock:
                already = full_name in self.loader._ram_cache
            if not already:
                try:
                    self.loader._load_raw_for_param(full_name)
                except Exception:
                    pass  # best-effort; main thread will retry

    def _route_sparse_mlp(self, hidden: torch.Tensor, layer_idx: int) -> Optional[torch.Tensor]:
        """Return active block indices [N, top_k] for this layer, or None (→ dense).

        Uses the learned encoder/decoder from the basis checkpoint to predict
        which MLP output blocks will have the largest norm for this hidden state,
        then returns the top-K block indices.  All tensors moved to GPU lazily.
        """
        routing = self._sparse_routing.get(layer_idx)
        if routing is None:
            return None

        # Transfer routing tensors for this layer from CPU to GPU (~12 MB).
        enc_w = routing["enc_w"].to(device=self.device, non_blocking=True)  # [R, H]
        enc_b = routing["enc_b"].to(device=self.device, non_blocking=True)  # [R]
        dec   = routing["dec"].to(device=self.device, non_blocking=True)    # [B, R, S]

        N = hidden.shape[0] * hidden.shape[1]
        h = hidden.view(N, -1).to(dtype=enc_w.dtype)

        latent = F.linear(h, enc_w, enc_b)                           # [N, R]
        B, R, S = dec.shape
        # predicted output per block: [N, B, S]  (fp32 for numerical precision)
        pred = torch.einsum("nr,brs->nbs", latent.float(), dec.float())
        scores = pred.norm(dim=-1)                                    # [N, B]
        active_blocks = scores.topk(self._sparse_top_k, dim=-1).indices  # [N, K]
        return active_blocks

    def _get_sparse_4bit_param(self, full_name: str) -> Dict[str, Any]:
        cached = self._sparse_param_cache.get(full_name)
        if cached is not None:
            return cached

        raw_weight, quant_aux = self.loader._load_raw_for_param(full_name)
        if not quant_aux:
            raise RuntimeError(f"Sparse 4-bit path expected quantized weights for {full_name}")

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        absmax = quant_state.absmax
        if quant_state.nested:
            absmax = bnb_functional.dequantize_blockwise(quant_state.absmax, quant_state.state2)
            absmax = absmax + quant_state.offset

        # Precompute block-major views for CPU gather
        in_features = int(quant_state.shape[1])
        out_features = int(quant_state.shape[0])
        block_size = self._sparse_block_size
        
        bytes_per_row = in_features // 2
        bytes_per_block = bytes_per_row * block_size
        blocks_per_col = out_features // block_size
        packed_blocks = raw_weight.view(blocks_per_col, bytes_per_block).contiguous()
        
        absmax_per_row = in_features // int(quant_state.blocksize)
        absmax_per_block = absmax_per_row * block_size
        absmax_blocks = absmax.view(blocks_per_col, absmax_per_block).contiguous()

        cached = {
            "packed_weight": raw_weight.reshape(-1).contiguous(),
            "packed_blocks": packed_blocks,
            "absmax": absmax.to(dtype=torch.float32).contiguous(),
            "absmax_blocks": absmax_blocks.to(dtype=torch.float32).contiguous(),
            "code": quant_state.code.to(dtype=torch.float32).contiguous(),
            "out_features": out_features,
            "in_features": in_features,
            "quant_block_size": int(quant_state.blocksize),
            "quant_type": str(quant_state.quant_type),
            "dtype": quant_state.dtype,
        }
        self._sparse_param_cache[full_name] = cached
        return cached

    def _load_optional_bias(
        self,
        full_name: str,
        bias_ref: Optional[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if bias_ref is None:
            return None
        bias_name = full_name[:-7] + ".bias" if full_name.endswith(".weight") else ""
        if not bias_name:
            return bias_ref.to(device=device, dtype=dtype, non_blocking=True)
        bias = self.loader.load_parameter(bias_name)
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
            return mlp(hidden)  # not a standard LlamaMLP — run dense
        if not (hasattr(gate_proj, "weight") and gate_proj.weight is not None):
            return mlp(hidden)  # quantised linear without accessible .weight

        block_size = self._sparse_block_size
        N, H = hidden.shape[0] * hidden.shape[1], hidden.shape[-1]
        h = hidden.view(N, H)

        # Expand block indices to individual neuron indices.
        # active_blocks: [N, K]  →  active_neurons: [N, K*block_size]
        neuron_offsets = torch.arange(block_size, device=active_blocks.device)  # [S]
        active_neurons = (
            active_blocks.unsqueeze(-1) * block_size + neuron_offsets
        ).reshape(N, -1)                                              # [N, K*S]

        # For single-token generation (N=1) the unique set == the full set.
        # For N>1 take the union so every token's active neurons are covered.
        if N == 1:
            unique_neurons = active_neurons.squeeze(0)                # [K*S]
        else:
            unique_neurons = active_neurons.unique()                  # [≤N*K*S]

        def _is_4bit_linear(linear: nn.Module) -> bool:
            weight = getattr(linear, "weight", None)
            return bool(torch.is_tensor(weight) and getattr(weight, "quant_state", None) is not None)

        def _load_quantized_weight_cpu(param_name: str) -> Optional[torch.Tensor]:
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
        setattr(gate_proj, "_sparse_param_name", f"{prefix}gate_proj.weight")
        setattr(up_proj, "_sparse_param_name", f"{prefix}up_proj.weight")
        setattr(down_proj, "_sparse_param_name", f"{prefix}down_proj.weight")
        gate = _project_active_inputs(gate_proj, unique_neurons)    # [N, K*S]
        up   = _project_active_inputs(up_proj, unique_neurons)      # [N, K*S]
        act  = F.silu(gate) * up                                    # [N, K*S]
        del gate, up
        out  = _project_active_outputs(act, down_proj, unique_neurons)  # [N, H]
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

    def _sparse_mlp_forward_fast(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
        active_blocks: torch.Tensor,
    ) -> torch.Tensor:
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return self._sparse_mlp_forward(layer_idx, mlp, hidden, active_blocks)

        block_size = self._sparse_block_size
        flat_hidden = hidden.view(-1, hidden.shape[-1])
        prefix = f"model.layers.{int(layer_idx)}.mlp."

        gate_param = self._get_sparse_4bit_param(f"{prefix}gate_proj.weight")
        up_param = self._get_sparse_4bit_param(f"{prefix}up_proj.weight")
        down_param = self._get_sparse_4bit_param(f"{prefix}down_proj.weight")

        cpu_blocks = active_blocks.cpu().view(-1)
        if flat_hidden.shape[0] > 1:
            cpu_blocks = cpu_blocks.unique()
        
        num_active_blocks = cpu_blocks.shape[0]
        K_S = num_active_blocks * block_size
        H_in = gate_param["in_features"]
        
        def _gather_and_dequant(param: Dict[str, Any], bias_param: Optional[torch.Tensor]) -> torch.Tensor:
            p_blocks = param["packed_blocks"][cpu_blocks].reshape(-1)
            a_blocks = param["absmax_blocks"][cpu_blocks].reshape(-1)
            p_gpu = p_blocks.to(device=self.device, non_blocking=False).contiguous()
            a_gpu = a_blocks.to(device=self.device, non_blocking=False).contiguous()

            out_gpu = torch.empty((K_S, H_in), dtype=self.dtype, device=self.device)
            _bnb_dequant_impl(
                p_gpu, a_gpu, param["quant_block_size"], param["quant_type"], out_gpu.dtype, out=out_gpu
            )
            # No torch.cuda.synchronize() here — the subsequent F.linear enqueues
            # on the same CUDA stream, so it sees the completed dequant result.

            bias_gpu = None
            if bias_param is not None:
                neuron_offsets = torch.arange(block_size, device="cpu")
                active_neurons = (cpu_blocks.unsqueeze(-1) * block_size + neuron_offsets).reshape(-1)
                bias_gpu = bias_param[active_neurons].to(device=self.device, dtype=self.dtype)

            # F.linear(x, W) -> x @ W.T
            return F.linear(flat_hidden, out_gpu, bias_gpu)

        gate = _gather_and_dequant(gate_param, getattr(gate_proj, "bias", None))
        up = _gather_and_dequant(up_param, getattr(up_proj, "bias", None))
        
        activated = F.silu(gate) * up
        del gate, up

        # Column-block gather for down_proj: only transfer the active columns' NF4
        # bytes (~8.7 MB) instead of the full 436 MB weight, then decode on CPU
        # and transfer the FP16 result (~33 MB) to GPU.
        H_out = down_param["out_features"]   # 16384 (hidden_size)
        I_in  = down_param["in_features"]    # 53248 (intermediate_size)
        qbs   = down_param["quant_block_size"]  # 64
        code_cpu = down_param["code"]        # [16] NF4 code table on CPU

        # Gather NF4 bytes for the active column blocks.
        # down_param["packed_weight"] is flat [H_out * I_in // 2]; view as [H_out, I_in//2].
        bytes_per_row  = I_in // 2           # 26624
        bytes_per_cblk = block_size // 2     # 16  (32 elements per column block = 16 bytes)
        raw_2d = down_param["packed_weight"].view(H_out, bytes_per_row)  # [H_out, I_in//2]
        col_b_starts = cpu_blocks * bytes_per_cblk                        # [K] byte start per block
        col_b_range  = torch.arange(bytes_per_cblk, dtype=torch.long)    # [16]
        col_b_idx    = (col_b_starts.unsqueeze(-1) + col_b_range.unsqueeze(0)).reshape(-1)  # [K*16]
        gathered_packed = raw_2d[:, col_b_idx].reshape(H_out, num_active_blocks, bytes_per_cblk)
        # gathered_packed: [H_out, K, bytes_per_cblk]

        # Gather absmax: one entry per (row, column-block), which covers block_size=32 elements.
        # Original absmax covers quant_block_size=64 elements per group; each column block
        # is the first or second half of that group (blocks come in pairs sharing an absmax).
        absmax_per_row = I_in // qbs         # 832 absmax entries per row
        absmax_2d = down_param["absmax"].view(H_out, absmax_per_row)     # [H_out, 832]
        absmax_col = absmax_2d[:, cpu_blocks // 2].unsqueeze(-1)         # [H_out, K, 1]

        # Decode NF4 nibbles → FP16 on CPU.
        # Even neuron in block (n % 2 == 0) → high nibble; odd → low nibble.
        hi = ((gathered_packed >> 4) & 0x0F).long()   # [H_out, K, bytes_per_cblk]
        lo = (gathered_packed & 0x0F).long()           # [H_out, K, bytes_per_cblk]
        # Interleave into [H_out, K, block_size]: positions 0,2,4… = hi; 1,3,5… = lo
        decoded = torch.stack([code_cpu[hi], code_cpu[lo]], dim=-1).reshape(H_out, num_active_blocks, block_size)
        # Scale and reshape to [H_out, K*block_size]
        down_weight_active_cpu = (decoded * absmax_col).reshape(H_out, K_S).to(dtype=self.dtype)

        # Transfer gathered FP16 columns to GPU (~33 MB vs 436 MB for full weight).
        down_weight_active = down_weight_active_cpu.to(device=self.device, non_blocking=False)

        down_bias = getattr(down_proj, "bias", None)
        bias_gpu = None if down_bias is None else down_bias.to(device=self.device, dtype=self.dtype)

        out = F.linear(activated, down_weight_active, bias_gpu)
        return out.view_as(hidden)

    def _dense_mlp_forward_streaming_fast(
        self,
        layer_idx: int,
        mlp: nn.Module,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        down_proj = getattr(mlp, "down_proj", None)
        if gate_proj is None or up_proj is None or down_proj is None:
            return self._dense_mlp_forward_streaming(mlp, hidden)
        if self._mlp_proj_staging is None:
            # No staging buffer (sparse mode or VRAM-constrained): fall back to
            # the streaming dense path which transfers one weight at a time.
            return self._dense_mlp_forward_streaming(mlp, hidden)

        flat_hidden = hidden.view(-1, hidden.shape[-1])
        prefix = f"model.layers.{int(layer_idx)}.mlp."

        # Reuse the pre-allocated projection staging buffer. This avoids a large
        # cudaMalloc/free per layer invocation and keeps the collector on a stable
        # memory footprint during long streaming runs.
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
            # copy_() does a direct CPU→GPU DMA — no intermediate GPU tensor needed.
            weight_view.copy_(raw.to(dtype=self.dtype))
            del raw
            bias_gpu = self._load_optional_bias(
                param_name,
                getattr(linear, "bias", None),
                device=x.device,
                dtype=x.dtype,
            )
            y = F.linear(x, weight_view.to(dtype=x.dtype), bias_gpu)
            del bias_gpu
            return y

        gate = _linear_stream(flat_hidden, gate_proj, f"{prefix}gate_proj.weight")
        up = _linear_stream(flat_hidden, up_proj, f"{prefix}up_proj.weight")
        act = F.silu(gate) * up
        del gate, up
        out = _linear_stream(act, down_proj, f"{prefix}down_proj.weight")
        return out.view_as(hidden)

    # ── Sparse Attention Head helpers ─────────────────────────────────────────

    def _get_attn_active_heads(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return sorted active head indices [K] for this layer, or None (dense).

        Blends static calibration importance with a live Taylor state_z signal.
        state_z[0, g, :].norm() reflects how much KV group g has accumulated
        context — groups that haven't fired much are down-weighted.

        Cost: 8 × norm([64]) + topk(128) ≈ 2 µs per token per layer.
        """
        static_indices = self._attn_active_head_indices.get(layer_idx)
        if static_indices is None:
            return None  # layer not profiled → dense

        taylor_cache = self._taylor_caches[layer_idx]
        if taylor_cache is None:
            return static_indices  # first token — no state yet, use static ranking

        static_imp = self._attn_head_importance.get(layer_idx)
        if static_imp is None:
            return static_indices

        # state_z: [1, num_kv_heads, feature_dim] float32 on GPU
        state_z = taylor_cache.state_z
        num_kv_heads = state_z.shape[1]
        group_norms = state_z[0].norm(dim=-1).float()                  # [num_kv_heads]

        # Expand KV-group norms to per-query-head: 128Q / 8KV = 16 query heads per group
        heads_per_group = static_imp.shape[0] // num_kv_heads          # 16
        head_norms = group_norms.repeat_interleave(heads_per_group)    # [num_heads] on GPU

        # Normalise each signal to [0,1] independently, then multiply.
        norm_s = static_imp.to(device=state_z.device) / static_imp.max().clamp_min(1e-8)
        norm_d = head_norms / head_norms.max().clamp_min(1e-8)
        combined = norm_s * norm_d  # [num_heads]

        K = self._attn_active_heads
        return torch.topk(combined, k=min(K, combined.shape[0]), largest=True).indices.sort().values

    def _get_sparse_4bit_attn_meta(self, full_name: str, *, head_dim: int) -> Dict[str, Any]:
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
            "code":            quant_state.code.to(dtype=torch.float32).contiguous(),  # [16]
            # Flat dequantised absmax — ~16 MB for 405B q_proj.  Caching this avoids
            # repeating the nested dequant on every token (expensive CPU op).
            "absmax_flat":     absmax.to(dtype=torch.float32).contiguous(),
        }
        self._attn_sparse_param_meta[full_name] = cached
        return cached

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
            return  # CPU-only mode: caller falls back to full dense load

        prefix = f"model.layers.{int(layer_idx)}."
        q_name = f"{prefix}self_attn.q_proj.weight"
        o_name = f"{prefix}self_attn.o_proj.weight"

        meta_q = self._get_sparse_4bit_attn_meta(q_name, head_dim=head_dim)
        meta_o = self._get_sparse_4bit_attn_meta(o_name, head_dim=head_dim)

        active_cpu = active_heads.cpu()  # keep on CPU for gather indexing
        K = int(active_cpu.shape[0])

        # ── q_proj: Row-block gather ──────────────────────────────────────────
        q_skel = self._layer_skeleton.self_attn.q_proj.weight  # [num_heads*head_dim, hidden]
        q_skel.zero_()  # fast GPU memset — inactive heads contribute 0

        q_raw, _ = self.loader._load_raw_for_param(q_name)       # RAM cache hit
        bytes_per_head_q = meta_q["in_features"] // 2 * head_dim  # 1.048 MB for 405B
        packed_2d_q = q_raw.view(meta_q["num_heads_total"], bytes_per_head_q)  # view, no copy
        gathered_q = packed_2d_q[active_cpu].reshape(-1).contiguous()          # [K*bph]

        absmax_per_head_q = head_dim * meta_q["in_features"] // meta_q["quant_block_size"]
        absmax_2d_q = meta_q["absmax_flat"].view(meta_q["num_heads_total"], absmax_per_head_q)
        gathered_absmax_q = absmax_2d_q[active_cpu].reshape(-1).contiguous()   # [K*aph]

        n_q = gathered_q.numel()
        self._nf4_staging[:n_q].copy_(gathered_q, non_blocking=False)
        absmax_q_gpu = gathered_absmax_q.to(device=self.device, non_blocking=False)
        q_partial = self._attn_q_head_staging.view(K * head_dim, hidden_size)
        _bnb_dequant_impl(
            self._nf4_staging[:n_q], absmax_q_gpu,
            meta_q["quant_block_size"], meta_q["quant_type"], self.dtype, out=q_partial,
        )
        active_gpu = active_heads.to(device=self.device)
        row_idx = (
            active_gpu.unsqueeze(-1) * head_dim
            + torch.arange(head_dim, device=self.device)
        ).reshape(-1)   # [K*head_dim]
        q_skel[row_idx] = q_partial

        # ── o_proj: Column-block gather (CPU decode, same as down_proj) ───────
        o_skel = self._layer_skeleton.self_attn.o_proj.weight  # [hidden, num_heads*head_dim]
        o_skel.zero_()

        o_raw, _ = self.loader._load_raw_for_param(o_name)
        H_out     = meta_o["out_features"]                     # hidden_size
        H_in      = meta_o["in_features"]                      # num_heads * head_dim
        qbs       = meta_o["quant_block_size"]                 # 64
        code_cpu  = meta_o["code"]                             # [16]

        bytes_per_row_o     = H_in // 2                        # 8192
        bytes_per_head_col  = head_dim // 2                    # 64 bytes per head-col per row
        raw_2d_o = o_raw.view(H_out, bytes_per_row_o)          # [H_out, 8192] — no copy

        col_starts = active_cpu * bytes_per_head_col           # [K]
        col_range  = torch.arange(bytes_per_head_col)          # [64]
        col_idx    = (col_starts.unsqueeze(-1) + col_range).reshape(-1)  # [K*64]
        gathered_o = raw_2d_o[:, col_idx].reshape(H_out, K, bytes_per_head_col)

        # Absmax: head_dim=128 spans 2 quant groups (qbs=64) per row.
        absmax_per_head_col = head_dim // qbs                  # 2
        absmax_per_row_o = H_in // qbs                         # 256
        absmax_2d_o = meta_o["absmax_flat"].view(H_out, absmax_per_row_o)
        a_starts = active_cpu * absmax_per_head_col            # [K]
        a_range  = torch.arange(absmax_per_head_col)           # [0, 1]
        a_idx    = (a_starts.unsqueeze(-1) + a_range).reshape(-1)  # [K*2]
        absmax_o = absmax_2d_o[:, a_idx].reshape(H_out, K, absmax_per_head_col, 1)

        # Decode NF4 nibbles on CPU — identical pattern to down_proj in _sparse_mlp_forward_fast.
        hi = ((gathered_o >> 4) & 0x0F).long()  # [H_out, K, 64]
        lo = (gathered_o & 0x0F).long()          # [H_out, K, 64]
        decoded = torch.stack([code_cpu[hi], code_cpu[lo]], dim=-1).reshape(H_out, K, head_dim)
        # Scale: reshape to [H_out, K, 2, qbs] to broadcast with [H_out, K, 2, 1].
        o_weight_cpu = (
            decoded.reshape(H_out, K, absmax_per_head_col, qbs) * absmax_o
        ).reshape(H_out, K * head_dim).to(dtype=self.dtype)

        o_active = o_weight_cpu.to(device=self.device, non_blocking=False)
        col_idx_gpu = (
            active_gpu.unsqueeze(-1) * head_dim
            + torch.arange(head_dim, device=self.device)
        ).reshape(-1)   # [K*head_dim]
        o_skel[:, col_idx_gpu] = o_active

    def _load_layer(self, layer_idx: int) -> LlamaDecoderLayer:
        self._layer_skeleton.layer_idx = int(layer_idx)
        if hasattr(self._layer_skeleton, "self_attn"):
            self._layer_skeleton.self_attn.layer_idx = int(layer_idx)

        skeleton_state = self._layer_skeleton.state_dict()

        # Capture floating-point param keys on first call (same for every layer).
        if self._layer_param_keys is None:
            self._layer_param_keys = [k for k, v in skeleton_state.items() if v.is_floating_point()]

        # Determine if sparse attention is active for this layer.
        # _get_attn_active_heads returns None when no importance data is loaded
        # or for layers not profiled — both fall back to full dense loading.
        _active_attn_heads = self._get_attn_active_heads(layer_idx)
        _head_dim = int(getattr(self.config, "head_dim", 128))
        _hidden   = int(getattr(self.config, "hidden_size", 16384))
        # Keys to skip in the main loop (handled by _load_sparse_attn_heads below).
        _skip_attn: set = (
            {"self_attn.q_proj.weight", "self_attn.o_proj.weight"}
            if _active_attn_heads is not None
            else set()
        )

        # ── Tier 1: RAM cache → GPU  (miss falls through to SSD on first pass) ──
        prefix = f"model.layers.{int(layer_idx)}."
        for k, dest in skeleton_state.items():
            full_name = f"{prefix}{k}"
            if k.startswith("mlp.") and dest.is_floating_point():
                continue
            if k in _skip_attn:
                continue  # handled below by _load_sparse_attn_heads
            if dest.is_floating_point():
                self.loader.load_parameter_into(
                    name=full_name,
                    out=dest,
                    dtype=self.dtype,
                    staging=self._nf4_staging,
                    absmax_staging=self._absmax_staging,
                    nested_absmax_staging=self._nested_absmax_staging,
                    state2_absmax_staging=self._state2_absmax_staging,
                    code_staging=self._code_staging,
                )
            else:
                raw = self.loader.load_parameter(full_name)
                dest.copy_(raw)

        # Sparse attention head loading: only transfer NF4 bytes for K active heads.
        # q_proj and o_proj are zeroed first; only the active head rows/columns are filled.
        if _active_attn_heads is not None:
            self._load_sparse_attn_heads(
                layer_idx, _active_attn_heads, head_dim=_head_dim, hidden_size=_hidden,
            )

        # Kick off RAM prefetch for the next layer on the background thread.
        next_idx = layer_idx + 1
        if self._prefetch_executor is not None and next_idx < self.num_layers:
            self._prefetch_executor.submit(self._prefetch_layer, next_idx)

        return self._layer_skeleton

    def _release_modules(self, *modules: nn.Module) -> None:
        # Avoid aggressive GC/empty_cache in the hot path
        pass

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        *,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
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
        capture_layers: Optional[Sequence[int]] = None,
        use_attention_cache: bool = True,
    ) -> tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1 or int(token_ids.shape[1]) != 1:
            raise RuntimeError("StreamingLlamaRuntime.forward_token currently supports batch_size=1 and seq_len=1 only")

        capture_set = {int(idx) for idx in capture_layers or []}
        captures: Dict[int, Dict[str, torch.Tensor]] = {}
        printed_progress = False
        # Index on CPU (avoids 3.9 GiB GPU allocation), move the resulting 32 KB vector to GPU.
        hidden = self.embed_tokens(token_ids.cpu()).to(device=self.device, dtype=self.dtype)
        position_ids = torch.tensor([[int(position_index)]], device=self.device, dtype=torch.long)
        rope = self.rotary_emb(hidden, position_ids)

        for layer_idx in range(self.num_layers):
            printed_progress = True
            if torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"  [layer {layer_idx + 1}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB (alloc/reserv)",
                    end="\r",
                    flush=True,
                )
            else:
                print(f"  [layer {layer_idx + 1}/{self.num_layers}] loading...", end="\r", flush=True)
            if self._debug_steps:
                print(f"[debug] layer {layer_idx}: load", flush=True)
            layer = self._load_layer(layer_idx)
            taylor_attn: Optional[GQATaylorSSDSelfAttention] = None
            residual = hidden
            hidden_norm = layer.input_layernorm(hidden)

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
            else:
                if self._debug_steps:
                    print(f"[debug] layer {layer_idx}: attn_forward_dense", flush=True)
                # No-cache path: used by collect_dense_mlp_rows (use_attention_cache=False)
                # to avoid accumulating ~4 MB Taylor-SSD state per layer × 126 layers
                # which would exhaust the 200 MB pool headroom by layer ~48 and crash.
                attn_out, _attn_weights = layer.self_attn(
                    hidden_states=hidden_norm,
                    position_embeddings=rope,
                    attention_mask=None,
                    past_key_values=self._dense_cache if use_attention_cache else None,
                    cache_position=position_ids.view(-1),
                )

            hidden = residual + attn_out
            residual = hidden
            mlp_input = layer.post_attention_layernorm(hidden)

            if self._debug_steps:
                print(f"[debug] layer {layer_idx}: route", flush=True)
            _active_blocks = self._route_sparse_mlp(mlp_input, layer_idx)
            if _active_blocks is not None:
                if self._debug_steps:
                    print(f"[debug] layer {layer_idx}: sparse_mlp", flush=True)
                mlp_out = self._sparse_mlp_forward_fast(layer_idx, layer.mlp, mlp_input, _active_blocks)
            else:
                if self._debug_steps:
                    print(f"[debug] layer {layer_idx}: dense_mlp", flush=True)
                mlp_out = self._dense_mlp_forward_streaming_fast(layer_idx, layer.mlp, mlp_input)

            if layer_idx in capture_set:
                captures[layer_idx] = {
                    "mlp_input": mlp_input.detach().cpu(),
                    "dense_mlp_out": mlp_out.detach().cpu(),
                }
            hidden = residual + mlp_out
            self._release_modules(layer, *( [taylor_attn] if taylor_attn is not None else [] ))

        if printed_progress:
            print("", flush=True)
        hidden = self.norm(hidden)
        # Move the 32 KB hidden vector to CPU for lm_head projection (weight stays on CPU).
        if self.lm_head is None:
            logits = torch.zeros((hidden.shape[0], hidden.shape[1], 1), dtype=torch.float32)
        else:
            logits = self.lm_head(hidden.cpu().to(dtype=self.lm_head.weight.dtype)).float()
        return logits, captures

    def forward_sequence(
        self,
        token_ids: torch.LongTensor,
        *,
        selected_layers_set: Optional[set] = None,
    ) -> None:
        """Run a full-sequence calibration pass with a layer-first loop.

        Each of the 126 layers is loaded from RAM→GPU exactly once per sequence.
        Within each layer, tokens are processed one at a time using the same
        call signature as forward_token() (seq_len=1, no KV cache).
        No KV cache means each token attends only to itself — acceptable for head
        importance calibration where relative per-head magnitudes matter, not
        autoregressive quality.  MLP residual is zero (already handled by
        zero-passthrough fallback when _mlp_proj_staging is None).

        Speedup vs forward_token(): seq_len× fewer PCIe layer loads.

        selected_layers_set: if provided, layers not in this set are skipped entirely
        (no PCIe load, hidden states pass through unchanged). Saves ~(1 - K/N)× bandwidth.
        """
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1:
            raise RuntimeError("forward_sequence requires batch_size=1")

        seq_len = int(token_ids.shape[1])
        all_hidden = self.embed_tokens(token_ids.cpu()).to(device=self.device, dtype=self.dtype)
        position_ids = torch.arange(seq_len, device=self.device, dtype=torch.long)
        # Pre-allocate second buffer once — avoids cudaMalloc on every layer iteration.
        next_hidden = torch.empty_like(all_hidden)

        for layer_idx in range(self.num_layers):
            # Skip PCIe load entirely for layers not being profiled.
            if selected_layers_set is not None and layer_idx not in selected_layers_set:
                continue

            if torch.cuda.is_available() and layer_idx % 10 == 0:
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"  [seq layer {layer_idx + 1}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB",
                    end="\r", flush=True,
                )
            layer = self._load_layer(layer_idx)

            # Process one token at a time (batch=1, seq=1).
            # The batch-as-tokens trick (batch=seq_len) causes the Taylor local-attention
            # to allocate [seq_len, 128_heads, local_window, head_dim] = ~268 MB tensors,
            # overflowing VRAM and causing Windows WDDM to page to system RAM.
            # Single-token processing keeps all Taylor state tensors at [1, ...] = tiny.
            for pos in range(seq_len):
                h = all_hidden[:, pos : pos + 1, :]                     # [1, 1, hidden]
                position_ids_tok = position_ids[pos : pos + 1].unsqueeze(0)  # [1, 1]
                rope_tok = self.rotary_emb(h, position_ids_tok)
                h_norm = layer.input_layernorm(h)
                attn_out, _ = layer.self_attn(
                    hidden_states=h_norm,
                    position_embeddings=rope_tok,
                    attention_mask=None,
                    past_key_values=None,
                )
                next_hidden[:, pos : pos + 1, :] = h + attn_out

            all_hidden, next_hidden = next_hidden, all_hidden  # swap — no allocation
            self._release_modules(layer)

        print("", flush=True)

    def _forward_prefill(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Process all prompt tokens with each of the 126 layers loaded only once.

        Instead of the naive loop (load layer N for token 1, load layer N for
        token 2, …), we invert the loops: load layer N once, then run all prompt
        tokens through it sequentially before unloading.  For a P-token prompt
        this reduces layer loads from P×126 to 126.

        Attention is still processed one token at a time (Taylor-SSD is a
        recurrent operator; running seq_len>1 through it allocates ~268 MB
        local-attention tensors per layer and overflows 8 GB VRAM).  MLP is
        batched over all tokens at once — the sparse path transfers only the
        active-block NF4 bytes regardless of seq_len.
        """
        seq_len = int(token_ids.shape[1])
        all_hidden = self.embed_tokens(token_ids.cpu()).to(device=self.device, dtype=self.dtype)
        # [1, seq_len, hidden]
        next_hidden = torch.empty_like(all_hidden)
        position_ids_all = torch.arange(seq_len, device=self.device, dtype=torch.long)

        for layer_idx in range(self.num_layers):
            if torch.cuda.is_available():
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved() / 1e9
                print(
                    f"  [prefill layer {layer_idx + 1}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB",
                    end="\r", flush=True,
                )
            layer = self._load_layer(layer_idx)

            # ── Attention: one token at a time ────────────────────────────
            use_taylor = layer_idx in self.taylor_layer_set
            if use_taylor:
                taylor_attn = self._shared_taylor_attn
                taylor_attn.layer_idx = layer_idx

            for pos in range(seq_len):
                h = all_hidden[:, pos : pos + 1, :]           # [1, 1, hidden]
                pos_ids = position_ids_all[pos : pos + 1].unsqueeze(0)  # [1, 1]
                rope_tok = self.rotary_emb(h, pos_ids)
                h_norm = layer.input_layernorm(h)

                if use_taylor:
                    attn_out, _, present = taylor_attn(
                        hidden_states=h_norm,
                        position_ids=pos_ids,
                        position_embeddings=rope_tok,
                        past_key_value=self._taylor_caches[layer_idx],
                        use_cache=True,
                    )
                    self._taylor_caches[layer_idx] = present
                else:
                    attn_out, _ = layer.self_attn(
                        hidden_states=h_norm,
                        position_embeddings=rope_tok,
                        attention_mask=None,
                        past_key_values=self._dense_cache,
                        cache_position=pos_ids.view(-1),
                    )

                next_hidden[:, pos : pos + 1, :] = h + attn_out

            all_hidden, next_hidden = next_hidden, all_hidden  # swap — no allocation
            # all_hidden is now the post-attention residual for all tokens

            # ── MLP: batched over all tokens ──────────────────────────────
            residual = all_hidden
            mlp_input = layer.post_attention_layernorm(all_hidden)
            _active_blocks = self._route_sparse_mlp(mlp_input, layer_idx)
            if _active_blocks is not None:
                mlp_out = self._sparse_mlp_forward_fast(layer_idx, layer.mlp, mlp_input, _active_blocks)
            else:
                mlp_out = self._dense_mlp_forward_streaming_fast(layer_idx, layer.mlp, mlp_input)
            all_hidden = residual + mlp_out

            self._release_modules(layer)

        print("", flush=True)
        all_hidden = self.norm(all_hidden)
        # Only the last token's logits are needed to start generation.
        logits = self.lm_head(
            all_hidden[:, -1:, :].cpu().to(dtype=self.lm_head.weight.dtype)
        ).float()
        return logits

    def generate(
        self,
        input_ids: torch.LongTensor,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: float = 1.0,
    ) -> torch.LongTensor:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise RuntimeError("StreamingLlamaRuntime.generate currently supports batch_size=1 only")
        if self.lm_head is None:
            raise RuntimeError("StreamingLlamaRuntime.generate requires materialize_lm_head=True")

        generated = input_ids.to(device=self.device)
        self.reset_caches()
        logits: Optional[torch.Tensor] = None
        with torch.no_grad():
            prompt_len = int(generated.shape[1])
            if prompt_len > 1:
                print(f"[prompt] batched prefill: {prompt_len} tokens × 1 layer pass", flush=True)
                logits = self._forward_prefill(generated)
            else:
                logits, _ = self.forward_token(generated[:, 0:1], position_index=0)
            if logits is None:
                raise RuntimeError("No prompt tokens were provided")

            for _step_idx in range(int(max_new_tokens)):
                next_token = self._sample_next_token(
                    logits[:, -1, :],
                    do_sample=bool(do_sample),
                    temperature=float(temperature),
                    top_k=top_k,
                    top_p=float(top_p),
                ).view(1, 1)
                generated = torch.cat([generated, next_token.to(device=self.device, dtype=generated.dtype)], dim=-1)
                if eos_token_id is not None and int(next_token.item()) == int(eos_token_id):
                    break
                logits, _ = self.forward_token(next_token, position_index=int(generated.shape[1]) - 1)
        return generated

    def collect_dense_mlp_rows(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        selected_layers: Sequence[int],
        layer_x: Dict[int, List[torch.Tensor]],
        layer_y: Dict[int, List[torch.Tensor]],
        layer_rows: Dict[int, int],
        max_rows: int,
    ) -> None:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            raise RuntimeError(
                "StreamingLlamaRuntime.collect_dense_mlp_rows currently supports batch_size=1 only"
            )
        if not selected_layers:
            return
        if self._mlp_proj_staging is None:
            raise RuntimeError(
                "collect_dense_mlp_rows: _mlp_proj_staging is None — VRAM was insufficient to "
                "allocate the MLP staging buffer at startup. All collected dense_mlp_out values "
                "would be zeros, producing a useless checkpoint. Free VRAM or reduce other "
                "pre-allocated buffers before collecting."
            )

        valid_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(input_ids.shape[1])
        self.reset_caches()
        with torch.no_grad():
            selected_set = {int(idx) for idx in selected_layers}
            all_hidden = self.embed_tokens(input_ids[:, :valid_len].cpu()).to(device=self.device, dtype=self.dtype)
            next_hidden = torch.empty_like(all_hidden)
            attn_hidden = torch.empty_like(all_hidden)
            position_ids = torch.arange(valid_len, device=self.device, dtype=torch.long)
            printed_progress = False

            for layer_idx in range(self.num_layers):
                pending_layers = [idx for idx in selected_layers if int(layer_rows[int(idx)]) < int(max_rows)]
                if not pending_layers:
                    break
                rows_done = sum(1 for idx in selected_layers if int(layer_rows[int(idx)]) >= int(max_rows))
                printed_progress = True
                status = (
                    f"[collect] layer {layer_idx+1}/{self.num_layers} "
                    f"({rows_done}/{len(selected_layers)} layers saturated, {len(pending_layers)} still collecting)"
                )
                print(status.ljust(120), end="\r", flush=True)
                layer = self._load_layer(layer_idx)

                for pos in range(valid_len):
                    h = all_hidden[:, pos : pos + 1, :]
                    position_ids_tok = position_ids[pos : pos + 1].unsqueeze(0)
                    rope_tok = self.rotary_emb(h, position_ids_tok)
                    h_norm = layer.input_layernorm(h)
                    attn_out, _ = layer.self_attn(
                        hidden_states=h_norm,
                        position_embeddings=rope_tok,
                        attention_mask=None,
                        past_key_values=None,
                    )
                    attn_hidden[:, pos : pos + 1, :] = h + attn_out

                mlp_input = layer.post_attention_layernorm(attn_hidden)
                mlp_out = self._dense_mlp_forward_streaming_fast(layer_idx, layer.mlp, mlp_input)

                if int(layer_idx) in selected_set:
                    remain = int(max_rows) - int(layer_rows[int(layer_idx)])
                    if remain > 0:
                        x = mlp_input.reshape(-1, mlp_input.shape[-1]).float()
                        y = mlp_out.reshape(-1, mlp_out.shape[-1]).float()
                        take = min(int(remain), int(x.shape[0]))
                        if take > 0:
                            layer_x[int(layer_idx)].append(x[:take].detach().cpu())
                            layer_y[int(layer_idx)].append(y[:take].detach().cpu())
                            layer_rows[int(layer_idx)] += int(take)

                next_hidden.copy_(attn_hidden + mlp_out)
                all_hidden, next_hidden = next_hidden, all_hidden
                self._release_modules(layer)
            if printed_progress:
                print("", flush=True)
