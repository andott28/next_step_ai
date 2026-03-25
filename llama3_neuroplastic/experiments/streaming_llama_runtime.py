from __future__ import annotations

import concurrent.futures
import gc
import json
import os
import threading
from collections import defaultdict
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
        self._ram_cache_enabled: bool = True
        self._ram_cache_lock = threading.Lock()

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
                    out[key] = handle.get_tensor(key)
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
            if cached is not None:
                return cached

        aux_names = self._quant_aux_by_base.get(full_name, [])
        tensors = self._load_exact_tensors([full_name, *aux_names])
        if full_name not in tensors:
            raise KeyError(f"Tensor '{full_name}' not found in safetensors index")

        weight = tensors[full_name]
        quant_aux = {k: v for k, v in tensors.items() if k != full_name}

        if self._ram_cache_enabled:
            # Store as plain CPU tensors — no pin_memory() to avoid CUDA driver
            # interactions from the background prefetch thread, which would race
            # with GPU kernels on the main thread and risk OOM on Windows.
            # Plain RAM→GPU DMA is still 5–10× faster than SSD→GPU.
            with self._ram_cache_lock:
                self._ram_cache[full_name] = (weight, quant_aux)
            return weight, quant_aux

        return weight, quant_aux

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

        # Create QuantState on CPU — fast dict parsing, no GPU memory allocated.
        # The small internal tensors (absmax, state2, code) stay on CPU and are
        # selectively copied into pre-allocated GPU buffers below.
        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))

        if quant_state.nested:
            if absmax_staging is not None and nested_absmax_staging is not None and state2_absmax_staging is not None and code_staging is not None:
                # --- Zero-alloc GPU path for nested (doubly-quantized) absmax ---
                # 1. Copy the nested absmax uint8 bytes to pre-allocated GPU buffer
                n_nested = quant_state.absmax.numel()
                nested_gpu = nested_absmax_staging[:n_nested]
                nested_gpu.copy_(quant_state.absmax)

                # 2. Copy state2 scales to pre-allocated GPU buffer
                n_s2 = quant_state.state2.absmax.numel()
                s2_gpu = state2_absmax_staging[:n_s2]
                s2_gpu.copy_(quant_state.state2.absmax)

                # 3. Copy dequant codebook to pre-allocated GPU buffer
                code_staging[:quant_state.state2.code.numel()].copy_(quant_state.state2.code)
                code_gpu = code_staging[:quant_state.state2.code.numel()]

                # 4. Dequantize nested absmax ON GPU → output to absmax_staging
                absmax = absmax_staging[:n_nested]
                bnb_functional.dequantize_blockwise(
                    nested_gpu,
                    absmax=s2_gpu,
                    code=code_gpu,
                    out=absmax,
                    blocksize=quant_state.state2.blocksize,
                )
                absmax.add_(quant_state.offset)
            else:
                # Fallback: no staging → let bitsandbytes allocate (slower path)
                quant_state_gpu = bnb_functional.QuantState.from_dict(quant_aux, device=out.device)
                absmax = bnb_functional.dequantize_blockwise(quant_state_gpu.absmax, quant_state_gpu.state2)
                absmax += quant_state_gpu.offset
                del quant_state_gpu
            if absmax.dtype != torch.float32:
                absmax = absmax.float()
        else:
            # Non-nested: absmax is already float32, just copy to GPU staging
            if absmax_staging is not None:
                n_abs = quant_state.absmax.numel()
                absmax = absmax_staging[:n_abs]
                absmax.copy_(quant_state.absmax)
            else:
                absmax = quant_state.absmax.to(device=out.device)
            if absmax.dtype != torch.float32:
                absmax = absmax.float()

        if n * 2 != out.numel():
            raise RuntimeError(
                f"Shape mismatch for '{full_name}': {n} NF4 bytes → {n * 2} elements "
                f"but out has {out.numel()} elements (shape {out.shape})"
            )

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
        sparse_basis_path: Optional[str] = None,
        sparse_top_k: Optional[int] = None,
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
        self.loader = ShardedSafetensorLoader(self.snapshot_dir)
        self.loader._ram_cache_enabled = bool(ram_cache)
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

        self.norm = LlamaRMSNorm(int(self.config.hidden_size), eps=float(self.config.rms_norm_eps)).to(
            device=self.device,
            dtype=self.dtype,
        )
        self.norm.weight.data.copy_(
            self.loader.load_parameter("model.norm.weight").to(device=self.device, dtype=self.dtype)
        )
        self.norm.requires_grad_(False)

        # Keep lm_head on CPU alongside embed_tokens — projection is done on CPU at end of forward.
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

        # Background thread pool (1 worker) that warms the RAM cache for the
        # next layer while the GPU processes the current one.
        self._prefetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="layer_prefetch"
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
        _ffn = int(getattr(self.config, "intermediate_size", _h * 4))
        _max_nf4_bytes = max(_h * _h, _ffn * _h) // 2  # NF4 = 2 values per byte
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
                    "enc_w": _state["encoder_weight"].to(dtype=self.dtype),  # [R, H]
                    "enc_b": _state["encoder_bias"].to(dtype=self.dtype),    # [R]
                    "dec":   _state["decoder_blocks"].to(dtype=self.dtype),  # [B, R, S]
                }
            _pct = int(round(self._sparse_top_k * 100 / max(_num_blocks, 1)))
            print(f"[sparse] loaded routing for {len(self._sparse_routing)}/{self.num_layers} layers "
                  f"| top_k={self._sparse_top_k}/{_num_blocks} blocks ({_pct}%) "
                  f"| block_size={self._sparse_block_size}", flush=True)

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

    def _sparse_mlp_forward(
        self,
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

        def _project_active_inputs(linear: nn.Module, neurons: torch.Tensor) -> torch.Tensor:
            bias = getattr(linear, "bias", None)
            weight = getattr(linear, "weight", None)
            if _is_4bit_linear(linear):
                quant_state = weight.quant_state
                # Params4bit advanced indexing silently dequantizes the whole matrix.
                # Dequantize one projection at a time instead so peak memory stays bounded.
                dense_weight = bnb_functional.dequantize_4bit(weight.t(), quant_state).t()
                proj_weight = dense_weight.index_select(0, neurons)
                del dense_weight
            else:
                proj_weight = weight.index_select(0, neurons)
            if bias is not None:
                bias = bias.index_select(0, neurons)
            return F.linear(h, proj_weight, bias)

        def _project_active_outputs(x: torch.Tensor, linear: nn.Module, neurons: torch.Tensor) -> torch.Tensor:
            bias = getattr(linear, "bias", None)
            weight = getattr(linear, "weight", None)
            if _is_4bit_linear(linear):
                quant_state = weight.quant_state
                dense_weight = bnb_functional.dequantize_4bit(weight.t(), quant_state).t()
                proj_weight = dense_weight.index_select(1, neurons)
                del dense_weight
            else:
                proj_weight = weight.index_select(1, neurons)
            return F.linear(x, proj_weight, bias)

        gate = _project_active_inputs(gate_proj, unique_neurons)    # [N, K*S]
        up   = _project_active_inputs(up_proj, unique_neurons)      # [N, K*S]
        act  = F.silu(gate) * up                                    # [N, K*S]
        del gate, up
        out  = _project_active_outputs(act, down_proj, unique_neurons)  # [N, H]
        return out.view_as(hidden)

    def _load_layer(self, layer_idx: int) -> LlamaDecoderLayer:
        self._layer_skeleton.layer_idx = int(layer_idx)
        if hasattr(self._layer_skeleton, "self_attn"):
            self._layer_skeleton.self_attn.layer_idx = int(layer_idx)

        skeleton_state = self._layer_skeleton.state_dict()

        # Capture floating-point param keys on first call (same for every layer).
        if self._layer_param_keys is None:
            self._layer_param_keys = [k for k, v in skeleton_state.items() if v.is_floating_point()]

        # ── Tier 1: RAM cache → GPU  (miss falls through to SSD on first pass) ──
        prefix = f"model.layers.{int(layer_idx)}."
        for k, dest in skeleton_state.items():
            full_name = f"{prefix}{k}"
            if dest.is_floating_point():
                self.loader.load_parameter_into(
                    full_name, dest, dtype=self.dtype,
                    staging=self._nf4_staging,
                    absmax_staging=self._absmax_staging,
                    nested_absmax_staging=self._nested_absmax_staging,
                    state2_absmax_staging=self._state2_absmax_staging,
                    code_staging=self._code_staging,
                )
            else:
                raw = self.loader.load_parameter(full_name)
                dest.copy_(raw)

        # Kick off RAM prefetch for the next layer on the background thread.
        next_idx = layer_idx + 1
        if next_idx < self.num_layers:
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
            if int(layer_idx) % 8 == 0 or int(layer_idx) == self.num_layers - 1:
                printed_progress = True
                if torch.cuda.is_available():
                    alloc_gb = torch.cuda.memory_allocated() / 1e9
                    reserv_gb = torch.cuda.memory_reserved() / 1e9
                    print(f"  [layer {layer_idx}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB (alloc/reserv)", end="\r", flush=True)
                else:
                    print(f"  [layer {layer_idx}/{self.num_layers}] loading...", end="\r", flush=True)
            layer = self._load_layer(layer_idx)
            taylor_attn: Optional[GQATaylorSSDSelfAttention] = None
            residual = hidden
            hidden_norm = layer.input_layernorm(hidden)

            if layer_idx in self.taylor_layer_set and use_attention_cache:
                taylor_attn = GQATaylorSSDSelfAttention.from_llama_attention(
                    source_attn=layer.self_attn,
                    layer_idx=layer_idx,
                    feature_map=self.taylor_feature_map,
                    local_window=self.taylor_local_window,
                    feature_dim=self.taylor_feature_dim,
                    state_decay=self.taylor_state_decay,
                ).to(device=self.device)
                taylor_attn.eval()
                attn_out, _attn_weights, present = taylor_attn(
                    hidden_states=hidden_norm,
                    position_ids=position_ids,
                    position_embeddings=rope,
                    past_key_value=self._taylor_caches[layer_idx],
                    use_cache=True,
                )
                self._taylor_caches[layer_idx] = present
            else:
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

            _active_blocks = self._route_sparse_mlp(mlp_input, layer_idx)
            if _active_blocks is not None:
                mlp_out = self._sparse_mlp_forward(layer.mlp, mlp_input, _active_blocks)
            else:
                mlp_out = layer.mlp(mlp_input)

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
        logits = self.lm_head(hidden.cpu().to(dtype=self.lm_head.weight.dtype)).float()
        return logits, captures

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

        generated = input_ids.to(device=self.device)
        self.reset_caches()
        logits: Optional[torch.Tensor] = None
        with torch.no_grad():
            for pos in range(int(generated.shape[1])):
                print(f"[prompt] processing token {pos+1}/{generated.shape[1]}...")
                logits, _ = self.forward_token(generated[:, pos : pos + 1], position_index=pos)
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
                print(f"[gen] step {_step_idx+1}/{max_new_tokens} (pos {generated.shape[1]-1})...")
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

        valid_len = int(attention_mask[0].sum().item()) if attention_mask is not None else int(input_ids.shape[1])
        self.reset_caches()
        with torch.no_grad():
            for pos in range(valid_len):
                pending_layers = [idx for idx in selected_layers if int(layer_rows[int(idx)]) < int(max_rows)]
                if not pending_layers:
                    break
                rows_done = sum(1 for idx in selected_layers if int(layer_rows[int(idx)]) >= int(max_rows))
                print(
                    f"[collect] token {pos+1}/{valid_len} "
                    f"({rows_done}/{len(selected_layers)} layers saturated, {len(pending_layers)} still collecting)",
                    flush=True,
                )
                _logits, captures = self.forward_token(
                    input_ids[:, pos : pos + 1],
                    position_index=pos,
                    capture_layers=pending_layers,
                    use_attention_cache=False,
                )
                for layer_idx in pending_layers:
                    capture = captures.get(int(layer_idx))
                    if capture is None:
                        continue
                    remain = int(max_rows) - int(layer_rows[int(layer_idx)])
                    if remain <= 0:
                        continue
                    x = capture["mlp_input"].reshape(-1, capture["mlp_input"].shape[-1]).float()
                    y = capture["dense_mlp_out"].reshape(-1, capture["dense_mlp_out"].shape[-1]).float()
                    take = min(int(remain), int(x.shape[0]))
                    if take <= 0:
                        continue
                    layer_x[int(layer_idx)].append(x[:take])
                    layer_y[int(layer_idx)].append(y[:take])
                    layer_rows[int(layer_idx)] += int(take)
