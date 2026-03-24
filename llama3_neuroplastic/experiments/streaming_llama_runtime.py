from __future__ import annotations

import gc
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import bitsandbytes.functional as bnb_functional
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
    def __init__(self, snapshot_dir: Path) -> None:
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
        self._shard_handles: Dict[str, Any] = {}

    def _load_exact_tensors(self, names: Sequence[str]) -> Dict[str, torch.Tensor]:
        requested = [str(name) for name in names if str(name) in self._available_names]
        by_shard: Dict[str, List[str]] = defaultdict(list)
        for name in requested:
            by_shard[self.weight_map[name]].append(name)

        out: Dict[str, torch.Tensor] = {}
        for shard_name, shard_keys in by_shard.items():
            if shard_name not in self._shard_handles:
                shard_path = self.snapshot_dir / shard_name
                self._shard_handles[shard_name] = safe_open(str(shard_path), framework="pt", device="cpu")
            handle = self._shard_handles[shard_name]
            for key in shard_keys:
                out[key] = handle.get_tensor(key)
        return out

    def load_parameter(self, name: str) -> torch.Tensor:
        full_name = str(name)
        tensors = self._load_exact_tensors([full_name, *self._quant_aux_by_base.get(full_name, [])])
        if full_name not in tensors:
            raise KeyError(f"Tensor '{full_name}' not found in safetensors index")

        weight = tensors[full_name]
        quant_aux = {key: value for key, value in tensors.items() if key != full_name}
        if not quant_aux:
            return weight

        quant_state = bnb_functional.QuantState.from_dict(quant_aux, device=torch.device("cpu"))
        return bnb_functional.dequantize_4bit(weight, quant_state=quant_state)

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
    ) -> None:
        self.snapshot_dir = _resolve_snapshot_dir(model_name_or_path, local_files_only=bool(local_files_only))
        self.config = AutoConfig.from_pretrained(str(self.snapshot_dir), local_files_only=bool(local_files_only))
        if str(getattr(self.config, "model_type", "")) != "llama":
            raise RuntimeError(f"Streaming runtime only supports llama models, got {self.config.model_type!r}")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.loader = ShardedSafetensorLoader(self.snapshot_dir)
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

        self.embed_tokens = nn.Embedding(
            int(self.config.vocab_size),
            int(self.config.hidden_size),
            padding_idx=getattr(self.config, "pad_token_id", None),
        ).to(device=self.device, dtype=self.dtype)
        self.embed_tokens.weight.data.copy_(
            self.loader.load_parameter("model.embed_tokens.weight").to(device=self.device, dtype=self.dtype)
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

        self.lm_head = nn.Linear(int(self.config.hidden_size), int(self.config.vocab_size), bias=False)
        lm_head_weight_name = (
            "lm_head.weight"
            if "lm_head.weight" in self.loader.weight_map
            else "model.embed_tokens.weight"
        )
        if lm_head_weight_name == "model.embed_tokens.weight":
            # Tied embeddings: share the already-resident embed_tokens weight, no extra VRAM
            self.lm_head.weight = self.embed_tokens.weight
        else:
            self.lm_head.to(device=self.device, dtype=self.dtype)
            self.lm_head.weight.data.copy_(
                self.loader.load_parameter(lm_head_weight_name).to(device=self.device, dtype=self.dtype)
            )
        self.lm_head.requires_grad_(False)

        self.rotary_emb = LlamaRotaryEmbedding(self.config, device=self.device)
        self._taylor_caches: List[Optional[TaylorSSDLayerCache]] = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None
        self._layer_skeleton = LlamaDecoderLayer(self.config, layer_idx=0).to(device=self.device, dtype=self.dtype)
        for p in self._layer_skeleton.parameters():
            p.requires_grad = False
        self._layer_skeleton.eval()

    def reset_caches(self) -> None:
        self._taylor_caches = [None for _ in range(self.num_layers)]
        self._dense_cache = DynamicCache(config=self.config) if DynamicCache is not None else None

    def _load_layer(self, layer_idx: int) -> LlamaDecoderLayer:
        self._layer_skeleton.layer_idx = int(layer_idx)
        if hasattr(self._layer_skeleton, "self_attn"):
             self._layer_skeleton.self_attn.layer_idx = int(layer_idx)
        
        expected_keys = list(self._layer_skeleton.state_dict().keys())
        state = self.loader.load_module_state(
            prefix=f"model.layers.{int(layer_idx)}.",
            expected_keys=expected_keys,
            dtype=self.dtype,
        )
        # Load onto GPU directly where possible or move after
        state_gpu = {k: v.to(device=self.device, non_blocking=True) for k, v in state.items()}
        self._layer_skeleton.load_state_dict(state_gpu, strict=True)
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
    ) -> tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1 or int(token_ids.shape[1]) != 1:
            raise RuntimeError("StreamingLlamaRuntime.forward_token currently supports batch_size=1 and seq_len=1 only")

        capture_set = {int(idx) for idx in capture_layers or []}
        captures: Dict[int, Dict[str, torch.Tensor]] = {}
        hidden = self.embed_tokens(token_ids.to(device=self.device))
        position_ids = torch.tensor([[int(position_index)]], device=self.device, dtype=torch.long)
        rope = self.rotary_emb(hidden, position_ids)

        for layer_idx in range(self.num_layers):
            if int(layer_idx) % 8 == 0 or int(layer_idx) == self.num_layers - 1:
                print(f"  [layer {layer_idx}/{self.num_layers}] loading...", end="\r", flush=True)
            layer = self._load_layer(layer_idx)
            taylor_attn: Optional[GQATaylorSSDSelfAttention] = None
            residual = hidden
            hidden_norm = layer.input_layernorm(hidden)

            if layer_idx in self.taylor_layer_set:
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
                attn_out, _attn_weights = layer.self_attn(
                    hidden_states=hidden_norm,
                    position_embeddings=rope,
                    attention_mask=None,
                    past_key_values=self._dense_cache,
                    cache_position=position_ids.view(-1),
                )

            hidden = residual + attn_out
            residual = hidden
            mlp_input = layer.post_attention_layernorm(hidden)
            mlp_out = layer.mlp(mlp_input)
            if layer_idx in capture_set:
                captures[layer_idx] = {
                    "mlp_input": mlp_input.detach().cpu(),
                    "dense_mlp_out": mlp_out.detach().cpu(),
                }
            hidden = residual + mlp_out
            self._release_modules(layer, *( [taylor_attn] if taylor_attn is not None else [] ))

        hidden = self.norm(hidden)
        logits = self.lm_head(hidden.to(dtype=self.lm_head.weight.dtype)).float()
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
                _logits, captures = self.forward_token(
                    input_ids[:, pos : pos + 1],
                    position_index=pos,
                    capture_layers=pending_layers,
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
