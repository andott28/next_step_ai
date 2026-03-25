#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

try:
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:  # pragma: no cover
    from streaming_llama_runtime import StreamingLlamaRuntime

try:
    from ..sca_sparse_config import SCASparseConfig
    from ..sca_sparse_mlp import SparseLlamaMLP
    from ..gqa_taylor_ssd import GQATaylorSSDSelfAttention
except ImportError:  # pragma: no cover
    from sca_sparse_config import SCASparseConfig
    from sca_sparse_mlp import SparseLlamaMLP
    from gqa_taylor_ssd import GQATaylorSSDSelfAttention


def _safe_console_text(text: str) -> str:
    return str(text).encode("cp1252", errors="replace").decode("cp1252")


def _parse_layer_selection(spec: str | None) -> Optional[List[int]]:
    if spec is None or str(spec).strip() == "":
        return None
    out: set[int] = set()
    for part in str(spec).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise ValueError(f"Invalid layer range: {token}")
            out.update(range(start, end + 1))
        else:
            out.add(int(token))
    return sorted(out)


def _noop_route_fn(hidden_states: torch.Tensor, layer_idx: int) -> Any:
    raise RuntimeError(f"route_fn should not be called for learned-basis streaming layer {layer_idx}")


class StreamingLearnedBasisRuntime(StreamingLlamaRuntime):
    def __init__(
        self,
        *,
        learned_basis_checkpoint: str,
        sca_basis_top_k: int = 0,
        sca_top_k: int = 0,
        verbose_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.verbose_progress = bool(verbose_progress)

        blob = torch.load(str(learned_basis_checkpoint), map_location="cpu")
        if not isinstance(blob, dict):
            raise RuntimeError("learned-basis checkpoint must be a dict")

        config_payload = blob.get("config", {})
        if not isinstance(config_payload, dict):
            config_payload = {}
        layer_states = blob.get("layer_states", {})
        if not isinstance(layer_states, dict) or not layer_states:
            raise RuntimeError("learned-basis checkpoint missing non-empty 'layer_states'")

        raw_selection = blob.get("layer_selection")
        if isinstance(raw_selection, list) and raw_selection:
            self.learned_basis_layers = [int(v) for v in raw_selection]
        else:
            self.learned_basis_layers = sorted(int(v) for v in layer_states.keys())
        self.learned_basis_layer_set = set(self.learned_basis_layers)

        basis_rank = int(config_payload.get("basis_rank", 32))
        basis_top_k = int(sca_basis_top_k) if int(sca_basis_top_k) > 0 else max(1, basis_rank // 8)
        block_top_k = int(sca_top_k) if int(sca_top_k) > 0 else 6
        block_size = int(config_payload.get("block_size", 32))

        basis_rank_by_layer = config_payload.get("basis_rank_by_layer", {})
        basis_top_k_by_layer = config_payload.get("basis_top_k_by_layer", {})
        top_k_by_layer = config_payload.get("top_k_by_layer", {})
        if not isinstance(basis_rank_by_layer, dict):
            basis_rank_by_layer = {}
        if not isinstance(basis_top_k_by_layer, dict):
            basis_top_k_by_layer = {}
        if not isinstance(top_k_by_layer, dict):
            top_k_by_layer = {}

        self.learned_basis_config = SCASparseConfig(
            hidden_size=int(self.config.hidden_size),
            block_size=block_size,
            basis_rank=basis_rank,
            basis_top_k=basis_top_k,
            top_k=block_top_k,
            adaptive_top_k=False,
            adaptive_top_k_min=block_top_k,
            adaptive_top_k_max=block_top_k,
            use_cuda=bool(torch.cuda.is_available()),
            spmm_impl="dense",
            sparse_placement="learned_basis",
            routing_mode="semantic_latent",
            basis_rank_by_layer={int(k): int(v) for k, v in basis_rank_by_layer.items()},
            basis_top_k_by_layer={int(k): int(v) for k, v in basis_top_k_by_layer.items()},
            top_k_by_layer={int(k): int(v) for k, v in top_k_by_layer.items()},
            stability_dense_fallback_threshold=0.0,
            soft_mask=True,
        )
        self.learned_basis_config.canonicalize_for_num_layers(self.num_layers)

        self._learned_basis_state_by_layer: Dict[int, Dict[str, torch.Tensor]] = {
            int(layer_idx): payload
            for layer_idx, payload in ((int(k), v) for k, v in layer_states.items())
            if isinstance(payload, dict)
        }
        self._sparse_mlp_wrapper = SparseLlamaMLP(
            base_mlp=self._layer_skeleton.mlp,
            config=self.learned_basis_config,
            layer_idx=0,
            route_fn=_noop_route_fn,
            enabled_fn=lambda _layer_idx: True,
        )
        # Keep one fixed sparse wrapper on the runtime device and reuse its
        # storage across layers. Replacing parameter storage every layer
        # causes avoidable CUDA allocator churn on 405B.
        self._sparse_mlp_wrapper.to(device=self.device, dtype=self.dtype)
        self._sparse_mlp_wrapper.eval()
        for param in self._sparse_mlp_wrapper.parameters():
            param.requires_grad = False
        self._loaded_sparse_basis_layer_idx: Optional[int] = None

    @staticmethod
    def _copy_payload_tensor_(
        dest: torch.Tensor,
        payload: Dict[str, Any],
        key: str,
        *,
        layer_idx: int,
    ) -> None:
        tensor = payload.get(key)
        if tensor is None:
            raise RuntimeError(f"learned-basis checkpoint missing '{key}' for layer {layer_idx}")
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"learned-basis checkpoint entry '{key}' for layer {layer_idx} must be a tensor")
        if tuple(tensor.shape) != tuple(dest.shape):
            raise RuntimeError(
                f"learned-basis payload shape mismatch for '{key}' at layer {layer_idx}: "
                f"expected {tuple(dest.shape)}, got {tuple(tensor.shape)}"
            )
        with torch.no_grad():
            dest.copy_(tensor.detach(), non_blocking=False)

    def _load_sparse_basis_layer(self, layer_idx: int) -> None:
        if self._loaded_sparse_basis_layer_idx == int(layer_idx):
            return
        payload = self._learned_basis_state_by_layer.get(int(layer_idx))
        if payload is None:
            raise RuntimeError(f"learned-basis checkpoint missing layer {layer_idx}")
        self._sparse_mlp_wrapper.layer_idx = int(layer_idx)
        self._copy_payload_tensor_(
            self._sparse_mlp_wrapper.sparse_basis_encoder.weight,
            payload,
            "encoder_weight",
            layer_idx=int(layer_idx),
        )
        self._copy_payload_tensor_(
            self._sparse_mlp_wrapper.sparse_basis_encoder.bias,
            payload,
            "encoder_bias",
            layer_idx=int(layer_idx),
        )
        self._copy_payload_tensor_(
            self._sparse_mlp_wrapper.sparse_basis_decoder,
            payload,
            "decoder_blocks",
            layer_idx=int(layer_idx),
        )
        self._copy_payload_tensor_(
            self._sparse_mlp_wrapper.sparse_basis_bias,
            payload,
            "decoder_bias",
            layer_idx=int(layer_idx),
        )
        scale = payload.get("scale")
        with torch.no_grad():
            if scale is None:
                self._sparse_mlp_wrapper.sparse_basis_scale.fill_(1.0)
            else:
                if not torch.is_tensor(scale):
                    raise RuntimeError(f"learned-basis checkpoint entry 'scale' for layer {layer_idx} must be a tensor")
                if tuple(scale.shape) != tuple(self._sparse_mlp_wrapper.sparse_basis_scale.shape):
                    raise RuntimeError(
                        f"learned-basis payload shape mismatch for 'scale' at layer {layer_idx}: "
                        f"expected {tuple(self._sparse_mlp_wrapper.sparse_basis_scale.shape)}, got {tuple(scale.shape)}"
                    )
                self._sparse_mlp_wrapper.sparse_basis_scale.copy_(scale.detach(), non_blocking=False)
            self._sparse_mlp_wrapper.sparse_output_compensation_bias.zero_()
        self._loaded_sparse_basis_layer_idx = int(layer_idx)

    def forward_token(
        self,
        token_ids: torch.LongTensor,
        *,
        position_index: int,
        capture_layers: Optional[List[int]] = None,
        use_attention_cache: bool = True,
    ) -> tuple[torch.Tensor, Dict[int, Dict[str, torch.Tensor]]]:
        if token_ids.ndim != 2 or int(token_ids.shape[0]) != 1 or int(token_ids.shape[1]) != 1:
            raise RuntimeError("StreamingLearnedBasisRuntime.forward_token currently supports batch_size=1 and seq_len=1 only")

        capture_set = {int(idx) for idx in capture_layers or []}
        captures: Dict[int, Dict[str, torch.Tensor]] = {}
        printed_progress = False
        hidden = self.embed_tokens(token_ids.cpu()).to(device=self.device, dtype=self.dtype)
        position_ids = torch.tensor([[int(position_index)]], device=self.device, dtype=torch.long)
        rope = self.rotary_emb(hidden, position_ids)

        for layer_idx in range(self.num_layers):
            if self.verbose_progress and (int(layer_idx) % 8 == 0 or int(layer_idx) == self.num_layers - 1):
                printed_progress = True
                if torch.cuda.is_available():
                    alloc_gb = torch.cuda.memory_allocated() / 1e9
                    reserv_gb = torch.cuda.memory_reserved() / 1e9
                    print(
                        f"  [layer {layer_idx}/{self.num_layers}] VRAM {alloc_gb:.2f}/{reserv_gb:.2f} GB (alloc/reserv)",
                        end="\r",
                        flush=True,
                    )
                else:
                    print(f"  [layer {layer_idx}/{self.num_layers}] loading...", end="\r", flush=True)

            layer = self._load_layer(layer_idx)
            if torch.cuda.is_available() and int(layer_idx) % 32 == 31:
                torch.cuda.empty_cache()

            taylor_attn = None
            residual = hidden
            hidden_norm = layer.input_layernorm(hidden)

            if layer_idx in self.taylor_layer_set and use_attention_cache:
                taylor_attn = self._build_taylor_attention(layer, layer_idx)
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
                    past_key_values=self._dense_cache if use_attention_cache else None,
                    cache_position=position_ids.view(-1),
                )

            hidden = residual + attn_out
            residual = hidden
            mlp_input = layer.post_attention_layernorm(hidden)
            if layer_idx in self.learned_basis_layer_set:
                self._load_sparse_basis_layer(layer_idx)
                mlp_out = self._sparse_mlp_wrapper(mlp_input)
            else:
                mlp_out = layer.mlp(mlp_input)
            if layer_idx in capture_set:
                captures[layer_idx] = {
                    "mlp_input": mlp_input.detach().cpu(),
                    "dense_mlp_out": mlp_out.detach().cpu(),
                }
            hidden = residual + mlp_out
            self._release_modules(layer, *( [taylor_attn] if taylor_attn is not None else [] ))

        if printed_progress and self.verbose_progress:
            print("", flush=True)
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden.cpu().to(dtype=self.lm_head.weight.dtype)).float()
        return logits, captures

    def _build_taylor_attention(self, layer: Any, layer_idx: int) -> Any:
        return GQATaylorSSDSelfAttention.from_llama_attention(
            source_attn=layer.self_attn,
            layer_idx=layer_idx,
            feature_map=self.taylor_feature_map,
            local_window=self.taylor_local_window,
            feature_dim=self.taylor_feature_dim,
            state_decay=self.taylor_state_decay,
        ).to(device=self.device).eval()

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
            raise RuntimeError("StreamingLearnedBasisRuntime.generate currently supports batch_size=1 only")

        generated = input_ids.to(device=self.device)
        self.reset_caches()
        logits: Optional[torch.Tensor] = None
        with torch.no_grad():
            for pos in range(int(generated.shape[1])):
                if self.verbose_progress:
                    print(f"[prompt] processing token {pos+1}/{generated.shape[1]}...")
                logits, _ = self.forward_token(generated[:, pos : pos + 1], position_index=pos)
            if logits is None:
                raise RuntimeError("No prompt tokens were provided")

            for step_idx in range(int(max_new_tokens)):
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
                if self.verbose_progress:
                    print(f"[gen] step {step_idx+1}/{max_new_tokens} (pos {generated.shape[1]-1})...")
                logits, _ = self.forward_token(next_token, position_index=int(generated.shape[1]) - 1)
        return generated


def _render_chat_input(tokenizer: Any, messages: List[Dict[str, str]]) -> torch.LongTensor:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if torch.is_tensor(rendered):
                return rendered
        except Exception:
            pass
    prompt = ""
    for message in messages:
        prompt += f"{message['role']}: {message['content']}\n"
    prompt += "assistant: "
    return tokenizer(prompt, return_tensors="pt")["input_ids"]


def _generate_reply(
    runtime: StreamingLearnedBasisRuntime,
    tokenizer: Any,
    input_ids: torch.LongTensor,
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with torch.no_grad():
        generated = runtime.generate(
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tokenizer.eos_token_id,
            do_sample=bool(do_sample),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
        )
    elapsed = time.perf_counter() - t0
    prompt_len = int(input_ids.shape[-1])
    reply_ids = generated[:, prompt_len:]
    reply_text = tokenizer.decode(reply_ids[0], skip_special_tokens=True).strip()
    new_tokens = max(int(reply_ids.shape[-1]), 1)
    return {
        "latency_s": float(elapsed),
        "tok_s": float(new_tokens / max(elapsed, 1e-9)),
        "reply_text": reply_text,
        "reply_ids": reply_ids,
        "generated_ids": generated,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chat with the 405B streaming runtime using a learned-basis sparse MLP checkpoint.")
    p.add_argument("--model-name", type=str, required=True, help="HF repo id or local snapshot directory")
    p.add_argument("--learned-basis-checkpoint", type=str, required=True, help="Path to learned_basis_init_*.pt")
    p.add_argument("--prompt", action="append", default=[], help="Prompt to run; may be repeated")
    p.add_argument("--system", type=str, default="", help="Optional system message for chat mode")
    p.add_argument("--interactive", action="store_true", help="Start a simple stdin/stdout chat loop")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--taylor-layers", type=str, default="", help="Layer selection, for example '0-125'")
    p.add_argument(
        "--taylor-feature-map",
        type=str,
        default="hybrid_performer",
        choices=["hybrid_performer", "elu", "taylor"],
    )
    p.add_argument("--taylor-local-window", type=int, default=64)
    p.add_argument("--taylor-feature-dim", type=int, default=64)
    p.add_argument("--taylor-state-decay", type=float, default=1.0)
    p.add_argument("--sca-basis-top-k", type=int, default=0, help="0 = auto (basis_rank // 8)")
    p.add_argument("--sca-top-k", type=int, default=0, help="0 = auto (6)")
    p.add_argument("--verbose-progress", action="store_true", help="Print token/layer progress while streaming")
    p.add_argument("--dump-json", type=str, default="")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    taylor_layers = _parse_layer_selection(args.taylor_layers)

    runtime = StreamingLearnedBasisRuntime(
        model_name_or_path=str(args.model_name),
        learned_basis_checkpoint=str(args.learned_basis_checkpoint),
        sca_basis_top_k=int(args.sca_basis_top_k),
        sca_top_k=int(args.sca_top_k),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=torch.float16,
        taylor_layers=taylor_layers,
        taylor_feature_map=str(args.taylor_feature_map),
        taylor_local_window=int(args.taylor_local_window),
        taylor_feature_dim=int(args.taylor_feature_dim),
        taylor_state_decay=float(args.taylor_state_decay),
        local_files_only=bool(args.local_files_only),
        verbose_progress=bool(args.verbose_progress),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name),
        use_fast=True,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(
        "learned_basis_runtime="
        + json.dumps(
            {
                "layers_initialized": len(runtime.learned_basis_layers),
                "basis_rank": int(runtime.learned_basis_config.basis_rank),
                "basis_top_k": int(runtime.learned_basis_config.basis_top_k),
                "top_k": int(runtime.learned_basis_config.top_k),
            }
        )
    )

    rows: List[Dict[str, Any]] = []
    if args.interactive:
        messages: List[Dict[str, str]] = []
        if str(args.system).strip():
            messages.append({"role": "system", "content": str(args.system).strip()})
        print("Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_text = input("user> ").strip()
            except EOFError:
                break
            if not user_text:
                continue
            if user_text.lower() in {"exit", "quit"}:
                break
            messages.append({"role": "user", "content": user_text})
            input_ids = _render_chat_input(tokenizer, messages)
            try:
                result = _generate_reply(
                    runtime,
                    tokenizer,
                    input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                )
            except Exception:
                print("", flush=True)
                raise
            reply_text = str(result["reply_text"])
            print(f"assistant> {_safe_console_text(reply_text)}")
            print(f"[latency] {result['latency_s']:.3f}s ({result['tok_s']:.3f} tok/s)")
            messages.append({"role": "assistant", "content": reply_text})
            rows.append(
                {
                    "prompt": user_text,
                    "reply_text": reply_text,
                    "latency_s": float(result["latency_s"]),
                    "tok_s": float(result["tok_s"]),
                }
            )
    else:
        prompts = list(args.prompt) if args.prompt else ["Write one sentence about Norway."]
        for idx, prompt in enumerate(prompts):
            messages = []
            if str(args.system).strip():
                messages.append({"role": "system", "content": str(args.system).strip()})
            messages.append({"role": "user", "content": prompt})
            input_ids = _render_chat_input(tokenizer, messages)
            try:
                result = _generate_reply(
                    runtime,
                    tokenizer,
                    input_ids,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=bool(args.do_sample),
                    temperature=float(args.temperature),
                    top_k=int(args.top_k),
                    top_p=float(args.top_p),
                )
            except Exception:
                print("", flush=True)
                raise
            row = {
                "prompt_idx": int(idx),
                "prompt": prompt,
                "reply_text": str(result["reply_text"]),
                "latency_s": float(result["latency_s"]),
                "tok_s": float(result["tok_s"]),
            }
            rows.append(row)
            print(f"[prompt {idx}] latency_s={row['latency_s']:.3f} tok_s={row['tok_s']:.3f}")
            print(f"[prompt {idx}] text={_safe_console_text(row['reply_text'])}")

    if args.dump_json:
        out_path = Path(args.dump_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": str(args.model_name),
            "learned_basis_checkpoint": str(args.learned_basis_checkpoint),
            "rows": rows,
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote diagnostics: {out_path}")


if __name__ == "__main__":
    main()
