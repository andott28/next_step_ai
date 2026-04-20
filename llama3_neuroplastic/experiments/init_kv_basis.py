"""init_kv_basis.py
Calibration script to fit a learned routing basis for K/V column-block sparse
inference.  Mirrors the structure of init_learned_basis_from_dense_mlp.py but
targets the K and V projections rather than the MLP gate/up/down projections.

The output checkpoint contains per-layer routing tensors that the streaming
runtime uses to predict (before PCIe transfer) which ~10 % of W_K / W_V column
blocks are active for each decode token.

Usage example:
    python -m llama3_neuroplastic.experiments.init_kv_basis \\
        --model-path /models/llama405b \\
        --output-path results/kv_basis_r32.pt \\
        --basis-rank 32 \\
        --block-size 32 \\
        --top-k 51 \\
        --max-rows 8192 \\
        --layers all \\
        --device cuda
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from llama3_neuroplastic.layer_selection import parse_layer_selection

try:
    from ..basis_fitting import fit_layer_basis
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    try:
        from llama3_neuroplastic.basis_fitting import fit_layer_basis
        from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime
    except ImportError:
        from basis_fitting import fit_layer_basis
        from streaming_llama_runtime import StreamingLlamaRuntime






def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit K/V column-block routing basis for sparse streaming inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", type=str, required=True,
                   help="Path to a local snapshot directory or HuggingFace repo id.")
    p.add_argument("--output-path", type=str, required=True,
                   help="Where to write the output checkpoint (.pt).")
    p.add_argument("--basis-rank", type=int, default=32,
                   help="Rank of the learned routing encoder.")
    p.add_argument("--block-size", type=int, default=32,
                   help="Column-block size (number of input features per block).")
    p.add_argument("--top-k", type=int, default=51,
                   help="Number of column-blocks to activate per token.")
    p.add_argument("--max-rows", type=int, default=8192,
                   help="Maximum number of (x, k_out, v_out) rows to collect per layer.")
    p.add_argument("--layers", type=str, default="all",
                   help='Comma-separated layer indices or "all".')
    p.add_argument("--device", type=str, default="",
                   help="Torch device string (e.g. cuda, cpu). Auto-detected if empty.")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--calibration-text", type=str, default="",
                   help="Optional path to a .txt file used as calibration text.  "
                        "If empty, a built-in wikitext-style excerpt is used.")
    p.add_argument("--pca-method", type=str, default="auto",
                   choices=["auto", "lowrank", "incremental"])
    p.add_argument("--pca-batch-rows", type=int, default=1024)
    return p.parse_args()






def _parse_layer_selection(spec: str, total_layers: int) -> list[int]:
    return parse_layer_selection(spec, total_layers=int(total_layers)) or []






_FALLBACK_CALIBRATION_TEXT = (
    "The transformer architecture has become the dominant paradigm for large "
    "language models. Each decoder layer applies multi-head self-attention "
    "followed by a feed-forward network. The key and value projections map the "
    "post-layernorm hidden state to a lower-dimensional space used by the "
    "attention mechanism. Sparse approximations to these projections can "
    "dramatically reduce the PCIe bandwidth required for token generation on "
    "consumer GPU hardware. By routing only the most informative column blocks "
    "of W_K and W_V to the device, we preserve generation quality while "
    "achieving a five-to-ten times reduction in attention weight traffic. "
    "This calibration script measures the actual activation patterns of K and V "
    "to build the routing tables needed at inference time. "
)


def _load_calibration_texts(path: str) -> list[str]:
    if path.strip():
        text = Path(path).read_text(encoding="utf-8")

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [text]
    return [_FALLBACK_CALIBRATION_TEXT]






def _collect_kv_rows(
    runtime: StreamingLlamaRuntime,
    texts: list[str],
    selected_layers: list[int],
    *,
    max_rows: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Collect (x_post_layernorm, k_out, v_out) triples for each selected layer.

    We intercept:
      - layer.input_layernorm output  → x  [N, hidden_size]
      - layer.self_attn.k_proj output → k_out [N, kv_hidden]
      - layer.self_attn.v_proj output → v_out [N, kv_hidden]

    Uses forward hooks on the skeleton that is loaded layer-by-layer.
    The streaming runtime reuses a single LlamaDecoderLayer skeleton, so we
    register hooks, run the forward, then immediately remove them.
    """



    device = runtime.device
    dtype = runtime.dtype


    buffers: dict[int, dict[str, list[torch.Tensor]]] = {
        li: {"x": [], "k": [], "v": []} for li in selected_layers
    }
    rows_collected: dict[int, int] = {li: 0 for li in selected_layers}

    selected_set = set(selected_layers)

    for text in texts:




        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                str(runtime.snapshot_dir),
                local_files_only=True,
                use_fast=True,
            )
            token_ids = tok(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=False,
            )["input_ids"]
        except Exception:

            vocab_size = int(runtime._embed_weight_cpu.shape[0])
            raw = [ord(c) % vocab_size for c in text[:256]]
            token_ids = torch.tensor([raw], dtype=torch.long)

        if int(token_ids.shape[1]) < 2:
            continue

        token_ids = token_ids.to(device=torch.device("cpu"))



        with torch.no_grad():
            hidden = runtime._embed_tokens_cpu(token_ids).to(device=device, dtype=dtype)
            position_ids = torch.arange(int(token_ids.shape[1]), device=device, dtype=torch.long)

            for layer_idx in range(runtime.num_layers):

                pending = [li for li in selected_layers if rows_collected[li] < max_rows]
                if not pending:
                    break

                layer = runtime._load_layer(layer_idx)

                h_norm_captures: list[torch.Tensor] = []
                k_out_captures: list[torch.Tensor] = []
                v_out_captures: list[torch.Tensor] = []

                hooks: list[Any] = []

                if layer_idx in selected_set and rows_collected[layer_idx] < max_rows:
                    def _hook_ln(module: nn.Module, inp: Any, out: torch.Tensor,
                                 _buf: list = h_norm_captures) -> None:
                        _buf.append(out.detach().cpu())

                    def _hook_k(module: nn.Module, inp: Any, out: torch.Tensor,
                                _buf: list = k_out_captures) -> None:
                        _buf.append(out.detach().cpu())

                    def _hook_v(module: nn.Module, inp: Any, out: torch.Tensor,
                                _buf: list = v_out_captures) -> None:
                        _buf.append(out.detach().cpu())

                    hooks.append(layer.input_layernorm.register_forward_hook(_hook_ln))
                    if hasattr(layer.self_attn, "k_proj"):
                        hooks.append(layer.self_attn.k_proj.register_forward_hook(_hook_k))
                    if hasattr(layer.self_attn, "v_proj"):
                        hooks.append(layer.self_attn.v_proj.register_forward_hook(_hook_v))


                seq_len = int(hidden.shape[1])
                next_hidden = torch.empty_like(hidden)
                for pos in range(seq_len):
                    h = hidden[:, pos : pos + 1, :]
                    pos_ids = position_ids[pos : pos + 1].unsqueeze(0)
                    rope_tok = runtime.rotary_emb(h, pos_ids)
                    h_norm_tok = layer.input_layernorm(h)
                    try:
                        attn_out, _ = layer.self_attn(
                            hidden_states=h_norm_tok,
                            position_embeddings=rope_tok,
                            attention_mask=None,
                            past_key_values=None,
                            cache_position=pos_ids.view(-1),
                        )
                    except TypeError:
                        attn_out, _ = layer.self_attn(
                            hidden_states=h_norm_tok,
                            position_embeddings=rope_tok,
                            attention_mask=None,
                        )
                    next_hidden[:, pos : pos + 1, :] = h + attn_out

                hidden, next_hidden = next_hidden, hidden


                residual = hidden
                mlp_input = layer.post_attention_layernorm(hidden)
                try:
                    mlp_out = runtime._dense_mlp_forward_streaming_fast(layer_idx, layer.mlp, mlp_input)
                except Exception:
                    mlp_out = torch.zeros_like(mlp_input)
                hidden = residual + mlp_out


                for h_handle in hooks:
                    h_handle.remove()
                hooks.clear()


                if layer_idx in selected_set and h_norm_captures and k_out_captures and v_out_captures:

                    x_cat = torch.cat(h_norm_captures, dim=1).reshape(-1, h_norm_captures[0].shape[-1]).float()
                    k_cat = torch.cat(k_out_captures, dim=1).reshape(-1, k_out_captures[0].shape[-1]).float()
                    v_cat = torch.cat(v_out_captures, dim=1).reshape(-1, v_out_captures[0].shape[-1]).float()

                    remain = max_rows - rows_collected[layer_idx]
                    take = int(min(remain, int(x_cat.shape[0])))
                    if take > 0:
                        buffers[layer_idx]["x"].append(x_cat[:take])
                        buffers[layer_idx]["k"].append(k_cat[:take])
                        buffers[layer_idx]["v"].append(v_cat[:take])
                        rows_collected[layer_idx] += take


        if all(rows_collected[li] >= max_rows for li in selected_layers):
            break


    result: dict[int, dict[str, torch.Tensor]] = {}
    for li in selected_layers:
        buf = buffers[li]
        if buf["x"] and buf["k"] and buf["v"]:
            result[li] = {
                "x": torch.cat(buf["x"], dim=0),
                "k": torch.cat(buf["k"], dim=0),
                "v": torch.cat(buf["v"], dim=0),
            }
    return result






def _fit_kv_basis(
    x: torch.Tensor,
    kv_out: torch.Tensor,
    *,
    basis_rank: int,
    block_size: int,
    hidden_size: int,
    pca_method: str = "auto",
    pca_batch_rows: int = 1024,
) -> dict[str, Any]:
    """Fit routing basis for one K or V projection.

    Target: column-block norms of x (the input to W_K or W_V).
    We want to predict which blocks of x will produce large K/V outputs,
    so we regress the block-norm of x against x itself.

    Args:
        x:       [N, hidden_size]  post-input-layernorm activations
        kv_out:  [N, kv_hidden]    k_proj or v_proj output (unused in routing;
                                   kept for future quality diagnostics)
        basis_rank:  encoder rank
        block_size:  column block size (must divide hidden_size)
        hidden_size: input dimension to W_K / W_V

    Returns dict matching the checkpoint schema for one layer projection.
    """
    num_col_blocks = hidden_size // block_size
    N = int(x.shape[0])

    x_f = x.float()


    x_blocks = x_f.reshape(N, num_col_blocks, block_size)
    block_norms = x_blocks.norm(dim=-1)


    block_importance = block_norms.mean(dim=0)



    layer_result = fit_layer_basis(
        x=x_f,
        y=block_norms,
        basis_rank=int(basis_rank),
        block_size=1,
        pca_method=pca_method,
        pca_batch_rows=int(pca_batch_rows),
    )



    enc_w = layer_result["encoder_weight"]
    enc_b = layer_result["encoder_bias"]

    dec_raw = layer_result["decoder_blocks"]
    dec_flat = dec_raw.squeeze(-1)
    dec_norm_t = dec_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    dec_flat / dec_norm_t

    decoder_blocks = dec_flat.unsqueeze(-1).contiguous()
    decoder_bias = layer_result["decoder_bias"]

    return {
        "encoder_weight": enc_w.contiguous(),
        "encoder_bias": enc_b.contiguous(),
        "decoder_blocks": decoder_blocks.contiguous(),
        "decoder_bias": decoder_bias.contiguous(),
        "block_importance": block_importance.contiguous(),
        "explained_variance_ratio": float(layer_result.get("explained_variance_ratio", 0.0)),
    }






def main() -> None:
    args = _parse_args()


    if args.device.strip():
        device = torch.device(args.device.strip())
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[kv_basis] device={device}", flush=True)
    print(f"[kv_basis] model={args.model_path}", flush=True)
    print(f"[kv_basis] output={args.output_path}", flush=True)
    print(f"[kv_basis] basis_rank={args.basis_rank}  block_size={args.block_size}  top_k={args.top_k}", flush=True)


    runtime = StreamingLlamaRuntime(
        model_name_or_path=str(args.model_path),
        device=device,
        dtype=torch.float16,
        taylor_layers=[],
        local_files_only=bool(args.local_files_only),
        ram_cache=False,
        materialize_lm_head=False,
    )

    total_layers = int(runtime.num_layers)
    selected_layers = _parse_layer_selection(args.layers, total_layers)
    print(f"[kv_basis] selected {len(selected_layers)} layers: {selected_layers}", flush=True)

    hidden_size = int(getattr(runtime.config, "hidden_size", 16384))
    kv_hidden = int(getattr(runtime.config, "num_key_value_heads",
                             getattr(runtime.config, "num_attention_heads", 128)
                    ) * getattr(runtime.config, "head_dim", 128))
    num_col_blocks = hidden_size // int(args.block_size)

    print(f"[kv_basis] hidden_size={hidden_size}  kv_hidden={kv_hidden}  "
          f"num_col_blocks={num_col_blocks}", flush=True)


    texts = _load_calibration_texts(str(args.calibration_text))
    print(f"[kv_basis] {len(texts)} calibration document(s)", flush=True)


    print("[kv_basis] collecting K/V activations...", flush=True)
    collected = _collect_kv_rows(
        runtime,
        texts,
        selected_layers,
        max_rows=int(args.max_rows),
    )
    print(f"[kv_basis] collected data for {len(collected)} layer(s)", flush=True)


    layer_states: dict[str, Any] = {}
    for layer_idx in selected_layers:
        data = collected.get(layer_idx)
        if data is None:
            print(f"[kv_basis] layer {layer_idx}: no data collected — skipping", flush=True)
            continue
        x = data["x"]
        k = data["k"]
        data["v"]
        N = int(x.shape[0])
        print(f"[kv_basis] layer {layer_idx}: fitting basis on {N} rows...", flush=True)

        k_state = _fit_kv_basis(
            x, k,
            basis_rank=int(args.basis_rank),
            block_size=int(args.block_size),
            hidden_size=hidden_size,
            pca_method=str(args.pca_method),
            pca_batch_rows=int(args.pca_batch_rows),
        )

        layer_states[str(layer_idx)] = {
            "encoder_weight":    k_state["encoder_weight"],
            "encoder_bias":      k_state["encoder_bias"],
            "decoder_blocks":    k_state["decoder_blocks"],
            "decoder_bias":      k_state["decoder_bias"],
            "block_importance":  k_state["block_importance"],
        }
        print(
            f"[kv_basis] layer {layer_idx}: done — "
            f"explained_var={k_state['explained_variance_ratio']:.4f}  "
            f"enc_w={list(k_state['encoder_weight'].shape)}",
            flush=True,
        )


    config_payload = {
        "hidden_size": hidden_size,
        "kv_hidden": kv_hidden,
        "block_size": int(args.block_size),
        "num_col_blocks": num_col_blocks,
        "basis_rank": int(args.basis_rank),
        "top_k": int(args.top_k),
        "separate_k_v_routing": False,
    }
    checkpoint = {
        "config": config_payload,
        "layer_states": layer_states,
    }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(f"[kv_basis] saved checkpoint to {out_path}  ({len(layer_states)} layers)", flush=True)


if __name__ == "__main__":
    main()
