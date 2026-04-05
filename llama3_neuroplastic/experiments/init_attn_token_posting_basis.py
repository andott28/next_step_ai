from __future__ import annotations

"""init_attn_token_posting_basis.py

Offline calibration script: fit a per-(layer, KV-group) PCA basis over
post-RoPE cached keys and compute IDF weights per latent coordinate.

The output is loaded at inference time by the ``TokenPostingArchive`` to
route sparse archive queries via latent posting lists.

Design follows the same no-retrain, no-SGD philosophy as ``basis_fitting.py``
and ``init_kv_basis.py``:

  1. Run calibration texts through the model with a dense DynamicCache.
  2. After each forward pass, read the exact post-RoPE keys from the cache.
  3. For each retrieval layer and each KV group, run rank-R PCA on those keys.
  4. Compute a per-coordinate IDF weight from calibration statistics.
  5. Save the result as ``results/attn_token_posting_basis.pt``.

Usage
-----
    python -m llama3_neuroplastic.experiments.init_attn_token_posting_basis \\
        --model-path /models/llama405b \\
        --output-path results/attn_token_posting_basis.pt \\
        --basis-rank 32 \\
        --retrieval-start-layer 86 \\
        --max-rows 8192 \\
        --device cuda
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Imports with fallback paths
# ---------------------------------------------------------------------------

try:
    from .streaming_llama_runtime import StreamingLlamaRuntime
except ImportError:
    try:
        from llama3_neuroplastic.experiments.streaming_llama_runtime import StreamingLlamaRuntime
    except ImportError:
        from streaming_llama_runtime import StreamingLlamaRuntime  # type: ignore

try:
    from transformers.cache_utils import DynamicCache
except Exception:
    DynamicCache = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit PCA posting basis for token-posting sparse attention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-path", type=str, required=True,
                   help="Local model snapshot directory or HuggingFace repo id.")
    p.add_argument("--output-path", type=str, default="results/attn_token_posting_basis.pt")
    p.add_argument("--basis-rank", type=int, default=32,
                   help="PCA rank R per (layer, KV group).  32 or 64 recommended.")
    p.add_argument("--retrieval-start-layer", type=int, default=86,
                   help="First layer index for which the archive is active. "
                        "Earlier layers use standard dense KV cache.")
    p.add_argument("--retrieval-layers", type=str, default="",
                   help="Explicit comma-separated layer list.  "
                        "Overrides --retrieval-start-layer when provided.")
    p.add_argument("--max-rows", type=int, default=8192,
                   help="Maximum post-RoPE key rows collected per (layer, group).")
    p.add_argument("--idf-threshold", type=float, default=0.1,
                   help="Min |alpha| to count a token as active in coordinate r "
                        "(used for IDF document-frequency computation).")
    p.add_argument("--max-seq-len", type=int, default=512,
                   help="Truncate calibration sequences to this many tokens.")
    p.add_argument("--device", type=str, default="",
                   help="Torch device string (e.g. cuda, cpu).  Auto-detected if empty.")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--calibration-text", type=str, default="",
                   help="Optional path to a .txt file with calibration text.  "
                        "Paragraphs split on double-newline are used as separate sequences.")
    p.add_argument("--pca-method", type=str, default="auto",
                   choices=["auto", "lowrank", "incremental"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Layer selection
# ---------------------------------------------------------------------------

def _parse_retrieval_layers(
    explicit: str,
    retrieval_start_layer: int,
    total_layers: int,
) -> List[int]:
    if explicit.strip():
        out: set = set()
        for part in explicit.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                s, e = token.split("-", 1)
                out.update(range(int(s), int(e) + 1))
            else:
                out.add(int(token))
        return sorted(v for v in out if 0 <= v < total_layers)
    return list(range(int(retrieval_start_layer), int(total_layers)))


# ---------------------------------------------------------------------------
# Calibration text
# ---------------------------------------------------------------------------

_FALLBACK_CALIBRATION_TEXT = (
    "The transformer architecture processes sequences of tokens through "
    "successive layers of self-attention and feed-forward networks.  In the "
    "multi-head attention mechanism, each query attends to every cached key "
    "in the context.  For long contexts the key-value cache can contain tens "
    "of thousands of tokens, making dense full-context attention expensive.  "
    "Sparse retrieval methods can dramatically reduce this cost by identifying "
    "the small subset of past tokens that are most relevant to the current query.  "
    "A low-rank projection of the key space into a latent basis allows fast "
    "approximate nearest-neighbour search via inverted posting lists.  The "
    "exact (K, V) pairs for only the retrieved tokens are then transferred "
    "to the GPU for a small exact attention computation.  "
    "The Llama model family uses grouped-query attention in which 128 query "
    "heads share only 8 key-value groups, reducing the KV cache footprint.  "
    "This also means that one latent basis per KV group suffices.  "
    "Language model outputs range from factual recall to creative generation.  "
    "Efficient inference matters for interactive applications.  "
)


def _load_calibration_texts(path: str) -> List[str]:
    if path.strip():
        raw = Path(path).read_text(encoding="utf-8")
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        return paragraphs if paragraphs else [raw]
    # Repeat fallback a few times to get more calibration data.
    return [_FALLBACK_CALIBRATION_TEXT] * 4


# ---------------------------------------------------------------------------
# Key collection
# ---------------------------------------------------------------------------

def _collect_keys(
    runtime: StreamingLlamaRuntime,
    texts: List[str],
    retrieval_layers: List[int],
    *,
    max_rows: int,
    max_seq_len: int,
) -> Dict[int, torch.Tensor]:
    """Return a dict layer_idx → [N, G, D] float32 tensor of post-RoPE keys.

    We run each text through ``_forward_prefill`` (which populates
    ``runtime._dense_cache`` with exact post-RoPE keys), then read the cache.
    """
    G = int(getattr(runtime.config, "num_key_value_heads", 8))
    D = int(getattr(runtime.config, "head_dim", 128))

    # Accumulators: layer_idx → list of [T, G, D] tensors
    key_banks: Dict[int, List[torch.Tensor]] = {l: [] for l in retrieval_layers}
    rows_collected: Dict[int, int] = {l: 0 for l in retrieval_layers}
    selected_set = set(retrieval_layers)

    # Try to load a tokeniser from the snapshot.
    tok = None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            str(runtime.snapshot_dir),
            local_files_only=True,
            use_fast=True,
        )
    except Exception:
        pass

    for seq_idx, text in enumerate(texts):
        # Check early termination.
        if all(rows_collected[l] >= max_rows for l in retrieval_layers):
            break

        # Tokenise.
        try:
            if tok is not None:
                token_ids = tok(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(max_seq_len),
                    padding=False,
                )["input_ids"]    # [1, T]
            else:
                raise ValueError("no tokenizer")
        except Exception:
            vocab_size = int(runtime._embed_weight_cpu.shape[0]) if runtime._embed_weight_cpu is not None else 32000
            raw = [ord(c) % vocab_size for c in text[:max_seq_len]]
            token_ids = torch.tensor([raw], dtype=torch.long)

        if int(token_ids.shape[1]) < 2:
            continue

        print(
            f"[posting_basis] seq {seq_idx + 1}/{len(texts)}  "
            f"len={int(token_ids.shape[1])}",
            flush=True,
        )

        # Full prefill: populates runtime._dense_cache for all layers.
        with torch.no_grad():
            runtime.reset_caches()
            token_ids_gpu = token_ids.to(device=runtime.device)
            _ = runtime._forward_prefill(token_ids_gpu)

        # Read post-RoPE keys for each retrieval layer from the dense cache.
        dense_cache = runtime._dense_cache
        if dense_cache is None:
            print("[posting_basis] DynamicCache not available — skipping", flush=True)
            continue

        T = int(token_ids.shape[1])
        for l in retrieval_layers:
            if rows_collected[l] >= max_rows:
                continue
            kc = dense_cache.key_cache
            if l >= len(kc) or kc[l] is None:
                continue
            # key_cache[l]: [1, G, T, D]  (post-RoPE, exact)
            k_seq = kc[l][0].permute(1, 0, 2).contiguous().cpu().float()    # [T, G, D]
            remain = max_rows - rows_collected[l]
            take = min(int(k_seq.shape[0]), remain)
            key_banks[l].append(k_seq[:take])
            rows_collected[l] += take

        print(
            f"[posting_basis] collected: " +
            ", ".join(f"l{l}:{rows_collected[l]}" for l in retrieval_layers[:4]) +
            (" ..." if len(retrieval_layers) > 4 else ""),
            flush=True,
        )

    # Concatenate per layer.
    result: Dict[int, torch.Tensor] = {}
    for l in retrieval_layers:
        chunks = key_banks[l]
        if chunks:
            result[l] = torch.cat(chunks, dim=0)    # [N, G, D]
    return result


# ---------------------------------------------------------------------------
# PCA per (layer, KV group)
# ---------------------------------------------------------------------------

def _fit_group_basis(
    keys: np.ndarray,    # [N, D] float32
    *,
    rank: int,
    pca_method: str,
    idf_threshold: float,
) -> Dict[str, np.ndarray]:
    """Fit PCA basis for one KV group and compute IDF weights.

    Returns
    -------
    dict with keys:
        basis   : [rank, D]  — projection matrix U^T (orthonormal rows)
        idf     : [rank]     — IDF weight per latent coordinate
        key_mean: [D]        — mean key (for centering at inference)
    """
    N, D = keys.shape
    rank_eff = int(min(rank, N, D))

    key_mean = keys.mean(axis=0)           # [D]
    keys_c = keys - key_mean               # [N, D] centred

    # PCA via torch.pca_lowrank (SVD-based, fast for rank << D).
    keys_t = torch.from_numpy(keys_c)
    method = str(pca_method).strip().lower()
    if method == "auto":
        method = "incremental" if N >= max(4096, rank_eff * 8) else "lowrank"

    if method == "incremental":
        try:
            from sklearn.decomposition import IncrementalPCA
            ipca = IncrementalPCA(n_components=rank_eff, batch_size=max(rank_eff * 4, 512))
            for start in range(0, N, max(rank_eff * 4, 512)):
                chunk = keys_c[start : start + max(rank_eff * 4, 512)]
                if int(chunk.shape[0]) >= rank_eff:
                    ipca.partial_fit(chunk)
            V = torch.from_numpy(ipca.components_.copy())    # [rank_eff, D]
        except Exception:
            method = "lowrank"
    if method == "lowrank":
        _u, _s, V = torch.pca_lowrank(keys_t, q=rank_eff, center=False, niter=2)
        V = V.t()[:rank_eff]    # [rank_eff, D]

    # Pad to requested rank if needed.
    if rank_eff < rank:
        pad = torch.zeros(rank - rank_eff, D, dtype=V.dtype)
        V = torch.cat([V, pad], dim=0)    # [rank, D]

    basis = V.numpy().astype(np.float32)    # [rank, D]  orthonormal rows

    # Project all calibration keys into latent space: [N, rank]
    alpha = keys_c @ basis.T    # [N, rank]

    # IDF: log((1 + N) / (1 + df_r)) where df_r = tokens with |alpha_r| > threshold.
    df = (np.abs(alpha) > float(idf_threshold)).sum(axis=0).astype(np.float32)    # [rank]
    idf = np.log((1.0 + float(N)) / (1.0 + df)).astype(np.float32)    # [rank]

    return {
        "basis": basis,
        "idf": idf,
        "key_mean": key_mean.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    device = torch.device(args.device.strip()) if args.device.strip() else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    print(f"[posting_basis] device={device}", flush=True)
    print(f"[posting_basis] model={args.model_path}", flush=True)
    print(f"[posting_basis] output={args.output_path}", flush=True)

    # Build runtime with DynamicCache enabled (materialize_lm_head not needed).
    runtime = StreamingLlamaRuntime(
        model_name_or_path=str(args.model_path),
        device=device,
        dtype=torch.float16,
        taylor_layers=[],
        local_files_only=bool(args.local_files_only),
        ram_cache=True,
        materialize_lm_head=False,
    )

    total_layers = int(runtime.num_layers)
    G = int(getattr(runtime.config, "num_key_value_heads", 8))
    D = int(getattr(runtime.config, "head_dim", 128))

    retrieval_layers = _parse_retrieval_layers(
        args.retrieval_layers,
        args.retrieval_start_layer,
        total_layers,
    )
    print(
        f"[posting_basis] retrieval_layers: {retrieval_layers[0]}..{retrieval_layers[-1]} "
        f"({len(retrieval_layers)} layers)  G={G}  D={D}  rank={args.basis_rank}",
        flush=True,
    )

    # Load calibration texts.
    texts = _load_calibration_texts(str(args.calibration_text))
    print(f"[posting_basis] {len(texts)} calibration sequence(s)", flush=True)

    # Collect post-RoPE keys.
    print("[posting_basis] collecting post-RoPE keys via dense prefill ...", flush=True)
    key_data = _collect_keys(
        runtime,
        texts,
        retrieval_layers,
        max_rows=int(args.max_rows),
        max_seq_len=int(args.max_seq_len),
    )
    print(f"[posting_basis] collected keys for {len(key_data)} layer(s)", flush=True)

    # Fit PCA per (layer, KV group).
    layer_states: Dict[str, Any] = {}

    for l in retrieval_layers:
        keys_lgd = key_data.get(l)
        if keys_lgd is None:
            print(f"[posting_basis] layer {l}: no data — skipping", flush=True)
            continue

        N = int(keys_lgd.shape[0])
        print(f"[posting_basis] layer {l}: fitting PCA on {N}×{G}×{D} keys ...", flush=True)

        group_bases: List[np.ndarray] = []
        idf_weights: List[np.ndarray] = []
        key_means: List[np.ndarray] = []

        for g in range(G):
            keys_g = keys_lgd[:, g, :].numpy()    # [N, D]
            result = _fit_group_basis(
                keys_g,
                rank=int(args.basis_rank),
                pca_method=str(args.pca_method),
                idf_threshold=float(args.idf_threshold),
            )
            group_bases.append(result["basis"])
            idf_weights.append(result["idf"])
            key_means.append(result["key_mean"])

        layer_states[str(l)] = {
            "group_bases": group_bases,     # list of G arrays, each [rank, D]
            "idf_weights": idf_weights,     # list of G arrays, each [rank]
            "key_means": key_means,         # list of G arrays, each [D]
        }
        print(f"[posting_basis] layer {l}: done", flush=True)

    # Build and save checkpoint.
    config_payload: Dict[str, Any] = {
        "basis_rank": int(args.basis_rank),
        "num_kv_groups": G,
        "head_dim": D,
        "retrieval_layers": retrieval_layers,
        "retrieval_start_layer": int(args.retrieval_start_layer),
        "idf_threshold": float(args.idf_threshold),
        "total_layers": total_layers,
    }
    checkpoint = {
        "config": config_payload,
        "layer_states": layer_states,
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    print(
        f"[posting_basis] saved checkpoint → {out_path}  "
        f"({len(layer_states)} layers fitted)",
        flush=True,
    )


if __name__ == "__main__":
    main()
