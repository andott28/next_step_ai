from __future__ import annotations

"""token_posting_archive.py

CPU-side sparse token index for long-range sparse attention retrieval.

Replaces a dense DynamicCache for "retrieval layers" with three tiers:

  1. Sink buffer  – first ``num_sinks`` tokens; always on GPU; never evicted.
  2. GPU ring     – the most recent ``ring_size`` tokens; exact FP16 on GPU.
  3. CPU archive  – all older tokens; exact FP16, pinned CPU RAM.
                    Indexed by a per-(layer, kv_group, latent_coord) posting list.

During decode one archive probe per KV group selects up to ``candidates`` tokens
from the archive.  Those tokens' exact (K, V) pairs are H2D-transferred and
concatenated with the ring and sinks.  Exact attention is then run on that
small shortlist only — no dense full-context attention ever happens.

No SGD, no fine-tuning.  The PCA basis is fitted offline by
``init_attn_token_posting_basis.py`` (post-RoPE keys, per (layer, KV group)).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_pin(t: torch.Tensor) -> torch.Tensor:
    try:
        return t.pin_memory()
    except Exception:
        return t


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TokenPostingArchive:
    """Sparse token posting index for long-range streaming attention.

    One instance covers all retrieval layers for a single generation session.

    Parameters
    ----------
    retrieval_layers:
        Layer indices for which this archive is active.
    num_kv_groups:
        Number of KV groups (``num_key_value_heads`` in the model config).
    head_dim:
        Per-head dimension.
    basis_rank:
        PCA rank (R) used when projecting keys into latent space.
    ring_size:
        Number of tokens in the GPU exact-window ring (W).  Default 256.
    num_sinks:
        First ``num_sinks`` tokens always kept on GPU (never evicted).
    archive_capacity:
        Maximum tokens stored in the CPU archive per retrieval layer.
    token_topk:
        How many latent coordinates are stored per archived token.
        Trades index size against recall.  Default 8.
    r_query:
        How many latent query coordinates are probed per decode step.
        Default 6.
    candidates:
        Number of archive tokens (M) fetched per (layer, group) per step.
    device:
        GPU device for ring/sink buffers.
    dtype:
        FP16 for all K/V tensors.
    """

    def __init__(
        self,
        *,
        retrieval_layers: List[int],
        num_kv_groups: int,
        head_dim: int,
        basis_rank: int,
        ring_size: int = 256,
        num_sinks: int = 16,
        archive_capacity: int = 16384,
        token_topk: int = 8,
        r_query: int = 6,
        candidates: int = 64,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.retrieval_layers = list(retrieval_layers)
        self.num_kv_groups = int(num_kv_groups)
        self.head_dim = int(head_dim)
        self.basis_rank = int(basis_rank)
        self.ring_size = int(ring_size)
        self.num_sinks = int(num_sinks)
        self.archive_capacity = int(archive_capacity)
        self.token_topk = int(token_topk)
        self.r_query = int(r_query)
        self.candidates = int(candidates)
        self.device = device
        self.dtype = dtype

        # ── Basis / IDF (numpy float32, loaded from checkpoint) ──────────────
        # basis[l][g]    : ndarray [rank, head_dim]  — U^T so projection = basis @ key
        # idf[l][g]      : ndarray [rank]
        # key_mean[l][g] : ndarray [head_dim]  (centering before projection)
        self.basis: Dict[int, List[Optional[np.ndarray]]] = {}
        self.idf: Dict[int, List[Optional[np.ndarray]]] = {}
        self.key_mean: Dict[int, List[Optional[np.ndarray]]] = {}

        # ── Sink buffers (GPU) ────────────────────────────────────────────────
        self.sink_k: Dict[int, torch.Tensor] = {}     # [num_sinks, G, D]
        self.sink_v: Dict[int, torch.Tensor] = {}
        self.sink_count: Dict[int, int] = {}

        # ── Ring buffers (GPU, circular) ──────────────────────────────────────
        self.ring_k: Dict[int, torch.Tensor] = {}     # [ring_size, G, D]
        self.ring_v: Dict[int, torch.Tensor] = {}
        self.ring_head: Dict[int, int] = {}            # next write slot (0..ring_size-1)
        self.ring_count: Dict[int, int] = {}           # valid entries (0..ring_size)

        # ── Archive (CPU pinned, indexed) ─────────────────────────────────────
        self.archive_k_cpu: Dict[int, torch.Tensor] = {}  # [cap, G, D] FP16
        self.archive_v_cpu: Dict[int, torch.Tensor] = {}
        self.archive_count: Dict[int, int] = {}

        # ── Posting lists ─────────────────────────────────────────────────────
        # _post_tok[l][g][r]   : list of int   — archive indices
        # _post_coeff[l][g][r] : list of int   — Q8 latent coefficients (int8 range)
        # _post_scale[l][g][r] : list of float — per-token latent max_abs scale
        self._post_tok: Dict[int, List[List[List[int]]]] = {}
        self._post_coeff: Dict[int, List[List[List[int]]]] = {}
        self._post_scale: Dict[int, List[List[List[float]]]] = {}

        # ── Score / stamp scratch (CPU, per step) ────────────────────────────
        # score_buf[l][g] : float32 ndarray [archive_capacity]
        # stamp_buf[l][g] : int32  ndarray [archive_capacity]  (-1 = never touched)
        self._score_buf: Dict[int, List[np.ndarray]] = {}
        self._stamp_buf: Dict[int, List[np.ndarray]] = {}

        # Global decode step counter used to avoid clearing score/stamp buffers.
        self.step: int = 0

        self._init_buffers()

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_buffers(self) -> None:
        G = self.num_kv_groups
        D = self.head_dim
        R = self.basis_rank
        W = self.ring_size
        NS = self.num_sinks
        AC = self.archive_capacity

        for l in self.retrieval_layers:
            # Basis placeholders
            self.basis[l] = [None] * G
            self.idf[l] = [None] * G
            self.key_mean[l] = [None] * G

            # Sinks (GPU)
            self.sink_k[l] = torch.zeros(NS, G, D, dtype=self.dtype, device=self.device)
            self.sink_v[l] = torch.zeros(NS, G, D, dtype=self.dtype, device=self.device)
            self.sink_count[l] = 0

            # Ring (GPU)
            self.ring_k[l] = torch.zeros(W, G, D, dtype=self.dtype, device=self.device)
            self.ring_v[l] = torch.zeros(W, G, D, dtype=self.dtype, device=self.device)
            self.ring_head[l] = 0
            self.ring_count[l] = 0

            # Archive (CPU pinned)
            self.archive_k_cpu[l] = _try_pin(torch.zeros(AC, G, D, dtype=torch.float16))
            self.archive_v_cpu[l] = _try_pin(torch.zeros(AC, G, D, dtype=torch.float16))
            self.archive_count[l] = 0

            # Posting lists
            self._post_tok[l] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_coeff[l] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_scale[l] = [[[] for _ in range(R)] for _ in range(G)]

            # Scratch buffers
            self._score_buf[l] = [np.zeros(AC, dtype=np.float32) for _ in range(G)]
            self._stamp_buf[l] = [np.full(AC, -1, dtype=np.int32) for _ in range(G)]

    def load_basis(
        self,
        layer_idx: int,
        group_idx: int,
        basis: np.ndarray,
        idf: np.ndarray,
        key_mean: Optional[np.ndarray] = None,
    ) -> None:
        """Set the PCA basis for one (layer, kv_group).

        Parameters
        ----------
        basis    : [rank, head_dim] float32 — projection matrix U^T
        idf      : [rank] float32            — per-coordinate IDF weights
        key_mean : [head_dim] float32        — mean key (for centering)
        """
        if layer_idx not in self.basis:
            self.basis[layer_idx] = [None] * self.num_kv_groups
            self.idf[layer_idx] = [None] * self.num_kv_groups
            self.key_mean[layer_idx] = [None] * self.num_kv_groups
        self.basis[layer_idx][group_idx] = np.asarray(basis, dtype=np.float32)
        self.idf[layer_idx][group_idx] = np.asarray(idf, dtype=np.float32)
        self.key_mean[layer_idx][group_idx] = (
            np.asarray(key_mean, dtype=np.float32)
            if key_mean is not None
            else np.zeros(self.head_dim, dtype=np.float32)
        )

    def reset(self) -> None:
        """Reset all buffers.  Call at the start of each generation session."""
        G = self.num_kv_groups
        R = self.basis_rank
        AC = self.archive_capacity
        for l in self.retrieval_layers:
            self.sink_count[l] = 0
            self.ring_head[l] = 0
            self.ring_count[l] = 0
            self.archive_count[l] = 0
            self._post_tok[l] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_coeff[l] = [[[] for _ in range(R)] for _ in range(G)]
            self._post_scale[l] = [[[] for _ in range(R)] for _ in range(G)]
            for g in range(G):
                self._stamp_buf[l][g].fill(-1)
                self._score_buf[l][g].fill(0.0)
        self.step = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Token ingestion
    # ─────────────────────────────────────────────────────────────────────────

    def append_token(
        self,
        layer_idx: int,
        seq_pos: int,
        k_cpu: torch.Tensor,    # [G, D] FP16 CPU
        v_cpu: torch.Tensor,    # [G, D] FP16 CPU
    ) -> None:
        """Ingest one (K, V) pair into the appropriate tier.

        Routing:
          seq_pos < num_sinks            → sink buffer (GPU, never evicted)
          sink full, ring not full       → ring (GPU)
          ring full                      → evict oldest ring slot to CPU archive
                                           then write new token into the freed ring slot
        """
        # ── Sink tier ────────────────────────────────────────────────────────
        sc = self.sink_count[layer_idx]
        if sc < self.num_sinks:
            self.sink_k[layer_idx][sc] = k_cpu.to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            self.sink_v[layer_idx][sc] = v_cpu.to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            self.sink_count[layer_idx] = sc + 1
            return

        # ── Ring tier ─────────────────────────────────────────────────────────
        ring_full = self.ring_count[layer_idx] >= self.ring_size
        evict_slot = self.ring_head[layer_idx]

        if ring_full:
            # Evict oldest ring entry to CPU archive before overwriting.
            evict_k = self.ring_k[layer_idx][evict_slot].cpu()    # [G, D]
            evict_v = self.ring_v[layer_idx][evict_slot].cpu()
            self._archive_token(layer_idx, evict_k, evict_v)

        # Write new token into the ring slot.
        self.ring_k[layer_idx][evict_slot] = k_cpu.to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        self.ring_v[layer_idx][evict_slot] = v_cpu.to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        self.ring_head[layer_idx] = (evict_slot + 1) % self.ring_size
        if not ring_full:
            self.ring_count[layer_idx] += 1

    def _archive_token(
        self,
        layer_idx: int,
        k_cpu: torch.Tensor,    # [G, D] FP16 CPU
        v_cpu: torch.Tensor,
    ) -> None:
        """Write a token into the CPU archive and update posting lists."""
        n = self.archive_count[layer_idx]
        if n >= self.archive_capacity:
            return   # archive full — silently drop (oldest-first drop policy)

        # Store exact K/V.
        self.archive_k_cpu[layer_idx][n].copy_(k_cpu.half())
        self.archive_v_cpu[layer_idx][n].copy_(v_cpu.half())

        # Project each KV group's key through the PCA basis and quantize.
        k_f32 = k_cpu.float().numpy()      # [G, D]
        G = self.num_kv_groups
        R = self.basis_rank

        for g in range(G):
            basis_g = self.basis[layer_idx][g]    # [R, D] or None
            if basis_g is None:
                continue
            mean_g = self.key_mean[layer_idx][g]  # [D]
            k_g = k_f32[g] - mean_g               # [D] centred
            alpha = basis_g @ k_g                 # [R] latent coefficients

            # Q8 quantise: pack into [-127, 127]
            max_abs = float(np.abs(alpha).max())
            if max_abs > 1e-8:
                alpha_q8 = np.round(alpha * (127.0 / max_abs)).clip(-127, 127).astype(np.int8)
            else:
                alpha_q8 = np.zeros(R, dtype=np.int8)

            # Append only the top-k magnitude coordinates to keep index lean.
            top_k = min(self.token_topk, R)
            top_coords = np.argpartition(np.abs(alpha), -top_k)[-top_k:]
            for r in top_coords.tolist():
                coeff = int(alpha_q8[r])
                if coeff != 0:
                    self._post_tok[layer_idx][g][r].append(n)
                    self._post_coeff[layer_idx][g][r].append(coeff)
                    self._post_scale[layer_idx][g][r].append(max_abs)

        self.archive_count[layer_idx] = n + 1

    # ─────────────────────────────────────────────────────────────────────────
    # Archive probing
    # ─────────────────────────────────────────────────────────────────────────

    def _probe(
        self,
        layer_idx: int,
        group_idx: int,
        beta: np.ndarray,    # [rank] float32 — IDF-weighted query latent
        step: int,
    ) -> List[int]:
        """Accumulate approximate archive scores for candidate tokens.

        Uses stamp/score arrays to avoid clearing O(archive_capacity) memory
        every step.  Only touched tokens are scored; a list of their archive
        indices is returned.

        The scores are written into ``self._score_buf[layer_idx][group_idx]``.
        """
        idf = self.idf[layer_idx][group_idx]    # [R]
        weighted = beta * idf                   # [R]

        r_query = min(self.r_query, int(self.basis_rank))
        # Pick the r_query highest-magnitude weighted coordinates.
        top_r = np.argpartition(np.abs(weighted), -r_query)[-r_query:]

        score_buf = self._score_buf[layer_idx][group_idx]
        stamp_buf = self._stamp_buf[layer_idx][group_idx]
        touched: List[int] = []

        for r in top_r.tolist():
            beta_r = float(weighted[r])
            tok_list = self._post_tok[layer_idx][group_idx][r]
            coeff_list = self._post_coeff[layer_idx][group_idx][r]
            scale_list = self._post_scale[layer_idx][group_idx][r]
            for i in range(len(tok_list)):
                t = tok_list[i]
                if stamp_buf[t] != step:
                    stamp_buf[t] = step
                    score_buf[t] = 0.0
                    touched.append(t)
                score_buf[t] += beta_r * (float(coeff_list[i]) / 127.0) * float(scale_list[i])

        return touched

    def select_candidates(
        self,
        layer_idx: int,
        group_idx: int,
        q_cpu: np.ndarray,    # [head_dim] float32 — representative query
        step: int,
        M: Optional[int] = None,
    ) -> np.ndarray:
        """Return up to M archive indices most relevant to ``q_cpu``."""
        if M is None:
            M = self.candidates
        n_arch = self.archive_count[layer_idx]
        if n_arch == 0:
            return np.empty(0, dtype=np.int32)

        basis_g = self.basis[layer_idx][group_idx]
        if basis_g is None:
            return np.arange(min(M, n_arch), dtype=np.int32)

        mean_g = self.key_mean[layer_idx][group_idx]
        q_centred = q_cpu - mean_g              # [D]
        beta = basis_g @ q_centred             # [R]

        touched = self._probe(layer_idx, group_idx, beta, step)
        if not touched:
            return np.empty(0, dtype=np.int32)

        touched_arr = np.array(touched, dtype=np.int32)
        scores = self._score_buf[layer_idx][group_idx][touched_arr]
        order = np.argsort(scores)[::-1]
        if len(touched_arr) <= M:
            return touched_arr[order]

        # Pick top-M by accumulated score.
        best_local = np.argpartition(scores, -M)[-M:]
        selected = touched_arr[best_local]
        selected_scores = scores[best_local]
        selected_order = np.argsort(selected_scores)[::-1]
        return selected[selected_order]

    # ─────────────────────────────────────────────────────────────────────────
    # Shortlist assembly (main per-step call)
    # ─────────────────────────────────────────────────────────────────────────

    def fetch_shortlist_kv(
        self,
        layer_idx: int,
        group_idx: int,
        q_rep_gpu: torch.Tensor,    # [head_dim] GPU FP16 — representative query
        step: int,
        M: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assemble the full shortlist K, V for one (layer, group, step).

        Returns
        -------
        k_all : [T, head_dim]  GPU FP16
        v_all : [T, head_dim]  GPU FP16

        T = num_sinks_valid + num_archive_candidates + num_ring_tokens.
        Always includes current-step ring and sink tokens (exact local window).
        """
        g = group_idx
        parts_k: List[torch.Tensor] = []
        parts_v: List[torch.Tensor] = []

        # ── Sink tier (always exact, GPU) ─────────────────────────────────────
        sc = self.sink_count[layer_idx]
        if sc > 0:
            parts_k.append(self.sink_k[layer_idx][:sc, g, :])    # [sc, D]
            parts_v.append(self.sink_v[layer_idx][:sc, g, :])

        # ── Archive candidates (posting probe → H2D) ──────────────────────────
        q_cpu = q_rep_gpu.float().cpu().numpy()
        if q_cpu.ndim == 1:
            cand_ids = self.select_candidates(layer_idx, g, q_cpu, step, M)
        else:
            # Union per-head probes so one head's query does not suppress another.
            # Use disjoint stamp ids per query probe to avoid score-buffer collisions.
            score_best: Dict[int, float] = {}
            base_step = int(step) * 1024
            for qi in range(int(q_cpu.shape[0])):
                local_ids = self.select_candidates(layer_idx, g, q_cpu[qi], base_step + qi, M)
                if local_ids.size == 0:
                    continue
                local_scores = self._score_buf[layer_idx][g][local_ids]
                for idx, score in zip(local_ids.tolist(), local_scores.tolist()):
                    prev = score_best.get(int(idx))
                    if prev is None or float(score) > prev:
                        score_best[int(idx)] = float(score)
            if score_best:
                ordered = sorted(score_best.items(), key=lambda kv: kv[1], reverse=True)
                cand_ids = np.asarray([int(tok) for tok, _ in ordered[: int(M or self.candidates)]], dtype=np.int32)
            else:
                cand_ids = np.empty(0, dtype=np.int32)
        if len(cand_ids) > 0:
            # Slice pinned CPU tensors and async-copy to GPU.
            k_arch_cpu = self.archive_k_cpu[layer_idx][cand_ids, g, :]    # [C, D]
            v_arch_cpu = self.archive_v_cpu[layer_idx][cand_ids, g, :]
            parts_k.append(k_arch_cpu.to(device=self.device, non_blocking=True))
            parts_v.append(v_arch_cpu.to(device=self.device, non_blocking=True))

        # ── Ring tier (always exact, GPU) ─────────────────────────────────────
        rc = self.ring_count[layer_idx]
        if rc > 0:
            if rc < self.ring_size:
                parts_k.append(self.ring_k[layer_idx][:rc, g, :])
                parts_v.append(self.ring_v[layer_idx][:rc, g, :])
            else:
                # Full ring: all ring_size slots are valid.
                # Order does not matter for attention.
                parts_k.append(self.ring_k[layer_idx][:, g, :])
                parts_v.append(self.ring_v[layer_idx][:, g, :])

        if not parts_k:
            empty = torch.zeros(0, self.head_dim, dtype=self.dtype, device=self.device)
            return empty, empty

        k_all = torch.cat(parts_k, dim=0)    # [T, D]
        v_all = torch.cat(parts_v, dim=0)    # [T, D]
        return k_all, v_all

    # ─────────────────────────────────────────────────────────────────────────
    # Warm-up from DynamicCache (call once after prefill)
    # ─────────────────────────────────────────────────────────────────────────

    def warm_up_from_dense_cache(
        self,
        dense_cache: Any,
        seq_len: int,
    ) -> None:
        """Populate the archive from a DynamicCache that was filled during prefill.

        ``dense_cache.key_cache[l]`` has shape ``[1, G, T, D]`` (post-RoPE).
        Tokens are ingested in sequence order so the sink/ring/archive tiers
        are filled correctly: the first ``num_sinks`` tokens land in sinks,
        the next ``ring_size`` fill the ring, and everything older is in the
        CPU archive ready for posting-list retrieval.
        """
        if hasattr(dense_cache, "key_cache") and hasattr(dense_cache, "value_cache"):
            kc = dense_cache.key_cache
            vc = dense_cache.value_cache
            for l in self.retrieval_layers:
                if l >= len(kc) or kc[l] is None:
                    continue
                T = int(kc[l].shape[2])
                T = min(T, int(seq_len))
                # Shape: [1, G, T, D]  →  [T, G, D]
                k_seq = kc[l][0].permute(1, 0, 2).contiguous().cpu()    # [T, G, D]
                v_seq = vc[l][0].permute(1, 0, 2).contiguous().cpu()    # [T, G, D]
                for t in range(T):
                    self.append_token(l, t, k_seq[t], v_seq[t])
            return

        # Newer transformers DynamicCache APIs may hide storage internals and only
        # expose legacy conversion.
        if hasattr(dense_cache, "to_legacy_cache"):
            legacy = dense_cache.to_legacy_cache()
            if legacy is None:
                return
            for l in self.retrieval_layers:
                if l >= len(legacy):
                    continue
                layer_cache = legacy[l]
                if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) < 2:
                    continue
                k_layer = layer_cache[0]
                v_layer = layer_cache[1]
                if not (torch.is_tensor(k_layer) and torch.is_tensor(v_layer)):
                    continue
                if k_layer.ndim != 4:
                    continue
                T = min(int(k_layer.shape[2]), int(seq_len))
                if T <= 0:
                    continue
                k_seq = k_layer[0].permute(1, 0, 2).contiguous().cpu()
                v_seq = v_layer[0].permute(1, 0, 2).contiguous().cpu()
                for t in range(T):
                    self.append_token(l, t, k_seq[t], v_seq[t])
            return

        raise RuntimeError(
            "Unsupported cache type for token-posting warm-up: expected key_cache/value_cache "
            "or to_legacy_cache()."
        )
