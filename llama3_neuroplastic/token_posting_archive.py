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

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _try_pin(t: torch.Tensor) -> torch.Tensor:
    try:
        return t.pin_memory()
    except Exception:
        return t






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
        retrieval_layers: list[int],
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





        self.basis: dict[int, list[np.ndarray | None]] = {}
        self.idf: dict[int, list[np.ndarray | None]] = {}
        self.key_mean: dict[int, list[np.ndarray | None]] = {}


        self.sink_k: dict[int, torch.Tensor] = {}
        self.sink_v: dict[int, torch.Tensor] = {}
        self.sink_count: dict[int, int] = {}


        self.ring_k: dict[int, torch.Tensor] = {}
        self.ring_v: dict[int, torch.Tensor] = {}
        self.ring_head: dict[int, int] = {}
        self.ring_count: dict[int, int] = {}


        self.archive_k_cpu: dict[int, torch.Tensor] = {}
        self.archive_v_cpu: dict[int, torch.Tensor] = {}
        self.archive_count: dict[int, int] = {}
        self.archive_generation: dict[int, np.ndarray] = {}





        self._post_tok: dict[int, np.ndarray] = {}
        self._post_gen: dict[int, np.ndarray] = {}
        self._post_coeff: dict[int, np.ndarray] = {}
        self._post_scale: dict[int, np.ndarray] = {}
        self._post_head: dict[int, np.ndarray] = {}
        self._post_count: dict[int, np.ndarray] = {}




        self._score_buf: dict[int, list[np.ndarray]] = {}
        self._stamp_buf: dict[int, list[np.ndarray]] = {}


        self.step: int = 0

        self._init_buffers()





    def _init_buffers(self) -> None:
        G = self.num_kv_groups
        D = self.head_dim
        R = self.basis_rank
        W = self.ring_size
        NS = self.num_sinks
        AC = self.archive_capacity

        for layer_idx in self.retrieval_layers:

            self.basis[layer_idx] = [None] * G
            self.idf[layer_idx] = [None] * G
            self.key_mean[layer_idx] = [None] * G


            self.sink_k[layer_idx] = torch.zeros(NS, G, D, dtype=self.dtype, device=self.device)
            self.sink_v[layer_idx] = torch.zeros(NS, G, D, dtype=self.dtype, device=self.device)
            self.sink_count[layer_idx] = 0


            self.ring_k[layer_idx] = torch.zeros(W, G, D, dtype=self.dtype, device=self.device)
            self.ring_v[layer_idx] = torch.zeros(W, G, D, dtype=self.dtype, device=self.device)
            self.ring_head[layer_idx] = 0
            self.ring_count[layer_idx] = 0


            self.archive_k_cpu[layer_idx] = _try_pin(torch.zeros(AC, G, D, dtype=torch.float16))
            self.archive_v_cpu[layer_idx] = _try_pin(torch.zeros(AC, G, D, dtype=torch.float16))
            self.archive_count[layer_idx] = 0
            self.archive_generation[layer_idx] = np.full(AC, -1, dtype=np.int64)


            self._post_tok[layer_idx] = np.zeros((G, R, AC), dtype=np.int32)
            self._post_gen[layer_idx] = np.zeros((G, R, AC), dtype=np.int64)
            self._post_coeff[layer_idx] = np.zeros((G, R, AC), dtype=np.int8)
            self._post_scale[layer_idx] = np.zeros((G, R, AC), dtype=np.float32)
            self._post_head[layer_idx] = np.zeros((G, R), dtype=np.int32)
            self._post_count[layer_idx] = np.zeros((G, R), dtype=np.int32)


            self._score_buf[layer_idx] = [np.zeros(AC, dtype=np.float32) for _ in range(G)]
            self._stamp_buf[layer_idx] = [np.full(AC, -1, dtype=np.int32) for _ in range(G)]

    def load_basis(
        self,
        layer_idx: int,
        group_idx: int,
        basis: np.ndarray,
        idf: np.ndarray,
        key_mean: np.ndarray | None = None,
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
        for layer_idx in self.retrieval_layers:
            self.sink_count[layer_idx] = 0
            self.ring_head[layer_idx] = 0
            self.ring_count[layer_idx] = 0
            self.archive_count[layer_idx] = 0
            self.archive_generation[layer_idx].fill(-1)
            self._post_tok[layer_idx] = np.zeros((G, R, self.archive_capacity), dtype=np.int32)
            self._post_gen[layer_idx] = np.zeros((G, R, self.archive_capacity), dtype=np.int64)
            self._post_coeff[layer_idx] = np.zeros((G, R, self.archive_capacity), dtype=np.int8)
            self._post_scale[layer_idx] = np.zeros((G, R, self.archive_capacity), dtype=np.float32)
            self._post_head[layer_idx] = np.zeros((G, R), dtype=np.int32)
            self._post_count[layer_idx] = np.zeros((G, R), dtype=np.int32)
            for g in range(G):
                self._stamp_buf[layer_idx][g].fill(-1)
                self._score_buf[layer_idx][g].fill(0.0)
        self.step = 0





    def append_token(
        self,
        layer_idx: int,
        seq_pos: int,
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
    ) -> None:
        """Ingest one (K, V) pair into the appropriate tier.

        Routing:
          seq_pos < num_sinks            → sink buffer (GPU, never evicted)
          sink full, ring not full       → ring (GPU)
          ring full                      → evict oldest ring slot to CPU archive
                                           then write new token into the freed ring slot
        """

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


        ring_full = self.ring_count[layer_idx] >= self.ring_size
        evict_slot = self.ring_head[layer_idx]

        if ring_full:

            evict_k = self.ring_k[layer_idx][evict_slot].cpu()
            evict_v = self.ring_v[layer_idx][evict_slot].cpu()
            self._archive_token(layer_idx, evict_k, evict_v)


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
        k_cpu: torch.Tensor,
        v_cpu: torch.Tensor,
    ) -> None:
        """Write a token into the CPU archive and update posting lists."""
        n = self.archive_count[layer_idx]
        write_pos = n % self.archive_capacity


        self.archive_k_cpu[layer_idx][write_pos].copy_(k_cpu.half())
        self.archive_v_cpu[layer_idx][write_pos].copy_(v_cpu.half())
        self.archive_generation[layer_idx][write_pos] = int(n)


        k_f32 = k_cpu.float().numpy()
        G = self.num_kv_groups
        R = self.basis_rank

        for g in range(G):
            basis_g = self.basis[layer_idx][g]
            if basis_g is None:
                continue
            mean_g = self.key_mean[layer_idx][g]
            k_g = k_f32[g] - mean_g
            alpha = basis_g @ k_g


            max_abs = float(np.abs(alpha).max())
            if max_abs > 1e-8:
                alpha_q8 = np.round(alpha * (127.0 / max_abs)).clip(-127, 127).astype(np.int8)
            else:
                alpha_q8 = np.zeros(R, dtype=np.int8)


            top_k = min(self.token_topk, R)
            top_coords = np.argpartition(np.abs(alpha), -top_k)[-top_k:]
            for r in top_coords.tolist():
                coeff = int(alpha_q8[r])
                if coeff != 0:
                    head = self._post_head[layer_idx][g, r]
                    self._post_tok[layer_idx][g, r, head] = write_pos
                    self._post_gen[layer_idx][g, r, head] = n
                    self._post_coeff[layer_idx][g, r, head] = coeff
                    self._post_scale[layer_idx][g, r, head] = max_abs
                    self._post_head[layer_idx][g, r] = (head + 1) % self.archive_capacity
                    self._post_count[layer_idx][g, r] += 1

        self.archive_count[layer_idx] = n + 1





    def _probe(
        self,
        layer_idx: int,
        group_idx: int,
        beta: np.ndarray,
        step: int,
    ) -> list[int]:
        """Accumulate approximate archive scores for candidate tokens.

        Uses stamp/score arrays to avoid clearing O(archive_capacity) memory
        every step.  Only touched tokens are scored; a list of their archive
        indices is returned.

        The scores are written into ``self._score_buf[layer_idx][group_idx]``.
        """
        idf = self.idf[layer_idx][group_idx]
        weighted = beta * idf

        r_query = min(self.r_query, int(self.basis_rank))

        top_r = np.argpartition(np.abs(weighted), -r_query)[-r_query:]

        score_buf = self._score_buf[layer_idx][group_idx]
        stamp_buf = self._stamp_buf[layer_idx][group_idx]
        touched: list[int] = []

        for r in top_r.tolist():
            beta_r = float(weighted[r])
            limit = min(self._post_count[layer_idx][group_idx, r], self.archive_capacity)
            if limit == 0:
                continue
                
            tok_arr = self._post_tok[layer_idx][group_idx, r, :limit]
            gen_arr = self._post_gen[layer_idx][group_idx, r, :limit]
            coeff_arr = self._post_coeff[layer_idx][group_idx, r, :limit]
            scale_arr = self._post_scale[layer_idx][group_idx, r, :limit]
            
            valid_mask = (self.archive_generation[layer_idx][tok_arr] == gen_arr)
            
            valid_toks = tok_arr[valid_mask]
            valid_coeffs = coeff_arr[valid_mask]
            valid_scales = scale_arr[valid_mask]
            
            new_mask = stamp_buf[valid_toks] != step
            if new_mask.any():
                new_toks = valid_toks[new_mask]
                stamp_buf[new_toks] = step
                score_buf[new_toks] = 0.0
                touched.extend(new_toks.tolist())
                
            score_buf[valid_toks] += beta_r * (valid_coeffs.astype(np.float32) / 127.0) * valid_scales

        return touched

    def select_candidates(
        self,
        layer_idx: int,
        group_idx: int,
        q_cpu: np.ndarray,
        step: int,
        M: int | None = None,
    ) -> np.ndarray:
        """Return up to M archive indices most relevant to ``q_cpu``."""
        if M is None:
            M = self.candidates
        n_arch = self.archive_count[layer_idx]
        if n_arch == 0:
            return np.empty(0, dtype=np.int32)
        valid_count = min(int(n_arch), int(self.archive_capacity))

        basis_g = self.basis[layer_idx][group_idx]
        if basis_g is None:
            first_generation = max(0, int(n_arch) - valid_count)
            slots = np.asarray(
                [generation % int(self.archive_capacity) for generation in range(first_generation, int(n_arch))],
                dtype=np.int32,
            )
            return slots[: min(int(M), int(slots.size))]

        mean_g = self.key_mean[layer_idx][group_idx]
        q_centred = q_cpu - mean_g
        beta = basis_g @ q_centred

        touched = self._probe(layer_idx, group_idx, beta, step)
        if not touched:
            return np.empty(0, dtype=np.int32)

        touched_arr = np.array(touched, dtype=np.int32)
        scores = self._score_buf[layer_idx][group_idx][touched_arr]
        order = np.argsort(scores)[::-1]
        if len(touched_arr) <= M:
            return touched_arr[order]


        best_local = np.argpartition(scores, -M)[-M:]
        selected = touched_arr[best_local]
        selected_scores = scores[best_local]
        selected_order = np.argsort(selected_scores)[::-1]
        return selected[selected_order]





    def fetch_shortlist_kv(
        self,
        layer_idx: int,
        group_idx: int,
        q_rep_gpu: torch.Tensor,
        step: int,
        M: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the full shortlist K, V for one (layer, group, step).

        Returns
        -------
        k_all : [T, head_dim]  GPU FP16
        v_all : [T, head_dim]  GPU FP16

        T = num_sinks_valid + num_archive_candidates + num_ring_tokens.
        Always includes current-step ring and sink tokens (exact local window).
        """
        g = group_idx
        parts_k: list[torch.Tensor] = []
        parts_v: list[torch.Tensor] = []


        sc = self.sink_count[layer_idx]
        if sc > 0:
            parts_k.append(self.sink_k[layer_idx][:sc, g, :])
            parts_v.append(self.sink_v[layer_idx][:sc, g, :])


        q_cpu = q_rep_gpu.float().cpu().numpy()
        if q_cpu.ndim == 1:
            cand_ids = self.select_candidates(layer_idx, g, q_cpu, step, M)
        else:


            score_best: dict[int, float] = {}
            base_step = int(step) * 1024
            for qi in range(int(q_cpu.shape[0])):
                local_ids = self.select_candidates(layer_idx, g, q_cpu[qi], base_step + qi, M)
                if local_ids.size == 0:
                    continue
                local_scores = self._score_buf[layer_idx][g][local_ids]
                for idx, score in zip(local_ids.tolist(), local_scores.tolist(), strict=False):
                    prev = score_best.get(int(idx))
                    if prev is None or float(score) > prev:
                        score_best[int(idx)] = float(score)
            if score_best:
                ordered = sorted(score_best.items(), key=lambda kv: kv[1], reverse=True)
                cand_ids = np.asarray([int(tok) for tok, _ in ordered[: int(M or self.candidates)]], dtype=np.int32)
            else:
                cand_ids = np.empty(0, dtype=np.int32)
        if len(cand_ids) > 0:

            k_arch_cpu = self.archive_k_cpu[layer_idx][cand_ids, g, :]
            v_arch_cpu = self.archive_v_cpu[layer_idx][cand_ids, g, :]
            parts_k.append(k_arch_cpu.to(device=self.device, non_blocking=True))
            parts_v.append(v_arch_cpu.to(device=self.device, non_blocking=True))


        rc = self.ring_count[layer_idx]
        if rc > 0:
            if rc < self.ring_size:
                parts_k.append(self.ring_k[layer_idx][:rc, g, :])
                parts_v.append(self.ring_v[layer_idx][:rc, g, :])
            else:


                parts_k.append(self.ring_k[layer_idx][:, g, :])
                parts_v.append(self.ring_v[layer_idx][:, g, :])

        if not parts_k:
            empty = torch.zeros(0, self.head_dim, dtype=self.dtype, device=self.device)
            return empty, empty

        k_all = torch.cat(parts_k, dim=0)
        v_all = torch.cat(parts_v, dim=0)
        return k_all, v_all





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
            for layer_idx in self.retrieval_layers:
                if layer_idx >= len(kc) or kc[layer_idx] is None:
                    continue
                T = int(kc[layer_idx].shape[2])
                T = min(T, int(seq_len))

                k_seq = kc[layer_idx][0].permute(1, 0, 2).contiguous().cpu()
                v_seq = vc[layer_idx][0].permute(1, 0, 2).contiguous().cpu()
                for t in range(T):
                    self.append_token(layer_idx, t, k_seq[t], v_seq[t])
            return



        if hasattr(dense_cache, "to_legacy_cache"):
            legacy = dense_cache.to_legacy_cache()
            if legacy is None:
                return
            for layer_idx in self.retrieval_layers:
                if layer_idx >= len(legacy):
                    continue
                layer_cache = legacy[layer_idx]
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
                    self.append_token(layer_idx, t, k_seq[t], v_seq[t])
            return

        raise RuntimeError(
            "Unsupported cache type for token-posting warm-up: expected key_cache/value_cache "
            "or to_legacy_cache()."
        )
