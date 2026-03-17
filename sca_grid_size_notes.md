# SCA `grid_size` vs. the Canonical `16×16×16` Lattice (and What Our Ablation Actually Tested)

This note clarifies what the Spatially Constrained Adapter (SCA) "grid" means in this repo, why `16×16×16` appears, and how to describe the `grid_size` ablations in `newtests.md` accurately in a paper.

## 1) Why `16×16×16` Exists (and Why It Is Not a Free Choice)

In the current implementation, block-center coordinates are derived from the model hidden size by mapping **each hidden dimension** onto a 3D lattice.

- For Llama-3.1-8B, `hidden_size = 4096`.
- `4096 = 16^3`, so the canonical lattice is `16×16×16`, with coordinates in `[0..15]^3`.

This is why `16` shows up: it is the unique cube-root that makes a **1:1 mapping** possible under this design.

If you "choose another grid size" while keeping the same "one hidden-dimension per voxel" idea:

- If `L^3 < 4096`: multiple hidden dimensions must collide to the same coordinate (many-to-one).
- If `L^3 > 4096`: some coordinates are unused (holes).

So, under the current mapping scheme, `16×16×16` is not a hyperparameter; it is implied by `hidden_size`.

## 2) What `grid_size` Actually Does in This Repo

Despite the name, `grid_size` in this repo is not "routing resolution".

### 2.1 Centers: fixed lattice then rescaled

`build_block_centers()` constructs coordinates on the canonical lattice implied by `hidden_size`, then **linearly rescales** them into `[0..grid_size-1]^3`.

- Canonical lattice: `[0..15]^3` for `hidden_size=4096`.
- Rescale: multiply coordinates by `(grid_size-1)/(base_grid-1)` where `base_grid=16`.

This is implemented in `llama3_neuroplastic/sca_sparse_config.py` (see the "Rescale" comment).

### 2.2 Queries: continuous values scaled into the same range

Queries are produced by a learned projection, then squashed with a sigmoid and scaled:

`query = sigmoid(spatial_proj(...)) * (grid_size - 1)`

This is implemented in `llama3_neuroplastic/neuroplastic_llama.py`.

### 2.3 Gating uses an RBF kernel of squared distance

Active blocks are selected using:

`score = exp(-||q - c||^2 / (2*sigma^2))`, then `top_k`.

This is implemented in `llama3_neuroplastic/sca_sparse_adapter.py`.

### 2.4 Key consequence: `grid_size` is mainly a scale/temperature knob

Because both the centers and queries are scaled into the same coordinate range, changing `grid_size` behaves mostly like changing the **effective bandwidth** of the RBF kernel.

If we denote the linear coordinate scale as:

- `s = (grid_size - 1) / (base_grid - 1)` with `base_grid=16`

then distances scale as `||q - c|| -> s * ||q - c||`, and the RBF looks like using an "effective sigma":

- `sigma_eff = sigma / s = sigma * (base_grid - 1) / (grid_size - 1)`

Examples (assuming `sigma=1` and `base_grid=16`):

- `grid_size=16`: `s=1`, so `sigma_eff = 1` (no rescale).
- `grid_size=2`: `s=1/15`, so `sigma_eff = 15` (very "soft" gating; scores become flatter).

Important nuance: even if a single forward pass would keep the *distance ordering* unchanged under pure rescaling, the system is **recursive** across layers (gating changes hidden states, which changes subsequent queries), and this repo also includes a deterministic tie-break `index_bias` inside the gating. So small changes can still show up.

## 3) What We Actually Ablated in `newtests.md`

The combined MRPC ablation in `newtests.md` sweeps:

- `top_k` (real sparsity knob): `{1, 2, 3, 4, 8}` plus `dense` (`top_k=128`)
- `grid_size` (in this repo, mostly `sigma_eff` / coordinate-extent): `{2, 4, 6, 8, 12, 16, 20}`

So the "grid sweep" is best described as:

- **an RBF bandwidth/temperature sweep via coordinate extent** (i.e., `sigma_eff` sweep),
not
- a "2×2×2 routing grid" or "reduced routing resolution".

## 4) Findings From the Combined MRPC Ablation (as recorded in `newtests.md`)

These are the observed outcomes from the combined table in `newtests.md`:

- Best accuracy observed in that run: `0.3229` (many settings tied).
- Some settings dropped to `0.3125` (one-sample difference at the run's sample granularity).
- Best speed among best-accuracy settings: `sparse, grid=2, top_k=3` with `0.75 samples/s` at `97.66%` sparsity.
- Dense baselines were around `0.69–0.70 samples/s` with the same tied accuracy (`0.3229`) for several grids.
- "No-grid" (`--disable-grid`) with `top_k=3` showed a small accuracy drop vs dense in the example run you pasted (`0.3125` vs `0.3229`) with only a small throughput gain.

Interpretation consistent with Section 2:

- `top_k` behaves like the real compute/sparsity knob.
- `grid_size` as currently implemented does not represent a change in discrete topology; it mainly modulates gating sharpness (`sigma_eff`) and can produce small differences via downstream dynamics and tie-breaking.

## 5) How to Describe This in a Paper (Suggested Wording)

Accurate phrasing for the current code:

- "We embed hidden dimensions on a canonical `16×16×16` lattice implied by `hidden_size=4096`."
- "`grid_size` rescales the coordinate extent of both query and center embeddings; this is equivalent to adjusting the effective RBF bandwidth (`sigma_eff`)."
- "Therefore, `grid_size` sweeps should be interpreted as a kernel-temperature/bandwidth ablation, not a routing-resolution ablation."

If the scientific goal is to ablate "spatial resolution/topology", the current design does not do that; it would require a different mapping (e.g., placing the **128 blocks** on an `L×L×L` lattice and varying `L`, or explicit quantization/aggregation of centers), rather than rescaling the same canonical lattice.

