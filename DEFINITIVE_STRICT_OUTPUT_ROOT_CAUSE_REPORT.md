# Definitive Strict-Output Root Cause Report

## Scope

This report answers one question only:

**Why are strict sparse-MLP outputs still repetitive / junk, and what fix is definitively required?**

The conclusion here is based on direct falsification tests, not training heuristics or speculation.

## Final conclusion

The strict-output bug is **not** primarily caused by:

- bad repetition-penalty heuristics
- sparse-attention paging
- the MLP bank path
- the original input-mask placement bug by itself
- the router simply picking the wrong blocks

The bug is this:

**The decode-critical dense MLP outputs are not sparse in the current `128 x 32` block basis at the budgets being used (`k=3`, and even far larger values).**

That means the current strict sparse-MLP scheme is asking the model to decode through a representation that does not exist in this basis at useful sparsity levels.

This was proven by testing the strongest possible oracle cases:

1. **Oracle routing** based on the dense teacher's actual block importance still failed.
2. **Exact dense MLP outputs**, masked down to top-`k` blocks, still failed.

If even exact dense outputs break when sparsified in this basis, then no routing tweak, no recalibration tweak, and no block-bank tweak can fix the current scheme at low `k`.

## What was tested

### 1. Baseline dense continuation

On the same prompt, dense hybrid baseline is coherent:

- prompt: `Describe a sparse MLP routing system.`
- dense continuation:
  - `The`
  - `system`
  - `has`
  - `a`

This confirms the base model/runtime is not the problem.

### 2. Placement fix was implemented and verified

I implemented post-gate sparse placements:

- `output_sparse`
- `intermediate_group`

Files changed:

- [sca_sparse_config.py](/c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/sca_sparse_config.py)
- [sca_sparse_mlp.py](/c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/sca_sparse_mlp.py)
- [neuroplastic_llama_gqa_mamba.py](/c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py)
- [run_hybrid_gqa_mamba_inference.py](/c:/Users/andre/Desktop/Overføre/next_step_ai/llama3_neuroplastic/run_hybrid_gqa_mamba_inference.py)

Unit validation passed:

- `36 passed`

### 3. Fast local placement probe

Artifact:

- [mlp_sparse_placement_probe.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/mlp_sparse_placement_probe.json)

Aggregate result across active layers:

- current input-mask formulation:
  - norm ratio `0.0025`
- input-mask, no output mask:
  - norm ratio `0.0137`
- output-side sparse down:
  - norm ratio `0.1462`
- intermediate-group mask:
  - norm ratio `0.1204`

This proves the old input-basis masking was structurally wrong.

### 4. Real strict probes after the placement fix

Artifacts:

- [strict_probe_output_sparse_v1.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/strict_probe_output_sparse_v1.json)
- [strict_probe_intermediate_group_v1.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/strict_probe_intermediate_group_v1.json)

Result:

- strict outputs were still junk / repetitive
- changing placement changed logits, but not enough to restore language

Example failure shapes:

- `879879 fame...`
- `492492erc...`
- `fame fame fame...`

Therefore:

- fixing placement was necessary
- but not sufficient

### 5. Oracle routing on the new placement

Artifact:

- [oracle_output_block_routing_probe.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/oracle_output_block_routing_probe.json)

Method:

- keep the new `output_sparse` formulation
- route using the dense teacher's actual top output blocks per layer/row

Result at `k=3`:

- still failed
- current route was bad
- oracle route was also bad

Example oracle continuation:

- `omik omik omik ...`

Therefore:

**the remaining failure is not simply "the router picked the wrong blocks."**

### 6. Oracle routing with larger `k`

Artifacts:

- [oracle_output_block_topk_sweep.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/oracle_output_block_topk_sweep.json)
- [oracle_output_block_topk_sweep_hi.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/oracle_output_block_topk_sweep_hi.json)

Method:

- use oracle output-block routing
- sweep `k`

Observed continuation quality:

- `k=3`: `omikomikomik...`
- `k=8`: `żżżż`
- `k=16`: `via via via via`
- `k=32`: `of  with with`
- `k=48`: `S of A`
- `k=64`: `1Wh`

This is important:

- even with perfect block choice, output-block sparsity at practical `k` values still breaks decode

### 7. Oracle intermediate-group routing

Artifact:

- [oracle_intermediate_group_topk_sweep.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/oracle_intermediate_group_topk_sweep.json)

Method:

- keep full input through `gate/up`
- sparsify intermediate activation groups
- choose groups using oracle teacher importance

Result:

- still failed across tested `k`
- examples:
  - `izardizard...`
  - `omikomikutenizard`

Therefore:

**moving the route target to intermediate groups also does not fix the core bug at low `k`.**

### 8. Strongest falsification: exact dense MLP output, then top-`k` mask

Artifacts:

- [dense_topk_mask_generation_probe.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/dense_topk_mask_generation_probe.json)
- [dense_topk_mask_generation_probe_hi.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/dense_topk_mask_generation_probe_hi.json)

Method:

- no sparse approximation at all
- compute the exact dense MLP output
- keep only the top-`k` output blocks
- feed that exact masked dense output into generation

This is the key test.

If this still breaks, then the bug is not approximation quality, not routing, and not training.

Observed results:

- `k=3`: broken
- `k=8`: broken
- `k=16`: broken
- `k=32`: broken
- `k=64`: broken
- `k=96`: first continuation that becomes recognizably English:
  - `What`
  - `is`
  - `the`
  - `difference`
- `k=128`: dense baseline:
  - `The`
  - `system`
  - `has`
  - `a`

This proves:

**The dense MLP output itself is not sparse enough in this block basis to support language decoding at low or moderate `k`.**

### 9. Layer-band isolation

Artifact:

- [dense_topk_mask_layerband_probe.json](/c:/Users/andre/Desktop/Overføre/next_step_ai/results/dense_topk_mask_layerband_probe.json)

Method:

- exact dense MLP output
- top-`k` mask with `k=3`
- apply masking only to:
  - lower active layers `0-11`
  - upper active layers `12-17`
  - all active layers `0-17`

Result:

- lower-only: still broken
- upper-only: still broken
- all-active: still broken

Examples:

- lower-only: `ñañańń`
- upper-only: `olateolateolateolate`

This proves:

**The issue is not confined to one small decode band. The current sparse block basis is not adequate for strict MLP sparsification across the active band at low `k`.**

## Definitive root cause

The current strict sparse-MLP runtime assumes that dense MLP outputs can be represented by a very small number of blocks in the current fixed residual/output block basis.

That assumption is false.

More precisely:

- the dense MLP output is too distributed across blocks
- keeping only a small number of blocks destroys token semantics
- this remains true even when:
  - the output is exact dense teacher output
  - the kept blocks are chosen by oracle teacher importance

So the bug is:

**Using low-`k` block sparsity on the current dense MLP output basis as a decode-time representation.**

That representation is invalid for language generation.

## Definitive fix

There is only one immediate fix that is proven by the experiments above:

### Immediate proven fix

**Do not use the current sparse MLP block-mask scheme in strict decode.**

In practice:

- disable sparse MLP in strict runtime
- keep dense MLP
- keep sparse attention if desired

Why this is definitive:

- dense baseline is coherent
- every tested low-`k` sparse MLP representation failed
- exact dense-top-`k` masking failed

So if the goal is "strict outputs must be correct," the current sparse MLP path must be removed from the decode path.

### What is not a real fix

These are **not** definitive fixes for the current scheme:

- retraining the same block-mask formulation harder
- changing the router only
- changing top-`k` from `3` to a modestly larger value
- using oracle routing only
- moving sparsity from input to output/intermediate while keeping the same low-`k` block representation

All of those were falsified directly.

## If MLP sparsity is still required

Then the current scheme must be replaced, not tuned.

The replacement must satisfy this requirement:

**The sparse representation must be learned in a basis where decode-critical MLP outputs are actually compressible.**

That means:

- not fixed top-`k` masking over the current `128 x 32` block basis
- not "keep a few blocks of the dense output"
- instead, a learned sparse substrate such as:
  - learned experts / MoE-style experts
  - learned low-rank sparse expansion basis
  - sparse adapters whose basis is trained for decode quality

But this is a redesign project, not a tweak to the current SCA block-mask runtime.

## Recommended action

### For immediate correctness

1. Remove sparse MLP from strict decode runtime.
2. Keep dense MLP.
3. Keep sparse attention if you still want attention-side savings.

### For future MLP sparsity work

1. Stop using the current fixed block-mask MLP representation for decode.
2. Start a new MLP sparsity design on a learned basis.
3. Re-evaluate sparsity there with autoregressive quality as the primary metric.

## Bottom line

The strict-output bug is not a small implementation mistake anymore.

It is a proven representational failure:

**the current low-`k` block-sparse MLP representation cannot carry language semantics in strict decode, even under oracle conditions.**

So the definitive fix is:

**remove this sparse MLP scheme from strict generation, or replace it with a different learned sparse representation.**
