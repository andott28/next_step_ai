405B Transition Implementation Plan
This plan bridges the gap between the proven 8B architecture (Hybrid Performer Attention + Sparse MLPs) and the final goal: running 405B out-of-core on an 8GB GPU.

Phase 1: The Final 8B Validations
1. The Combined Smoke Test (Completed)
Goal: Verify that the Hybrid Performer Attention and the PCA-based Sparse MLPs play well together in the same generation loop without degrading the manifold. Status: [x] Validated. The stable 3-7 layer band runs with 0% manifold degeneration using the new Hybrid Performer backend.

2. Wide-Band Analytical Alignment (Layers 3-31) (In Progress)
Goal: Validate exact $O(L)$ manifold stability across the entire 8B depth stack. Deep SGD is structurally forbidden; we interleave PCA-initialized sparse manifolds with dense anchors to strictly bound drift linearly rather than exponentially.

System Invariants (
llama3_neuroplastic/neuroplastic_llama_gqa_mamba.py
):

 Inject dense_anchor_stride into 
_sparse_layer_enabled
 to enforce periodic dense manifold resets.
Zero-Shot Deployment Constraints: The PCA initialized geometry (learned_basis_init.pt) must evaluate zero-shot. To absorb $O(L)$ variance-drop across 29 routing layers without breaking the no-training constraint, the deployment enforces three analytical preconditions:

 Mean-Shift Integrity: Activate sca_output_compensation_bias to analytically zero the cumulative dropout shift natively at deployment.
 Layer-Adaptive Capacity: Ingest 
sca_layer_profile.json
 to parameterize variable 
basis_rank
 and 
top_k
 per-layer, mitigating upper-layer geometric collapse.
 Closed-Form Export: Restrict recalibration_mode to export_only (or a strict 1-step local_mlp_geometry bias translation) to irrevocably bypass SGD propagation.
## Phase 2: Out-of-Core 405B Scale
3. Layer-by-Layer SSD Streaming Harness
Goal: Run the massive 405B model without OOMing the 16-64GB system RAM or 8GB VRAM. Implementation:
Status: [x] Verified on 8B ("Oslo" confirmed).
- Build a new standalone script run_streaming_inference.py.
Keep only the embedding layer, lm_head, and the Taylor Layer Cache (which is fixed-size $O(1)$) in VRAM permanently.
For each generation step: load layer_N weights from the 4-bit Safetensors physically on disk, move to GPU, run the backend.step(), move hidden state to the next step, delete layer_N from memory, force garbage collection, and load layer_N+1.
Because Sparse MLPs only fetch 3 blocks, and Attention KV is fixed $O(1)$, the only limit is the NVMe PCIe read speed.
4. Out-of-Core PCA Initialization (405B Target)
Goal: Compute the learned_basis_init.pt for 405B without loading the 405B into RAM.
Implementation Dimensions:
- Hidden Size: 16384 (4x scaling)
- Num Blocks: 512 (4x scaling)
- Block Size: 32 (fixed)
- Basis Rank: 96 (fixed)
Process:
- Adapt 
init_learned_basis_from_dense_mlp.py
 to use the SSD Streaming Harness.
- Pipe a stream of texts through the first 405B layer, collect output activations, run PCA, save to disk. Discard activations, load the next layer, repeat.