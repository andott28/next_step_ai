# Taylor Fluency Ablation

| Variant | distinct1_mean | rep_frac_mean | degenerate_frac | mean_tok_s |
|---|---:|---:|---:|---:|
| v0_dense_baseline | 1.000 | 0.000 | 0.000 | 1.618 |
| v1_plus_taylor_attention | 1.000 | 0.000 | 0.000 | 1.164 |
| v2_plus_sparse_routing | 1.000 | 0.000 | 0.000 | 0.801 |
| v3_plus_block_bank | 1.000 | 0.000 | 0.000 | 0.600 |
| v4_plus_router_checkpoint | 0.000 | 0.000 | 0.000 | 0.709 |

## Prompt 1
`Explain gravity briefly.`

| Variant | Continuation | degenerate |
|---|---|---:|
| v0_dense_baseline | ? | 0 |
| v1_plus_taylor_attention | Ex | 0 |
| v2_plus_sparse_routing | by | 0 |
| v3_plus_block_bank | The | 0 |
| v4_plus_router_checkpoint |  | 0 |

