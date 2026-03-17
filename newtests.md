## nr.1 failed results baseline vs neuroplastic

========================================
MINIMAL LIVE BENCHMARK SUMMARY Baseline
========================================
COLA       | 0.0820 (MCC)
SST2       | 0.5195 (Accuracy)
MRPC       | 0.6838 (Accuracy)
QQP        | 0.3280 (Accuracy)
STSB       | 0.0099 (Pearson)
MNLI       | 0.3310 (Accuracy)
QNLI       | 0.4740 (Accuracy)
RTE        | 0.5307 (Accuracy)
========================================

========================================
MINIMAL LIVE BENCHMARK SUMMARY Neuroplastic
========================================
COLA       | -0.0386 (MCC)
SST2       | 0.5069 (Accuracy)
MRPC       | 0.6250 (Accuracy)
QQP        | 0.3820 (Accuracy)
STSB       | -0.0532 (Pearson)
MNLI       | 0.3190 (Accuracy)
QNLI       | 0.4650 (Accuracy)
RTE        | 0.5415 (Accuracy)
========================================

## nr.2 failed results baseline vs neuroplastic
========================================
MINIMAL LIVE BENCHMARK SUMMARY Baseline
========================================
COLA       | 0.0127 (MCC)
SST2       | 0.5092 (Accuracy)
MRPC       | 0.6275 (Accuracy)
QQP        | 0.3610 (Accuracy)
STSB       | -0.0218 (Pearson)
MNLI       | 0.3310 (Accuracy)
QNLI       | 0.4660 (Accuracy)
RTE        | 0.5415 (Accuracy)
========================================

========================================
MINIMAL LIVE BENCHMARK SUMMARY Neuroplastic
========================================
COLA       | -0.0003 (MCC)
SST2       | 0.5092 (Accuracy)
MRPC       | 0.6324 (Accuracy)
QQP        | 0.3540 (Accuracy)
STSB       | -0.0217 (Pearson)
MNLI       | 0.3310 (Accuracy)
QNLI       | 0.4670 (Accuracy)
RTE        | 0.5487 (Accuracy)
========================================

## nr.3 + nr.4 MRPC SCA FLOP-saving grid/top-k ablation (combined)

source: `run_sca_flops_grid_ablation.py` (MRPC only)

**best accuracy observed (combined): `0.3229` (many settings tied)**  
**best speed among best-accuracy settings (combined): `sparse, grid=2, top_k=3`**  
**best combined pick: `sparse, grid=2, top_k=3` (`97.66%` sparsity, `0.75` samples/s, `+0.0000` dScore)**

combined rows from both runs (`grid=8,12,16,20` and `grid=2,4,6`):

==========================================================================================
SCA FLOP-SAVING + GRID ABLATION SUMMARY (MRPC)
==========================================================================================
Mode       Grid   TopK       Active%  Sparse%  Metric     Score    Sec       Samples/s  dScore   dS/s
sparse     8      1/128     0.78    99.22   Accuracy   0.3229   140.58    0.68       +0.0000  +0.11
sparse     8      2/128     1.56    98.44   Accuracy   0.3229   144.90    0.66       +0.0000  +0.09
sparse     8      3/128     2.34    97.66   Accuracy   0.3229   383.55    0.25       +0.0000  -0.32
sparse     8      4/128     3.12    96.88   Accuracy   0.3229   163.50    0.59       +0.0000  +0.02
sparse     8      8/128     6.25    93.75   Accuracy   0.3125   176.62    0.54       -0.0104  -0.03
sparse     12     1/128     0.78    99.22   Accuracy   0.3229   315.11    0.30       +0.0000  -0.26
sparse     12     2/128     1.56    98.44   Accuracy   0.3125   152.65    0.63       -0.0104  +0.06
sparse     12     3/128     2.34    97.66   Accuracy   0.3229   156.27    0.61       +0.0000  +0.05
sparse     12     4/128     3.12    96.88   Accuracy   0.3229   156.69    0.61       +0.0000  +0.05
sparse     12     8/128     6.25    93.75   Accuracy   0.3229   158.43    0.61       +0.0000  +0.04
sparse     16     1/128     0.78    99.22   Accuracy   0.3229   151.71    0.63       +0.0000  +0.09
sparse     16     2/128     1.56    98.44   Accuracy   0.3229   154.95    0.62       +0.0000  +0.08
sparse     16     3/128     2.34    97.66   Accuracy   0.3125   159.69    0.60       -0.0104  +0.06
sparse     16     4/128     3.12    96.88   Accuracy   0.3125   152.78    0.63       -0.0104  +0.09
sparse     16     8/128     6.25    93.75   Accuracy   0.3229   159.38    0.60       +0.0000  +0.06
sparse     20     1/128     0.78    99.22   Accuracy   0.3229   165.38    0.58       +0.0000  -0.01
sparse     20     2/128     1.56    98.44   Accuracy   0.3125   154.61    0.62       -0.0104  +0.03
sparse     20     3/128     2.34    97.66   Accuracy   0.3229   159.68    0.60       +0.0000  +0.01
sparse     20     4/128     3.12    96.88   Accuracy   0.3229   161.14    0.60       +0.0000  +0.01
sparse     20     8/128     6.25    93.75   Accuracy   0.3229   159.42    0.60       +0.0000  +0.01
dense      8      128/128     100.00  0.00    Accuracy   0.3229   168.02    0.57       n/a      n/a
dense      12     128/128     100.00  0.00    Accuracy   0.3229   169.48    0.57       n/a      n/a
dense      16     128/128     100.00  0.00    Accuracy   0.3229   178.48    0.54       n/a      n/a
dense      20     128/128     100.00  0.00    Accuracy   0.3229   162.63    0.59       n/a      n/a
sparse     2      1/128     0.78    99.22   Accuracy   0.3229   131.28    0.73       +0.0000  +0.04
sparse     2      2/128     1.56    98.44   Accuracy   0.3125   130.46    0.74       -0.0104  +0.05
**sparse     2      3/128     2.34    97.66   Accuracy   0.3229   128.66    0.75       +0.0000  +0.06**
sparse     2      4/128     3.12    96.88   Accuracy   0.3125   129.61    0.74       -0.0104  +0.05
sparse     2      8/128     6.25    93.75   Accuracy   0.3125   129.75    0.74       -0.0104  +0.05
sparse     4      1/128     0.78    99.22   Accuracy   0.3229   130.11    0.74       +0.0000  +0.04
sparse     4      2/128     1.56    98.44   Accuracy   0.3125   130.00    0.74       -0.0104  +0.04
sparse     4      3/128     2.34    97.66   Accuracy   0.3229   130.46    0.74       +0.0000  +0.04
sparse     4      4/128     3.12    96.88   Accuracy   0.3229   129.35    0.74       +0.0000  +0.04
sparse     4      8/128     6.25    93.75   Accuracy   0.3229   130.15    0.74       +0.0000  +0.04
sparse     6      1/128     0.78    99.22   Accuracy   0.3229   131.04    0.73       +0.0000  +0.04
sparse     6      2/128     1.56    98.44   Accuracy   0.3229   129.50    0.74       +0.0000  +0.05
sparse     6      3/128     2.34    97.66   Accuracy   0.3125   130.94    0.73       -0.0104  +0.04
sparse     6      4/128     3.12    96.88   Accuracy   0.3229   129.51    0.74       +0.0000  +0.05
sparse     6      8/128     6.25    93.75   Accuracy   0.3229   131.78    0.73       +0.0000  +0.04
dense      2      128/128     100.00  0.00    Accuracy   0.3229   139.52    0.69       n/a      n/a
dense      4      128/128     100.00  0.00    Accuracy   0.3229   137.35    0.70       n/a      n/a
dense      6      128/128     100.00  0.00    Accuracy   0.3229   138.81    0.69       n/a      n/a
==========================================================================================

# Fail
Setting                                                                       Loss        PPL      Tok/s    Step ms   VRAM GB
-----------------------------------------------------------------------------------------------------------------------------
np_on|spmm=dense|k=3|soft=0|inh=0.0000|ref=100|sig=1.0000                  13.9655 1161794.17      230.2     316.90      5.60    
np_on|spmm=dense|k=3|soft=1|inh=0.0000|ref=100|sig=1.0000                  13.9651 1161330.69      223.0     327.25      5.60    
np_on|spmm=dense|k=3|soft=1|inh=0.0000|ref=0|sig=1.0000                    13.9651 1161330.69      221.8     328.91      5.60    
np_on|spmm=cuda_spmm|k=3|soft=0|inh=0.0000|ref=0|sig=1.0000                13.9650 1161282.99       90.0     811.14      5.90    
np_on|spmm=cuda_spmm|k=3|soft=1|inh=0.0000|ref=0|sig=1.0000                13.9661 1162518.70       88.8     821.61      5.90    
np_on|spmm=cuda_spmm|k=3|soft=0|inh=0.0000|ref=100|sig=1.0000              13.9650 1161284.04       88.8     821.83      5.90    
np_on|spmm=cuda_spmm|k=3|soft=1|inh=0.0000|ref=100|sig=1.0000              13.9661 1162476.95       88.4     825.66      5.90  