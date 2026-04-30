import torch
import bitsandbytes.functional as B
import sys, os
sys.path.insert(0, os.path.abspath('.'))
from llama3_neuroplastic.triton_sparse_mlp import triton_sparse_input_linear_4bit

def main():
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    device=torch.device('cuda')
    torch.manual_seed(0)

    out_features=96
    in_features=160
    block_size=32
    quant_block_size=64
    top_k=in_features//block_size

    w=torch.randn(out_features,in_features,device=device,dtype=torch.float16)
    # Match project contract: quantize transposed weight
    packed, qs = B.quantize_4bit(w.t().contiguous(), blocksize=quant_block_size, quant_type='nf4')
    absmax = qs.absmax
    if getattr(qs,'nested',False):
        absmax = B.dequantize_blockwise(qs.absmax, qs.state2) + qs.offset
    absmax = absmax.to(device=device,dtype=torch.float32).contiguous()
    code = qs.code.to(device=device,dtype=torch.float32).contiguous()

    # Dense reference from bitsandbytes decode
    dense_w = B.dequantize_4bit(packed, qs).t().contiguous().to(dtype=torch.float16)

    rows=2
    x=torch.randn(rows,in_features,device=device,dtype=torch.float16)
    # Activate all blocks in order
    active_idx = torch.arange(top_k, device=device, dtype=torch.int32).view(1,-1).repeat(rows,1).contiguous()

    y_sparse = triton_sparse_input_linear_4bit(
        x,
        active_idx,
        packed_weight=packed.view(-1),
        absmax=absmax.view(-1),
        code=code,
        out_features=out_features,
        in_features=in_features,
        quant_block_size=quant_block_size,
        bias=None,
        block_size=block_size,
        quant_weight_ref=None,
    )
    y_ref = x @ dense_w.t()

    diff = (y_sparse - y_ref).abs()
    print('max_abs_err', float(diff.max().item()))
    print('mean_abs_err', float(diff.mean().item()))
    if float(diff.max().item()) > 0.15:
        raise SystemExit('FAIL: possible NF4 layout/transposition mismatch')
    print('PASS: sparse-input 4bit path numerically consistent on non-square matrix')

if __name__=='__main__':
    main()
