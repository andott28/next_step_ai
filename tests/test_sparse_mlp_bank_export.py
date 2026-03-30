import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_DIR = os.path.join(ROOT, "llama3_neuroplastic")
if LLAMA_DIR not in sys.path:
    sys.path.insert(0, LLAMA_DIR)

from run_sparse_mlp_bank_export import _parse_layer_selection, _resolve_dtype  # noqa: E402


def test_parse_layer_selection_supports_ranges_and_lists():
    assert _parse_layer_selection("") is None
    assert _parse_layer_selection(None) is None
    assert _parse_layer_selection("8-10,12,14-15") == [8, 9, 10, 12, 14, 15]


def test_resolve_dtype_aliases():
    assert _resolve_dtype("float16") == torch.float16
    assert _resolve_dtype("fp16") == torch.float16
    assert _resolve_dtype("bfloat16") == torch.bfloat16
    assert _resolve_dtype("bf16") == torch.bfloat16
    assert _resolve_dtype("float32") == torch.float32
