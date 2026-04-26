from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MlpBlockBankLayout:
    layer_idx: int
    hidden_size: int
    intermediate_size: int
    block_size: int
    num_blocks: int

    def __post_init__(self) -> None:
        layer_idx = int(self.layer_idx)
        hidden_size = int(self.hidden_size)
        intermediate_size = int(self.intermediate_size)
        block_size = int(self.block_size)
        num_blocks = int(self.num_blocks)
        if layer_idx < 0:
            raise RuntimeError(f"layer_idx must be >= 0, got {layer_idx}")
        if hidden_size <= 0:
            raise RuntimeError(f"hidden_size must be > 0, got {hidden_size}")
        if intermediate_size <= 0:
            raise RuntimeError(f"intermediate_size must be > 0, got {intermediate_size}")
        if block_size <= 0:
            raise RuntimeError(f"block_size must be > 0, got {block_size}")
        if intermediate_size % block_size != 0:
            raise RuntimeError(
                f"intermediate_size={intermediate_size} is not divisible by block_size={block_size}"
            )
        expected_num_blocks = intermediate_size // block_size
        if num_blocks != expected_num_blocks:
            raise RuntimeError(
                f"num_blocks={num_blocks} does not match intermediate_size/block_size={expected_num_blocks}"
            )

    def block_bounds(self, block_idx: int) -> tuple[int, int]:
        index = int(block_idx)
        if index < 0 or index >= int(self.num_blocks):
            raise RuntimeError(f"block index out of range: {index} (num_blocks={int(self.num_blocks)})")
        start = index * int(self.block_size)
        stop = start + int(self.block_size)
        return start, stop

    def as_dict(self) -> dict[str, int]:
        return {
            "layer_idx": int(self.layer_idx),
            "hidden_size": int(self.hidden_size),
            "intermediate_size": int(self.intermediate_size),
            "block_size": int(self.block_size),
            "num_blocks": int(self.num_blocks),
        }


def build_intermediate_mlp_block_bank_layout(
    *,
    layer_idx: int,
    hidden_size: int,
    intermediate_size: int,
    block_size: int,
    num_blocks: int | None = None,
) -> MlpBlockBankLayout:
    block_size_i = int(block_size)
    intermediate_i = int(intermediate_size)
    inferred_num_blocks = intermediate_i // max(block_size_i, 1)
    return MlpBlockBankLayout(
        layer_idx=int(layer_idx),
        hidden_size=int(hidden_size),
        intermediate_size=intermediate_i,
        block_size=block_size_i,
        num_blocks=int(inferred_num_blocks if num_blocks is None else num_blocks),
    )


def validate_intermediate_mlp_block_bank_params(
    *,
    layout: MlpBlockBankLayout,
    gate_in_features: int,
    gate_out_features: int,
    up_in_features: int,
    up_out_features: int,
    down_in_features: int,
    down_out_features: int,
) -> None:
    if int(gate_in_features) != int(layout.hidden_size) or int(up_in_features) != int(layout.hidden_size):
        raise RuntimeError(
            f"Layer {int(layout.layer_idx)} block-bank mismatch: gate/up in_features must equal hidden_size="
            f"{int(layout.hidden_size)} (got gate={int(gate_in_features)}, up={int(up_in_features)})."
        )
    if int(gate_out_features) != int(layout.intermediate_size) or int(up_out_features) != int(layout.intermediate_size):
        raise RuntimeError(
            f"Layer {int(layout.layer_idx)} block-bank mismatch: gate/up out_features must equal intermediate_size="
            f"{int(layout.intermediate_size)} (got gate={int(gate_out_features)}, up={int(up_out_features)})."
        )
    if int(down_out_features) != int(layout.hidden_size) or int(down_in_features) != int(layout.intermediate_size):
        raise RuntimeError(
            f"Layer {int(layout.layer_idx)} block-bank mismatch: down_proj expected "
            f"out={int(layout.hidden_size)}, in={int(layout.intermediate_size)}; "
            f"got out={int(down_out_features)}, in={int(down_in_features)}."
        )
