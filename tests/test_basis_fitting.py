import pytest
import torch

from llama3_neuroplastic.basis_fitting import fit_layer_basis


@pytest.mark.parametrize("pca_method", ["lowrank", "incremental"])
def test_fit_layer_basis_returns_expected_shapes(pca_method: str) -> None:
    if pca_method == "incremental":
        pytest.importorskip("sklearn.decomposition")
    torch.manual_seed(0)
    rows = 192
    input_dim = 12
    hidden_size = 16
    basis_rank = 4
    block_size = 4
    x = torch.randn(rows, input_dim)
    y = x @ torch.randn(input_dim, hidden_size) + (0.01 * torch.randn(rows, hidden_size))

    fitted = fit_layer_basis(
        x,
        y,
        basis_rank=basis_rank,
        block_size=block_size,
        pca_method=pca_method,
        pca_batch_rows=32,
    )

    assert fitted["encoder_weight"].shape == (basis_rank, input_dim)
    assert fitted["encoder_bias"].shape == (basis_rank,)
    assert fitted["decoder_blocks"].shape == (hidden_size // block_size, basis_rank, block_size)
    assert fitted["decoder_bias"].shape == (hidden_size // block_size, block_size)
    assert fitted["rank_effective"] == basis_rank
    assert 0.0 <= float(fitted["explained_variance_ratio"]) <= 1.01
    assert fitted["pca_method"] == pca_method


def test_fit_layer_basis_auto_prefers_incremental_for_tall_matrices() -> None:
    pytest.importorskip("sklearn.decomposition")
    torch.manual_seed(1)
    x = torch.randn(4096, 8)
    y = x @ torch.randn(8, 16)

    fitted = fit_layer_basis(
        x,
        y,
        basis_rank=4,
        block_size=4,
        pca_method="auto",
        pca_batch_rows=128,
    )

    assert fitted["pca_method"] == "incremental"
