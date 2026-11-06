import torch

from bisurfgnn.models.symmetric_bixt import SymmetricBiXT


def _assert_valid_close(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> None:
    torch.testing.assert_close(left[mask], right[mask], atol=1e-6, rtol=1e-6)


def test_bixt_padding_invariance_and_drug_exchange_equivariance() -> None:
    torch.manual_seed(3)
    module = SymmetricBiXT(hidden_dim=16, num_heads=4, dropout=0.0).eval()
    tokens_a = torch.randn(2, 5, 16)
    tokens_b = torch.randn(2, 7, 16)
    mask_a = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    mask_b = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0]], dtype=torch.bool
    )

    output = module(tokens_a, tokens_b, mask_a, mask_b, return_attention=True)
    changed_a = tokens_a.clone()
    changed_b = tokens_b.clone()
    changed_a[~mask_a] = 1000.0 * torch.randn_like(changed_a[~mask_a])
    changed_b[~mask_b] = 1000.0 * torch.randn_like(changed_b[~mask_b])
    changed = module(changed_a, changed_b, mask_a, mask_b)

    _assert_valid_close(output.tokens_a, changed.tokens_a, mask_a)
    _assert_valid_close(output.tokens_b, changed.tokens_b, mask_b)
    torch.testing.assert_close(output.pair, changed.pair, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(output.tokens_a[~mask_a]) == 0
    assert torch.count_nonzero(output.tokens_b[~mask_b]) == 0

    swapped = module(tokens_b, tokens_a, mask_b, mask_a, return_attention=True)
    _assert_valid_close(output.tokens_a, swapped.tokens_b, mask_a)
    _assert_valid_close(output.tokens_b, swapped.tokens_a, mask_b)
    torch.testing.assert_close(output.pair, swapped.pair, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        output.attention_ab,
        swapped.attention_ba,
        atol=1e-6,
        rtol=1e-6,
    )


def test_bixt_rejects_empty_drug() -> None:
    module = SymmetricBiXT(hidden_dim=8, num_heads=2)
    tokens = torch.randn(1, 3, 8)
    valid = torch.ones(1, 3, dtype=torch.bool)
    empty = torch.zeros(1, 3, dtype=torch.bool)
    try:
        module(tokens, tokens, empty, valid)
    except ValueError as error:
        assert "valid atom" in str(error)
    else:
        raise AssertionError("An empty molecular graph must be rejected")

