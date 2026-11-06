import torch

from bisurfgnn.data.batch import GraphBatch
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN
from bisurfgnn.models.compact_bisurf_additive import AdditiveCompactBiSurfGNN

ATOM_DIM = 62
CELL_DIM = 53
KWARGS = dict(
    atom_feature_dim=ATOM_DIM,
    cell_feature_dim=CELL_DIM,
    hidden_dim=64,
    attention_heads=4,
    dropout=0.1,
    reference_mode="observed_hsa",
    monotherapy_upper_bound=3.0,
    residual_monotherapy_input=True,
)


def _banks(unique_drugs: int = 5, atoms: int = 7, cells: int = 3, nodes: int = 6):
    torch.manual_seed(0)
    drugs = GraphBatch(
        node_features=torch.rand(unique_drugs, atoms, ATOM_DIM),
        adjacency=torch.rand(unique_drugs, atoms, atoms).round(),
        node_mask=torch.ones(unique_drugs, atoms, dtype=torch.bool),
    )
    context = GraphBatch(
        node_features=torch.rand(cells, nodes, CELL_DIM),
        adjacency=torch.rand(cells, nodes, nodes).round(),
        node_mask=torch.ones(cells, nodes, dtype=torch.bool),
    )
    return drugs, context


def _points(count: int = 11, unique_drugs: int = 5, cells: int = 3, dose_scale: float = 1.0):
    generator = torch.Generator().manual_seed(3)
    return (
        torch.randint(0, unique_drugs, (count,), generator=generator),
        torch.randint(0, unique_drugs, (count,), generator=generator),
        torch.randint(0, cells, (count,), generator=generator),
        (torch.rand(count, 2, generator=generator) + 0.2) * dose_scale,
        torch.rand(count, generator=generator) * 1.5,
        torch.rand(count, generator=generator) * 1.5,
        torch.ones(count, dtype=torch.bool),
    )


def _perturb_residual_head(model: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(7)
    head = model.interaction_decoder.residual_head
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * 0.5)
        head.bias.fill_(0.3)


def test_additive_initial_parameters_match_compact_bisurf() -> None:
    """The only difference from v2 is the output map, not the network or its initialisation."""
    torch.manual_seed(41)
    anchored = CompactBiSurfGNN(**KWARGS)
    torch.manual_seed(41)
    additive = AdditiveCompactBiSurfGNN(**KWARGS)
    expected, actual = anchored.state_dict(), additive.state_dict()
    assert expected.keys() == actual.keys()
    assert all(torch.equal(expected[name], actual[name]) for name in expected)


def test_additive_untrained_prediction_is_observed_hsa() -> None:
    """The zero-initialised residual head keeps the epoch -1 candidate equal to HSA."""
    drugs, context = _banks()
    a, b, cell, doses, mono_a, mono_b, mask = _points()
    model = AdditiveCompactBiSurfGNN(**KWARGS)

    output = model(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    assert torch.equal(output.prediction, torch.minimum(mono_a, mono_b))


def test_additive_output_adds_gated_decoder_output_to_reference() -> None:
    drugs, context = _banks()
    a, b, cell, doses, mono_a, mono_b, mask = _points()
    torch.manual_seed(41)
    anchored = CompactBiSurfGNN(**KWARGS).eval()
    torch.manual_seed(41)
    additive = AdditiveCompactBiSurfGNN(**KWARGS).eval()
    _perturb_residual_head(anchored)
    _perturb_residual_head(additive)

    old = anchored(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    new = additive(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    assert torch.equal(new.interaction_logit, old.interaction_logit)
    assert torch.equal(new.reference, old.reference)
    assert torch.allclose(new.prediction, new.reference + new.boundary_gate * new.interaction_logit, atol=1e-7)
    assert not torch.allclose(new.prediction, old.prediction, atol=1e-4)


def test_additive_output_is_exchange_symmetric_in_eval_mode() -> None:
    drugs, context = _banks()
    a, b, cell, doses, mono_a, mono_b, mask = _points()
    model = AdditiveCompactBiSurfGNN(**KWARGS).eval()
    _perturb_residual_head(model)

    direct = model(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    swapped = model(drugs, context, b, a, cell, doses.flip(-1), mono_b, mono_a, mask, mask)
    assert torch.allclose(direct.prediction, swapped.prediction, atol=1e-6)


def test_additive_output_is_not_clipped_to_the_monotherapy_bound() -> None:
    drugs, context = _banks()
    a, b, cell, doses, mono_a, mono_b, mask = _points(dose_scale=5.0)
    model = AdditiveCompactBiSurfGNN(**KWARGS).eval()
    with torch.no_grad():
        model.interaction_decoder.residual_head.bias.fill_(5.0)

    output = model(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    assert bool((output.prediction > 3.0).all())


def test_additive_gradient_reaches_the_residual_head() -> None:
    drugs, context = _banks()
    a, b, cell, doses, mono_a, mono_b, mask = _points()
    model = AdditiveCompactBiSurfGNN(**KWARGS)

    output = model(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    output.prediction.sum().backward()
    grad = model.interaction_decoder.residual_head.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
