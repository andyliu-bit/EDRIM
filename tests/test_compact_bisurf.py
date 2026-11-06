import pytest
import torch

from bisurfgnn.data.batch import GraphBatch
from bisurfgnn.independent_graph_benchmark import (
    BISURF_MODEL_SLUG,
    MODEL_FAMILIES,
    MODEL_SLUG,
    IndependentGraphConfig,
)
from bisurfgnn.models.compact_bisurf import CompactBiSurfGNN

ATOM_DIM = 62
CELL_DIM = 53


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


def _points(count: int = 11, unique_drugs: int = 5, cells: int = 3):
    generator = torch.Generator().manual_seed(3)
    return (
        torch.randint(0, unique_drugs, (count,), generator=generator),
        torch.randint(0, unique_drugs, (count,), generator=generator),
        torch.randint(0, cells, (count,), generator=generator),
        torch.rand(count, 2, generator=generator) + 0.2,
        torch.rand(count, generator=generator) * 1.5,
        torch.rand(count, generator=generator) * 1.5,
        torch.ones(count, dtype=torch.bool),
    )


def _model(**overrides) -> CompactBiSurfGNN:
    kwargs = dict(
        atom_feature_dim=ATOM_DIM,
        cell_feature_dim=CELL_DIM,
        hidden_dim=64,
        attention_heads=4,
        reference_mode="observed_hsa",
        monotherapy_upper_bound=3.0,
    )
    kwargs.update(overrides)
    return CompactBiSurfGNN(**kwargs)


def test_compact_bisurf_is_exchange_symmetric_in_eval_mode() -> None:
    """H1 requires the bidirectional exchange to be invariant to drug order."""
    drugs, context = _banks()
    a, b, cell, doses, mono_a, mono_b, mask = _points()
    model = _model().eval()

    direct = model(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    swapped = model(
        drugs, context, b, a, cell, doses.flip(-1), mono_b, mono_a, mask, mask
    )
    assert torch.allclose(direct.prediction, swapped.prediction, atol=1e-6)


def test_compact_bisurf_anchors_on_observed_hsa_and_can_exceed_one() -> None:
    """The D1 endpoint keeps ratios above 1, so the head must be able to reach them."""
    drugs, context = _banks()
    a, b, cell, doses, mono_a, mono_b, mask = _points()
    model = _model()

    output = model(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    assert torch.allclose(output.reference, torch.minimum(mono_a, mono_b), atol=1e-6)
    assert torch.isfinite(output.prediction).all()
    assert bool((output.prediction <= 3.0).all())

    # A monotherapy anchor above the declared bound is rejected, not truncated.
    with pytest.raises(ValueError, match=r"must lie in \[0,1.0\]"):
        _model(monotherapy_upper_bound=1.0)(
            drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask
        )


def test_compact_bisurf_deduplicates_pairs_without_changing_results() -> None:
    """Pair-level BiXT is deduplicated for cost; repeated pairs must still agree."""
    drugs, context = _banks()
    model = _model().eval()
    a = torch.tensor([0, 1, 0, 1, 0])
    b = torch.tensor([2, 3, 2, 3, 2])
    cell = torch.tensor([0, 0, 0, 0, 0])
    doses = torch.full((5, 2), 0.6)
    mono = torch.full((5,), 0.7)
    mask = torch.ones(5, dtype=torch.bool)

    output = model(drugs, context, a, b, cell, doses, mono, mono, mask, mask)
    # Rows 0, 2 and 4 are the same pair, cell and dose; rows 1 and 3 are another.
    assert torch.allclose(output.prediction[[0, 2, 4]], output.prediction[[0, 0, 0]], atol=1e-6)
    assert torch.allclose(output.prediction[[1, 3]], output.prediction[[1, 1]], atol=1e-6)


def test_bisurf_capacity_matches_the_frozen_control_band() -> None:
    """hidden_dim 136 keeps BiSurfGNN inside the same matched-capacity band."""
    model = _model(hidden_dim=136)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    target = 2_031_839
    assert abs(count - target) / target < 0.02


def test_graph_config_rejects_indivisible_attention_width() -> None:
    with pytest.raises(Exception, match="divisible by attention_heads"):
        IndependentGraphConfig(architecture=BISURF_MODEL_SLUG, hidden_dim=137, attention_heads=4)


def test_graph_config_defaults_to_the_pooled_architecture() -> None:
    """Existing configs must keep producing the pooled model and its family label."""
    config = IndependentGraphConfig()
    assert config.architecture == MODEL_SLUG
    assert MODEL_FAMILIES[config.architecture].endswith("no_cross_attention")
    assert MODEL_FAMILIES[BISURF_MODEL_SLUG].startswith("shared_drug_gnn_symmetric_bixt")


def test_compact_bisurf_starts_exactly_at_the_observed_anchor() -> None:
    """Phase A requires the untrained model to be the zero-parameter reference.

    The runner refuses to fit unless the initial prediction equals the anchor to
    within a tight tolerance, so that the epoch -1 candidate is genuinely the
    exact HSA reference and a fold is allowed to decline to learn.
    """
    drugs, context = _banks()
    a, b, cell, doses, mono_a, mono_b, mask = _points()
    model = _model().eval()

    output = model(drugs, context, a, b, cell, doses, mono_a, mono_b, mask, mask)
    assert torch.allclose(
        output.prediction, output.reference, rtol=2e-6, atol=2e-7
    )


@pytest.mark.parametrize("conditioned", [False, True])
def test_monotherapy_information_path_at_fixed_hsa(conditioned: bool) -> None:
    drugs, context = _banks()
    model = _model(residual_monotherapy_input=conditioned).eval()
    # A zero residual head would make any input test vacuous; activate it first.
    torch.nn.init.normal_(model.interaction_decoder.residual_head.weight, std=0.1)
    a, b, cell = torch.tensor([0]), torch.tensor([1]), torch.tensor([0])
    doses, ma = torch.tensor([[0.5, 0.7]]), torch.tensor([0.4])
    mb = torch.tensor([0.6], requires_grad=True)
    mask = torch.tensor([True])
    first = model(drugs, context, a, b, cell, doses, ma, mb, mask, mask)
    second = model(drugs, context, a, b, cell, doses, ma, torch.tensor([0.9]), mask, mask)
    torch.testing.assert_close(first.reference, second.reference)
    if conditioned:
        assert abs((first.interaction_logit - second.interaction_logit).item()) > 1e-7
        first.interaction_logit.sum().backward()
        assert mb.grad is not None and mb.grad.abs().item() > 1e-7
    else:
        torch.testing.assert_close(first.interaction_logit, second.interaction_logit, rtol=0, atol=0)


def test_conditioned_decoder_symmetry_boundary_and_nonzero_training_gradient() -> None:
    drugs, context = _banks()
    a, b, cell, doses, ma, mb, mask = _points()
    model = _model(residual_monotherapy_input=True).eval()
    torch.nn.init.normal_(model.interaction_decoder.residual_head.weight, std=0.1)
    doses[0, 0] = 0
    direct = model(drugs, context, a, b, cell, doses, ma, mb, mask, mask)
    swapped = model(drugs, context, b, a, cell, doses.flip(-1), mb, ma, mask, mask)
    torch.testing.assert_close(direct.prediction, swapped.prediction, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(direct.prediction[0], direct.reference[0], atol=0, rtol=0)
    direct.prediction.square().mean().backward()
    gradient = model.interaction_decoder.shared_dose_arm[1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient[:, -1].abs().sum() > 0


def test_legacy_default_weights_remain_compatible_and_new_version_is_explicit() -> None:
    legacy = _model(hidden_dim=136)
    explicit = _model(hidden_dim=136, residual_monotherapy_input=False)
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    assert sum(p.numel() for p in legacy.parameters()) == 2_049_542
    new = _model(hidden_dim=136, residual_monotherapy_input=True)
    assert sum(p.numel() for p in new.parameters()) == 2_049_816
    with pytest.raises(RuntimeError, match="size mismatch"):
        new.load_state_dict(legacy.state_dict(), strict=True)
    with pytest.raises(TypeError, match="boolean"):
        _model(residual_monotherapy_input="false")
