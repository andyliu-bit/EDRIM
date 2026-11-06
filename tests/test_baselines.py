from dataclasses import replace

import pytest
import torch

from bisurfgnn.config import DataConfig, ModelConfig
from bisurfgnn.data import SyntheticSurfaceDataset, collate_surface_samples, swap_drug_order
from bisurfgnn.models import (
    IndependentPooledGNN,
    MorganHallmarkResidualMLP,
    ObservedBlissReference,
    ObservedHSAReference,
    ObservedProductReference,
)


def _configs() -> tuple[DataConfig, ModelConfig]:
    return (
        DataConfig(
            num_drugs=8,
            num_cells=4,
            num_surfaces=12,
            dose_levels=(0.0, 0.5, 1.0),
            atom_feature_dim=10,
            cell_feature_dim=5,
            min_atoms=4,
            max_atoms=8,
            pathway_nodes=6,
            noise_std=0.0,
        ),
        ModelConfig(
            hidden_dim=16,
            graph_layers=2,
            attention_heads=4,
            dropout=0.0,
            dose_frequencies=3,
        ),
    )


def _batch():
    data_config, _ = _configs()
    dataset = SyntheticSurfaceDataset(data_config, seed=37)
    return collate_surface_samples(dataset.samples[:27])


def _assert_surface_contract(output, swapped_output) -> None:
    assert bool(torch.isfinite(output.prediction).all())
    assert bool(((output.prediction >= 0) & (output.prediction <= 1)).all())
    torch.testing.assert_close(
        output.prediction,
        swapped_output.prediction,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        output.reference,
        swapped_output.reference,
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(output.mono_a, swapped_output.mono_b)
    torch.testing.assert_close(output.mono_b, swapped_output.mono_a)
    boundary = output.boundary_gate == 0
    assert bool(boundary.any())
    torch.testing.assert_close(output.prediction[boundary], output.reference[boundary])


def _assert_finite_nonzero_parameter_gradients(model, prediction, target) -> None:
    loss = (prediction - target).square().mean()
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_observed_bliss_is_exact_symmetric_and_input_differentiable() -> None:
    batch = _batch()
    mono_a = batch.mono_response_a.detach().clone().requires_grad_(True)
    mono_b = batch.mono_response_b.detach().clone().requires_grad_(True)
    differentiable_batch = replace(
        batch,
        mono_response_a=mono_a,
        mono_response_b=mono_b,
    )
    baseline = ObservedBlissReference()

    output = baseline(differentiable_batch)
    swapped = baseline(swap_drug_order(differentiable_batch))

    product = ObservedProductReference()(differentiable_batch)
    hsa = ObservedHSAReference()(differentiable_batch)
    torch.testing.assert_close(product.prediction, output.prediction)
    torch.testing.assert_close(
        hsa.prediction,
        torch.minimum(mono_a, mono_b),
    )

    torch.testing.assert_close(output.prediction, mono_a * mono_b)
    _assert_surface_contract(output, swapped)
    output.prediction.sum().backward()
    assert mono_a.grad is not None and bool(torch.isfinite(mono_a.grad).all())
    assert mono_b.grad is not None and bool(torch.isfinite(mono_b.grad).all())


def test_observed_baselines_reject_missing_monotherapy_anchors() -> None:
    batch = _batch()
    missing = batch.mono_mask_a.clone()
    missing[0] = False
    incomplete = replace(batch, mono_mask_a=missing)

    with pytest.raises(ValueError, match="require both monotherapy anchors"):
        ObservedBlissReference()(incomplete)
    with pytest.raises(ValueError, match="require both monotherapy anchors"):
        ObservedHSAReference()(incomplete)


def test_morgan_hallmark_mlp_is_anchored_symmetric_and_differentiable() -> None:
    torch.manual_seed(41)
    batch = _batch()
    batch_size = batch.doses.shape[0]
    fingerprint_a = torch.rand(batch_size, 32)
    fingerprint_b = torch.rand(batch_size, 32)
    hallmark = torch.randn(batch_size, 12)
    model = MorganHallmarkResidualMLP(
        fingerprint_dim=32,
        hallmark_dim=12,
        hidden_dim=24,
        dose_frequencies=3,
        dropout=0.0,
    ).eval()

    output = model(
        fingerprint_a,
        fingerprint_b,
        hallmark,
        batch.doses,
        batch.mono_response_a,
        batch.mono_response_b,
        batch.mono_mask_a,
        batch.mono_mask_b,
    )
    swapped = model(
        fingerprint_b,
        fingerprint_a,
        hallmark,
        batch.doses.flip(dims=(-1,)),
        batch.mono_response_b,
        batch.mono_response_a,
        batch.mono_mask_b,
        batch.mono_mask_a,
    )

    _assert_surface_contract(output, swapped)
    _assert_finite_nonzero_parameter_gradients(model, output.prediction, batch.response)


def test_morgan_hallmark_mlp_supports_a_frozen_hsa_anchor() -> None:
    batch = _batch()
    batch_size = batch.doses.shape[0]
    model = MorganHallmarkResidualMLP(
        fingerprint_dim=16,
        hallmark_dim=10,
        hidden_dim=12,
        reference_mode="observed_hsa",
    ).eval()
    fingerprint_a = torch.rand(batch_size, 16)
    fingerprint_b = torch.rand(batch_size, 16)
    hallmark = torch.randn(batch_size, 10)

    output = model(
        fingerprint_a,
        fingerprint_b,
        hallmark,
        batch.doses,
        batch.mono_response_a,
        batch.mono_response_b,
        batch.mono_mask_a,
        batch.mono_mask_b,
    )
    swapped = model(
        fingerprint_b,
        fingerprint_a,
        hallmark,
        batch.doses.flip(dims=(-1,)),
        batch.mono_response_b,
        batch.mono_response_a,
        batch.mono_mask_b,
        batch.mono_mask_a,
    )

    torch.testing.assert_close(
        output.reference,
        torch.minimum(batch.mono_response_a, batch.mono_response_b),
    )
    _assert_surface_contract(output, swapped)


def test_independent_pooled_gnn_is_anchored_symmetric_and_differentiable() -> None:
    torch.manual_seed(43)
    data_config, model_config = _configs()
    batch = _batch()
    model = IndependentPooledGNN(data_config, model_config).eval()

    output = model(batch)
    swapped = model(swap_drug_order(batch))

    _assert_surface_contract(output, swapped)
    _assert_finite_nonzero_parameter_gradients(model, output.prediction, batch.response)

    hsa_model = IndependentPooledGNN(
        data_config,
        model_config,
        reference_mode="observed_hsa",
    ).eval()
    hsa_output = hsa_model(batch)
    torch.testing.assert_close(
        hsa_output.reference,
        torch.minimum(batch.mono_response_a, batch.mono_response_b),
    )
    torch.testing.assert_close(
        hsa_output.prediction,
        hsa_model(swap_drug_order(batch)).prediction,
        atol=2e-6,
        rtol=2e-6,
    )


def test_anchored_output_reduces_to_unit_interval_at_bound_one() -> None:
    """B=1 must reproduce the original unit-interval parameterisation exactly."""
    from bisurfgnn.models.baselines import _anchored_output, _safe_logit

    doses = torch.tensor([[0.7, 1.3], [0.0, 0.9], [1.1, 0.4]], dtype=torch.float32)
    mono_a = torch.tensor([0.8, 0.5, 0.95], dtype=torch.float32)
    mono_b = torch.tensor([0.6, 0.7, 0.30], dtype=torch.float32)
    mask = torch.ones(3, dtype=torch.bool)
    logit = torch.tensor([0.4, -1.2, 2.0], dtype=torch.float32)

    output = _anchored_output(doses, mono_a, mono_b, mask, mask, logit, "observed_hsa", 1.0)

    reference = torch.minimum(mono_a, mono_b)
    gate = (1.0 - torch.exp(-2.0 * doses[:, 0])) * (1.0 - torch.exp(-2.0 * doses[:, 1]))
    expected_shifted = torch.sigmoid(
        torch.log(reference.clamp(1e-6, 1.0 - 1e-6))
        - torch.log1p(-reference.clamp(1e-6, 1.0 - 1e-6))
        + gate * logit
    )
    expected = torch.where(gate == 0, reference, expected_shifted)
    assert torch.allclose(output.prediction, expected, atol=1e-7)


def test_anchored_output_admits_responses_above_one_when_bound_widens() -> None:
    """D1: with a wider declared bound the reference survives and predictions can exceed 1.

    Under the clipped endpoint the reference was squeezed into (0,1) by the logit
    and the sigmoid capped every prediction at 1, so an unclipped target above 1
    was unreachable by construction.
    """
    from bisurfgnn.models.baselines import _anchored_output

    doses = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    mono_a = torch.tensor([1.8, 1.2], dtype=torch.float32)
    mono_b = torch.tensor([1.5, 2.2], dtype=torch.float32)
    mask = torch.ones(2, dtype=torch.bool)
    logit = torch.tensor([1.5, 1.5], dtype=torch.float32)

    output = _anchored_output(doses, mono_a, mono_b, mask, mask, logit, "observed_hsa", 3.0)

    # The HSA reference itself is retained above 1 instead of being clamped.
    assert torch.allclose(output.reference, torch.tensor([1.5, 1.2]), atol=1e-6)
    # A positive interaction logit can now push the prediction above 1.
    assert bool((output.prediction > 1.0).all())
    assert bool((output.prediction < 3.0).all())

    # The same inputs under the old unit bound are rejected outright.
    with pytest.raises(ValueError, match=r"must lie in \[0,1.0\]"):
        _anchored_output(doses, mono_a, mono_b, mask, mask, logit, "observed_hsa", 1.0)


def test_morgan_mlp_carries_the_declared_bound_into_predictions() -> None:
    model = MorganHallmarkResidualMLP(
        fingerprint_dim=16,
        hallmark_dim=8,
        hidden_dim=32,
        reference_mode="observed_hsa",
        monotherapy_upper_bound=3.0,
    )
    assert model.monotherapy_upper_bound == 3.0
    fingerprint_a = torch.rand(4, 16)
    fingerprint_b = torch.rand(4, 16)
    hallmark = torch.rand(4, 8)
    doses = torch.rand(4, 2) + 0.5
    mono_a = torch.tensor([1.6, 0.4, 1.1, 0.9], dtype=torch.float32)
    mono_b = torch.tensor([1.2, 0.8, 1.9, 0.3], dtype=torch.float32)
    mask = torch.ones(4, dtype=torch.bool)

    output = model(
        fingerprint_a, fingerprint_b, hallmark, doses, mono_a, mono_b, mask, mask
    )
    assert torch.isfinite(output.prediction).all()
    assert bool((output.prediction <= 3.0).all())
    assert torch.allclose(output.reference, torch.minimum(mono_a, mono_b), atol=1e-6)
