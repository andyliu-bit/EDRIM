from dataclasses import replace

import pytest
import torch

from bisurfgnn.config import DataConfig, ModelConfig
from bisurfgnn.data import SyntheticSurfaceDataset, collate_surface_samples, swap_drug_order
from bisurfgnn.losses import BiSurfLoss
from bisurfgnn.models import BiSurfGNN


def _configs() -> tuple[DataConfig, ModelConfig]:
    data = DataConfig(
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
    )
    model = ModelConfig(
        hidden_dim=16,
        graph_layers=2,
        attention_heads=4,
        dropout=0.0,
        dose_frequencies=3,
    )
    return data, model


def test_full_model_is_symmetric_and_respects_zero_dose_boundaries() -> None:
    data_config, model_config = _configs()
    dataset = SyntheticSurfaceDataset(data_config, seed=5)
    batch = collate_surface_samples(dataset.samples[:27])
    model = BiSurfGNN(data_config, model_config).eval()

    output = model(batch)
    swapped = model(swap_drug_order(batch))
    torch.testing.assert_close(output.prediction, swapped.prediction, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output.reference, swapped.reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output.mono_a, swapped.mono_b, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(output.log_variance, swapped.log_variance, atol=2e-6, rtol=2e-6)

    boundary = output.boundary_gate == 0
    assert bool(boundary.any())
    torch.testing.assert_close(output.prediction[boundary], output.reference[boundary])
    dose_a_zero = batch.doses[:, 0] == 0
    dose_b_zero = batch.doses[:, 1] == 0
    torch.testing.assert_close(
        output.prediction[dose_a_zero], output.mono_b[dose_a_zero]
    )
    torch.testing.assert_close(
        output.prediction[dose_b_zero], output.mono_a[dose_b_zero]
    )
    assert bool(((output.prediction >= 0) & (output.prediction <= 1)).all())


def test_model_backward_has_finite_nonzero_gradients() -> None:
    data_config, model_config = _configs()
    dataset = SyntheticSurfaceDataset(data_config, seed=11)
    batch = collate_surface_samples(dataset.samples[:18])
    model = BiSurfGNN(data_config, model_config)
    loss = BiSurfLoss(monotherapy_weight=0.25, variance_weight=0.02)(model(batch), batch)
    loss.total.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)


def test_observed_monotherapy_reference_is_symmetric_and_respects_boundaries() -> None:
    data_config, model_config = _configs()
    dataset = SyntheticSurfaceDataset(data_config, seed=17)
    batch = collate_surface_samples(dataset.samples[:27])
    observed_a = torch.exp(-0.23 * batch.doses[:, 0])
    observed_b = torch.exp(-0.41 * batch.doses[:, 1])
    batch = replace(
        batch,
        mono_response_a=observed_a,
        mono_response_b=observed_b,
    )

    predicted_model = BiSurfGNN(data_config, model_config).eval()
    observed_model = BiSurfGNN(
        data_config,
        model_config,
        reference_mode="observed_monotherapy",
    ).eval()
    observed_model.load_state_dict(predicted_model.state_dict())

    predicted = predicted_model(batch)
    observed = observed_model(batch)
    swapped = observed_model(swap_drug_order(batch))

    torch.testing.assert_close(observed.reference, observed_a * observed_b)
    torch.testing.assert_close(observed.mono_a, predicted.mono_a)
    torch.testing.assert_close(observed.mono_b, predicted.mono_b)
    torch.testing.assert_close(observed.prediction, swapped.prediction, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(observed.reference, swapped.reference, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(observed.mono_a, swapped.mono_b, atol=2e-6, rtol=2e-6)

    boundary = observed.boundary_gate == 0
    dose_a_zero = batch.doses[:, 0] == 0
    dose_b_zero = batch.doses[:, 1] == 0
    assert bool(boundary.any())
    torch.testing.assert_close(observed.prediction[boundary], observed.reference[boundary])
    torch.testing.assert_close(observed.prediction[dose_a_zero], observed_b[dose_a_zero])
    torch.testing.assert_close(observed.prediction[dose_b_zero], observed_a[dose_b_zero])


def test_observed_monotherapy_reference_rejects_incomplete_masks() -> None:
    data_config, model_config = _configs()
    dataset = SyntheticSurfaceDataset(data_config, seed=23)
    batch = collate_surface_samples(dataset.samples[:9])
    incomplete_mask = batch.mono_mask_b.clone()
    incomplete_mask[0] = False
    batch = replace(batch, mono_mask_b=incomplete_mask)
    model = BiSurfGNN(
        data_config,
        model_config,
        reference_mode="observed_monotherapy",
    )

    with pytest.raises(ValueError, match="requires both monotherapy masks"):
        model(batch)


def test_model_rejects_unknown_reference_mode() -> None:
    data_config, model_config = _configs()

    with pytest.raises(ValueError, match="Unsupported reference_mode"):
        BiSurfGNN(data_config, model_config, reference_mode="unknown")  # type: ignore[arg-type]
