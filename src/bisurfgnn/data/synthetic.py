from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..config import DataConfig
from .batch import GraphItem, SurfaceSample


def _connected_random_graph(
    nodes: int,
    feature_dim: int,
    generator: torch.Generator,
    edge_probability: float = 0.18,
) -> GraphItem:
    features = torch.randn(nodes, feature_dim, generator=generator)
    adjacency = torch.eye(nodes)
    for index in range(nodes - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0
    random_edges = torch.rand(nodes, nodes, generator=generator) < edge_probability
    random_edges = torch.triu(random_edges, diagonal=1)
    adjacency = torch.maximum(adjacency, random_edges.float() + random_edges.T.float())
    return GraphItem(node_features=features, adjacency=adjacency)


def _pathway_graph(nodes: int, feature_dim: int, generator: torch.Generator) -> GraphItem:
    features = torch.randn(nodes, feature_dim, generator=generator)
    adjacency = torch.eye(nodes)
    for index in range(nodes):
        neighbor = (index + 1) % nodes
        adjacency[index, neighbor] = 1.0
        adjacency[neighbor, index] = 1.0
    if nodes > 4:
        adjacency[0, nodes // 2] = 1.0
        adjacency[nodes // 2, 0] = 1.0
    return GraphItem(node_features=features, adjacency=adjacency)


def _safe_logit(value: Tensor) -> Tensor:
    value = value.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(value) - torch.log1p(-value)


class SyntheticSurfaceDataset(Dataset[SurfaceSample]):
    """Deterministic, learnable response surfaces for pipeline verification.

    The generator is not a biological simulator. It encodes the same invariances and
    zero-dose boundaries expected from the production model so implementation errors
    can be detected before real data are connected.
    """

    def __init__(self, config: DataConfig, seed: int = 17) -> None:
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(seed)

        self.drug_graphs = [
            _connected_random_graph(
                nodes=int(
                    torch.randint(
                        config.min_atoms,
                        config.max_atoms + 1,
                        (1,),
                        generator=generator,
                    ).item()
                ),
                feature_dim=config.atom_feature_dim,
                generator=generator,
            )
            for _ in range(config.num_drugs)
        ]
        self.cell_graphs = [
            _pathway_graph(config.pathway_nodes, config.cell_feature_dim, generator)
            for _ in range(config.num_cells)
        ]

        latent_dim = 8
        drug_projection = torch.randn(
            config.atom_feature_dim, latent_dim, generator=generator
        ) / math.sqrt(config.atom_feature_dim)
        cell_projection = torch.randn(
            config.cell_feature_dim, latent_dim, generator=generator
        ) / math.sqrt(config.cell_feature_dim)
        drug_latents = [
            torch.tanh(graph.node_features.mean(dim=0) @ drug_projection)
            for graph in self.drug_graphs
        ]
        cell_latents = [
            torch.tanh(graph.node_features.mean(dim=0) @ cell_projection)
            for graph in self.cell_graphs
        ]

        possible_surfaces = list(
            itertools.product(
                itertools.combinations(range(config.num_drugs), 2),
                range(config.num_cells),
            )
        )
        if config.num_surfaces > len(possible_surfaces):
            raise ValueError(
                f"Requested {config.num_surfaces} surfaces, but only "
                f"{len(possible_surfaces)} unique drug-pair/cell surfaces exist"
            )
        selection = torch.randperm(len(possible_surfaces), generator=generator)[
            : config.num_surfaces
        ].tolist()

        self.samples: list[SurfaceSample] = []
        for surface_id, selected_index in enumerate(selection):
            (drug_a_id, drug_b_id), cell_id = possible_surfaces[selected_index]
            latent_a = drug_latents[drug_a_id]
            latent_b = drug_latents[drug_b_id]
            latent_cell = cell_latents[cell_id]

            rate_a = 0.65 + 1.35 * torch.sigmoid(2.0 * (latent_a * latent_cell).mean())
            rate_b = 0.65 + 1.35 * torch.sigmoid(2.0 * (latent_b * latent_cell).mean())
            chemistry = (latent_a * latent_b * latent_cell).sum() / math.sqrt(latent_dim)
            context_alignment = ((latent_a + latent_b) * latent_cell).mean()
            interaction_strength = 1.8 * torch.tanh(chemistry + 0.4 * context_alignment)

            for dose_a, dose_b in itertools.product(
                config.dose_levels, config.dose_levels
            ):
                dose_a_tensor = torch.tensor(float(dose_a), dtype=torch.float32)
                dose_b_tensor = torch.tensor(float(dose_b), dtype=torch.float32)
                mono_a = torch.exp(-rate_a * dose_a_tensor)
                mono_b = torch.exp(-rate_b * dose_b_tensor)
                reference = mono_a * mono_b
                boundary_gate = (1.0 - torch.exp(-2.0 * dose_a_tensor)) * (
                    1.0 - torch.exp(-2.0 * dose_b_tensor)
                )
                dose_shape = 1.0 + 0.20 * torch.tanh(dose_a_tensor + dose_b_tensor)
                dose_shape = dose_shape + 0.08 * (dose_a_tensor - dose_b_tensor).square()
                noise = config.noise_std * torch.randn((), generator=generator) * boundary_gate
                response = torch.sigmoid(
                    _safe_logit(reference)
                    + boundary_gate * dose_shape * interaction_strength
                    + noise
                )
                self.samples.append(
                    SurfaceSample(
                        drug_a=self.drug_graphs[drug_a_id],
                        drug_b=self.drug_graphs[drug_b_id],
                        cell=self.cell_graphs[cell_id],
                        doses=torch.tensor([dose_a, dose_b], dtype=torch.float32),
                        mono_response_a=mono_a.float(),
                        mono_response_b=mono_b.float(),
                        mono_mask_a=torch.tensor(True),
                        mono_mask_b=torch.tensor(True),
                        response=response.float(),
                        surface_id=surface_id,
                        drug_a_id=drug_a_id,
                        drug_b_id=drug_b_id,
                        cell_id=cell_id,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SurfaceSample:
        return self.samples[index]
