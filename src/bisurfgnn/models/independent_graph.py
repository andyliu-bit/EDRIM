from __future__ import annotations

import torch
from torch import Tensor, nn

from ..data.batch import GraphBatch
from .baselines import BaselineOutput, _anchored_output, _validate_observed_monotherapy
from .graph_encoder import GraphEncoder
from .surface_head import DoseFeatures


class CompactIndependentPooledGNN(nn.Module):
    """Independent pooled-GNN baseline over compact, unique graph banks.

    Each drug or cell graph is encoded once per call. Point-level indices then
    gather those unique encodings for the dose-response rows. The two drug arms
    share all parameters and are combined with symmetric statistics, so swapping
    the aligned drug, dose, and monotherapy inputs leaves the prediction unchanged.
    """

    def __init__(
        self,
        atom_feature_dim: int,
        cell_feature_dim: int,
        hidden_dim: int,
        graph_layers: int = 2,
        dose_frequencies: int = 4,
        dropout: float = 0.0,
        reference_mode: str = "observed_hsa",
        monotherapy_upper_bound: float = 1.0,
    ) -> None:
        super().__init__()
        if reference_mode not in {"observed_product", "observed_hsa"}:
            raise ValueError(
                "reference_mode must be 'observed_product' or 'observed_hsa'"
            )
        if atom_feature_dim < 1 or cell_feature_dim < 1:
            raise ValueError("Graph feature dimensions must be positive")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.atom_feature_dim = int(atom_feature_dim)
        self.cell_feature_dim = int(cell_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.reference_mode = reference_mode
        self.monotherapy_upper_bound = float(monotherapy_upper_bound)

        self.drug_encoder = GraphEncoder(
            input_dim=self.atom_feature_dim,
            hidden_dim=hidden_dim,
            layers=graph_layers,
            dropout=dropout,
        )
        self.context_encoder = GraphEncoder(
            input_dim=self.cell_feature_dim,
            hidden_dim=hidden_dim,
            layers=graph_layers,
            dropout=dropout,
        )
        self.dose_features = DoseFeatures(dose_frequencies)
        arm_dim = hidden_dim + self.dose_features.output_dim + 1
        self.shared_arm = nn.Sequential(
            nn.LayerNorm(arm_dim),
            nn.Linear(arm_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.residual_network = nn.Sequential(
            nn.LayerNorm(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # The untrained model is exactly the chosen observed reference in
        # residual-logit space while retaining gradients for the first update.
        nn.init.zeros_(self.residual_network[-1].weight)
        nn.init.zeros_(self.residual_network[-1].bias)

    @staticmethod
    def _validate_graph_bank(
        graph: GraphBatch,
        *,
        name: str,
        expected_feature_dim: int,
    ) -> None:
        features = graph.node_features
        adjacency = graph.adjacency
        mask = graph.node_mask
        if features.ndim != 3 or adjacency.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                f"{name} shapes must be [G,N,F], [G,N,N], and [G,N]"
            )
        graph_count, nodes, feature_dim = features.shape
        if graph_count < 1:
            raise ValueError(f"{name} must contain at least one graph")
        if feature_dim != expected_feature_dim:
            raise ValueError(
                f"{name} expected feature dimension {expected_feature_dim}, got {feature_dim}"
            )
        if adjacency.shape != (graph_count, nodes, nodes) or mask.shape != (
            graph_count,
            nodes,
        ):
            raise ValueError(f"{name} component shapes do not agree")
        if not features.is_floating_point() or not adjacency.is_floating_point():
            raise TypeError(f"{name} features and adjacency must be floating-point tensors")
        if adjacency.dtype != features.dtype:
            raise TypeError(f"{name} features and adjacency must share a dtype")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} node_mask must be boolean")
        if not (features.device == adjacency.device == mask.device):
            raise ValueError(f"{name} components must be on the same device")
        if not bool(torch.isfinite(features).all()) or not bool(
            torch.isfinite(adjacency).all()
        ):
            raise ValueError(f"{name} features and adjacency must be finite")
        if not bool(mask.any(dim=1).all()):
            raise ValueError(f"Every graph in {name} must contain at least one valid node")

    @staticmethod
    def _validate_point_index(
        index: Tensor,
        *,
        name: str,
        point_count: int,
        graph_count: int,
        device: torch.device,
    ) -> None:
        if index.ndim != 1 or index.shape != (point_count,):
            raise ValueError(f"{name} must have shape [B]")
        if index.dtype != torch.long:
            raise TypeError(f"{name} must use torch.long indices")
        if index.device != device:
            raise ValueError(f"{name} must be on the same device as the graph banks")
        if bool((index < 0).any()) or bool((index >= graph_count).any()):
            raise IndexError(f"{name} contains an out-of-range graph index")

    @staticmethod
    def _validate_embedding_bank(
        embeddings: Tensor,
        *,
        name: str,
        expected_dim: int,
    ) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"{name} must have shape [G,H]")
        if embeddings.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one embedding")
        if embeddings.shape[1] != expected_dim:
            raise ValueError(
                f"{name} expected embedding dimension {expected_dim}, "
                f"got {embeddings.shape[1]}"
            )
        if not embeddings.is_floating_point():
            raise TypeError(f"{name} must be floating-point")
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError(f"{name} must be finite")

    def _validate_pooled_inputs(
        self,
        drug_embeddings: Tensor,
        context_embeddings: Tensor,
        drug_a_index: Tensor,
        drug_b_index: Tensor,
        cell_index: Tensor,
        doses: Tensor,
        mono_a: Tensor,
        mono_b: Tensor,
        mask_a: Tensor,
        mask_b: Tensor,
    ) -> None:
        self._validate_embedding_bank(
            drug_embeddings,
            name="drug_embeddings",
            expected_dim=self.hidden_dim,
        )
        self._validate_embedding_bank(
            context_embeddings,
            name="context_embeddings",
            expected_dim=self.hidden_dim,
        )
        if drug_embeddings.device != context_embeddings.device:
            raise ValueError("Drug and context embeddings must be on the same device")
        if drug_embeddings.dtype != context_embeddings.dtype:
            raise TypeError("Drug and context embeddings must share a dtype")

        _validate_observed_monotherapy(
            doses,
            mono_a,
            mono_b,
            mask_a,
            mask_b,
            self.monotherapy_upper_bound,
        )
        point_count = doses.shape[0]
        if point_count < 1:
            raise ValueError("A pooled batch must contain at least one point")
        embedding_device = drug_embeddings.device
        point_tensors = (
            doses,
            mono_a,
            mono_b,
            mask_a,
            mask_b,
        )
        if any(tensor.device != embedding_device for tensor in point_tensors):
            raise ValueError("Point tensors and embeddings must be on the same device")
        embedding_dtype = drug_embeddings.dtype
        if not doses.is_floating_point() or doses.dtype != embedding_dtype:
            raise TypeError("doses must be floating-point and match the embedding dtype")
        if (
            not mono_a.is_floating_point()
            or not mono_b.is_floating_point()
            or mono_a.dtype != embedding_dtype
            or mono_b.dtype != embedding_dtype
        ):
            raise TypeError(
                "Observed monotherapy responses must be floating-point and match "
                "the embedding dtype"
            )
        if mask_a.dtype != torch.bool or mask_b.dtype != torch.bool:
            raise TypeError("Observed monotherapy masks must be boolean")

        self._validate_point_index(
            drug_a_index,
            name="drug_a_index",
            point_count=point_count,
            graph_count=drug_embeddings.shape[0],
            device=embedding_device,
        )
        self._validate_point_index(
            drug_b_index,
            name="drug_b_index",
            point_count=point_count,
            graph_count=drug_embeddings.shape[0],
            device=embedding_device,
        )
        self._validate_point_index(
            cell_index,
            name="cell_index",
            point_count=point_count,
            graph_count=context_embeddings.shape[0],
            device=embedding_device,
        )

    def encode_drugs(self, drug_graphs: GraphBatch) -> Tensor:
        """Encode one compact drug graph bank to pooled embeddings."""

        self._validate_graph_bank(
            drug_graphs,
            name="drug_graphs",
            expected_feature_dim=self.atom_feature_dim,
        )
        return self.drug_encoder(drug_graphs).pooled

    def encode_context(self, cell_graphs: GraphBatch) -> Tensor:
        """Encode one compact cell/context graph bank to pooled embeddings."""

        self._validate_graph_bank(
            cell_graphs,
            name="cell_graphs",
            expected_feature_dim=self.cell_feature_dim,
        )
        return self.context_encoder(cell_graphs).pooled

    def _arm(self, drug: Tensor, dose: Tensor, mono_response: Tensor) -> Tensor:
        return self.shared_arm(
            torch.cat(
                [drug, self.dose_features(dose), mono_response.unsqueeze(-1)],
                dim=-1,
            )
        )

    def forward_pooled(
        self,
        drug_embeddings: Tensor,
        context_embeddings: Tensor,
        drug_a_index: Tensor,
        drug_b_index: Tensor,
        cell_index: Tensor,
        doses: Tensor,
        mono_a: Tensor,
        mono_b: Tensor,
        mask_a: Tensor,
        mask_b: Tensor,
    ) -> BaselineOutput:
        """Decode point rows from precomputed pooled graph embeddings.

        Callers may cache embeddings only while the encoder parameters and model
        mode remain fixed. This is intended for evaluation over multiple point
        chunks after a checkpoint has been selected.
        """

        self._validate_pooled_inputs(
            drug_embeddings,
            context_embeddings,
            drug_a_index,
            drug_b_index,
            cell_index,
            doses,
            mono_a,
            mono_b,
            mask_a,
            mask_b,
        )
        drug_a = drug_embeddings[drug_a_index]
        drug_b = drug_embeddings[drug_b_index]
        context = context_embeddings[cell_index]

        arm_a = self._arm(drug_a, doses[:, 0], mono_a)
        arm_b = self._arm(drug_b, doses[:, 1], mono_b)
        invariant = torch.cat(
            [arm_a + arm_b, torch.abs(arm_a - arm_b), arm_a * arm_b, context],
            dim=-1,
        )
        interaction_logit = self.residual_network(invariant).squeeze(-1)
        return _anchored_output(
            doses,
            mono_a,
            mono_b,
            mask_a,
            mask_b,
            interaction_logit,
            self.reference_mode,
            self.monotherapy_upper_bound,
        )

    def forward(
        self,
        drug_graphs: GraphBatch,
        cell_graphs: GraphBatch,
        drug_a_index: Tensor,
        drug_b_index: Tensor,
        cell_index: Tensor,
        doses: Tensor,
        mono_a: Tensor,
        mono_b: Tensor,
        mask_a: Tensor,
        mask_b: Tensor,
    ) -> BaselineOutput:
        drug_embeddings = self.encode_drugs(drug_graphs)
        context_embeddings = self.encode_context(cell_graphs)
        return self.forward_pooled(
            drug_embeddings,
            context_embeddings,
            drug_a_index,
            drug_b_index,
            cell_index,
            doses,
            mono_a,
            mono_b,
            mask_a,
            mask_b,
        )


__all__ = ["CompactIndependentPooledGNN"]
