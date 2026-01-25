"""Synthetic model family generator definitions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from generators.base_channels import (
    aklt_like_base,
    cluster_like_base,
    near_critical_base,
    periodic_base,
    random_gapped_base,
    reducible_block_base,
)
from generators.expand_tensor import compress_kraus


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kraus_count: int = 16
    reducible_only_dim: int | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "aklt_like": ModelSpec("aklt_like"),
    "cluster_like": ModelSpec("cluster_like", kraus_count=16),
    "random_gapped": ModelSpec("random_gapped", kraus_count=16),
    "near_critical": ModelSpec("near_critical", kraus_count=16),
    "periodic": ModelSpec("periodic", kraus_count=16),
    "reducible_block": ModelSpec("reducible_block", kraus_count=12, reducible_only_dim=16),
}


def list_models() -> list[str]:
    return list(MODEL_SPECS.keys())


def _base_ops_for(model: str) -> list[np.ndarray]:
    if model == "aklt_like":
        return aklt_like_base()
    if model == "cluster_like":
        return cluster_like_base()
    if model == "random_gapped":
        return random_gapped_base()
    if model == "near_critical":
        return near_critical_base()
    if model == "periodic":
        return periodic_base()
    if model == "reducible_block":
        return reducible_block_base()
    raise ValueError(f"Unknown model '{model}'")


def generate_kraus(model: str, bond_dim: int, seed: int) -> list[np.ndarray]:
    spec = MODEL_SPECS.get(model)
    if spec is None:
        raise ValueError(f"Unknown model '{model}'")
    if spec.reducible_only_dim is not None and bond_dim != spec.reducible_only_dim:
        raise ValueError(f"Model '{model}' only supports D={spec.reducible_only_dim}")
    base_ops = _base_ops_for(model)
    return compress_kraus(base_ops, bond_dim, seed, spec.kraus_count)
