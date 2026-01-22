"""Physical-model MPS generators (AKLT, cluster, DMRG/TEBD-loaded inputs)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _mix_with_identity(kraus: list[np.ndarray], epsilon: float) -> list[np.ndarray]:
    if epsilon <= 0:
        return kraus
    dim = kraus[0].shape[0]
    mix = np.sqrt(epsilon) * np.eye(dim, dtype=complex)
    out = [np.sqrt(1 - epsilon) * op for op in kraus]
    out.append(mix)
    return out


def _block_diagonal_repeat(kraus: list[np.ndarray], scale: int) -> list[np.ndarray]:
    if scale <= 1:
        return kraus
    out: list[np.ndarray] = []
    for op in kraus:
        dim = op.shape[0]
        mat = np.zeros((dim * scale, dim * scale), dtype=complex)
        for idx in range(scale):
            start = idx * dim
            mat[start : start + dim, start : start + dim] = op
        out.append(mat)
    return out


def _load_payload_from_path(path: Path) -> Any:
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        if "kraus" in data:
            return data["kraus"]
        if "mps" in data:
            return data["mps"]
        if "tensors" in data:
            return data["tensors"]
        if "transfer" in data:
            return data["transfer"]
        if len(data.files) == 1:
            return data[data.files[0]]
        return {name: data[name] for name in data.files}
    payload = np.load(path, allow_pickle=True)
    if isinstance(payload, np.ndarray) and payload.dtype == object:
        if payload.shape == ():
            return payload.item()
        return list(payload)
    return payload


def _kraus_from_site_tensor(tensor: np.ndarray) -> list[np.ndarray]:
    if tensor.ndim != 3:
        raise ValueError("Expected a single site tensor with three indices.")
    dims = tensor.shape
    min_axis = int(np.argmin(dims))
    if min_axis == 0:
        return [tensor[idx] for idx in range(dims[0])]
    if min_axis == 1:
        return [tensor[:, idx, :] for idx in range(dims[1])]
    return [tensor[:, :, idx] for idx in range(dims[2])]


def _kraus_from_mps_payload(payload: Any) -> list[np.ndarray]:
    if isinstance(payload, dict):
        for key in ("kraus", "mps", "tensors", "sites"):
            if key in payload:
                payload = payload[key]
                break
        else:
            raise ValueError("Unsupported MPS payload dictionary; expected keys like 'kraus' or 'mps'.")

    if isinstance(payload, (list, tuple)):
        if not payload:
            raise ValueError("Empty MPS payload.")
        if all(isinstance(item, np.ndarray) and item.ndim == 2 for item in payload):
            return list(payload)
        if all(isinstance(item, np.ndarray) and item.ndim == 3 for item in payload):
            return _kraus_from_site_tensor(payload[0])
        raise ValueError("Unsupported MPS payload list format.")

    if isinstance(payload, np.ndarray):
        if payload.ndim == 2:
            return [payload]
        if payload.ndim == 3:
            return _kraus_from_site_tensor(payload)
        if payload.ndim == 4:
            return _kraus_from_site_tensor(payload[0])
        if payload.ndim == 1 and payload.dtype == object:
            return _kraus_from_mps_payload(list(payload))

    raise ValueError("Unsupported MPS payload; expected site tensors or Kraus operators.")


def _superop_to_choi(superop: np.ndarray) -> np.ndarray:
    if superop.ndim != 2 or superop.shape[0] != superop.shape[1]:
        raise ValueError("Transfer operator must be a square matrix.")
    dim = int(np.sqrt(superop.shape[0]))
    if dim * dim != superop.shape[0]:
        raise ValueError("Transfer operator size must be a perfect square.")
    tensor = superop.reshape(dim, dim, dim, dim, order="F")
    return tensor.transpose(0, 2, 1, 3).reshape(dim * dim, dim * dim, order="F")


def _kraus_from_superop(superop: np.ndarray, tol: float = 1e-12) -> list[np.ndarray]:
    choi = _superop_to_choi(superop)
    choi = (choi + choi.conj().T) / 2
    eigvals, eigvecs = np.linalg.eigh(choi)
    dim = int(np.sqrt(superop.shape[0]))
    kraus: list[np.ndarray] = []
    for idx, value in enumerate(eigvals):
        if value > tol:
            vec = eigvecs[:, idx] * np.sqrt(value)
            kraus.append(vec.reshape(dim, dim, order="F"))
    if not kraus:
        raise ValueError("No Kraus operators recovered from transfer operator.")
    return kraus


def mps_kraus(
    mps_input: list[np.ndarray] | np.ndarray | dict[str, Any] | str | Path,
    epsilon: float = 0.0,
    scale: int = 1,
) -> list[np.ndarray]:
    """Build Kraus operators from DMRG/TEBD MPS data (file path or in-memory)."""
    payload = _load_payload_from_path(Path(mps_input)) if isinstance(mps_input, (str, Path)) else mps_input
    kraus = _kraus_from_mps_payload(payload)
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)


def transfer_kraus(
    transfer_input: np.ndarray | str | Path,
    epsilon: float = 0.0,
    scale: int = 1,
    tol: float = 1e-12,
) -> list[np.ndarray]:
    """Recover Kraus operators from a transfer/superoperator matrix."""
    payload = _load_payload_from_path(Path(transfer_input)) if isinstance(transfer_input, (str, Path)) else transfer_input
    kraus = _kraus_from_superop(np.asarray(payload), tol=tol)
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)


def aklt_kraus(epsilon: float = 0.0, scale: int = 1) -> list[np.ndarray]:
    s2 = np.sqrt(2 / 3)
    s1 = np.sqrt(1 / 3)
    a_plus = s2 * np.array([[0, 1], [0, 0]], dtype=complex)
    a_zero = -s1 * np.array([[1, 0], [0, -1]], dtype=complex)
    a_minus = -s2 * np.array([[0, 0], [1, 0]], dtype=complex)
    kraus = [a_plus, a_zero, a_minus]
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)


def cluster_kraus(epsilon: float = 0.0, scale: int = 1) -> list[np.ndarray]:
    inv_sqrt2 = 1 / np.sqrt(2)
    a0 = inv_sqrt2 * np.array([[1, 0], [1, 0]], dtype=complex)
    a1 = inv_sqrt2 * np.array([[0, 1], [0, -1]], dtype=complex)
    kraus = [a0, a1]
    kraus = _block_diagonal_repeat(kraus, scale)
    return _mix_with_identity(kraus, epsilon)


def tfim_kraus(
    mps_input: list[np.ndarray] | np.ndarray | dict[str, Any] | str | Path,
    epsilon: float = 0.0,
    scale: int = 1,
) -> list[np.ndarray]:
    """TFIM Kraus operators from DMRG/TEBD MPS input."""
    return mps_kraus(mps_input, epsilon=epsilon, scale=scale)


def xxz_kraus(
    mps_input: list[np.ndarray] | np.ndarray | dict[str, Any] | str | Path,
    epsilon: float = 0.0,
    scale: int = 1,
) -> list[np.ndarray]:
    """XXZ Kraus operators from DMRG/TEBD MPS input."""
    return mps_kraus(mps_input, epsilon=epsilon, scale=scale)


def j1j2_kraus(
    mps_input: list[np.ndarray] | np.ndarray | dict[str, Any] | str | Path,
    epsilon: float = 0.0,
    scale: int = 1,
) -> list[np.ndarray]:
    """J1-J2 Kraus operators from DMRG/TEBD MPS input."""
    return mps_kraus(mps_input, epsilon=epsilon, scale=scale)


def bose_hubbard_kraus(
    mps_input: list[np.ndarray] | np.ndarray | dict[str, Any] | str | Path,
    epsilon: float = 0.0,
    scale: int = 1,
) -> list[np.ndarray]:
    """Bose-Hubbard Kraus operators from DMRG/TEBD MPS input."""
    return mps_kraus(mps_input, epsilon=epsilon, scale=scale)


def kitaev_chain_kraus(
    mps_input: list[np.ndarray] | np.ndarray | dict[str, Any] | str | Path,
    epsilon: float = 0.0,
    scale: int = 1,
) -> list[np.ndarray]:
    """Kitaev chain Kraus operators from DMRG/TEBD MPS input."""
    return mps_kraus(mps_input, epsilon=epsilon, scale=scale)
