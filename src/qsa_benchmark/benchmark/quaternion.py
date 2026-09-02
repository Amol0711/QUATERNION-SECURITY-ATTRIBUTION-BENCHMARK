from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def _canonical_frame_from_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Return a deterministic proper eigenframe for a real symmetric 3x3 matrix.

    Repeated eigenspaces are canonicalized by projecting the ordered Cartesian
    basis into each eigenspace and applying deterministic Gram--Schmidt.  This
    removes dependence on an arbitrary LAPACK basis within a repeated
    eigenspace.  Column signs and final orientation are then fixed canonically.
    """

    symmetric = np.asarray(matrix, dtype=float)
    if symmetric.shape != (3, 3):
        raise ValueError("matrix must have shape (3, 3)")
    symmetric = 0.5 * (symmetric + symmetric.T)

    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    spectral_scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    cluster_tolerance = 128.0 * np.finfo(float).eps * spectral_scale
    vector_tolerance = 256.0 * np.finfo(float).eps

    frame_columns: list[np.ndarray] = []
    start = 0
    while start < 3:
        stop = start + 1
        while stop < 3 and abs(eigenvalues[stop] - eigenvalues[start]) <= cluster_tolerance:
            stop += 1

        subspace = eigenvectors[:, start:stop]
        multiplicity = stop - start
        if multiplicity == 1:
            frame_columns.append(subspace[:, 0].copy())
        else:
            projector = subspace @ subspace.T
            local_basis: list[np.ndarray] = []
            candidates = [np.eye(3)[:, index] for index in range(3)]
            candidates.extend(subspace[:, index] for index in range(multiplicity))
            for candidate in candidates:
                vector = projector @ candidate
                for basis_vector in local_basis:
                    vector = vector - np.dot(basis_vector, vector) * basis_vector
                norm = float(np.linalg.norm(vector))
                if norm > vector_tolerance:
                    local_basis.append(vector / norm)
                if len(local_basis) == multiplicity:
                    break
            if len(local_basis) != multiplicity:
                raise RuntimeError("failed to canonicalize repeated eigenspace")
            frame_columns.extend(local_basis)
        start = stop

    frame = np.column_stack(frame_columns)
    for column in range(3):
        frame[:, column] /= np.linalg.norm(frame[:, column])

    for column in range(3):
        index = int(np.argmax(np.abs(frame[:, column])))
        if frame[index, column] < 0:
            frame[:, column] *= -1
    if np.linalg.det(frame) < 0:
        frame[:, -1] *= -1
    return frame


def _order_invariant_scatter(samples: np.ndarray) -> np.ndarray:
    """Return N*sum(xx^T)-sum(x)sum(x)^T using integer accumulation.

    The positive scalar factor relative to the covariance does not affect its
    eigenvectors.  Integer aggregation makes the matrix exactly invariant to
    pixel order before conversion of the final 3x3 matrix to floating point.
    """

    integer_samples = np.asarray(samples, dtype=np.int64).reshape(-1, 3)
    count = int(len(integer_samples))
    if count == 0:
        return np.zeros((3, 3), dtype=float)

    maximum_coordinate = int(np.max(np.abs(integer_samples)))
    safe_count = int(math.isqrt(np.iinfo(np.int64).max // max(1, maximum_coordinate**2)))
    if count <= safe_count:
        sums = integer_samples.sum(axis=0, dtype=np.int64)
        second_moment = integer_samples.T @ integer_samples
        scatter = count * second_moment - np.outer(sums, sums)
        return scatter.astype(float)

    # Very large images can exceed int64 in N*sum(xx^T).  Aggregate the nine
    # scalar entries with Python integers in that exceptional regime.
    sums_py = [sum(int(value) for value in integer_samples[:, index]) for index in range(3)]
    scatter_py = np.empty((3, 3), dtype=object)
    for row in range(3):
        for column in range(3):
            cross_sum = sum(
                int(x) * int(y)
                for x, y in zip(
                    integer_samples[:, row], integer_samples[:, column], strict=True
                )
            )
            scatter_py[row, column] = count * cross_sum - sums_py[row] * sums_py[column]
    return np.asarray(scatter_py, dtype=float)


def _order_invariant_second_moment(samples: np.ndarray) -> np.ndarray:
    """Return an exact-order-invariant 3x3 integer second-moment matrix."""

    integer_samples = np.asarray(samples, dtype=np.int64).reshape(-1, 3)
    if len(integer_samples) == 0:
        return np.zeros((3, 3), dtype=float)
    return (integer_samples.T @ integer_samples).astype(float)


def proper_frame_from_pca(image: np.ndarray) -> np.ndarray:
    samples = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
    scatter = _order_invariant_scatter(samples)
    return _canonical_frame_from_symmetric(scatter)


def random_so3(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def fixed_axis_rotation(angle: float = 0.73) -> np.ndarray:
    axis = np.array([1.0, 2.0, 3.0])
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle).as_matrix()


def caseii_rotation(t: float, scale: float = 1.0) -> np.ndarray:
    # The exact Case-II quaternion propagator q(t)=exp(e1 t)exp(-e3 t^2/2)
    # acts on pure quaternions through the corresponding composed SO(3) rotations.
    rx = Rotation.from_rotvec(np.array([2.0 * scale * t, 0.0, 0.0])).as_matrix()
    rz = Rotation.from_rotvec(np.array([0.0, 0.0, -scale * t * t])).as_matrix()
    return rx @ rz


def caseii_schedule(length: int, scale: float = 0.55, states: int = 8) -> list[np.ndarray]:
    if states < 2:
        raise ValueError("states must be at least two")
    times = np.linspace(0.15, 1.05, states)
    return [caseii_rotation(float(t), scale=scale) for t in times]


def curvature_frame(image: np.ndarray, scale: float = 0.55, states: int = 8) -> np.ndarray:
    samples = np.asarray(image, dtype=np.int64).reshape(-1, 3) - 128
    schedule = caseii_schedule(len(samples), scale=scale, states=states)
    gram = np.zeros((3, 3), dtype=float)
    for index, rotation in enumerate(schedule):
        block = samples[index::states]
        if len(block):
            second_moment = _order_invariant_second_moment(block)
            gram += rotation @ second_moment @ rotation.T
    return _canonical_frame_from_symmetric(gram)


def curvature_proxy(scale: float = 0.55, points: int = 256) -> float:
    times = np.linspace(0.15, 1.05, points)
    mats = np.stack([caseii_rotation(float(t), scale) for t in times])
    first = np.gradient(mats, axis=0)
    second = np.gradient(first, axis=0)
    return float(np.mean(np.linalg.norm(second, axis=(1, 2))))
