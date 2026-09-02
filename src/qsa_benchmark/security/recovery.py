"""Exact finite witnesses for recovery of reused static 3-D rotations."""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations, product
from typing import Iterable, Sequence

Scalar = Fraction
Vector3 = tuple[Scalar, Scalar, Scalar]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _f(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value, 1)


def as_vector3(value: Sequence[int | Fraction]) -> Vector3:
    if len(value) != 3:
        raise ValueError("expected a three-vector")
    return (_f(value[0]), _f(value[1]), _f(value[2]))


def cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def determinant(matrix: Matrix3) -> Fraction:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def transpose(matrix: Matrix3) -> Matrix3:
    return tuple(tuple(matrix[r][c] for r in range(3)) for c in range(3))  # type: ignore[return-value]


def inverse(matrix: Matrix3) -> Matrix3:
    det = determinant(matrix)
    if det == 0:
        raise ValueError("singular matrix")
    a, b, c = matrix
    cof = (
        (b[1] * c[2] - b[2] * c[1], -(b[0] * c[2] - b[2] * c[0]), b[0] * c[1] - b[1] * c[0]),
        (-(a[1] * c[2] - a[2] * c[1]), a[0] * c[2] - a[2] * c[0], -(a[0] * c[1] - a[1] * c[0])),
        (a[1] * b[2] - a[2] * b[1], -(a[0] * b[2] - a[2] * b[0]), a[0] * b[1] - a[1] * b[0]),
    )
    adj = transpose(cof)
    return tuple(tuple(value / det for value in row) for row in adj)  # type: ignore[return-value]


def matmul(left: Matrix3, right: Matrix3) -> Matrix3:
    right_t = transpose(right)
    return tuple(
        tuple(sum(left[i][k] * right_t[j][k] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


def matvec(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(sum(matrix[i][j] * vector[j] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def columns(*vectors: Vector3) -> Matrix3:
    if len(vectors) != 3:
        raise ValueError("three columns required")
    return tuple(tuple(vectors[c][r] for c in range(3)) for r in range(3))  # type: ignore[return-value]


def recover_rotation_from_two_correspondences(
    x1: Sequence[int | Fraction],
    x2: Sequence[int | Fraction],
    y1: Sequence[int | Fraction],
    y2: Sequence[int | Fraction],
) -> Matrix3:
    """Recover R exactly from y_i = R x_i for noncollinear x_1,x_2."""
    x1v, x2v, y1v, y2v = map(as_vector3, (x1, x2, y1, y2))
    x3, y3 = cross(x1v, x2v), cross(y1v, y2v)
    if x3 == (0, 0, 0) or y3 == (0, 0, 0):
        raise ValueError("correspondence pairs must be noncollinear")
    return matmul(columns(y1v, y2v, y3), inverse(columns(x1v, x2v, x3)))


def proper_signed_permutation_matrices() -> tuple[Matrix3, ...]:
    matrices: list[Matrix3] = []
    for perm in permutations(range(3)):
        for signs in product((-1, 1), repeat=3):
            rows = []
            for r in range(3):
                row = [Fraction(0) for _ in range(3)]
                row[perm[r]] = Fraction(signs[r])
                rows.append(tuple(row))
            matrix = tuple(rows)  # type: ignore[assignment]
            if determinant(matrix) == 1:
                matrices.append(matrix)  # type: ignore[arg-type]
    if len(matrices) != 24:
        raise AssertionError("expected 24 proper signed permutation rotations")
    return tuple(matrices)
