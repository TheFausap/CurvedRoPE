"""Small matrix tools for RoPE-like metric-preserving transforms.

The point of this module is clarity, not speed. It keeps the first experiments
free of heavy ML dependencies while we settle the algebra.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = Sequence[float]
Matrix = Sequence[Sequence[float]]


def dot(x: Vector, y: Vector) -> float:
    return sum(a * b for a, b in zip(x, y, strict=True))


def matvec(matrix: Matrix, vector: Vector) -> list[float]:
    return [dot(row, vector) for row in matrix]


def matmul(left: Matrix, right: Matrix) -> list[list[float]]:
    columns = list(zip(*right, strict=True))
    return [[dot(row, column) for column in columns] for row in left]


def transpose(matrix: Matrix) -> list[list[float]]:
    return [list(column) for column in zip(*matrix, strict=True)]


def bilinear(x: Vector, metric: Matrix, y: Vector) -> float:
    """Return x^T metric y."""

    return dot(x, matvec(metric, y))


def euclidean_rope_2d(position: float, theta: float = 1.0) -> list[list[float]]:
    """Ordinary 2D RoPE rotation."""

    angle = position * theta
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, -s], [s, c]]


def lorentz_boost_2d(position: float, rapidity: float = 1.0) -> list[list[float]]:
    """2D Lorentz boost preserving diag(1, -1)."""

    amount = position * rapidity
    c = math.cosh(amount)
    s = math.sinh(amount)
    return [[c, s], [s, c]]


def apply(transform: Matrix, vector: Vector) -> list[float]:
    return matvec(transform, vector)


def metric_relative_error(
    query: Vector,
    key: Vector,
    metric: Matrix,
    transform_m: Matrix,
    transform_n: Matrix,
    transform_delta: Matrix,
) -> float:
    """Compare B(G_m q, G_n k) with B(q, G_(n-m) k)."""

    left = bilinear(apply(transform_m, query), metric, apply(transform_n, key))
    right = bilinear(query, metric, apply(transform_delta, key))
    return abs(left - right)

