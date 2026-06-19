"""Path-ordered non-abelian RoPE prototypes.

Ordinary RoPE uses one shared generator, so position is represented by
``G_t = exp(t A)``. This module explores the more language-shaped case where
each step may use a different group element:

``P_t = S_(t-1) ... S_1 S_0``.

For orthogonal steps, the relative score identity becomes:

``<P_m q, P_n k> = <q, P_m^T P_n k>``.

Unlike ordinary RoPE, ``P_m^T P_n`` depends on the ordered path between tokens,
not just the scalar distance ``n - m``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

from .metric import Matrix, Vector, bilinear, matmul, matvec, transpose


def identity(size: int) -> list[list[float]]:
    return [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]


def rotation_x(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def rotation_y(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def rotation_z(angle: float) -> list[list[float]]:
    c = math.cos(angle)
    s = math.sin(angle)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def role_rotation(role: str, angle: float) -> list[list[float]]:
    """Map a toy linguistic role to a non-commuting SO(3) rotation."""

    if role == "noun":
        return rotation_x(angle)
    if role == "verb":
        return rotation_y(angle)
    if role == "modifier":
        return rotation_z(angle)
    raise ValueError(f"Unknown role: {role}")


def path_prefixes(steps: Iterable[Matrix], size: int) -> list[list[list[float]]]:
    """Return prefix transforms ``P_t`` for sequentially applied steps."""

    prefix = identity(size)
    prefixes = [prefix]
    for step in steps:
        prefix = matmul(step, prefix)
        prefixes.append(prefix)
    return prefixes


def relative_path_transform(prefix_m: Matrix, prefix_n: Matrix) -> list[list[float]]:
    """Return the orthogonal relative transform ``P_m^T P_n``."""

    return matmul(transpose(prefix_m), prefix_n)


def path_relative_error(
    query: Vector,
    key: Vector,
    metric: Matrix,
    prefix_m: Matrix,
    prefix_n: Matrix,
) -> float:
    """Compare ``B(P_m q, P_n k)`` with ``B(q, P_m^T P_n k)``."""

    left = bilinear(matvec(prefix_m, query), metric, matvec(prefix_n, key))
    relative = relative_path_transform(prefix_m, prefix_n)
    right = bilinear(query, metric, matvec(relative, key))
    return abs(left - right)


def max_abs_difference(left: Matrix, right: Matrix) -> float:
    return max(
        abs(left_value - right_value)
        for left_row, right_row in zip(left, right, strict=True)
        for left_value, right_value in zip(left_row, right_row, strict=True)
    )


def apply_path(path: Sequence[Matrix]) -> list[list[float]]:
    """Collapse an ordered path into one transform."""

    if not path:
        raise ValueError("Path must contain at least one step")
    prefix = identity(len(path[0]))
    for step in path:
        prefix = matmul(step, prefix)
    return prefix

