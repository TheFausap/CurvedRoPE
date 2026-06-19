"""CurvedRoPE research prototypes."""

from .metric import (
    bilinear,
    euclidean_rope_2d,
    lorentz_boost_2d,
    matmul,
    metric_relative_error,
)
from .path import (
    apply_path,
    path_prefixes,
    path_relative_error,
    relative_path_transform,
    role_rotation,
)

__all__ = [
    "apply_path",
    "bilinear",
    "euclidean_rope_2d",
    "lorentz_boost_2d",
    "matmul",
    "metric_relative_error",
    "path_prefixes",
    "path_relative_error",
    "relative_path_transform",
    "role_rotation",
]
