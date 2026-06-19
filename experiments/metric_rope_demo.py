from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curved_rope.metric import (
    bilinear,
    euclidean_rope_2d,
    lorentz_boost_2d,
    matvec,
    metric_relative_error,
)


def main() -> None:
    q = [0.7, -1.2]
    k = [1.5, 0.25]

    euclidean_metric = [[1.0, 0.0], [0.0, 1.0]]
    lorentz_metric = [[1.0, 0.0], [0.0, -1.0]]

    m = 3.0
    n = 11.0

    euclidean_error = metric_relative_error(
        q,
        k,
        euclidean_metric,
        euclidean_rope_2d(m, theta=0.2),
        euclidean_rope_2d(n, theta=0.2),
        euclidean_rope_2d(n - m, theta=0.2),
    )

    lorentz_error = metric_relative_error(
        q,
        k,
        lorentz_metric,
        lorentz_boost_2d(m, rapidity=0.05),
        lorentz_boost_2d(n, rapidity=0.05),
        lorentz_boost_2d(n - m, rapidity=0.05),
    )

    boosted_q = matvec(lorentz_boost_2d(n, rapidity=0.05), q)

    print(f"Euclidean relative-position error: {euclidean_error:.3e}")
    print(f"Lorentz relative-position error:   {lorentz_error:.3e}")
    print(f"Lorentz metric norm before:         {bilinear(q, lorentz_metric, q):.6f}")
    print(f"Lorentz metric norm after:          {bilinear(boosted_q, lorentz_metric, boosted_q):.6f}")


if __name__ == "__main__":
    main()
