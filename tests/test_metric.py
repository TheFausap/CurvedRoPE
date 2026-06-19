import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curved_rope.metric import (
    bilinear,
    euclidean_rope_2d,
    lorentz_boost_2d,
    matvec,
    metric_relative_error,
)


class MetricRoPETest(unittest.TestCase):
    def test_euclidean_rope_preserves_relative_position_identity(self) -> None:
        metric = [[1.0, 0.0], [0.0, 1.0]]
        q = [0.7, -1.2]
        k = [1.5, 0.25]

        error = metric_relative_error(
            q,
            k,
            metric,
            euclidean_rope_2d(3.0, theta=0.2),
            euclidean_rope_2d(11.0, theta=0.2),
            euclidean_rope_2d(8.0, theta=0.2),
        )

        self.assertLess(error, 1e-12)

    def test_lorentz_boost_preserves_relative_position_identity(self) -> None:
        metric = [[1.0, 0.0], [0.0, -1.0]]
        q = [0.7, -1.2]
        k = [1.5, 0.25]

        error = metric_relative_error(
            q,
            k,
            metric,
            lorentz_boost_2d(3.0, rapidity=0.05),
            lorentz_boost_2d(11.0, rapidity=0.05),
            lorentz_boost_2d(8.0, rapidity=0.05),
        )

        self.assertLess(error, 1e-12)

    def test_lorentz_metric_norm_is_preserved(self) -> None:
        metric = [[1.0, 0.0], [0.0, -1.0]]
        q = [0.7, -1.2]
        boosted = matvec(lorentz_boost_2d(2.5, rapidity=0.3), q)

        before = bilinear(q, metric, q)
        after = bilinear(boosted, metric, boosted)

        self.assertLess(abs(before - after), 1e-12)


if __name__ == "__main__":
    unittest.main()
