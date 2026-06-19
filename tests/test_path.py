import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curved_rope.path import (
    apply_path,
    max_abs_difference,
    path_prefixes,
    path_relative_error,
    role_rotation,
)


class PathRoPETest(unittest.TestCase):
    def test_path_relative_identity_holds_for_orthogonal_prefixes(self) -> None:
        q = [0.4, -0.8, 1.2]
        k = [1.1, 0.3, -0.5]
        metric = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        roles = ["noun", "verb", "modifier", "noun", "verb", "modifier"]
        steps = [role_rotation(role, angle=0.12) for role in roles]
        prefixes = path_prefixes(steps, size=3)

        error = path_relative_error(q, k, metric, prefixes[1], prefixes[5])

        self.assertLess(error, 1e-12)

    def test_noncommuting_steps_make_order_visible(self) -> None:
        noun_then_verb = apply_path(
            [role_rotation("noun", 0.25), role_rotation("verb", 0.25)]
        )
        verb_then_noun = apply_path(
            [role_rotation("verb", 0.25), role_rotation("noun", 0.25)]
        )

        self.assertGreater(max_abs_difference(noun_then_verb, verb_then_noun), 1e-2)


if __name__ == "__main__":
    unittest.main()
