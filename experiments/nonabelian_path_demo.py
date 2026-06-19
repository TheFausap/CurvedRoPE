from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curved_rope.path import (
    apply_path,
    max_abs_difference,
    path_prefixes,
    path_relative_error,
    relative_path_transform,
    role_rotation,
)


def main() -> None:
    q = [0.4, -0.8, 1.2]
    k = [1.1, 0.3, -0.5]
    metric = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    roles = ["noun", "verb", "modifier", "noun", "verb", "modifier"]
    steps = [role_rotation(role, angle=0.12) for role in roles]
    prefixes = path_prefixes(steps, size=3)

    m = 1
    n = 5
    error = path_relative_error(q, k, metric, prefixes[m], prefixes[n])
    relative = relative_path_transform(prefixes[m], prefixes[n])

    noun_then_verb = apply_path(
        [role_rotation("noun", 0.25), role_rotation("verb", 0.25)]
    )
    verb_then_noun = apply_path(
        [role_rotation("verb", 0.25), role_rotation("noun", 0.25)]
    )
    order_gap = max_abs_difference(noun_then_verb, verb_then_noun)

    print(f"Path-relative score error:          {error:.3e}")
    print(f"Same steps, different order gap:    {order_gap:.3e}")
    print("Relative transform P_m^T P_n:")
    for row in relative:
        print("  " + " ".join(f"{value: .4f}" for value in row))


if __name__ == "__main__":
    main()

