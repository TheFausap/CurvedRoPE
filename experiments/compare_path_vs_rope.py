from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This experiment needs NumPy. Try the bundled runtime printed by "
        "`codex_app.load_workspace_dependencies`, or install numpy locally."
    ) from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from curved_rope.path import apply_path, role_rotation


ROLES = ("noun", "verb", "modifier")
ROLE_TO_ID = {role: index for index, role in enumerate(ROLES)}


@dataclass(frozen=True)
class Split:
    features: np.ndarray
    labels: np.ndarray


def unique_role_sequences(repeats_per_role: int = 3) -> list[tuple[str, ...]]:
    seed = tuple(role for role in ROLES for _ in range(repeats_per_role))
    return sorted(set(itertools.permutations(seed)))


def path_matrix(sequence: tuple[str, ...], angle: float = 0.45) -> np.ndarray:
    steps = [role_rotation(role, angle) for role in sequence]
    return np.array(apply_path(steps), dtype=np.float64)


def semantic_path_score(sequence: tuple[str, ...]) -> float:
    """Toy composition target: does the ordered path send a probe upward?"""

    probe = np.array([1.0, -0.35, 0.2], dtype=np.float64)
    direction = path_matrix(sequence) @ probe
    return float(direction[2])


def standard_rope_features(sequence: tuple[str, ...]) -> np.ndarray:
    length = len(sequence)
    frequencies = np.array([1.0, 0.25, 0.0625], dtype=np.float64)
    angles = length * frequencies
    return np.concatenate([np.cos(angles), np.sin(angles)])


def bag_plus_rope_features(sequence: tuple[str, ...]) -> np.ndarray:
    counts = np.zeros(len(ROLES), dtype=np.float64)
    for role in sequence:
        counts[ROLE_TO_ID[role]] += 1.0
    counts /= len(sequence)
    return np.concatenate([standard_rope_features(sequence), counts])


def nonabelian_path_features(sequence: tuple[str, ...]) -> np.ndarray:
    return path_matrix(sequence).reshape(-1)


def make_splits(
    feature_fn,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 7,
) -> tuple[Split, Split, Split]:
    sequences = unique_role_sequences()
    scores = np.array([semantic_path_score(sequence) for sequence in sequences])
    threshold = float(np.median(scores))
    labels = (scores > threshold).astype(np.float64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(sequences))

    train_end = int(train_fraction * len(indices))
    validation_end = int((train_fraction + validation_fraction) * len(indices))
    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    features = np.array([feature_fn(sequence) for sequence in sequences], dtype=np.float64)
    train_mean = features[train_indices].mean(axis=0, keepdims=True)
    train_std = features[train_indices].std(axis=0, keepdims=True)
    train_std = np.where(train_std < 1e-8, 1.0, train_std)
    features = (features - train_mean) / train_std

    return (
        Split(features[train_indices], labels[train_indices]),
        Split(features[validation_indices], labels[validation_indices]),
        Split(features[test_indices], labels[test_indices]),
    )


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def binary_cross_entropy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    eps = 1e-8
    p = np.clip(probabilities, eps, 1.0 - eps)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def accuracy(probabilities: np.ndarray, labels: np.ndarray) -> float:
    predictions = probabilities >= 0.5
    return float(np.mean(predictions == labels))


def train_logistic_probe(
    train: Split,
    *,
    steps: int = 700,
    learning_rate: float = 0.25,
    weight_decay: float = 1e-3,
) -> tuple[np.ndarray, float]:
    weights = np.zeros(train.features.shape[1], dtype=np.float64)
    bias = 0.0

    for _ in range(steps):
        logits = train.features @ weights + bias
        probabilities = sigmoid(logits)
        error = probabilities - train.labels
        grad_w = train.features.T @ error / len(train.labels) + weight_decay * weights
        grad_b = float(np.mean(error))
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    return weights, bias


def evaluate(name: str, feature_fn) -> dict[str, float | str]:
    train, validation, test = make_splits(feature_fn)
    weights, bias = train_logistic_probe(train)

    train_probabilities = sigmoid(train.features @ weights + bias)
    validation_probabilities = sigmoid(validation.features @ weights + bias)
    test_probabilities = sigmoid(test.features @ weights + bias)

    return {
        "name": name,
        "train_acc": accuracy(train_probabilities, train.labels),
        "val_acc": accuracy(validation_probabilities, validation.labels),
        "test_acc": accuracy(test_probabilities, test.labels),
        "test_loss": binary_cross_entropy(test_probabilities, test.labels),
    }


def main() -> None:
    print("Synthetic task: predict an order-dependent composed-path label")
    print("Dataset: all unique permutations of 3 nouns, 3 verbs, 3 modifiers")
    print("Length and bag-of-role counts are identical for every example.")
    print()

    results = [
        evaluate("standard_rope", standard_rope_features),
        evaluate("standard_rope_plus_bag", bag_plus_rope_features),
        evaluate("nonabelian_path_rope", nonabelian_path_features),
    ]

    print(f"{'model':<24} {'train':>8} {'val':>8} {'test':>8} {'test_loss':>10}")
    for row in results:
        print(
            f"{row['name']:<24} "
            f"{row['train_acc']:>8.3f} "
            f"{row['val_acc']:>8.3f} "
            f"{row['test_acc']:>8.3f} "
            f"{row['test_loss']:>10.3f}"
        )


if __name__ == "__main__":
    main()
