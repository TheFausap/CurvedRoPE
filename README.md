# CurvedRoPE

CurvedRoPE is a small research sandbox for asking whether rotary positional
embeddings can be generalized beyond ordinary Euclidean rotations.

Standard RoPE applies a position-dependent orthogonal/unitary transform to query
and key vectors. The attention score remains compatible with relative position
because the dot product between two transformed vectors depends on the relative
rotation:

```text
<R_m q, R_n k> = <q, R_(n-m) k>
```

This repo explores the broader pattern:

```text
pairing(G_m q, G_n k) = pairing(q, G_(n-m) k)
```

where `G_t` is a one-parameter group action and `pairing` may be Euclidean,
Hermitian, Lorentzian, symplectic, or induced by a curved manifold.

## First Research Questions

1. Can attention use metric-aware pairings instead of the Euclidean dot product?
2. Are indefinite spaces, such as Lorentzian geometry, useful for positional
   signals that include scale-like or causal structure?
3. Can complex hyperbolic or Mobius-style transformations provide richer phase
   behavior than unit complex rotations?
4. What constraints keep the resulting logits numerically stable?
5. Which variants preserve the key RoPE property that only relative position
   matters?

## Candidate Families

- Euclidean/unitary RoPE: ordinary block rotations and complex phases.
- Metric RoPE: transformations preserving a learned or fixed bilinear form.
- Lorentz RoPE: boosts in spaces with signature `(d-1, 1)`.
- Complex hyperbolic RoPE: transformations related to `U(p, q)` or `SU(1, 1)`.
- Symplectic RoPE: phase-space transformations preserving a skew form.
- Manifold transport RoPE: query/key vectors transported along geodesics before
  scoring.

## Repo Layout

- `docs/research-map.md`: mathematical framing and experiment plan.
- `docs/nonabelian-path-rope.md`: adventurous path-ordered RoPE direction.
- `docs/tiny-model-comparison.md`: first small comparison against standard RoPE.
- `src/curved_rope/`: dependency-light prototype code.
- `experiments/metric_rope_demo.py`: runnable invariant checks.
- `experiments/nonabelian_path_demo.py`: order-sensitive path RoPE demo.
- `experiments/compare_path_vs_rope.py`: optional NumPy comparison probe.
- `tests/`: regression tests for the algebraic invariants.

## Quick Start

```bash
python3 experiments/metric_rope_demo.py
python3 experiments/nonabelian_path_demo.py
python3 -m unittest
```
