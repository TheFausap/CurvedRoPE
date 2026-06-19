# Research Map

## Core Abstraction

RoPE is not just "rotate the vector." It is a representation of the additive
position group:

```text
t -> G_t = exp(t A)
```

For ordinary RoPE, `A` is block skew-symmetric, so `G_t` is orthogonal. In a
complex view, `A` is diagonal imaginary frequency and `G_t` is unitary phase
rotation.

The useful relative-position identity appears when the score pairing is
preserved by `G_t`:

```text
B(G_t x, G_t y) = B(x, y)
```

Then:

```text
B(G_m q, G_n k) = B(q, G_(n-m) k)
```

This suggests a general recipe:

1. Choose a vector space and pairing `B`.
2. Choose generators `A` that are skew with respect to `B`.
3. Apply `G_position = exp(position * A)` to queries and keys.
4. Score with the same pairing `B`, or project back to a Euclidean-compatible
   score if the model architecture requires it.

## Geometry Families

### 1. Euclidean and Complex Unitary

This is the baseline. `B(x, y) = x^T y`, and valid generators satisfy:

```text
A^T + A = 0
```

The complex equivalent uses a Hermitian pairing and skew-Hermitian generators.

### 2. Indefinite Metric / Lorentzian

Use a metric matrix `M` with mixed signs:

```text
B(x, y) = x^T M y
```

Valid generators satisfy:

```text
A^T M + M A = 0
```

This includes ordinary rotations and hyperbolic boosts. A boost can amplify one
light-cone direction while contracting another, which may encode directional or
causal distance differently from circular phases.

Risk: logits can become unstable if Euclidean norms grow while the indefinite
metric remains preserved.

### 3. Complex Hyperbolic

The complex version uses an indefinite Hermitian form:

```text
H(z, w) = z* J w
```

Transformations in `U(p, q)` preserve `H`. The `SU(1, 1)` case is closely tied
to automorphisms of the Poincare disk and could produce position-dependent
phase plus radial effects.

Risk: the model may need explicit normalization or bounded coordinates.

### 4. Symplectic

A symplectic form `Omega` is skew-symmetric:

```text
S^T Omega S = Omega
```

This is natural for phase-space transformations. It does not directly provide a
symmetric attention score, but it may be useful as an auxiliary phase or as part
of a mixed symmetric-plus-symplectic kernel.

### 5. Manifold Transport

Treat queries and keys as tangent vectors on a manifold. Position moves a base
point along a curve, and vectors are parallel transported before scoring.

This is the most geometrically faithful approach, but also the most expensive.
Good first cases are constant-curvature spaces: sphere, hyperbolic space, and
products of both.

### 6. Non-Abelian Path RoPE

Instead of representing position with one scalar-indexed group element, let each
token or local state produce a group element and accumulate an ordered prefix:

```text
P_t = S_(t-1) ... S_1 S_0
```

Then attention compares two tokens through the relative path:

```text
B(P_m q, P_n k) = B(q, P_m^-1 P_n k)
```

If the group is non-abelian, the ordered structure between the two tokens
matters. This makes it a candidate for syntax, scope, discourse moves, and
composition, rather than pure distance encoding. See
`docs/nonabelian-path-rope.md`.

## Initial Experiment Plan

1. Verify invariants for fixed metric-preserving transformations.
2. Compare relative-position identity under Euclidean dot product versus the
   correct metric-aware pairing.
3. Measure logit scale drift for Lorentz boosts.
4. Test whether non-abelian path products distinguish ordered synthetic
   language-like structures.
5. Implement PyTorch attention kernels once the algebraic prototypes are clear.
6. Train tiny language models or synthetic retrieval tasks to see whether any
   geometry improves extrapolation.

## Practical Constraints

- If scoring uses ordinary dot products, non-Euclidean transforms may lose the
  relative-position identity.
- If scoring uses indefinite metrics, logits can be negative or high magnitude
  in unfamiliar ways.
- A useful implementation probably needs careful normalization, per-head metric
  choices, and constraints on generator magnitude.
