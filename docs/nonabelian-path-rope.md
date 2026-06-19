# Non-Abelian Path RoPE

## Motivation

Ordinary RoPE is excellent when the only geometric fact we need is relative
offset:

```text
G_t = exp(t A)
B(G_m q, G_n k) = B(q, G_(n-m) k)
```

This treats the path between token `m` and token `n` as a scalar distance. For
language, that may be too thin. Two spans can have the same length but very
different structure:

```text
adjective -> noun -> verb
noun -> relative clause -> verb
topic -> aside -> return
```

A non-abelian path representation keeps the order of intermediate events.

## Core Idea

Let each local step produce a group element:

```text
S_i in G
```

The prefix transform at token `t` is the ordered product:

```text
P_t = S_(t-1) ... S_1 S_0
```

Queries and keys are transformed by their prefix:

```text
q_t' = P_t q_t
k_t' = P_t k_t
```

If the group preserves the score pairing `B`, then:

```text
B(P_m q, P_n k) = B(q, P_m^-1 P_n k)
```

For orthogonal groups:

```text
P_m^-1 = P_m^T
```

The relative term is no longer a function of `n - m`; it is a learned ordered
path from `m` to `n`.

## Why Non-Abelian?

In an abelian group:

```text
AB = BA
```

Only the counts or total displacement matter. In a non-abelian group:

```text
AB != BA
```

So the model can distinguish:

```text
noun -> verb
verb -> noun
```

even when both paths have the same length and the same local pieces.

## Language Hypothesis

The non-abelian path can represent ordered composition signals such as:

- syntactic role transitions
- phrase and clause boundary crossings
- scope entry and scope exit
- discourse topic shifts
- semantic updates accumulated through a span

This suggests a model where some attention heads use ordinary RoPE for stable
relative distance, while a few exploratory heads use path-relative transforms.

## Minimal Attention Sketch

For head `h`:

```text
P_t^h = S_(t-1)^h ... S_0^h
score(m, n) = <P_m^h q_m, P_n^h k_n>
            = <q_m, (P_m^h)^T P_n^h k_n>
```

The local step can be produced from token/layer state:

```text
S_t^h = exp(A_t^h)
A_t^h = sum_j alpha_(t,j)^h B_j^h
```

where each basis generator `B_j` belongs to a Lie algebra such as `so(3)`,
`su(2)`, `so(p,q)`, or `u(p,q)`.

## Candidate Groups

### SO(3)

Good first prototype. Compact, stable, and visibly non-commutative.

### SU(2)

Quaternion-style rotations. Also compact and stable, with a natural relation to
3D rotations.

### SO(p, q)

More adventurous. Adds hyperbolic boosts and hierarchy-like scale behavior, but
needs normalization because Euclidean norms can grow.

### U(p, q)

The complex hyperbolic version. Potentially combines phase, hierarchy, and
ordered composition.

## Implementation Warnings

- Prefix products are sequential, so naive computation is not parallel-friendly.
- Compact groups are numerically safer for early experiments.
- Non-compact groups need rapidity clamps or normalization.
- If `S_t` depends on token content, causal training can compute prefixes
  left-to-right, but bidirectional encoders need a policy for both directions.
- The model may need a gate initialized near zero so ordinary RoPE remains the
  initial behavior.

## First Experiment

The current prototype uses toy `SO(3)` role rotations:

```text
noun     -> x-axis rotation
verb     -> y-axis rotation
modifier -> z-axis rotation
```

This is not a final language model design. It is a small algebraic test showing
that:

1. path-relative scoring preserves the expected invariant
2. the same steps in a different order produce a different relative transform

