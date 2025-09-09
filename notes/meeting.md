# Project Guide — flow-matching-for-time-series
Note: This top section is a living, tracked guide for onboarding and structure.

## Overview
- Problem: Surrogate modeling for fluid systems using flow matching and progressive distillation to accelerate inference while maintaining fidelity.
- Goals: Fast, high-quality next-state prediction; reduce solver steps via distillation; mitigate blurriness using Sobolev losses.
- Contributions:
  - Novel distillation method to directly learn macro-step evolution for faster generation.
  - Physics-informed Sobolev loss to reduce blur in fluid fields.

## Theory

### Main Argument (Why and What)
- Flow Matching (and diffusion) is well-suited to partially observed dynamical systems: it can plausibly fill in missing information and produce realistic successors.
- Deterministic regimes are, in principle, "straight-path" transports; a single explicit Euler step could suffice if the learned velocity exactly matches the ground truth.
- Practical issues that motivate our methods:
  1) Stochasticity and modeling error require multiple solver steps, reducing throughput versus single-step surrogates.
  2) L2 training often yields blurry fields; sharp fluid features are penalized too weakly by pixel-wise losses.
- Remedies: Distillation to collapse many steps into one; Sobolev losses to preserve sharp structures.

### Preliminaries (Notation and FM Objectives)

- Bridging state with prior and target: with $x\_0 \sim p\_0$, $x\_1 \sim p\_1$, $t \sim U[0,1]$,

$$
x\_t = t\, x\_1 + (1-t)\, x\_0.
$$

- Flow Matching (FM): learn a velocity field $v\_\theta$ by

$$
\min_{\theta} \; E[\, \| v\_\theta(x\_t, t) - (x\_1 - x\_0) \|\_2^2 \,], \quad \frac{dx}{dt} = v\_\theta(x, t),\; t \in [0,1].
$$

- Conditional FM (CFM) for time series: condition on previous state $y = x\_1^-$ to predict next $x\_1^+$; with $x\_t = t\, x\_1^+ + (1-t)\, x\_0$,

$$
\min_{\theta} \; E[\, \| v\_\theta(x\_t, y, t) - (x\_1^+ - x\_0) \|\_2^2 \,].
$$

### Distillation Methods (Two Paths)

1) Progressive Distillation (PD; teacher–student velocity)
- Idea: match a frozen teacher's $K$ fine steps with one macro step from a student.
- Sample $\Delta \in (0,1]$, $t \sim U[0,1-\Delta]$, construct $x\_t$ as above.
- Teacher rollout (no grad):

$$
x\_T = rollout\_K(v\_T, x\_t, y, t, \Delta)
$$

  K steps of $\Delta/K$.

- Student macro step:

$$
x\_S = step\_1(v\_S, x\_t, y, t, \Delta).
$$

- Losses:

$$
L\_{L2} = \| x\_T - x\_S \|\_2^2, \quad L\_{Sob} = \alpha \| e \|\_2^2 + \beta \| grad\; e \|\_2^2,\; e = x\_T - x\_S.
$$

- Outcome: a student that advances by macro increments with fewer inference steps.

2) Flow-level (Semigroup) Distillation (learn $F\_\delta$ directly)
