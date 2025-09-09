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
- Deterministic regimes are, in principle, “straight-path” transports; a single explicit Euler step could suffice if the learned velocity exactly matches the ground truth.
- Practical issues that motivate our methods:
  1) Stochasticity and modeling error require multiple solver steps, reducing throughput versus single-step surrogates.
  2) L2 training often yields blurry fields; sharp fluid features are penalized too weakly by pixel-wise losses.
- Remedies: Distillation to collapse many steps into one; Sobolev losses to preserve sharp structures.

### Preliminaries (Notation and FM Objectives)
- Bridging state with prior and target: with $\bm x_0 \sim p_0$, $\bm x_1 \sim p_1$, $\bm t \sim \mathcal U[0,1]$,

  $$
  \bm x_t = \bm t\, \bm x_1 + (1-\bm t)\, \bm x_0.
  $$

- Flow Matching (FM): learn a velocity field $v_\theta$ by

  $$
  \min_{\theta} \; \mathbb E\big[\, \lVert v_\theta(\bm x_t, t) - (\bm x_1 - \bm x_0) \rVert_2^2 \,\big],
  \qquad \frac{d\bm x}{dt} = v_\theta(\bm x, t),\; t \in [0,1].
  $$

- Conditional FM (CFM) for time series: condition on previous state $\bm y = \bm x_1^-$ to predict next $\bm x_1^+$; with $\bm x_t = \bm t\, \bm x_1^+ + (1-\bm t)\, \bm x_0$,

  $$
  \min_{\theta} \; \mathbb E\big[\, \lVert v_\theta(\bm x_t, \bm y, t) - (\bm x_1^+ - \bm x_0) \rVert_2^2 \,\big].
  $$

### Distillation Methods (Two Paths)

1) Progressive Distillation (PD; teacher–student velocity)
- Idea: match a frozen teacher’s $K$ fine steps with one macro step from a student.
- Sample $\Delta \in (0,1]$, $t \sim \mathcal U[0,1-\Delta]$, construct $\bm x_t$ as above.
- Teacher rollout (no grad):

  $$
  \bm x_T = \operatorname{rollout}_K\big(v_T, \bm x_t, \bm y, t, \Delta\big)\quad (K\text{ steps of }\Delta/K).
  $$

- Student macro step:

  $$
  \bm x_S = \operatorname{step}_1\big(v_S, \bm x_t, \bm y, t, \Delta\big).
  $$

- Losses:

  $$
  L_{\text{L2}} = \lVert \bm x_T - \bm x_S \rVert_2^2, \qquad
  L_{\text{Sob}} = \alpha \lVert \bm e \rVert_2^2 + \beta \lVert \nabla \bm e \rVert_2^2,\; \bm e = \bm x_T - \bm x_S.
  $$

- Outcome: a student that advances by macro increments with fewer inference steps.

2) Flow-level (Semigroup) Distillation (learn $F_\delta$ directly)
- Aim: learn a flow map $F_\delta(\bm x_t, t)$ that obeys three properties: identity ($F_0(\bm x_t,t)=\bm x_t$), velocity consistency ($\tfrac{\partial}{\partial \delta}F_\delta|_{\delta=0}=v_t$), and semigroup ($F_{a+b}=F_a\!\circ F_b$ with time shift).
- Parameterization (Euler baseline + $\delta^2$ correction):

  $$
  F_\delta^{\xi}(\bm x_t, t) = \bm x_t + \delta\, v_t(\bm x_t) + \delta^2\,\big(\phi_\delta^{\xi}(\bm x_t, t) - v_t(\bm x_t)\big).
  $$

- Semigroup-consistency objective with stop-grad on the RHS target:

  $$
  \min_{\xi}\; \mathbb E\Big[\, \big\lVert F_\delta^{\xi}(\bm x_t, t) - \operatorname{sg}\big(F_{\delta/2}^{\xi}(\,F_{\delta/2}^{\xi}(\bm x_t, t),\, t+\tfrac{\delta}{2}\,)\big) \big\rVert_2^2 \,\Big].
  $$

  Flow-map properties and why they matter (with equations):

  1) Identity at zero increment

  $$
  F_0(\bm x_t, t) = \bm x_t.
  $$

  Advancing by zero time changes nothing. Our parameterization enforces this exactly since plugging $\delta=0$ into

  $$
  F_\delta^{\xi}(\bm x_t, t) = \bm x_t + \delta\, v_t(\bm x_t) + \delta^2 \big(\phi_\delta^{\xi}(\bm x_t, t) - v_t(\bm x_t)\big)
  $$

  yields $F_0^{\xi}(\bm x_t, t) = \bm x_t$.

  2) Consistency with the velocity (local correctness)

  $$
  \left. \frac{\partial}{\partial \delta} F_\delta(\bm x_t, t) \right|_{\delta=0} = v_t(\bm x_t).
  $$

  For the true ODE solution $\bm x(t)$ with $\dot{\bm x}(t) = v_t(\bm x(t))$ and $F_\delta(\bm x_t,t)=\bm x(t+\delta)$, the derivative at $\delta=0$ equals $\dot{\bm x}(t)$. Our parameterization matches this to first order: differentiating $F_\delta^{\xi}$ at $\delta=0$ gives

  $$
  \left.\frac{\partial}{\partial \delta} F_\delta^{\xi}(\bm x_t, t)\right|_{\delta=0} = v_t(\bm x_t),
  $$

  because the $\delta^2$ term vanishes at first order.

  3) Semigroup (composition) property with time shift

  $$
  F_{a+b}(\bm x_t, t) = F_a\!\big( F_b(\bm x_t, t),\, t+b \big), \quad a,b \in \mathbb R\,.
  $$

  Meaning: advancing by $b$ then by $a$ equals a single advance by $a+b$, provided the second map starts at time $t+b$. For the true ODE flow this holds under standard well-posedness (e.g., Lipschitz $v_t$). Our semigroup loss is a practical enforcement of this law for the learned $F^{\xi}$.

  Useful corollaries and limits:
  - Associativity across multiple steps follows from the semigroup law with appropriate time shifts.
  - Invertibility for small $\delta$ in well-posed regimes: $F_{-\delta}(\,F_{\delta}(\bm x_t, t),\, t+\delta\,) = \bm x_t$.
  - Small-step expansion recovers the velocity: $F_\delta(\bm x_t, t) = \bm x_t + \delta\, v_t(\bm x_t) + \mathcal O(\delta^2)$.

  - Outcome: a one-step flow operator consistent across compositions, approximating the teacher’s integrated dynamics.

### Sobolev Loss (Sharper Fields)
- Weighted Sobolev norm to emphasize gradients:

  $$
  \lVert \bm u \rVert_S^2 = \alpha \lVert \bm u \rVert_2^2 + \beta \lVert \nabla \bm u \rVert_2^2, \qquad \alpha=\alpha(t),\; \beta=\beta(t)\;\text{optional}.
  $$

- Use in FM and/or PD objectives to penalize gradient mismatches that appear as blur.

### Practical Notes (How We Train)
- Schedules: cosine annealing with $T_{\max}$ equal to the PD stage iteration budget (one LR cycle per stage).
- Checkpoints: save student-only weights; resume PD with non-strict loading and a separately provided teacher.
- Sampling: choose $\Delta$ from uniform, beta-shaped, or fixed macro steps with jitter; set $t\sim\mathcal U[0,1-\Delta]$.
- Efficiency: teacher forward is no-grad; batch size can be increased to use GPU memory without affecting teacher correctness.

## Repo Map (brief)
- `src/fmfts/experiments/`: training scripts (`trainer.py`), orchestration (`multistage_pd.py`), per-experiment params.
- `src/fmfts/utils/models/`: core models and PD mixin (`cfm_velocity_pd.py`).
- `notes/`: local notes; `notes/daily/` untracked; this guide tracked.
- `README.md`, `pyproject.toml`: project intro and dependencies.

## Setup
- Environment: PyTorch CUDA; see `README.md` and `pyproject.toml`.
- Run example: `python src/fmfts/experiments/trainer.py ns2d velocity_pd --new --teacher <path> --max-iters 10000`.

## Data
- Datasets: ns2d, ks2d, rti3d (full/sliced). Loading via `experiments/<exp>/training_parameters.py` and dataset classes.
- Paths: configured in per-experiment params; ensure availability on the cluster.
- Trained coverage (so far):
  - KS-2D at 256×256
  - Compressible NS-2D periodic at 64×64
  - RTI-3D full at 32×32×32
  - RTI-3D slices at 128×128

## Models
- VelocityModelNS2D: UNet-based conditional velocity field.
- VelocityPDNS2D: student model with a frozen teacher via `DistilledVelocityMixin`.
- Losses: L2 or Sobolev (configurable).

## Training
- Key script: `src/fmfts/experiments/trainer.py` with `--teacher`, `--max-iters`, cosine LR schedule.
- Checkpointing: saves student-only weights periodically and on PD stage end.
- Resume: non-strict load for PD to ignore missing teacher keys.

## Distillation/Orchestration
- Progressive distillation: K teacher fine steps vs. 1 student macro step, Δ and t sampled per batch.
- Multistage: `multistage_pd.py` runs multiple stages, promoting the student as next teacher.

## Experiments
- Logs: `experiments/train.out` for end-to-end runs; TensorBoard under `<exp>/runs`.
- Known issues: host RAM OOM if retaining autograd graphs in logging (fixed in `trainer.py`).

## Reproducibility
- Seeds and budgets: set in training parameters; stage iters via `--max-iters`.
- Hardware: GPU utilization typically low for teacher (no-grad); batch size can be increased.

## Known Issues & Troubleshooting
- OOM at ~80k iters: caused by logging Tensors in moving averages; fix applied to cast to floats.

## Glossary
- PD: Progressive Distillation; Δ: macro step size; K: teacher fine steps.

## Changelog
- 2025-09-09: Added tracked guide scaffold; documented OOM fix in `trainer.py` (untested at time of note).
