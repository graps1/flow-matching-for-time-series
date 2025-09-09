# Project Guide — flow-matching-for-time-series

Note: This top section is a living, tracked guide for onboarding and structure.

## Overview
- Problem: Surrogate modeling for fluid systems using flow matching and progressive distillation to accelerate inference while maintaining fidelity.
- Goals: Fast, high-quality next-state prediction; reduce solver steps via distillation; mitigate blurriness using Sobolev losses.

## Theory
- Flow Matching (FM): Learn a time-conditioned velocity field that transports samples from an easy prior p0 to a target p1 along straight bridging paths. With x0 ~ p0, x1 ~ p1, and t ~ U[0,1], define x_t = t x1 + (1 - t) x0. Train
  - min_θ E[ || v_θ(x_t, t) - (x1 - x0) ||^2 ]
  - Prediction solves the ODE dx/dt = v_θ(x, t) from t=0 to 1; in deterministic regimes, few steps may suffice.
- Conditional FM (CFM) for time series: Given previous state y = x1^- and next state x1^+ from the same trajectory, condition the velocity on y:
  - min_θ E[ || v_θ(x_t, y, t) - (x1^+ - x0) ||^2 ] where x_t = t x1^+ + (1 - t) x0.
- Progressive Distillation (PD): Compress K fine ODE steps of a frozen teacher into one macro step of a student.
  - Sample Δ ∈ [0,1], t ∈ [0, 1-Δ], and form x_t as above using p0 and x1.
  - Teacher rollout: x_T = rollout_K(v_T, x_t, y, t, Δ) via K fine steps of size Δ/K (no grad, teacher frozen).
  - Student step: x_S = step_1(v_S, x_t, y, t, Δ) using one macro step with the student velocity.
  - Loss (L2): L = || x_T - x_S ||^2. Optionally use Sobolev: L_S = α ||e||_2^2 + β ||∇e||_2^2 with e = x_T - x_S.
  - Intuition: The student learns to match the teacher’s flow over macro increments, enabling fewer inference steps.
- Flow-level distillation (semigroup alternative): Parameterize a flow map F_δ(x_t, t) with Euler baseline plus δ^2 correction and enforce semigroup consistency.
  - F_δ^ξ(x_t, t) = x_t + δ v_t(x_t) + δ^2 (φ_δ^ξ(x_t, t) - v_t(x_t)).
  - Semigroup loss: min_ξ E[ || F_δ^ξ(x_t, t) - sg( F_{δ/2}^ξ( F_{δ/2}^ξ( x_t, t ), t + δ/2 ) ) ||^2 ].
  - Relation to PD: PD distills a velocity’s macro step via rollout matching; semigroup distillation directly learns consistent flow maps F.
- Sobolev loss for deblurring: Define weighted Sobolev norm
  - ||u||_S^2 = α ||u||_2^2 + β ||∇u||_2^2, with optional time-dependent weights α(t), β(t).
  - Apply in FM training and/or PD matching objective to penalize gradient mismatches that manifest as blur in fluid fields.
- Schedules and budgets: Use cosine annealing with T_max equal to the PD stage iteration budget to complete one LR cycle per stage. Stage termination saves student-only checkpoints; teacher weights are never trained.
- Checkpoint structure and resume: PD checkpoints contain student weights only (e.g., keys not starting with teacher_velocity.*). Resume PD with non-strict loading; provide a teacher checkpoint separately.
- Δ, t sampling: Δ sampled from uniform, beta-shaped, or fixed-macro-step with jitter; t sampled uniformly on [0, 1-Δ]. Design samplers to balance coverage and match intended inference step sizes.

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

*** End of Guide ***

- ICLR deadline Sep 19th
- Vacation starting Sep 10th
- => a bit more than two weeks left

**Summary**

- I have successfully trained a bunch of flow matching models on:
    - the 2d Kuramoto-Sivashinsky equation, size 256 x 256
    - a compressible 2d Navier-Stokes setting w/ periodic BCs, size 64 x 64
    - the full 3d Rayleigh-Taylor instability compressed to a size of 32 x 32 x 32
    - slices of the 3d Rayleigh-Taylor instability, size 128 x 128
- Contributions:
    - a novel distillation method (i.e., a method that directly learns the solution of the flow matching ODE) for faster output generation
    - a (physics-informed) Sobolev loss that avoids blurry outputs

**Main argument**

Using flow matching (or denoising diffusion) is really promising for the surrogate modeling of (partially observed) systems

- quality-wise, these methods are capable of dealing w/ missing information and naturally generate plausible successor states.
- in the case of fully deterministic systems, the flow matching ODE can -- in theory -- be solved with a single iteration of explicit Euler, since the transition distribution is a Dirac delta centered at the successor and the flow matching paths are therefore completely straight. => In this case, a flow matching model is as efficient as a deterministic surrogate model(!)

But: 

1. for non-deterministic systems, one needs more solver steps. About 10 explicit Euler / 5 midpoint steps usually suffice, but this is still ~5x slower than a single-step model.
1. (this has nothing to do with the previous argument:) results often look blurry due to the use of the l2 norm for training. This is something that we want to avoid, since sharp features frequently appear in fluid dynamics applications.

So:

- Problem 1) motivates the use of a distillation-based approach.
- Problem 2) motivates the use of a different loss function, in particular a Sobolev loss that incorporates (coordinate-space) gradients.

**Preliminaries**

In flow matching, one learns the flow taking samples from the initial Gaussian distribution $p_0$ to the target distribution $p_1$. This is done by solving the optimization problem
$$
    \min_\theta \mathbb E_{p_0(\bm x_0), p_1(\bm x_1), U(\bm t)} \lVert v_t^\theta(\bm t \bm x_1 + (1-\bm t)\bm x_0) - (\bm x_1 - \bm x_0) \rVert^2.
$$
In our case, $\bm x_1$ is a fluid field consisting of velocities and a pressures, perhaps also densities. In order to do prediction, i.e., in order to move from one fluid state to the next, we use a conditional model of the form
$$
    \min_\theta \mathbb E_{p_0(\bm x_0), p_1(\bm x_1^-, \bm x_1^+), U(\bm t)} \lVert v_t^\theta(\bm t \bm x_1^+ + (1-\bm t)\bm x_0 ∣ \bm x_1^-) - (\bm x_1^+ - \bm x_0) \rVert^2,
$$
where $\bm x_1^+$ is now the next and $\bm x_1^-$ the previous fluid state. Bot $\bm x_1^-$ and $\bm x_1^+$ are drawn from the same trajectory.

### 1. The distillation method 

Say we have trained a flow matching velocity model $v_t(x_t) = v_t^\theta(x_t)$, which is assumed wlog to be unconditional. To generate new approx. samples from $p_1$, one samples $x_0 ∼ p_0$ from the Gaussian distribution and then solve the ODE
$$
    \dot {x}_t = v_t(x_t), \quad\quad(*)
$$
from $t = 0$ to $t = 1$. Associated with the velocity field $v^\theta$ is the underlying *flow* that it generates:
$$
    F_\delta(x_t, t) = x_{t + \delta}
$$
when $x_t$ follows the ODE $(*)$. If we *had* access to a model computing $F$, we could compute the solution of $x_0$ in a single step by evaluating $F_1(x_0, 0)$.

But we don't have access to it, so we have to learn it. The first insight is that we can characterize the flow by three properties:

1. (identity when no increment) $F_0(x_t, t) = x_t$.
1. (consistency w/ velocity field) $\frac d {d\delta} F^\xi_\delta(x_t, t) |_{\delta = 0} = v_t(x_t)$.
1. (semigroup property) $F_{a+b}(x_t, t) = F_a(F_b(x_t,t), t+b)$.

Then, we parametrize a neural network:
$$
    F^\xi_\delta(x_t, t) = \underbrace{x_t + \delta v_t(x_t)}_{\text{explicit Euler step}} + \underbrace{\delta^2 (\phi^\xi_\delta(x_t,t) - v_t(x_t))}_{\text{learned correction}}.
$$

This already ensures properties 1) and 2). To get the third property, we are optimizing the following "semigroup" loss:
$$
    \min_\xi \mathbb E_{\bm \delta, \bm t, \bm x_1, \bm x_0}  \lVert F_{\bm \delta}^\xi(\bm x_{\bm t}, \bm t) - \text{sg}(F_{\bm \delta/2}^\xi(F_{\bm \delta/2}^\xi(\bm x_{\bm t}, \bm t), \bm t + \bm \delta/2)) \rVert^2.
$$
Here, the $\text{sg}$ is the "stopgrad" operation. I'm utilizing it here since the rhs is a more accurate version of what we're trying to learn, i.e. closer to the true flow $F_{\bm \delta}(\bm x_{\bm t}, \bm t)$.
In practice, we are initializing $\phi$ (the "corrector" network) with a copy of $v$, where we modify the input weights to add an additional channel for $\delta$.


## 2. Avoiding blurryness in generated states 

- When being trained, both the velocity model (trained w/ flow matching) and the distilled flow (trained w/ semigroup loss) tend to produce blurry results.
- This is likely due to the use of the l2 norm in image space.

The Sobolev norm addresses this issue: In our case, $x_1$ is a fluid field, i.e., it is a differentiable function of the form $x_1 : \R^d \rightarrow \R^n$, where $d$ is the coordinate dimension ($d  = 2$ or $d = 3$) and $n$ is the number of fields, e.g. $n = 3$ when there is one pressure and a 2d velocity field.

One can define a weighted Sobolev norm by taking
$$
    \lVert x_1 \rVert^2_S = \alpha \lVert x_1 \rVert^2_2 + \beta \lVert \nabla x_1 \rVert^2_2,
$$
where $\alpha$ and $\beta$ are constants. Since it includes the gradient of $x_1$, it is more sensitive to changes of $x_1$ in the coordinate domain. In other words, the difference $\lVert x_1 - x_1' \rVert^2_S$ is now larger if the gradients of $x_1$ and $x_1'$ don't match, which is the case if $x_1$ is, for instance, a blurry version of $x_1'$.

This norm can be extended to the case where $x_t$ is a not fully denoised fluid field:
$$
    \lVert x_t \rVert^2_{S_t} = \alpha_t \lVert x_t \rVert^2_2 + \beta_t \lVert \nabla x_t \rVert^2_2,
$$
where $\alpha_t$ and $\beta_t$ are now allowed to depend on the time, e.g., we can let $\alpha_t$ stay constant and increase $\beta_t$ as $t \rightarrow 1$ in order to put a larger priority on "deblurring" less noisy fluid fields. 

The **first** way one can use this norm is to train the velocity model with this new norm instead:
$$
    \min_\theta \mathbb E_{p_0(\bm x_0), p_1(\bm x_1), U(\bm t)} \lVert v_t^\theta(\bm t \bm x_1 + (1-\bm t)\bm x_0) - (\bm x_1 - \bm x_0) \rVert^2_{S_{\bm t}}.
$$
This is actually sound, in the sense that $v^\theta$ is learning the correct velocity field.

The **second** way one can use this is to improve training of the distillation model: The new objective simply becomes:
$$
    \min_\xi \mathbb E_{\bm \delta, \bm t, \bm x_1, \bm x_0}  \lVert F_{\bm \delta}^\xi(\bm x_{\bm t}, \bm t) - \text{sg}(F_{\bm \delta/2}^\xi(F_{\bm \delta/2}^\xi(\bm x_{\bm t}, \bm t), \bm t + \bm \delta/2)) \rVert^2_{S_{\bm t}}.
$$
