# Project Guide: Flow‑Matching for Time Series

Last updated: 2025-09-12

This guide explains the project end‑to‑end so newcomers can understand the goals, theory, code, and workflows. It is intended to be kept current and to double as a research notebook toward a future paper.

Quick links (repo paths):
- `src/fmfts/utils/models/cfm_velocity_pd.py`
- `src/fmfts/utils/models/time_series_model.py`
- `src/fmfts/experiments/ns2d/models.py`
- `src/fmfts/experiments/trainer.py`
- `src/fmfts/experiments/ns2d/training_parameters.py`
- `src/fmfts/experiments/ns2d/testing_parameters.py`
- `src/fmfts/experiments/train.sh`
- `src/fmfts/experiments/eval.py`
- `notes/refs.bib`

Table of Contents
- [Overview](#overview)
- [Theory](#theory)
- [Open Questions / Future Work](#open-questions--future-work)
- [Models](#models)
- [Training](#training)
- [Distillation / Orchestration](#distillation--orchestration)
- [Experiments](#experiments)
- [Reproducibility](#reproducibility)
- [Known Issues & Troubleshooting](#known-issues--troubleshooting)
- [Decision Log](#decision-log)
- [References](#references)

---

## Overview
- Problem: Conditional flow‑matching for time‑series forecasting/reconstruction, with physics‑motivated setups (e.g., 2D Navier–Stokes, NS2D).
- Experiments: `ns2d`, `ks2d`, `rti3d_full`, `rti3d_sliced` supported via a shared trainer/evaluator.
- Goals: Accurate and stable one‑step predictions and multi‑step rollouts; support progressive distillation (PD) to compress K fine steps into 1 macro step.
- Scope: Reusable velocity/flow/single‑step/PD models, training and evaluation harnesses, and config‑driven experiments.
- Claims (draft): Flow‑matching objectives with suitable schedulers and priors can simplify training vs. diffusion in some regimes. [CITE]

Notes:
- Revise when objectives or scope change (e.g., new loss variants, new datasets).

---

## Theory
Purpose: formal, code‑aligned description of our method with equations. Math follows the GitHub conventions in `notes/latex-github-rules.md`.

### Problem Setup
- Data: `(y, x_1) ~ p_data`, where `y` is a conditioner (past context or features) and `x_1` is the target at final time.
- Latent prior: `x_0 ~ p_0` (Normal(0, I) in code).
- Time: `t in [0, 1]` parameterizes an interpolation between `x_0` and `x_1`.
- Velocity field (student): `dx/dt = v_θ(x, y, t)` with `v_θ` implemented by `VelocityModelNS2D.forward(x,y,t)`.

Bridge (training states):
- Define a linear bridge to sample supervised states on the path:
  - `x_t = (1 − t) x_0 + t x_1`, with `t ~ p(t)` and `x_0 ~ p_0`.
  - In code: see `DistilledVelocityMixin.compute_loss` where `x_t` is formed from `(x_0, x_1, t)`.

### Conditional Flow Matching via Progressive Distillation
Goal: train `v_θ` so that a one‑step macro update over `Δ` matches the effect of rolling the teacher `v_T` for `K` fine steps of size `Δ/K` starting from the same `(x_t, y, t)`.

Sampling of macro step and start time:
- `(Δ, t) ~ p(Δ, t)`. We support samplers:
  - Uniform: `Δ ~ U(δ_min, 1)`, `t ~ U(0, 1−Δ)`.
  - Fixed‑macrostep (around `1/S` with jitter).
  - Beta‑biased: `Δ = δ_min + Beta(α, β) · (1−δ_min)`; `α < β` biases toward small `Δ`.
  - See `uniform_delta_sampler`, `fixed_macrostep_sampler`, `beta_delta_sampler` in `cfm_velocity_pd.py`.

Shared integrators (teacher and student use the same):
- Euler: `x⁺ = x + Δ · v(x, y, t)`.
- Midpoint: `k0 = v(x,y,t)`, `x_mid = x + (Δ/2) k0`, `x⁺ = x + Δ · v(x_mid, y, t + Δ/2)`.
- RK4: classical 4‑stage update. Implemented in `_velocity_step`.

Teacher rollout vs. Student step (notation):
- Teacher K‑step rollout over `Δ`:
  - `x_T = Rollout_K(x_t; y, t, Δ; v_T)` where each substep uses `_velocity_step(·, Δ/K)` and time increases by `Δ/K`.
  - Code: `_rollout_velocity(self.teacher_velocity, x_t, y, t, Δ, steps=K, method)`.
- Student single macro step:
  - `x_S = Step_1(x_t; y, t, Δ; v_θ)` using `_velocity_step(self, x_t, y, t, Δ, method)`.

### Training Objectives
We match the teacher’s macro‑effect at the sampled `(x_t, y, t, Δ)`.

- L2 objective:
  - `L(θ) = E[ || x_S − x_T ||_2^2 ]`, expectation over `(y, x_1) ~ p_data`, `x_0 ~ p_0`, `(t, Δ) ~ p(Δ, t)`.
  - Selected via `loss = "l2"`.

- Sobolev objective (stabilizes spatial structure):
  - `L(θ) = E[ α || x_S − x_T ||_2^2 + β · t^2 · || ∇(x_S − x_T) ||_2^2 ]`.
  - Gradients are over spatial axes; implemented by `utils/loss_fn.sobolev`.
  - In our NS2D defaults we often prefer L2 for PD; Flow/Velocity can use Sobolev.

Expanded expectation (for clarity):
```
L(θ) = E_{(y,x1)~p_data} E_{x0~p0} E_{(t,Δ)~p} [ ℓ( Step_1(x_t; y,t,Δ; v_θ) − Rollout_K(x_t; y,t,Δ; v_T) ) ]
where x_t = (1−t) x0 + t x1.
```

### Schedules, Biases, and Trade‑offs
- `Δ` distribution controls difficulty: small `Δ` encourages local consistency (easier), larger `Δ` encourages long updates (harder, more informative).
- `K` (teacher fine steps): larger `K` produces a more accurate teacher target but increases compute.
- Integrator choice: Euler (fastest, most biased), Midpoint (good bias‑variance trade‑off), RK4 (more accurate, costlier).

### Relation to PF‑ODEs and Diffusion (intuition)
- CFM learns vector fields that transport a simple prior to the data along a chosen path (the bridge). Our PD variant supervises macro‑dynamics induced by a teacher velocity rather than estimating stochastic scores.
- Sampling is ODE‑based (no SDE noise); stability depends on the trained vector field and the rollout integrator.

### Practical Notes (code alignment)
- Teacher weights are frozen (`requires_grad=False`) and moved to the active device before use; student checkpoints exclude teacher keys.
- The bridge `x_t` provides supervision on in‑between states, improving stability versus endpoint‑only targets.
- For fair targets, the teacher and student share the same integrator during training.

---

## Open Questions / Future Work
- What priors best stabilize long‑horizon rollouts? [idea]
- Can multistage training improve sample efficiency for NS2D? [idea]
- Evaluation: standardized metrics for physical consistency vs. predictive error. [idea]
- Ablations: loss variants, noise/time parameterizations, architectures. [idea]

---

## Models
Entry points: `src/fmfts/experiments/ns2d/models.py`, `src/fmfts/utils/models/cfm_velocity_pd.py`.

- VelocityModelNS2D:
  - Base velocity model with UNet backbone: `UNet(in_ch=2*4+1, out_ch=4, features=..., padding=("circular","circular"), nl=ReLU)`.
  - Forward: concat `[x, y, t]` along channels; predicts velocity field at `t`.
  - Features default `(64, 96, 128)` in class; NS2D training params often use `(64, 96, 96, 128)`.

- FlowModelNS2D:
  - Wraps a trained `VelocityModelNS2D`; `phi_net` clones velocity UNet with `+1` input channel for Δ.
  - `phi(x, y, t, Δ)`: concat `[x, y, t, Δ]`; used for single macro flow step.

- SingleStepModelNS2D:
  - Uses a deep copy of the velocity UNet as `phi_net` for Δ‑agnostic single‑step mapping.
  - `phi(x, y)` with `t=0` channel injected (zeros).

- VelocityPDNS2D (student):
  - Inherits `VelocityModelNS2D` + `DistilledVelocityMixin`.
  - Args: `teacher` (frozen velocity), `K` fine steps to distill, integrator `method` (`euler|midpoint|rk4`), optional `delta_sampler`, `log_delta_t`.
  - Loss: `l2` or `sobolev` between teacher rollout and student macro‑step.

- DistilledVelocityMixin (loss+rollout utilities):
  - `_velocity_step`, `_rollout_velocity` implement integrators; samplers: `uniform_delta_sampler`, `fixed_macrostep_sampler(S,jitter)`, `beta_delta_sampler(alpha,beta)`.
  - `compute_loss` builds bridge `x_t`, runs teacher K steps vs. student 1 step, computes loss.

---

## Training
Code: `src/fmfts/experiments/trainer.py`, params in `src/fmfts/experiments/ns2d/training_parameters.py`.

- CLI:
  - Velocity: `python src/fmfts/experiments/trainer.py ns2d velocity --new`
  - Flow: `python src/fmfts/experiments/trainer.py ns2d flow --new`
  - SingleStep: `python src/fmfts/experiments/trainer.py ns2d single_step --new`
  - Progressive Distillation (student):
    - Default teacher path: `ns2d/trained_models/state_velocity_teacher1.pt` (override with `--teacher PATH`).
    - Example: `python src/fmfts/experiments/trainer.py ns2d velocity_pd --new --teacher ns2d/trained_models/state_velocity.pt --max-iters 10000`

- Params (NS2D defaults):
  - `velocity`: features `(64,96,96,128)`, loss `sobolev`, `lr_max=5e-5`, `lr_min=1e-5`, `batch_size=4`.
  - `flow`: loss `sobolev`, `steps=5`, `lr_max=1e-5`, `lr_min=1e-6`, `batch_size=4`.
  - `single_step`: loss `sobolev`, `steps=10`, `method=midpoint`, `lr_max=5e-5`, `lr_min=1e-5`, `batch_size=4`.
  - `velocity_pd`: loss `l2`, features `(64,96,96,128)`, `K=2`, `delta_sampler=beta_delta_sampler`, `lr_max=2e-4`, `lr_min=1e-5`, `training_kwargs.max_iters=10000`.

- Loop & scheduler:
  - Uses `Adam(lr=lr_max)` and `CosineAnnealingLR(T_max=500)` normally; for `velocity_pd`, `T_max` equals `max_iters` (or `--max-iters`).
  - Writes TensorBoard logs under `{experiment}/runs` (e.g., `ns2d/runs`).
  - Evaluates test loss every 10 iters on a single batch.

- Checkpoints & resume:
  - Saves: `{experiment}/trained_models/state_{modeltype}.pt` and timestamped copies in `{experiment}/checkpoints/` every 10k iters and at PD early stop.
  - Resume: run without `--new` to load `state_{modeltype}.pt` (optimizer lr reset to `lr_max`).
  - PD checkpoints store student weights only (teacher params excluded).

- Datasets:
  - Train/test datasets built from `params["dataset"]` (e.g., `DatasetNS2D(**kwargs)`), batch sizes per section above.

- Slurm example: see `src/fmfts/experiments/train.sh` for Alvis cluster directives and eval commands.

---

## Distillation / Orchestration
- Single‑stage PD: `trainer.py` distills `K` fine teacher steps into one student macro‑step over random Δ and t.
- Multistage PD: `src/fmfts/experiments/multistage_pd.py` orchestrates repeated PD stages, promoting the previous student to the new teacher.
  - Defaults (`ns2d.training_parameters["multistage_pd"]`): `stages=3`, `stage_iters=[50000,50000,50000]`, `initial_teacher=state_velocity_teacher1.pt`.
  - Example: `python src/fmfts/experiments/multistage_pd.py ns2d --stages 3 --stage-iters 50000 50000 50000`
  - Produces `ns2d/trained_models/stage_{i}_student.pt` and a `multistage_manifest.json` lineage file.

---

## Experiments
Evaluation: `src/fmfts/experiments/eval.py` (configs under `params["eval"]`).

- Modes: `one_step` (independent predictions), `rollout` (autoregressive), controlled by `--mode`.
- Models: choose via `--models velocity,flow,velocity_pd`; optional `--checkpoints` to point to specific files.
- Metrics: `l2`, `sob` (configurable reducers). Optional `use_denormalized` to compute in physical units if dataset provides means/std.
- Artifacts: CSV `metrics.csv`, JSON `summary.json`, and a PDF report with aggregates, curves, and snapshot visualizations.
- Example (from `train.sh`):
  - `python src/fmfts/experiments/eval.py ns2d --models velocity_pd,flow,velocity \
     --checkpoints ns2d/trained_models/stage_1_student.pt,ns2d/trained_models/state_flow.pt,ns2d/trained_models/state_velocity_teacher1.pt \
     --mode one_step,rollout`

---

## Reproducibility
- Seeds: default `42` in eval; training relies on DataLoader shuffling and PyTorch defaults.
- Device: eval defaults to `cuda`; trainer sets default device to `cuda`.
- Artifacts: runs/checkpoints under `{experiment}/runs` and `{experiment}/trained_models`.
- Environment: ensure `PYTHONPATH=src` (or install the package) when invoking scripts.

---

## Known Issues & Troubleshooting
- PD checkpoints intentionally exclude teacher weights; loading uses `strict=False` in places to avoid key mismatches.
- When starting `flow`/`single_step`/`velocity_pd`, trainer expects a trained velocity backbone at `.../state_velocity.pt`.
- Large rollouts can drift; consider increasing K during PD or biasing Δ smaller via `beta_delta_sampler(alpha<beta)`.
- If CUDA OOM, reduce batch size (`training_parameters`) or steps; consider gradient accumulation (not yet implemented here).

---

## Decision Log
- [2025-09-12] Adopted beta‑biased Δ sampler for PD (alpha=0.5, beta=1.0) in NS2D to emphasize small macro‑steps; PD checkpoints remain student‑only to simplify reloads.

Guideline: add TOC and update the “Last updated” line on every edit.

---

## References
Maintain BibTeX entries in `notes/refs.bib`. Cite inline via keys. If network access is needed to fetch metadata (arXiv/Crossref/DOI), request approval first and record commands in the daily note.

Placeholder citations: [CITE]
