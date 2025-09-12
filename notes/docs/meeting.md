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
- Problem: Flow matching for time series forecasting/reconstruction with physics‑informed or domain‑specific constraints (e.g., 2D Navier–Stokes configurations).
- Goals: Build accurate, stable models; enable multistage curricula; support reproducible experiments and evaluation.
- Scope: Utilities for continuous/decomposed vector fields, training/eval harnesses, and configuration‑driven experiments.
- Key claims: Simpler training dynamics vs. diffusion in certain regimes; compatibility with physics priors. [CITE]
- Alternatives considered: Score‑based diffusion, teacher‑student distillation, and autoregressive baselines. [CITE]

Notes:
- This section should be revised when objectives or scope meaningfully change.

---

## Theory
Purpose: Formal explanation of the approach with intuitive guidance and derivations. Keep GitHub‑friendly math (see `notes/latex-github-rules.md`).

- Setup and notation: Define data, time parameterization, and vector fields. [CITE]
- Objective: Flow matching losses used (conditional, probability‑flow, etc.). [CITE]
- Relationship to diffusion/ODE/SDE views; boundary conditions and priors. [CITE]
- Training dynamics: Schedules, discretizations, and stability notes. [CITE]
- Assumptions/limitations: Stationarity, Markovity, data regimes. [CITE]

TODO:
- Fill in core equations and symbol definitions based on current implementation.
- Add references after literature pass; use `notes/refs.bib` with BibTeX keys.

---

## Open Questions / Future Work
- What priors best stabilize long‑horizon rollouts? [idea]
- Can multistage training improve sample efficiency for NS2D? [idea]
- Evaluation: standardized metrics for physical consistency vs. predictive error. [idea]
- Ablations: loss variants, noise/time parameterizations, architectures. [idea]

---

## Models
Code entry points:
- `src/fmfts/utils/models/cfm_velocity_pd.py` — conditional flow/velocity parameterization utilities.
- `src/fmfts/utils/models/time_series_model.py` — generic time‑series model components.
- `src/fmfts/experiments/ns2d/models.py` — NS2D‑specific architectures and wrappers.

Document for each major model:
- Architecture: layers, normalization, positional/time encodings, parameter counts.
- Assumptions/inductive biases: invariances, constraints, physics priors.
- Variants: sizes, optional modules, activation choices.
- Checkpoints: naming/location conventions and compatibility notes.

TODO: List current model classes with brief descriptions and typical configs.

---

## Training
Code entry points:
- `src/fmfts/experiments/trainer.py` — training loop, logging, checkpointing.
- `src/fmfts/experiments/ns2d/training_parameters.py` — experiment parameters.
- `src/fmfts/experiments/train.sh` — shell wrapper for runs.

Document:
- Parameters and schedules: learning rate, time sampling, loss weights, augmentation.
- Resume semantics: what’s restored (optimizer, schedulers, RNG seeds), how to resume.
- Logging and artifacts: where runs/checkpoints/metrics are stored.
- Failure/oom patterns and mitigation strategies.

TODO: Add canonical command lines for common runs and expected runtimes.

---

## Distillation / Orchestration
- Pipeline stages: teacher/student roles, budgets, and curriculum.
- Data flows: how intermediate representations or pseudo‑labels are produced/used.
- Scheduling: stage transitions, stopping criteria, and resources.

TODO: Describe any multistage procedures currently used in `src/fmfts/experiments/multistage_pd.py` if applicable.

---

## Experiments
Code entry points:
- `src/fmfts/experiments/ns2d/testing_parameters.py` — eval configs.
- `src/fmfts/experiments/eval.py` — evaluation harness and metrics.

Document:
- Datasets/splits, preprocessing, and normalization.
- Current results with metrics; include links to artifacts.
- Failed/abandoned experiments and reasons.

TODO: Record baseline numbers and define comparison protocols.

---

## Reproducibility
- Seeds and determinism settings; any non‑deterministic kernels noted.
- Hardware and expected runtimes per experiment size.
- Exact environment and dependency versions (if tracked elsewhere, link/reference).

---

## Known Issues & Troubleshooting
- Common OOM cases and batch/microbatch guidance.
- Instabilities: exploding/vanishing norms; suggested gradient clipping/normalization.
- Dataset pitfalls: shapes, masks, padding, time alignment.

---

## Decision Log
Short dated rationales for major design choices.

- [YYYY-MM-DD] Example: changed loss definition to improve stability on NS2D.

Guideline: add TOC and update the “Last updated” line on every edit.

---

## References
Maintain BibTeX entries in `notes/refs.bib`. Cite inline via keys. If network access is needed to fetch metadata (arXiv/Crossref/DOI), request approval first and record commands in the daily note.

Placeholder citations: [CITE]

