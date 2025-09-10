from fmfts.experiments.ns2d.models import (
    VelocityModelNS2D,
    SingleStepModelNS2D,
    FlowModelNS2D,
    VelocityPDNS2D,  
)
from fmfts.dataloader.ns2d import DatasetNS2D
from fmfts.utils.models.cfm_velocity_pd import uniform_delta_sampler, fixed_macrostep_sampler, beta_delta_sampler

params = {
    "flow": {
        "model_kwargs": {
            "loss": "sobolev",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 5,
        },
        "lr_max": 1e-5,
        "lr_min": 1e-6,
        "cls": FlowModelNS2D,
    },
    "velocity": {
        "model_kwargs": {
            "features": (64, 96, 96, 128),
            "loss": "sobolev",
        },
        "training_kwargs": {
            "batch_size": 4,
        },
        "lr_max": 5e-5,
        "lr_min": 1e-5,
        "cls": VelocityModelNS2D,
    },
    "single_step": {
        "model_kwargs": {
            "loss": "sobolev",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 10,
            "method": "midpoint",
        },
        "lr_max": 5e-5,
        "lr_min": 1e-5,
        "cls": SingleStepModelNS2D,
    },
    "velocity_pd": {
        "cls": VelocityPDNS2D,
        "model_kwargs": {
            "features": (64, 96, 96, 128),
            "loss": "l2",
            "K": 2,
            # Use uniform Δ sampler by default for PD (can be overridden per stage)
            "delta_sampler": beta_delta_sampler, #by default uniform_sampler
        },
        "lr_max": 2e-4,
        "lr_min": 1e-5,
        # Training-only controls for PD (not passed to the optimizer directly)
        # Note: train loop will read max_iters only for velocity_pd and stop after this many steps.
        "training_kwargs": {
            "max_iters": 10000,
        },
    },

    # Multistage PD orchestration defaults (used by multistage_pd.py)
    "multistage_pd": {
        "modeltype": "velocity_pd",
        # Number of PD stages to run
        "stages": 3,
        # Iterations per stage; if a single int is provided, it applies to all stages
        "stage_iters": [50000, 50000, 50000],
        # Initial teacher checkpoint filename (relative to experiments/ns2d/trained_models)
        # Typically the base velocity model trained earlier
        "initial_teacher": "state_velocity_teacher1.pt",
    },
    "dataset": {
        "cls": DatasetNS2D,
        "kwargs": {},
    },
    # Evaluation defaults (overridable via CLI)
    "eval": {
        "common": {
            "device": "cuda",
            "seed": 42,
            #"batch_size": 4,  #8
            "samples": 16,  #64
        },
        "models": {
            "velocity":    {"checkpoint": "ns2d/trained_models/state_velocity.pt"},
            "flow":        {"checkpoint": "ns2d/trained_models/state_flow.pt"},
            "velocity_pd": {"checkpoint": "ns2d/trained_models/stage_1_student.pt"},
        },
        "one_step": {
            "steps":  {"velocity": 50, "flow": 1, "velocity_pd": 50},
            "method": {"velocity": "midpoint", "velocity_pd": "midpoint"}
        },
        "rollout": {
            "length": 10,  #32
            "latent_policy": "fixed_latent",  # fixed_latent | deterministic | new_latent
            "steps":  {"velocity": 50, "flow": 1, "velocity_pd": 50},
            "method": {"velocity": "midpoint", "velocity_pd": "midpoint"}
        },
        "metrics": {
            "enabled": ["l2", "sob"],
            "params": {"sob": {"alpha": 1.0, "beta": 0.0}}
        },
        "artifacts": {
            "pdf": {"filename": "eval_report.pdf", "dpi": 150, "max_figs_per_model": 6, "include_rollout_curves": True},
            "save_csv": True,
            "save_json": True
        }
    },

}
