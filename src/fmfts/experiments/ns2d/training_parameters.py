from fmfts.experiments.ns2d.models import (
    VelocityModelNS2D,
    SingleStepModelNS2D,
    DeterministicModelNS2D
)
from fmfts.dataloader.ns2d import DatasetNS2D


params = {
    "velocity": {
        "model_kwargs": {
            "features": (128, 196, 196),
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 32,
        },
        "lr_max": 1e-5,
        "lr_min": 1e-5,
        "cls": VelocityModelNS2D,
    },
    "single_step": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 32,
            "steps": 10,
            "method": "midpoint",
        },
        "lr_max": 5e-5,
        "lr_min": 1e-5,
        "cls": SingleStepModelNS2D,
    },
    "deterministic": {
        "model_kwargs": { 
            "features": (128, 196, 196),
        },
        "training_kwargs": {
            "batch_size": 32,
        },
        "optimizer_init": {
            "lr": 1e-4,
        },
        "cls": DeterministicModelNS2D,
    },
    "rectifier": {
        "training_kwargs": {
            "batch_size": 8,
            "steps": 10, 
            "method": "midpoint",
        },
        "optimizer_init": { "lr": 1e-5 },
        # "cls": Rectifier
    },
    "add": {
        "training_kwargs": { 
            "w_distillation": 0.9,
            "w_R1": 10.,
            "generator_rate": 1,
        },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 5e-5 },
    },
    "velocity_pd": {
        "model_kwargs": {
            "K": 2,
            "stage": 3,
        },
        "training_kwargs": {
            "batch_size": 8,
        },
        "optimizer_init": {
            "lr": 1e-5,
        },
    },

    "dataset": {
        "cls": DatasetNS2D,
        "kwargs": {},
    },
}
