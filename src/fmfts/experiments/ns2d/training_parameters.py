from fmfts.experiments.ns2d.models import (
    VelocityModelNS2D,
    DirectDistillationModelNS2D,
    DeterministicModelNS2D
)
from fmfts.dataloader.ns2d import DatasetNS2D


params = {

    "velocity": {
        "model_kwargs": { "features": (128, 196, 196), },
        "training_kwargs": { "batch_size": 32, },
        "optimizer_init": { "lr": 1e-4 },
        "cls": VelocityModelNS2D,
    },

    "dir_dist": {
        "model_kwargs": { },
        "training_kwargs": {
            "batch_size": 32,
            "steps": 10,
            "method": "midpoint",
        },
        "optimizer_init": { "lr": 1e-4 },
        "cls": DirectDistillationModelNS2D,
    },

    "deterministic": {
        "model_kwargs": {  "features": (128, 196, 196), },
        "training_kwargs": { "batch_size": 32, },
        "optimizer_init": { "lr": 1e-4 },
        "cls": DeterministicModelNS2D,
    },

    "rectifier": {
        "training_kwargs": {
            "batch_size": 8,
            "steps": 10, 
            "method": "midpoint",
        },
        "optimizer_init": { "lr": 1e-5 },
    },

    "add": {
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 5e-5 },
        "model_kwargs": { 
            "lmbda": 0.9,
            "gamma": 10.,
            "generator_rate": 1,
        },
    },

    "prog_dist": {
        "model_kwargs": { "K": 2, "stage": 3, },
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr": 1e-5 },
    },

    "dataset": {
        "cls": DatasetNS2D,
        "kwargs": {},
    },

}
