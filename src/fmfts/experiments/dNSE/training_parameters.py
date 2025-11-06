from fmfts.experiments.dNSE.models import (
    VelocityModelNS2D,
    DirectDistillationModelNS2D,
    DeterministicModelNS2D
)
from fmfts.dataloader.dNSE import DatasetNS2D


params = {

    "velocity": {
        "model_kwargs": { "features": (128, 196, 196), },
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr": 1e-5 },
        "cls": VelocityModelNS2D,
    },

    "dir_dist": {
        "model_kwargs": { },
        "training_kwargs": {
            "batch_size": 8,
            "steps": 10,
            "method": "midpoint",
        },
        "optimizer_init": { "lr": 1e-5 },
        "cls": DirectDistillationModelNS2D,
    },

    "deterministic": {
        "model_kwargs": {  "features": (128, 196, 196), },
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr": 1e-5 },
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

    # for +150k iterations
    # "add": {
    #     "training_kwargs": { "batch_size": 8, },
    #     "optimizer_init": { "lr_G": 5e-6, "lr_D": 5e-5 },
    #     "model_kwargs": { 
    #         "lmbda": 0.9,
    #         "gamma": 5.,
    #         "generator_rate": 5,
    #     },
    # },

    # for +20k iterations
    "add": {
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 1e-5 },
        "model_kwargs": { 
            "lmbda": 0.9,
            "gamma": 5.,
            "generator_rate": 5,
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
