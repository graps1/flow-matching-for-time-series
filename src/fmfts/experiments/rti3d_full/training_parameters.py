from fmfts.experiments.rti3d_full.models import VelocityModelFullRTI3D, \
                                                  DirectDistillationModelFullRTI3D, \
                                                  DeterministicModelFullRTI3D
from fmfts.dataloader.rti3d_full import DatasetFullRTI3D


params = {

    "velocity": {
        "model_kwargs": { "features": (128, 196), },
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr": 1e-5 },
        "cls": VelocityModelFullRTI3D
    },

    "dir_dist": {
        "model_kwargs": { },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 10,
            "method": "midpoint"
        },
        "optimizer_init": { "lr": 1e-5 },
        "cls": DirectDistillationModelFullRTI3D
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
        "training_kwargs": { "batch_size": 4, }, # up to 100k
        "model_kwargs": { 
            "lmbda": 0.9,
            "gamma": 25.,
            "generator_rate": 10,
        },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 1e-5 },
    },
    
    "prog_dist": {
        "model_kwargs": { "K": 2, "stage": 3, },
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr": 1e-5 },
    },

    "deterministic": {
        "model_kwargs": { "features": (128, 196), },
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr": 1e-5, },
        "cls": DeterministicModelFullRTI3D,
    },

    "dataset": {
        "cls": DatasetFullRTI3D,
        "kwargs": {
            "At" : 3/4,
            "dt" : 5, 
            "dx" : 4,
            "dy" : 4, 
            "dz" : 4,
            "include_timestamp" : True
        }
    }

}