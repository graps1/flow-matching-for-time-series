from fmfts.experiments.rti3d_sliced.models import (VelocityModelSlicedRTI3D, 
                                                   DirectDistillationModelSlicedRTI3D)
from fmfts.dataloader.rti3d_sliced import DatasetSlicedRTI3D

params = {
    "velocity": {
        "model_kwargs": {
            "features": (128, 256, 256, 128),
            # "features": (196, 256, 384),
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 32,
        },
        "optimizer_init": { "lr": 1e-5 },
        "cls": VelocityModelSlicedRTI3D
    },
    "dir_dist": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 10,
            "method": "midpoint"
        },
        "optimizer_init": { "lr": 1e-5 },
        "cls": DirectDistillationModelSlicedRTI3D
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
        "training_kwargs": { 
            "w_distillation": 0.0,
            "w_R1": 25.,
            "generator_rate": 1,
        },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 1e-4 },
    },


    "dataset": {
        "cls": DatasetSlicedRTI3D,
        "kwargs": {
            "At" : 3/4,
            "dt" : 5, 
            "dy" : 1,
            "dz" : 1, 
            "include_timestamp" : True
        }
    }
}