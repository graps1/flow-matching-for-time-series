from fmfts.experiments.rti3d_full.models import VelocityModelFullRTI3D, \
                                                  DirectDistillationModelFullRTI3D, \
                                                  DeterministicModelFullRTI3D
from fmfts.dataloader.rti3d_full import DatasetFullRTI3D


params = {

    "velocity": {
        "model_kwargs": { "features": (128, 196), },
        "training_kwargs": { "batch_size": 32, },
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
        "training_kwargs": { 
            "w_distillation": 0.9,
            "w_R1": 25.,
            "generator_rate": 1,
        },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 5e-5 },
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