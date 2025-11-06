from fmfts.experiments.sRTI.models import (VelocityModelSlicedRTI3D, 
                                           DirectDistillationModelSlicedRTI3D,
                                           DeterministicModelSlicedRTI3D)
from fmfts.dataloader.sRTI import DatasetSlicedRTI3D

params = {

    "velocity": {
        "model_kwargs": { "features": (128, 256, 256, 128), },
        "training_kwargs": { "batch_size": 32, },
        "optimizer_init": { "lr": 1e-5 },
        "cls": VelocityModelSlicedRTI3D
    },

    "dir_dist": {
        "model_kwargs": { },
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
    
    "prog_dist": {
        "model_kwargs": { "K": 2, "stage": 3, },
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr": 1e-5 },
    },
    
    "add": {
        "training_kwargs": { "batch_size": 8, },
        "model_kwargs": { 
            "lmbda": 0.0,
            "gamma": 25.,
            "generator_rate": 5,
        },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 5e-5 },
    },

    "deterministic": {
        "model_kwargs": { "features": (128, 256, 256, 128), },
        "training_kwargs": { "batch_size": 8, },
        "optimizer_init": { "lr": 1e-5, },
        "cls": DeterministicModelSlicedRTI3D,
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