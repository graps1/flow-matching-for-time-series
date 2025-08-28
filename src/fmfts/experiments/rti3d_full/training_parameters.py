from fmfts.experiments.rti3d_full.models import VelocityModelFullRTI3D, \
                                                  SingleStepModelFullRTI3D, \
                                                  FlowModelFullRTI3D
from fmfts.dataloader.rti3d_full import DatasetFullRTI3D

params = {
    "flow": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 4,
        },
        "lr_max": 1e-5,
        "lr_min": 1e-5,
        "cls": FlowModelFullRTI3D,
    },
    "velocity": {
        "model_kwargs": {
            "features": (128, 196),
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
        },
        "lr_max": 5e-5,
        "lr_min": 5e-5,
        "cls": VelocityModelFullRTI3D
    },
    "single_step": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 10,
            "method": "midpoint"
        },
        "lr_max": 1e-5,
        "lr_min": 1e-5,
        "cls": SingleStepModelFullRTI3D
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