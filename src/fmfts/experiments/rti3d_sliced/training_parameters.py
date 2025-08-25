from fmfts.experiments.rti3d_sliced.models import VelocityModelSlicedRTI3D, \
                                                  SingleStepModelSlicedRTI3D, \
                                                  FlowModelSlicedRTI3D
from fmfts.dataloader.rti3d_sliced import DatasetSlicedRTI3D

params = {
    "flow": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 2,
        },
        "lr_max": 5e-5,
        "lr_min": 1e-5,
        "cls": FlowModelSlicedRTI3D,
    },
    "velocity": {
        "model_kwargs": {
            "features": (128, 196, 196, 256),
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
        },
        "lr_max": 1e-5,
        "lr_min": 5e-6,
        "cls": VelocityModelSlicedRTI3D
    },
    "single_step": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 10,
        },
        "lr_max": 5e-5,
        "lr_min": 1e-5,
        "cls": SingleStepModelSlicedRTI3D
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