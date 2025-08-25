from fmfts.experiments.ns2d.models import   VelocityModelNS2D, \
                                            SingleStepModelNS2D, \
                                            FlowModelNS2D
from fmfts.dataloader.ns2d import DatasetNS2D

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
        "cls": VelocityModelNS2D
    },
    "single_step": {
        "model_kwargs": { 
            "loss": "sobolev",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 10,
            "method": "midpoint"
        },
        "lr_max": 5e-5,
        "lr_min": 1e-5,
        "cls": SingleStepModelNS2D
    },
    "dataset": {
        "cls": DatasetNS2D,
        "kwargs": { }
    }
}