from fmfts.experiments.rti3d_sliced.models import VelocityModelSlicedRTI3D, \
                                                  SingleStepModelSlicedRTI3D, \
                                                  FlowModelSlicedRTI3D
from fmfts.dataloader.rti3d_sliced import DatasetSlicedRTI3D
from fmfts.utils.models.cfm_rectifier import Rectifier
from fmfts.utils.models.add import AdversarialDiffusionDistillation

params = {
    "flow": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 4,
        },
        "optimizer_init": { "lr": 1e-5 },
        "cls": FlowModelSlicedRTI3D,
    },
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
    "single_step": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 10,
            "method": "midpoint"
        },
        "optimizer_init": { "lr": 1e-5 },
        "cls": SingleStepModelSlicedRTI3D
    },
    "rectifier": {
        "training_kwargs": {
            "batch_size": 8,
            "steps": 10, 
            "method": "midpoint",
        },
        "optimizer_init": { "lr": 1e-5 },
        "cls": Rectifier
    },
    "add": {
        "training_kwargs": { 
            "w_distillation": 0.0,
            "w_R1": 25.,
            "generator_rate": 1,
        },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 1e-4 },
        "cls": AdversarialDiffusionDistillation
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