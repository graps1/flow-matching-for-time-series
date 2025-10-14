from fmfts.experiments.rti3d_full.models import VelocityModelFullRTI3D, \
                                                  SingleStepModelFullRTI3D, \
                                                  FlowModelFullRTI3D, \
                                                  DeterministicModelFullRTI3D
from fmfts.dataloader.rti3d_full import DatasetFullRTI3D
from fmfts.utils.models.add import AdversarialDiffusionDistillation
from fmfts.utils.models.cfm_rectifier import Rectifier


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
            "batch_size": 32,
        },
        "lr_max": 1e-5,
        "lr_min": 1e-5,
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
            "w_distillation": 0.9,
            "w_R1": 25.,
            "generator_rate": 1,
        },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 5e-5 },
        "cls": AdversarialDiffusionDistillation
    },
    "deterministic": {
        "model_kwargs": { 
            "features": (128, 196),
        },
        "training_kwargs": {
            "batch_size": 8,
        },
        "optimizer_init": {
            "lr": 1e-5,
        },
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