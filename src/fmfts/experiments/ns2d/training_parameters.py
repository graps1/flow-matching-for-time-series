from fmfts.experiments.ns2d.models import   VelocityModelNS2D, \
                                            SingleStepModelNS2D, \
                                            FlowModelNS2D
from fmfts.dataloader.ns2d import DatasetNS2D
from fmfts.utils.models.add import AdversarialDiffusionDistillation


params = {
    "flow": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 32,
            "steps": 10,
            "p_geometric": 0.99
        },
        "lr_max": 1e-4,
        "lr_min": 1e-4,
        "cls": FlowModelNS2D,
    },
    "velocity": {
        "model_kwargs": {
            "features": (128, 196, 196),
            # "features": (128, 196),
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 32,
        },
        "lr_max": 1e-5,
        "lr_min": 1e-5,
        "cls": VelocityModelNS2D
    },
    "single_step": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 32,
            "steps": 10,
            "method": "midpoint"
        },
        "lr_max": 5e-5,
        "lr_min": 5e-5,
        "cls": SingleStepModelNS2D
    },
    "add": {
        "training_kwargs": { 
            "w_distillation": 0.0,
            "w_R1": 10.,
            "generator_rate": 1,
        },
        "optimizer_init": { "lr_G": 1e-6, "lr_D": 5e-5 },
        "cls": AdversarialDiffusionDistillation
    },
    "dataset": {
        "cls": DatasetNS2D,
        "kwargs": { }
    }
}