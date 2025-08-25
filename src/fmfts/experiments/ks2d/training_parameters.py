import os
from dataclasses import dataclass, field

from fmfts.experiments.ks2d.models import   VelocityModelKS2D
from fmfts.dataloader.ks2d import DatasetKS2D

params = {
    "velocity": {
        "model_kwargs": {
            "features": (64, 128),
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
        },
        "lr_max": 1e-5,
        "lr_min": 1e-6,
        "cls": VelocityModelKS2D
    },
    "dataset": {
        "cls": DatasetKS2D,
        "kwargs": { }
    }
}