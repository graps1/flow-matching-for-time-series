import os

import torch
import torch.optim
from fmfts.experiments.trainer import Trainer
from fmfts.dataloader.ks2d import DatasetKS2D
from fmfts.experiments.ks2d.models import VelocityModelKS2D

if __name__ == "__main__":
    torch.set_default_device("cuda")

    model = VelocityModelKS2D(features = (64, 128))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    trainer = Trainer(
        model_name = "velocity",
        checkpoints_dir = "./checkpoints",
        runs_dir = "./runs",
        model_dir = "./trained_models",
        dataset = DatasetKS2D(mode = "train"),
        dataset_test = DatasetKS2D(mode = "test"),
        model = model,
        optimizer = optimizer,
        training_kwargs = dict(batch_size=8),
        load_model_if_exists = True,
        load_optimizer_if_exists = True
    )
    
    trainer.loop()