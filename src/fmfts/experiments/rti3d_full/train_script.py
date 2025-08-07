import argparse
import torch
import torch.optim
from fmfts.experiments.trainer import Trainer
from fmfts.dataloader.rti3d_full import DatasetFullRTI3D
from fmfts.experiments.rti3d_full.models import VelocityModelFullRTI3D


features = (64, 96, 128)
lr_max = 1e-4
lr_min = 1e-5
batch_size = 16
At = 3/4
dt, dx, dy, dz = 5, 4, 4, 4
include_timestamp = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modeltype", help="must be either 'velocity' or 'flow'")
    parser.add_argument("--new", help="creates and trains a new model", action="store_true")

    args = parser.parse_args()
    assert args.modeltype in [ "velocity", "flow" ]

    print(f"creating new model: {args.new}")
    torch.set_default_device("cuda")

    if args.modeltype == "velocity":
        model = VelocityModelFullRTI3D(features = features)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)

        trainer = Trainer(
            model_name = "velocity",
            checkpoints_dir = "./checkpoints",
            runs_dir = "./runs",
            model_dir = "./trained_models",
            model = model,
            optimizer = optimizer,
            dataset =      DatasetFullRTI3D(mode = "train", At=At, dt=dt, dx=dx, dy=dy, dz=dz, include_timestamp=include_timestamp),
            dataset_test = DatasetFullRTI3D(mode = "test" , At=At, dt=dt, dx=dx, dy=dy, dz=dz, include_timestamp=include_timestamp),
            lr_min = lr_min,
            training_kwargs = dict(batch_size=batch_size),
            load_model_if_exists = not args.new,
            load_optimizer_if_exists = not args.new
        )
        
    trainer.loop()