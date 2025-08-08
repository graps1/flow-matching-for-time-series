import argparse
import torch
import torch.optim
from fmfts.experiments.trainer import Trainer
from fmfts.dataloader.rti3d_sliced import DatasetSlicedRTI3D
from fmfts.experiments.rti3d_sliced.models import VelocityModelSlicedRTI3D


features_velocity = (128, 196, 196, 256)
lr_max = 1e-4
lr_min = 1e-5
batch_size = 4
At = 3/4
dt, dy, dz = 5, 1, 1 
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
        model = VelocityModelSlicedRTI3D(features = features_velocity)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)

        trainer = Trainer(
            model_name = args.modeltype,
            checkpoints_dir = "./checkpoints",
            runs_dir = "./runs",
            model_dir = "./trained_models",
            model = model,
            optimizer = optimizer,
            dataset =      DatasetSlicedRTI3D(mode = "train", At=At, dt=dt, dy=dy, dz=dz, include_timestamp=include_timestamp),
            dataset_test = DatasetSlicedRTI3D(mode = "test" , At=At, dt=dt, dy=dy, dz=dz, include_timestamp=include_timestamp),
            lr_min = lr_min,
            training_kwargs = dict(batch_size=batch_size),
            load_model_if_exists = not args.new,
            load_optimizer_if_exists = not args.new
        )
        
    trainer.loop()