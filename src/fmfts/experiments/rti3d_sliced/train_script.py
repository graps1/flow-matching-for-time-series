import argparse
import torch
import torch.optim
from fmfts.experiments.trainer import Trainer
from fmfts.dataloader.rti3d_sliced import DatasetSlicedRTI3D
from fmfts.experiments.rti3d_sliced.models import VelocityModelSlicedRTI3D,  \
                                                  SingleStepModelSlicedRTI3D, \
                                                  FlowModelSlicedRTI3D


features_velocity = (128, 196, 196, 256)
features_single_step = (128, 196, 256)
features_flow = (128, 196, 256)
lr_max = 1e-4
lr_min = 1e-5
batch_size = 4
At = 3/4
dt, dy, dz = 5, 1, 1 
include_timestamp = True
steps_single_step = 10
steps_flow = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modeltype", help="must be either 'velocity' or 'single_step' or 'flow'")
    parser.add_argument("--new", help="creates and trains a new model", action="store_true")

    args = parser.parse_args()
    assert args.modeltype in [ "velocity", "single_step", "flow" ]

    print(f"creating new model: {args.new}")
    torch.set_default_device("cuda")

    if args.modeltype == "velocity":
        model = VelocityModelSlicedRTI3D(features = features_velocity)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
        training_kwargs = dict(batch_size=batch_size)
    elif args.modeltype in ["flow", "single_step"]:
        velocity_model = VelocityModelSlicedRTI3D(features = features_velocity)
        velocity_model_path = "./trained_models/model_velocity.pt"
        try:    velocity_model.load_state_dict(torch.load(velocity_model_path, weights_only=True))
        except: raise Exception(f"couldn't load velocity model ({velocity_model_path})")
        
        if args.modeltype == "single_step":
            model = SingleStepModelSlicedRTI3D(velocity_model = velocity_model, features = features_single_step)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
            training_kwargs = dict(batch_size=batch_size, steps=steps_single_step)
        else:
            model = FlowModelSlicedRTI3D(velocity_model = velocity_model, features = features_flow)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
            training_kwargs = dict(batch_size=batch_size, steps=steps_flow)

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
        training_kwargs = training_kwargs,
        load_model_if_exists = not args.new,
        load_optimizer_if_exists = not args.new
    )
        
    trainer.loop()