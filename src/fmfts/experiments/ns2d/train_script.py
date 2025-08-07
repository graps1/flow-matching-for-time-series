import argparse
import torch
import torch.optim
from fmfts.experiments.trainer import Trainer
from fmfts.dataloader.ns2d import DatasetNS2D
from fmfts.experiments.ns2d.models import VelocityModelNS2D, FlowModelNS2D

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("modeltype", help="must be either 'velocity' or 'flow'")
    parser.add_argument("--new", help="creates and trains a new model", action="store_true")

    args = parser.parse_args()
    assert args.modeltype in [ "velocity", "flow" ]

    print(f"creating new model: {args.new}")
    torch.set_default_device("cuda")

    if args.modeltype == "velocity":
        model = VelocityModelNS2D(features = (64, 96, 96, 128))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        trainer = Trainer(
            model_name = "velocity",
            checkpoints_dir = "./checkpoints",
            runs_dir = "./runs",
            model_dir = "./trained_models",
            dataset = DatasetNS2D(mode = "train"),
            dataset_test = DatasetNS2D(mode = "test"),
            model = model,
            optimizer = optimizer,
            lr_min = 1e-5,
            training_kwargs = dict(batch_size=16),
            load_model_if_exists = not args.new,
            load_optimizer_if_exists = not args.new
        )
    elif args.modeltype == "flow":
        velocity_model = VelocityModelNS2D(features = (64, 96, 96, 128))
        velocity_model_path = "./trained_models/model_velocity.pt"
        try:    velocity_model.load_state_dict(torch.load(velocity_model_path, weights_only=True))
        except: raise Exception(f"couldn't load velocity model ({velocity_model_path})")

        flow_model = FlowModelNS2D(velocity_model, features = (64, 96, 96, 128))
        optimizer = torch.optim.Adam(flow_model.parameters(), lr=1e-4)

        trainer = Trainer(
            model_name = "flow",
            checkpoints_dir = "./checkpoints",
            runs_dir = "./runs",
            model_dir = "./trained_models",
            dataset = DatasetNS2D(mode = "train"),
            dataset_test = DatasetNS2D(mode = "test"),
            model = flow_model,
            optimizer = optimizer,
            lr_min = 1e-5,
            training_kwargs = dict(batch_size=8, steps=5),
            load_model_if_exists = not args.new,
            load_optimizer_if_exists = not args.new
        )
        
    trainer.loop()