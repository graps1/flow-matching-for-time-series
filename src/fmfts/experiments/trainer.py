import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import argparse
import pprint

import fmfts.experiments.rti3d_sliced.training_parameters as rti3d_sliced_params
import fmfts.experiments.rti3d_full.training_parameters as rti3d_full_params
import fmfts.experiments.ns2d.training_parameters as ns2d_params
import fmfts.experiments.ks2d.training_parameters as ks2d_params

experiment2params = {
    "rti3d_sliced": rti3d_sliced_params.TrainingParameters(),
    "rti3d_full": rti3d_full_params.TrainingParameters(),
    "ns2d": ns2d_params.TrainingParameters(),
    "ks2d": ks2d_params.TrainingParameters()
}
modeltypes = [ "velocity", "single_step", "flow" ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", help=f"must be in {list(experiment2params.keys())}")
    parser.add_argument("modeltype", help=f"must be in {list(modeltypes)}")
    parser.add_argument("--new", help="creates and trains a new model", action="store_true")

    args = parser.parse_args()
    assert args.modeltype in modeltypes, f"modeltype must be in {list(modeltypes)}"
    assert args.experiment in experiment2params, f"experiment must be in {list(experiment2params.keys())}"
    params = experiment2params[args.experiment]

    print(f"creating new model: {args.new}")
    print("parameters:")
    pprint.pprint(params)

    torch.set_default_device("cuda")

    if args.modeltype == "velocity":
        model = params.VelocityCls(features = params.features_velocity)
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_max)
        training_kwargs = dict(batch_size=params.batch_size)
    elif args.modeltype in ["flow", "single_step"]:
        velocity_model = params.VelocityCls(features = params.features_velocity)
        velocity_model_path = f"{params.model_dir}/model_velocity.pt"
        try:    velocity_model.load_state_dict(torch.load(velocity_model_path, weights_only=True))
        except: raise Exception(f"couldn't load velocity model ({velocity_model_path})")
        
        if args.modeltype == "single_step":
            model = params.SingleStepCls(velocity_model = velocity_model, features = params.features_single_step)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_max)
            training_kwargs = dict(batch_size=params.batch_size, steps=params.steps_single_step)
        else:
            model = params.FlowCls(velocity_model = velocity_model, features = params.features_flow)
            optimizer = torch.optim.Adam(model.parameters(), lr=params.lr_max)
            training_kwargs = dict(batch_size=params.batch_size, steps=params.steps_flow)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=params.lr_min)
    writer = SummaryWriter(params.runs_dir)

    model_path = f"{params.model_dir}/model_{args.modeltype}.pt"
    optimizer_path = f"{params.model_dir}/optimizer_{args.modeltype}.pt"

    if not args.new:
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, weights_only=True))
            print(f"loaded model {model_path}")
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path, weights_only=True))
            for g in optimizer.param_groups: g['lr'] = params.lr_max
            print(f"loaded optimizer {optimizer_path}")

    dataset_train = params.DatasetCls(mode = "train", **params.kwargs_dataset_cls)
    dataset_test  = params.DatasetCls(mode = "test" , **params.kwargs_dataset_cls)
    loss_print_decay = 0.99

    dataloader_test = DataLoader(
        dataset_test, 
        batch_size=1, 
        shuffle=True, 
        num_workers=0,  
        generator=torch.Generator(device='cuda'))
    
    
    loss_test_avg = None
    loss_train_avg = None
    starting_time = time.time()

    # TRAINING LOOP

    for ctr, loss_train in enumerate(model.train_model(dataset_train, optimizer, **training_kwargs)):

        if ctr % 10 == 0:
            with torch.no_grad():
                y1, x1 = next(iter(dataloader_test))
                loss_test = model.compute_loss(y1, x1).item()
                writer.add_scalars(f"loss_{args.modeltype}", { "train": loss_train, "test": loss_test}, ctr)
                if ctr == 0:    loss_test_avg = loss_test
                else:           loss_test_avg = loss_print_decay * loss_test_avg  + (1 - loss_print_decay) * loss_test
            
        if ctr == 0:    loss_train_avg = loss_train
        else:           loss_train_avg = loss_print_decay * loss_train_avg + (1 - loss_print_decay) * loss_train

        time_passed = time.time() - starting_time
        seconds = int(time_passed) % 60
        minutes = int(time_passed / 60) % 60
        hours   = int(time_passed / (60*60)) % 24
        days    = int(time_passed / (24*60*60))

        pprint.pprint({
            "modeltype": args.modeltype,
            "loss/train": f"{loss_train_avg:.4e}",
            "loss/test": f"{loss_test_avg:.4e}",
            "lr": f"{lr_scheduler.get_last_lr()[0]:.4e}",
            "time_passed": f"{days}d {hours}h {minutes}m {seconds}s",
            "iter": ctr
        })

        lr_scheduler.step()
        
        if ctr % 500 == 0: 
            timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":","_").replace("-","_")

            print(f"saving model @ {model_path}")
            print(f"saving optimizer @ {optimizer_path}")
            print(f"saving checkpoint model @ {params.checkpoints_dir}/model_{args.modeltype}_{timestamp}.pt")
            print(f"saving checkpoint optimizer @ {params.checkpoints_dir}/optimizer_{args.modeltype}_{timestamp}.pt")

            torch.save(model.state_dict(), f"{params.checkpoints_dir}/model_{args.modeltype}_{timestamp}.pt")
            torch.save(model.state_dict(), model_path)
            time.sleep(0.1)
            torch.save(optimizer.state_dict(), f"{params.checkpoints_dir}/optimizer_{args.modeltype}_{timestamp}.pt")
            torch.save(optimizer.state_dict(), optimizer_path)
