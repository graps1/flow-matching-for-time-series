import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import argparse
import pprint

from fmfts.experiments.rti3d_sliced.training_parameters import params as rti3d_sliced_params
from fmfts.experiments.rti3d_full.training_parameters import params as rti3d_full_params
from fmfts.experiments.ns2d.training_parameters import params as ns2d_params
from fmfts.experiments.ks2d.training_parameters import params as ks2d_params

experiment2params = {
    "rti3d_sliced": rti3d_sliced_params,
    "rti3d_full": rti3d_full_params,
    "ns2d": ns2d_params,
    "ks2d": ks2d_params,
}
modeltypes = [ "velocity", "single_step", "flow" ]

if __name__ == "__main__":
    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", help=f"must be in {list(experiment2params.keys())}")
    parser.add_argument("modeltype", help=f"must be in {list(modeltypes)}")
    parser.add_argument("--new", help="creates and trains a new model", action="store_true")

    args = parser.parse_args()
    assert args.modeltype in modeltypes, f"modeltype must be in {list(modeltypes)}"
    assert args.experiment in experiment2params, f"experiment must be in {list(experiment2params.keys())}"
    params = experiment2params[args.experiment]
    
    print("parameters:")
    modelparams = params[args.modeltype]
    pprint.pprint(modelparams)
    pprint.pprint(params["dataset"])

    print(f"creating new model: {'YES' if args.new else 'NO'}")
    state_dir = f"{args.experiment}/trained_models"
    state_path = f"{state_dir}/state_{args.modeltype}.pt"
    print(state_path)
            
    # initialize model
    model_kwargs = modelparams["model_kwargs"]
    if args.modeltype in ["flow", "single_step"]:
        state_velocity_path = f"{state_dir}/state_velocity.pt"
        try:    
            serialized_state_velocity = torch.load(state_velocity_path, weights_only=True)
            velocity_model = params["velocity"]["cls"](**params["velocity"]["model_kwargs"])
            velocity_model.load_state_dict(serialized_state_velocity['model'])
        except: 
            raise Exception(f"couldn't load velocity model ({state_velocity_path})")
        model_kwargs |= {"velocity_model": velocity_model}
    model = modelparams["cls"](**model_kwargs)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=modelparams["lr_max"])
    time_passed_init = 0.0
    ctr_init = 0

    # load state if not new
    if not args.new:
        serialized_state =  torch.load(state_path, weights_only=True)
        time_passed_init = serialized_state["time_passed"]
        model.load_state_dict(serialized_state["model"])
        optimizer.load_state_dict(serialized_state["optimizer"])
        ctr_init = serialized_state.get("tensorboard_ctr", 0)
        for g in optimizer.param_groups: 
            g['lr'] = modelparams["lr_max"]
            g['initial_lr'] = modelparams["lr_max"]
        print(f"loaded serialized state {state_path}")
    
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=modelparams["lr_min"])
    writer = SummaryWriter(f"{args.experiment}/runs")
    dataset_train = params["dataset"]["cls"](mode = "train", **params["dataset"]["kwargs"])
    dataset_test  = params["dataset"]["cls"](mode = "test" , **params["dataset"]["kwargs"])

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

    for ctr, loss_train in enumerate(model.train_model(dataset_train, optimizer, **modelparams["training_kwargs"])):
        loss_print_decay = min(1 - 1/(ctr+1), 0.999)

        if ctr % 10 == 0:
            with torch.no_grad():
                y1, x1 = next(iter(dataloader_test))
                loss_test = model.compute_loss(y1, x1, ctr).item()
                writer.add_scalars(f"loss_{args.modeltype}", { "train": loss_train, "test": loss_test}, ctr_init + ctr)
                if ctr == 0:    loss_test_avg = loss_test
                else:           loss_test_avg = loss_print_decay * loss_test_avg  + (1 - loss_print_decay) * loss_test
            
        if ctr == 0:    loss_train_avg = loss_train
        else:           loss_train_avg = loss_print_decay * loss_train_avg + (1 - loss_print_decay) * loss_train

        time_passed = time_passed_init + (time.time() - starting_time)
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
            "iter": ctr_init + ctr
        })

        lr_scheduler.step()
        
        if ctr % 500 == 0: 
            timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":","_").replace("-","_")
           
            serialized_state = { 
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "time_passed": time_passed,
                "tensorboard_ctr": ctr_init + ctr
            }

            print(f"saving checkpoint @ {state_path}")
            torch.save(serialized_state, state_path)
            time.sleep(0.1)
            print(f"saving checkpoint @ {args.experiment}/checkpoints/state_{args.modeltype}_{timestamp}.pt")
            torch.save(serialized_state, f"{args.experiment}/checkpoints/state_{args.modeltype}_{timestamp}.pt")
