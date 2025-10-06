import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader
# from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import argparse
import pprint

from fmfts.utils.models.cfm_rectifier import Rectifier
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
modeltypes = [ "velocity", "single_step", "flow", "rectifier", "add", "velocity_pd" ]

if __name__ == "__main__":
    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", help=f"must be in {list(experiment2params.keys())}")
    parser.add_argument("modeltype", help=f"must be in {list(modeltypes)}")
    parser.add_argument("--new", "-n", help="creates and trains a new model", action="store_true")
    parser.add_argument("--source", "-s", help="the savefile to load", default="")
    parser.add_argument("--out", "-o", help="the path where to save the file", default="")
    parser.add_argument("--teacher", default=None, help="Path to teacher checkpoint (for velocity_pd). Overrides default if provided.")
    parser.add_argument("--max-iters", type=int, default=None, help="Override max iterations for velocity_pd stage (takes precedence over training_parameters).")

    args = parser.parse_args()
    assert args.modeltype in modeltypes, f"modeltype must be in {list(modeltypes)}"
    assert args.experiment in experiment2params, f"experiment must be in {list(experiment2params.keys())}"
    params = experiment2params[args.experiment]
    
    print("parameters:")
    modelparams = params.get(args.modeltype, dict())
    pprint.pprint(modelparams)
    pprint.pprint(params["dataset"])

    print(f"creating new model: {'YES' if args.new else 'NO'}")
    state_dir = f"{args.experiment}/trained_models"
    state_path = f"{state_dir}/state_{args.modeltype}.pt" if args.source == "" else args.source
    target_path = state_path if args.out == "" else args.out
    # Ensure output directories exist
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(f"{args.experiment}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.experiment}/runs", exist_ok=True)
    print(state_path)
            
    # initialize model
    model_kwargs = modelparams.get("model_kwargs", dict())
    if args.modeltype == "rectifier":
        modelparams["cls"] = Rectifier

    if args.modeltype in ["flow", "single_step", "rectifier", "add", "velocity_pd"]:
        state_velocity_path = f"{state_dir}/state_velocity.pt"
        try:    
            serialized_state_velocity = torch.load(state_velocity_path, weights_only=True)
            velocity_model = params["velocity"]["cls"](**params["velocity"]["model_kwargs"])
            velocity_model.load_state_dict(serialized_state_velocity['model'])
        except: 
            raise Exception(f"couldn't load velocity model ({state_velocity_path})")
        
    if args.modeltype in ["flow", "single_step"]:
        model_kwargs |= {"velocity_model": velocity_model}
    else:
        # Allow custom teacher checkpoint path via --teacher; otherwise use default
        teacher1_path = args.teacher if args.teacher is not None else f"{state_dir}/state_velocity_teacher1.pt"
        serialized_state_teacher1 = torch.load(teacher1_path, weights_only=True)
        teacher = params["velocity"]["cls"](**params["velocity"]["model_kwargs"])
        teacher.load_state_dict(serialized_state_teacher1['model'])
        model_kwargs |= {"teacher": teacher}
    model = modelparams["cls"](**model_kwargs)

    # initialize optimizer
    optimizers = model.init_optimizers(**modelparams["optimizer_init"]) # torch.optim.AdamW(model.parameters(), lr=modelparams["lr_max"], weight_decay=0.0)
    time_passed_init = 0.0
    ctr_init = 0

    # load state if not new
    if not args.new:
        serialized_state =  torch.load(state_path, weights_only=True)
        time_passed_init = serialized_state["time_passed"]
        # For velocity_pd, checkpoints intentionally exclude teacher weights.
        # Load non-strictly to avoid errors on missing teacher keys.
        strict_load = args.modeltype != "velocity_pd"
        model.load_state_dict(serialized_state["model"], strict=strict_load)
        #optimizer.load_state_dict(serialized_state["optimizer"])
        ctr_init = serialized_state.get("tensorboard_ctr", 0)
        for k, o in optimizers.items(): o.load_state_dict(serialized_state["optimizer"][k])
        model.update_optimizers(optimizers, **modelparams["optimizer_init"])
        print(f"loaded serialized state {state_path}")

    if isinstance(model, Rectifier):
        model.advance()
        print(f"loaded rectifier. advancing to the next level")
    
    # Determine PD stage length (only for velocity_pd) and set scheduler T_max accordingly
    pd_max_iters = None
    if args.modeltype == "velocity_pd":
        pd_max_iters = modelparams.get("training_kwargs", {}).get("max_iters")
        if args.max_iters is not None:
            pd_max_iters = args.max_iters
    T_max = pd_max_iters if pd_max_iters is not None else 500
    # lr_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=modelparams["lr_min"])
    writer = SummaryWriter(f"{args.experiment}/runs")
    dataset_train = params["dataset"]["cls"](mode = "train", **params["dataset"]["kwargs"])
    dataset_test  = params["dataset"]["cls"](mode = "test" , **params["dataset"]["kwargs"])

    # dataloader_test = DataLoader(
    #     dataset_test, 
    #     batch_size=1, 
    #     shuffle=True, 
    #     num_workers=0,  
    #     generator=torch.Generator(device='cuda'))
    
    
    loss_test_avg = None
    loss_train_avg = None
    starting_time = time.time()

    # TRAINING LOOP

    for ctr, update in enumerate(model.train_model(dataset_train, dataset_test, optimizers, **modelparams["training_kwargs"])):
        loss_print_decay = min(1 - 1/(ctr+1), 0.999)

        # if ctr % 10 == 0:
        #     with torch.no_grad():
        #         y1, x1 = next(iter(dataloader_test))
        #         model.eval()
        #         loss_test = model.compute_loss(y1, x1, ctr).item()
        #         model.train()
        #         writer.add_scalars(f"loss_{args.modeltype}", { "train": loss_train, "test": loss_test}, ctr_init + ctr)
        #         if ctr == 0:    loss_test_avg = loss_test
        #         else:           loss_test_avg = loss_print_decay * loss_test_avg  + (1 - loss_print_decay) * loss_test
            
        # if ctr == 0:    loss_train_avg = loss_train
        # else:           loss_train_avg = loss_print_decay * loss_train_avg + (1 - loss_print_decay) * loss_train

        time_passed = time_passed_init + (time.time() - starting_time)
        seconds = int(time_passed) % 60
        minutes = int(time_passed / 60) % 60
        hours   = int(time_passed / (60*60)) % 24
        days    = int(time_passed / (24*60*60))

        pprint.pprint({
            "modeltype": args.modeltype,
            # "loss/train": f"{loss_train_avg:.4e}",
            # "loss/test": f"{loss_test_avg:.4e}",
            # "lr": f"{lr_scheduler.get_last_lr()[0]:.4e}",
            "update": update,
            "time_passed": f"{days}d {hours}h {minutes}m {seconds}s",
            "iter": ctr_init + ctr
        })

        # lr_scheduler.step()
        for k, v in update.items():
            writer.add_scalars(f"{args.modeltype}/{k}", v, ctr_init + ctr)

        # Stop only for velocity_pd if a max-iteration budget is specified
        if pd_max_iters is not None and (ctr_init + ctr + 1) >= pd_max_iters:
            # Final save on exit (student-only state)
            full_state_dict = model.state_dict()
            student_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith("teacher_velocity.")}
            serialized_state = {
                "model": student_state_dict,
                "optimizer": { k: o.state_dict() for k,o in optimizers.items() },
                "time_passed": time_passed,
                "tensorboard_ctr": ctr_init + ctr + 1,
            }
            print(f"reached max iters ({pd_max_iters}); saving final checkpoint @ {state_path}")
            torch.save(serialized_state, state_path)
            timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":","_").replace("-","_")
            print(f"saving final checkpoint @ {args.experiment}/checkpoints/state_{args.modeltype}_{timestamp}.pt")
            torch.save(serialized_state, f"{args.experiment}/checkpoints/state_{args.modeltype}_{timestamp}.pt")
            break
        
        if ctr % 10000 == 0: 
            timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":","_").replace("-","_")
            
            #Only save student model (without teacher)
            full_state_dict = model.state_dict()
            student_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith("teacher_velocity.")}
                    
            serialized_state = { 
                # "model": student_state_dict,
                "model": model.state_dict(),
                "optimizer": { k: o.state_dict() for k,o in optimizers.items() },
                "time_passed": time_passed,
                "tensorboard_ctr": ctr_init + ctr
            }

            print(f"saving checkpoint @ {target_path}")
            torch.save(serialized_state, target_path)
            time.sleep(0.1)
            fname = ".".join( target_path.split("/")[-1].split(".")[:-1] )
            print(f"saving checkpoint @ {args.experiment}/checkpoints/state_{fname}_{timestamp}.pt")
            torch.save(serialized_state, f"{args.experiment}/checkpoints/state_{fname}_{timestamp}.pt")
