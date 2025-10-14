import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter 
import time
import argparse
import pprint

from fmfts.utils.models.cfm_rectifier import Rectifier
from fmfts.utils.models.cfm_velocity_pd import ProgressiveDistillation

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
modes = [ "velocity", "single_step", "flow", "rectifier", "add", "velocity_pd", "deterministic" ]

if __name__ == "__main__":
    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument("experiment", help=f"must be in {list(experiment2params.keys())}")
    parser.add_argument("mode", help=f"must be in {list(modes)}")
    parser.add_argument("--new", "-n", help="creates and trains a new model", action="store_true")
    parser.add_argument("--checkpoint", "-c", help="the savefile to load", default="")
    parser.add_argument("--velocity", "-v", help="the savefile to load the velocity model from (if necessary for distillation)", default=None)
    parser.add_argument("--advance", "-a", help="whether to continue training or to advance to the next level (only for rectifier or velocity_pd)", action="store_true")
    parser.add_argument("--max-iter", "-m", help="maximum number of iterations to train for", type=int, default=1000000)

    args = parser.parse_args()
    assert args.mode in modes, f"mode must be in {list(modes)}"
    assert args.experiment in experiment2params, f"experiment must be in {list(experiment2params.keys())}"
    params = experiment2params[args.experiment]
    
    print("parameters:")
    modelparams = params.get(args.mode, dict())
    pprint.pprint(modelparams)
    pprint.pprint(params["dataset"])

    print(f"creating new model: {'YES' if args.new else 'NO'}")
    state_dir = f"{args.experiment}/trained_models"

    # Ensure output directories exist
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(f"{args.experiment}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.experiment}/runs", exist_ok=True)



    #region initialize model
    model_kwargs = modelparams.get("model_kwargs", dict())

    # adds class to modelparams (it's the same for all experiments)
    # for wrappers such as Rectifier and ProgressiveDistillation
    if args.mode == "rectifier":   modelparams["cls"] = Rectifier
    if args.mode == "velocity_pd": modelparams["cls"] = ProgressiveDistillation
    
    # loads velocity model
    if args.mode in ["flow", "single_step", "rectifier", "add", "velocity_pd"]:
        try:    
            serialized_state_velocity = torch.load(args.velocity, weights_only=True)
            velocity_model = params["velocity"]["cls"](**params["velocity"]["model_kwargs"])
            velocity_model.load_state_dict(serialized_state_velocity['model'])
        except: 
            raise Exception(f"couldn't load velocity model (path: {args.velocity})")

        model_kwargs |= {"velocity_model": velocity_model}

    model = modelparams["cls"](**model_kwargs)
    #endregion




    #region load state if not new
    optimizers = model.init_optimizers(**modelparams["optimizer_init"])
    time_passed_init = 0.0
    ctr_init = 0
    if not args.new:
        serialized_state =  torch.load(args.checkpoint, weights_only=True)
        time_passed_init = serialized_state["time_passed"]
        model.load_state_dict(serialized_state["model"], strict = args.mode != "velocity_pd")
        ctr_init = serialized_state.get("tensorboard_ctr", 0)
        for k, o in optimizers.items(): o.load_state_dict(serialized_state["optimizer"][k])
        model.update_optimizers(optimizers, **modelparams["optimizer_init"])
        print(f"loaded serialized state (path: {args.checkpoint})")

    if args.advance and args.mode in ["rectifier", "velocity_pd"]:
        print("advancing to the next level.")
        model.advance()

    #endregion


    #region TRAINING LOOP
    writer = SummaryWriter(f"{args.experiment}/runs")
    dataset_train = params["dataset"]["cls"](mode = "train", **params["dataset"]["kwargs"])
    dataset_test  = params["dataset"]["cls"](mode = "test" , **params["dataset"]["kwargs"])
    starting_time = time.time()
    for ctr, update in enumerate(model.train_model(dataset_train, dataset_test, optimizers, **modelparams["training_kwargs"])):



        #region status update (terminal and tensorboard) 
        time_passed = time_passed_init + (time.time() - starting_time)
        seconds = int(time_passed) % 60
        minutes = int(time_passed / 60) % 60
        hours   = int(time_passed / (60*60)) % 24
        days    = int(time_passed / (24*60*60))

        if ctr % 100 == 0:
            pprint.pprint({
                "model": str(model),
                "mode": args.mode,
                "filename": model.filename,
                "update": update,
                "time_passed": f"{days}d {hours}h {minutes}m {seconds}s",
                "iter": ctr_init + ctr
            })

        for k, v in update.items(): writer.add_scalars(f"{args.mode}/{k}", v, ctr_init + ctr)
        #endregion

        
        if ctr % 1000 == 0 or ctr > args.max_iter: 
            timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":","_").replace("-","_")
            
            full_state_dict = model.state_dict()
                    
            serialized_state = { 
                "model": model.state_dict(),
                "optimizer": { k: o.state_dict() for k,o in optimizers.items() },
                "time_passed": time_passed,
                "tensorboard_ctr": ctr_init + ctr
            }

            state_path = f"{state_dir}/{model.filename}"

            print(f"saving checkpoint @ {state_path}")
            torch.save(serialized_state, state_path)
            time.sleep(0.1)
            fname = ".".join( state_path.split("/")[-1].split(".")[:-1] )
            print(f"saving checkpoint @ {args.experiment}/checkpoints/{fname}__{timestamp}.pt")
            torch.save(serialized_state, f"{args.experiment}/checkpoints/{fname}__{timestamp}.pt")

        if ctr > args.max_iter: break
    #endregion
