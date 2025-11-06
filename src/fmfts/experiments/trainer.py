import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter 
import time
import argparse
import tqdm

from fmfts.utils.models.cfm_rectifier import Rectifier
from fmfts.utils.models.cfm_prog_dist import ProgressiveDistillation
from fmfts.utils.models.add import AdversarialDiffusionDistillation

from fmfts.experiments.sRTI.training_parameters import params as sRTI_params
from fmfts.experiments.dRTI.training_parameters import params as dRTI_params
from fmfts.experiments.dNSE.training_parameters import params as dNSE_params

import warnings
warnings.filterwarnings("ignore") 

experiment2params = {
    "sRTI": sRTI_params,
    "dRTI": dRTI_params,
    "dNSE": dNSE_params,
}

modes = [ "velocity", "dir_dist", "flow", "rectifier", "add", "prog_dist", "deterministic" ]

if __name__ == "__main__":
    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser()

    # general settings
    parser.add_argument("experiment", 
                        help=f"must be in {list(experiment2params.keys())}")
    parser.add_argument("mode", 
                        help=f"must be in {list(modes)}")
    parser.add_argument("--checkpoint", "-c", 
                        help="the savefile to load", 
                        default=None)
    parser.add_argument("--velocity", "-v", 
                        help="the savefile to load the velocity model from (if necessary for distillation)", 
                        default=None)
    parser.add_argument("--advance", "-a", 
                        help="whether to continue training or to advance to the next level (only for rectifier or prog_dist)", 
                        action="store_true")
    parser.add_argument("--max-iter", "-m", 
                        help="maximum number of iterations to train for", 
                        type=int, default=1000000)

    # additional parameters that can be set by the user
    parser.add_argument("--pd-stage", "-pds", 
                        help="overwrites the progressive distillation stage (only for mode=prog_dist)", 
                        default=None, type=int)
    parser.add_argument("--pd-k", "-pdk", 
                        help="overwrites the number of progressive distillation expert iterations (only for mode=prog_dist)", 
                        default=None, type=int)
    parser.add_argument("--add-lambda", "-addl", 
                        help="overwrites the distillation weight (lambda) (only for mode=add)", 
                        default=None, type=float)
    parser.add_argument("--add-gamma", "-addg", 
                        help="overwrites the gradient penalty weight (gamma) (only for mode=add)", 
                        default=None, type=float)

    args = parser.parse_args()
    assert args.mode in modes, f"mode must be in {list(modes)}"
    assert args.experiment in experiment2params, f"experiment must be in {list(experiment2params.keys())}"
    params = experiment2params[args.experiment]
    
    modelparams = params.get(args.mode, dict())
    model_kwargs = modelparams.get("model_kwargs", dict())

    #region overwrite parameters if specified by user
    if args.mode == "prog_dist":
        if args.pd_stage is not None:
            print(f"overwriting stage to stage={args.pd_stage}")
            model_kwargs["stage"] = args.pd_stage
        if args.pd_k is not None:
            print(f"overwriting number of expert iterations to K={args.pd_k}")
            model_kwargs["K"] = args.pd_k
    if args.mode == "add":
        if args.add_lambda is not None:
            print(f"overwriting lambda to lambda={args.add_lambda}")
            model_kwargs["lmbda"] = args.add_lambda
        if args.add_gamma is not None:
            print(f"overwriting gamma to gamma={args.add_gamma}")
            model_kwargs["gamma"] = args.add_gamma
    #endregion


    if not args.checkpoint: print(f"model parameters: {model_kwargs}")
    print(f"training parameters: {modelparams["training_kwargs"]}")
    print(f"optimizer parameters: {modelparams["optimizer_init"]}")
    print(f"dataset parameters: {params["dataset"].get("kwargs", dict())}")

    print(f"creating new model: {'NO' if args.checkpoint else 'YES'}")
    state_dir = f"{args.experiment}/trained_models"

    # Ensure output directories exist
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(f"{args.experiment}/checkpoints", exist_ok=True)
    os.makedirs(f"{args.experiment}/runs", exist_ok=True)


    #region initialize model

    # adds class to modelparams (it's the same for all experiments)
    # for wrappers such as Rectifier and ProgressiveDistillation
    if args.mode == "rectifier": modelparams["cls"] = Rectifier
    if args.mode == "prog_dist": modelparams["cls"] = ProgressiveDistillation
    if args.mode == "add":       modelparams["cls"] = AdversarialDiffusionDistillation
    
    # loads velocity model
    if args.mode in ["flow", "dir_dist", "rectifier", "add", "prog_dist"]:
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
    if args.checkpoint:
        serialized_state =  torch.load(args.checkpoint, weights_only=True)
        time_passed_init = serialized_state["time_passed"]
        model.load_state_dict(serialized_state["model"])
        ctr_init = serialized_state.get("tensorboard_ctr", 0)

        for k, o in optimizers.items(): o.load_state_dict(serialized_state["optimizer"][k])
        model.update_optimizers(optimizers, **modelparams["optimizer_init"])
        print(f"loaded serialized state (path: {args.checkpoint})")

    if args.advance and args.mode in ["rectifier", "prog_dist"]:
        print("advancing to the next level.")
        model.advance(**modelparams["optimizer_init"])

    #endregion


    #region TRAINING LOOP
    writer = SummaryWriter(f"{args.experiment}/runs")
    dataset_train = params["dataset"]["cls"](mode = "train", **params["dataset"]["kwargs"])
    dataset_test  = params["dataset"]["cls"](mode = "test" , **params["dataset"]["kwargs"])
    starting_time = time.time()
    iterator = iter(model.train_model(dataset_train, dataset_test, optimizers, **modelparams["training_kwargs"]))

    for ctr in (pbar := tqdm.tqdm(range(args.max_iter - ctr_init))):
        update = next(iterator)
        ctr_total = ctr_init + ctr

        #region status update (terminal and tensorboard) 
        time_passed = time_passed_init + (time.time() - starting_time)
        seconds = int(time_passed) % 60
        minutes = int(time_passed / 60) % 60
        hours   = int(time_passed / (60*60)) % 24
        days    = int(time_passed / (24*60*60))

        pbar.set_description(f"{str(model)} | total_iters = {ctr_total}/{args.max_iter}")
        for k, v in update.items(): writer.add_scalars(f"{args.mode}/{k}", v, ctr_init + ctr)
        
        if ctr % 1000 == 0 or ctr == args.max_iter - 1: 
            timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":","_").replace("-","_")
            
            full_state_dict = model.state_dict()
                    
            serialized_state = { 
                "model": model.state_dict(),
                "optimizer": { k: o.state_dict() for k,o in optimizers.items() },
                "time_passed": time_passed,
                "tensorboard_ctr": ctr_init + ctr
            }

            state_path = f"{state_dir}/{model.filename}"
            fname_checkpoint = ".".join( state_path.split("/")[-1].split(".")[:-1] )

            torch.save(serialized_state, state_path)
            time.sleep(0.1)
            torch.save(serialized_state, f"{args.experiment}/checkpoints/{fname_checkpoint}__{timestamp}.pt")
    #endregion
