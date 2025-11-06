import os
import torch
import datetime
import argparse

experiments = [ "dNSE", "sRTI", "dRTI" ]


parser = argparse.ArgumentParser()
parser.add_argument("--experiment", "-e", type=str, default=None, help="Specify a single experiment to print stats for.")
parser.add_argument("--prefix", "-p", type=str, default="", help="File prefix to filter models.")
args = parser.parse_args()

if args.experiment is not None:
    experiments = [args.experiment]

for experiment in experiments:
    print("="*10 + " " + experiment + " " +  "="*10)
    path = f"./{experiment}/trained_models/"
    files = os.listdir(path)
    for file in files:
        if file.startswith(args.prefix) and file.endswith(".pt"):
            size = os.path.getsize(os.path.join(path, file)) / (1024*1024)
            print(f"{file}:")
            state_dict = torch.load(os.path.join(path, file), weights_only=True)
            time_passed = state_dict.get("time_passed", None)
            number_iterations = state_dict.get("tensorboard_ctr", None)
            optimizer = state_dict.get("optimizer", None)
           
            if optimizer is not None: 
                stack = [ ("",k,optimizer) for k in optimizer.keys()]
                while len(stack)>0:
                    parent, key, d = stack.pop()
                    if key == "param_groups":
                        for group in d["param_groups"]:
                            print(f"\t{parent} learning rate: {group.get("lr", None)}")
                    elif isinstance(d[key], dict):
                        for k in d[key].keys():
                            stack.append((parent + "/" + str(key), k, d[key]))
            
            model = state_dict["model"]
            top_levels = { k.split(".")[0] for k in model.keys() }
            for tl in top_levels:
                total_params = 0
                for k in model.keys():
                    if k.startswith(tl + "."):
                        total_params += torch.tensor(model[k].shape).prod()
                print(f"\t{tl} parameters: {total_params/1e6:.2f} million")

            if time_passed is not None: 
                print(f"\ttime passed: {datetime.timedelta(seconds=time_passed)}")
                if number_iterations is not None:
                    print(f"\titerations/second: {number_iterations / time_passed:.2f}")
            print(f"\tnumber of iterations: {number_iterations}")
            #print(f"  optimizer state: betas = {optimizer["betas"]}, ")
            print(f"\tsize: {size:.2f} MB")
            