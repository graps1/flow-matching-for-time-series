import os
import torch
import datetime

experiments = [ "ks2d", "ns2d", "rti3d_sliced", "rti3d_full" ]
for experiment in experiments:
    print("="*10 + " " + experiment + " " +  "="*10)
    path = f"./{experiment}/trained_models/"
    files = os.listdir(path)
    for file in files:
        if file.endswith(".pt"):
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

            if time_passed is not None: print(f"\ttime passed: {datetime.timedelta(seconds=time_passed)}")
            print(f"\tnumber of iterations: {number_iterations}")
            #print(f"  optimizer state: betas = {optimizer["betas"]}, ")
            print(f"\tsize: {size:.2f} MB")
            