import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

# include_timestamp = True
# include_vertical_position = True
# 
# TRAIN_MODE = "velocity" # velocity
# settings = {
#     "one_step": {
#         "lr": 1e-4,
#         "modelpath": "one_step_model.pt",
#         "kwargs_train": { 
#             "steps": 15,
#             "batch_size": 4
#         }
#     },
#     "flow": {
#         "lr": 1e-4,
#         "modelpath": "flow_model.pt",
#         "kwargs_train": { 
#             "steps": 5,
#             "batch_size": 4
#         }
#     },
#     "velocity": {
#         "lr": 1e-4,
#         "modelpath": "velocity_model.pt",
#         "kwargs_train": { 
#             "batch_size": 4
#         }
#     }
# }[TRAIN_MODE]
# print("training mode:", TRAIN_MODE)
# print("settings:", settings)
# 
# torch.set_default_device("cuda")


# dataset = DatasetFullNS3D(
#     history=1, At=3/4, dt=5, dx=4, dy=4, dz=4, train=True, 
#     include_timestamp=include_timestamp)
# dataset_test = DatasetFullNS3D(
#     history=1, At=3/4, dt=5, dx=4, dy=4, dz=4, train=False, 
#     include_timestamp=include_timestamp)



# if os.path.exists(settings["modelpath"]):
#     self.model = torch.load(settings["modelpath"], weights_only=False)
#     print("loaded model")
# else:
#     print("couldn't load model")
#     if TRAIN_MODE == "velocity":
#         self.model = model.Velocity(
#             include_vertical_position=include_vertical_position, 
#             include_timestamp=include_timestamp
#         )

    # NOTE: NOT YET IMPLEMENTED
    # elif TRAIN_MODE == "flow":
    #     if os.path.exists("velocity_model.pt"):
    #         v = torch.load("velocity_model.pt", weights_only=False)
    #         u = model.Flow(velocity_model=v)
    #         print("loaded velocity model")
    #     else:
    #         print("couldn't load velocity model. Cannot train flow model! Exiting.")
    #         exit()
    # elif TRAIN_MODE == "one_step":
    #     if os.path.exists("velocity_model.pt"):
    #         v = torch.load("velocity_model.pt", weights_only=False)
    #         u = model.OneStepModel(velocity_model=v)
    #         print("loaded velocity model")
    #     else:
    #         print("couldn't load velocity model. Cannot train one step model! Exiting.")
    #         exit()

# if os.path.exists(f"optimizer_{TRAIN_MODE}.pt"):
#     opt = torch.optim.Adam(self.model.parameters(), lr=settings["lr"])
#     opt.load_state_dict(torch.load(f"optimizer_{TRAIN_MODE}.pt", weights_only=True))
# else:
#     opt = torch.optim.Adam(self.model.parameters(), lr=settings["lr"])
#     # opt = torch.optim.SGD(u.parameters(), lr=settings["lr"], momentum=0.99)



class Trainer:
    def __init__(self,
                 model_name,
                 dataset,
                 dataset_test,
                 model,
                 training_kwargs,
                 optimizer,
                 lr_min = 5e-6,
                 loss_decay = 0.99,
                 runs_dir = "./runs",
                 model_dir = "./", 
                 checkpoints_dir = "./checkpoints",
                 load_model_if_exists=True,
                 load_optimizer_if_exists=True):

        self.model_name = model_name
        self.model_dir = model_dir
        self.checkpoints_dir = checkpoints_dir
        self.dataset = dataset
        self.dataset_test = dataset_test
        self.model = model
        self.training_kwargs = training_kwargs
        self.optimizer = optimizer
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=500, eta_min=lr_min)
        self.writer = SummaryWriter(runs_dir)
        self.loss_decay = loss_decay

        self.model_path = f"{self.model_dir}/model_{self.model_name}.pt"
        self.optimizer_path = f"{self.model_dir}/optimizer_{self.model_name}.pt"
    
        if load_model_if_exists and os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
            print(f"loaded model {self.model_path}")
        if load_optimizer_if_exists and os.path.exists(self.optimizer_path):
            self.optimizer.load_state_dict(torch.load(self.optimizer_path, weights_only=True))
            print(f"loaded optimizer {self.optimizer_path}")

        self.dataloader_test = DataLoader(
            dataset_test, 
            batch_size=1, 
            shuffle=True, 
            num_workers=0,  
            generator=torch.Generator(device='cuda'))

    def loop(self):
        loss_test_avg = None
        loss_train_avg = None

        for ctr, loss_train in enumerate(self.model.train_model(self.dataset, self.optimizer, **self.training_kwargs)):

            if ctr % 10 == 0:
                with torch.no_grad():
                    y1, x1 = next(iter(self.dataloader_test))
                    loss_test = self.model.compute_loss(y1, x1).item()
                    self.writer.add_scalars(f"loss_{self.model_name}", { "train": loss_train, "test": loss_test}, ctr)
                    if ctr == 0:    loss_test_avg = loss_test
                    else:           loss_test_avg = self.loss_decay * loss_test_avg  + (1 - self.loss_decay) * loss_test
                
            if ctr == 0:    loss_train_avg = loss_train
            else:           loss_train_avg = self.loss_decay * loss_train_avg + (1 - self.loss_decay) * loss_train

            print(
                f"training {self.model_name} model (iter = {ctr}): "+\
                f"loss/train = {loss_train_avg:.4e} loss/test = {loss_test_avg:.4e} "+\
                f"lr = {self.lr_scheduler.get_last_lr()[0]:.4e}")

            self.lr_scheduler.step()
            
            if ctr % 500 == 0: 
                timestamp = datetime.datetime.now().isoformat().split(".")[0].replace(":","_").replace("-","_")

                print(f"saving model @ {self.model_path}")
                print(f"saving optimizer @ {self.optimizer_path}")
                print(f"saving checkpoint model @ {self.checkpoints_dir}/model_{self.model_name}_{timestamp}.pt")
                print(f"saving checkpoint optimizer @ {self.checkpoints_dir}/optimizer_{self.model_name}_{timestamp}.pt")

                torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/model_{self.model_name}_{timestamp}.pt")
                torch.save(self.model.state_dict(), self.model_path)
                time.sleep(0.1)
                torch.save(self.optimizer.state_dict(), f"{self.checkpoints_dir}/optimizer_{self.model_name}_{timestamp}.pt")
                torch.save(self.optimizer.state_dict(), self.optimizer_path)



