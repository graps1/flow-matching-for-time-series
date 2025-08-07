import os
import torch
import datetime
from torch.utils.tensorboard import SummaryWriter 
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import time



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
        starting_time = time.time()

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

            time_passed = time.time() - starting_time
            seconds = int(time_passed) % 60
            minutes = int(time_passed / 60) % 60
            hours   = int(time_passed / (60*60)) % 24
            days    = int(time_passed / (24*60*60))
            print(
                f"training {self.model_name} model (iter = {ctr}): "+\
                f"loss/train = {loss_train_avg:.4e} loss/test = {loss_test_avg:.4e} "+\
                f"lr = {self.lr_scheduler.get_last_lr()[0]:.4e} "+\
                f"time passed = {days}d {hours}h {minutes}m {seconds}s")

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



