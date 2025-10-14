import copy
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from fmfts.utils.models.cfm_velocity import VelocityModel
from fmfts.utils.models.time_series_model import TimeSeriesModel

class AdversarialDiffusionDistillation(TimeSeriesModel):
    def __init__(self, velocity_model : VelocityModel, lmbda = 0.5, gamma = 1e-4, generator_rate=10):
        super().__init__()
        self.v  = copy.deepcopy(velocity_model)
        self.G_ = copy.deepcopy(velocity_model)
        self.D_ = copy.deepcopy(velocity_model)
        self.lmbda = lmbda
        self.gamma = gamma
        self.generator_rate = generator_rate

    def additional_info(self):
        return { "lmbda": self.lmbda, "gamma": self.gamma, "grate": self.generator_rate }
    
    def init_optimizers(self, lr_G = 1e-5, lr_D = 1e-4):
        return { "G": torch.optim.Adam(self.G_.parameters(), lr=lr_G, betas=[0., 0.99]), 
                 "D": torch.optim.Adam(self.D_.parameters(), lr=lr_D, betas=[0., 0.99]) }
    
    def update_optimizers(self, optimizers, lr_G = 1e-5, lr_D = 1e-4):
        for g in optimizers["G"].param_groups:
            g["lr"] = lr_G
            g["initial_lr"] = lr_G
        for g in optimizers["D"].param_groups:
            g["lr"] = lr_D
            g["initial_lr"] = lr_D

    def forward(self, x, y, mode="generator"):
        assert mode in ["critic", "generator" ]
        if mode == "critic":    return self.D(x,y)
        if mode == "generator": return self.G(x,y)
    
    def sample(self, y1, x0=None, **kwargs):
        return self.G(x=x0, y=y1)
    
    def compute_loss(self, y1, x1, mode="generator"):
        assert mode in ["critic", "generator", "all" ]

        x0 = self.v.p0.sample(x1.shape).to(x1.device)
        x1_fake = self.G(x0, y1)
        
        if mode in ["all", "critic"]:
            d_fake = self.D(x1_fake.detach(), y1)
            d_real = self.D(x1, y1)
            R1 = torch.func.grad( lambda x1: self.D(x1, y1).sum() )(x1).pow(2).mean(dim=0).sum()
            loss_D = self.gamma * R1 + torch.clip(1 - d_fake, min=0).mean() + torch.clip(1 + d_real, min=0).mean()

        if mode in ["all", "generator"]: 
            d_fake = self.D(x1_fake, y1)
            loss_distillation = torch.tensor(0.0 )
            if self.lmbda > 0: 
                x1_true = self.v.sample(y1, x0 = x0, steps=10, method="euler")
                loss_distillation = (x1_true - x1_fake).pow(2).mean()
            loss_G = self.lmbda *  loss_distillation + (1 - self.lmbda) * d_fake.mean()
        
        if mode == "all":       return (loss_G, loss_distillation, loss_D, R1)
        if mode == "generator": return (loss_G, loss_distillation)
        if mode == "critic":    return (loss_D, R1)


    def G(self, x, y): return x + self.G_(x,y,torch.zeros(len(x)))
    def D(self, x, y): return self.D_(x,y,torch.zeros(len(x))).flatten(start_dim=1).mean(dim=1)
            
    def train_model(self, dataset_train, dataset_test, optimizers, batch_size=8, **kwargs):
        dataloader_train = DataLoader(
            dataset_train, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0,  
            generator=torch.Generator(device='cuda'))

        dataloader_test = DataLoader(
            dataset_test, 
            batch_size=1, 
            shuffle=True, 
            num_workers=0,  
            generator=torch.Generator(device='cuda'))

        dataiter_train = iter(dataloader_train)
        dataiter_test = iter(dataloader_test)
        opt_G = optimizers["G"]
        opt_D = optimizers["D"]

        ctr = 0
        while True:
            try:    y1, x1 = next(dataiter_train)
            except: y1, x1 = next(dataiter_train := iter(dataloader_train))
            
            ret = defaultdict(dict)

            if ctr % self.generator_rate == 0:
                opt_G.zero_grad()
                loss_G, loss_distillation = self.compute_loss(y1, x1, mode="generator")
                loss_G.backward()
                opt_G.step()

                ret["loss_G"]            |= { "train": loss_G.item() }
                ret["loss_distillation"] |= { "train": loss_distillation.item() }
            
            opt_D.zero_grad()
            loss_D, R1 = self.compute_loss(y1, x1, mode="critic")
            loss_D.backward()
            opt_D.step()

            ret["loss_D"] |= { "train": loss_D.item() }
            ret["R1"]     |= { "train": R1.item() }

            if ctr % 50 == 0:
                try:    y1, x1 = next(dataiter_test)
                except: y1, x1 = next(dataiter_test := iter(dataloader_test))
                with torch.no_grad():
                    loss_G, loss_distillation, loss_D, R1 = self.compute_loss(y1, x1, mode="all")
                    ret["loss_G"]            |= { "test": loss_G.item() }
                    ret["loss_D"]            |= { "test": loss_D.item() }
                    ret["loss_distillation"] |= { "test": loss_distillation.item() }
                    ret["R1"]                |= { "test": R1.item() }

            yield ret
            ctr += 1

