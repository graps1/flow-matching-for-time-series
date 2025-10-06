import copy
import torch
from torch.utils.data import DataLoader
from fmfts.utils.unet import ResNet
from fmfts.utils.models.cfm_velocity import VelocityModel
from collections import defaultdict

class AdversarialDiffusionDistillation(torch.nn.Module):
    def __init__(self, velocity_model : VelocityModel):
        super().__init__()
        self.v  = copy.deepcopy(velocity_model)
        self.G_ = copy.deepcopy(velocity_model)
        self.D_ = copy.deepcopy(velocity_model)
        # self.D_ = torch.nn.Sequential(
        #     ResNet(in_channels=2*5, out_channels=128, features=(64, 96, 96, 128), padding=("circular", "zeros"), nl=torch.nn.ReLU()),
        #     torch.nn.Flatten(),
        #     torch.nn.LazyLinear(out_features=64)
        # )
    
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

    def G(self, x, y): return x + self.G_(x,y,torch.zeros(len(x)))
    # def D(self, x, y): return self.D_(torch.cat((x,y),dim=1)) # self.D_(x,y,torch.zeros(len(x)))
    # def D(self, x, y): return self.D_(x,y,torch.zeros(len(x)))
    def D(self, x, y): return self.D_(x,y,torch.zeros(len(x))).flatten(start_dim=1).mean(dim=1)
            
    def train_model(self, dataset_train, dataset_test, optimizers, batch_size=8, w_distillation = 0.5, w_R1 = 1e-4, generator_rate=10, **kwargs):
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
        
            x0 = self.v.p0.sample(x1.shape).to(x1.device)
            x1_fake = self.G(x0, y1)
            d_real = self.D(x1, y1)
            d_fake = self.D(x1_fake, y1)

            ret = defaultdict(dict)

            if ctr % generator_rate == 0:
                opt_G.zero_grad()

                loss_distillation = torch.tensor(0.0 )
                if w_distillation > 0: 
                    x1_true = self.v.sample(y1, x0 = x0, steps=10, method="euler")
                    loss_distillation = (x1_true - x1_fake).pow(2).mean()
                
                loss_G = w_distillation *  loss_distillation + (1 - w_distillation) * d_fake.mean()
                loss_G.backward()
                opt_G.step()

                ret["loss_G"] |= { "train": loss_G.item() }
                ret["loss_distillation"] |= { "train": loss_distillation.item() }

            
            opt_D.zero_grad()
            # d_fake = self.D(x1_fake.detach(), y1)
            # R1 = torch.func.grad( lambda x: self.D(x, y1).mean() )(x1).pow(2).sum()
            # loss_D = w_R1 * R1 + torch.clip(1 - d_fake, min=0).mean() + torch.clip(1 + d_real, min=0).mean()
            d_fake = self.D(x1_fake.detach(), y1)
            R1 = torch.func.grad( lambda x1: self.D(x1, y1).sum() )(x1).pow(2).mean(dim=0).sum()
            loss_D = w_R1 * R1 + torch.clip(1 - d_fake, min=0).mean() + torch.clip(1 + d_real, min=0).mean()
            loss_D.backward()
            opt_D.step()

            ret["loss_D"] |= { "train": loss_D.item() }
            ret["gradient_R1"] |= { "train": R1.item() }

            if ctr % 50 == 0:
                try:    y1, x1 = next(dataiter_test)
                except: y1, x1 = next(dataiter_test := iter(dataloader_test))
                with torch.no_grad():
                    x0 = self.v.p0.sample(x1.shape).to(x1.device)
                    x1_fake = self.G(x0, y1)
                    loss_distillation = torch.tensor(0.0)
                    if w_distillation > 0: 
                        x1_true = self.v.sample(y1, x0, steps=10, method="euler")
                        loss_distillation = (x1_true - x1_fake).pow(2).mean()
                    d_fake = self.D(x1_fake, y1)
                    d_real = self.D(x1, y1)
                    loss_G = w_distillation * loss_distillation + (1-w_distillation) * d_fake.mean()
                    loss_D = torch.clip(1 - d_fake, min=0).mean() + torch.clip(1 + d_real, min=0).mean()
                    ret["loss_G"] |= { "test": loss_G.item() }
                    ret["loss_D"] |= { "test": loss_D.item() }

            yield ret
            ctr += 1

