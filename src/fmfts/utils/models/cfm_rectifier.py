import torch
import copy
from torch.utils.data import DataLoader
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.models.cfm_velocity import VelocityModel
from fmfts.utils.loss_fn import sobolev

class Rectifier(TimeSeriesModel):
    def __init__(self, velocity_model: VelocityModel):
        super().__init__()
        self.rectified_velocity_model = copy.deepcopy(velocity_model)
        self.base_velocity_model = copy.deepcopy(self.rectified_velocity_model)

    def advance(self):
        self.base_velocity_model = copy.deepcopy(self.rectified_velocity_model)
        for param in self.base_velocity_model.parameters(): param.requires_grad = False

    def forward(self, x, y, tx):
        return self.rectified_velocity_model.forward(x, y, tx)
        
    def compute_loss(self, y1, x1, steps=10, method="midpoint"):
        # x1 is ignored, but I'm still using it for consistency w/ other methods
        x0 = self.base_velocity_model.p0.sample(y1.shape).to(y1.device)
        x1 = torch.no_grad(self.base_velocity_model.sample)(y1, x0=x0, steps=steps, method=method)

        bs = y1.shape[0] 
        tx = torch.rand(bs)
        tx_ = tx.view(-1, *[1]*(x1.dim()-1)) 
        x = (1 - tx_)*x0 + tx_*x1

        v = self.rectified_velocity_model.forward(x, y1, tx)
        if self.base_velocity_model.loss_fn == "l2":        loss = ( v - (x1 - x0) ).pow(2).mean()
        elif self.base_velocity_model.loss_fn == "sobolev": loss = sobolev(v - (x1 - x0), alpha=1.0, beta=1.0, t=tx)
        return loss

    # def train_model(self, dataset, opt, batch_size=8, steps = 10, method = "midpoint"):
    #     dataloader = DataLoader(
    #         dataset, 
    #         batch_size=batch_size, 
    #         shuffle=True, 
    #         num_workers=0,  
    #         generator=torch.Generator(device='cuda'))

    #     dataiter = iter(dataloader)

    #     ctr = 0
    #     while True:
    #         try:    y1, _ = next(dataiter)
    #         except: y1, _ = next(dataiter := iter(dataloader))

    #         opt.zero_grad()
    #         loss = self.compute_loss(y1, None, ctr, steps=steps, method=method)
    #         loss.backward()
    #         opt.step()

    #         ctr += 1
    #         yield loss

