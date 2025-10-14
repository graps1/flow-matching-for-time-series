import torch
import copy
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.models.cfm_velocity import VelocityModel

class Rectifier(TimeSeriesModel):
    def __init__(self, velocity_model: VelocityModel):
        super().__init__()
        self.register_buffer("stage", torch.tensor(0))
        self.rectified_velocity_model = copy.deepcopy(velocity_model)
        self.advance()
    
    def additional_info(self):
        return { "stage": self.stage.item() }

    def advance(self):
        self.stage = self.stage + 1
        self.base_velocity_model = copy.deepcopy(self.rectified_velocity_model)
        for param in self.base_velocity_model.parameters(): param.requires_grad = False

    def forward(self, x, y, tx):
        return self.rectified_velocity_model.forward(x, y, tx)
    
    def sample(self, y, x0=None, steps=10, method="midpoint"):
        return self.rectified_velocity_model.sample(y, x0=x0, steps=steps, method=method)
        
    def compute_loss(self, y1, x1, steps=10, method="midpoint"):
        # x1 is ignored, but I'm still using it for consistency w/ other methods
        x0 = self.base_velocity_model.p0.sample(y1.shape).to(y1.device)
        x1 = torch.no_grad(self.base_velocity_model.sample)(y1, x0=x0, steps=steps, method=method)

        bs = y1.shape[0] 
        tx = torch.rand(bs)
        tx_ = tx.view(-1, *[1]*(x1.dim()-1)) 
        x = (1 - tx_)*x0 + tx_*x1

        v = self.rectified_velocity_model.forward(x, y1, tx)
        loss = ( v - (x1 - x0) ).pow(2).mean()
        return loss
