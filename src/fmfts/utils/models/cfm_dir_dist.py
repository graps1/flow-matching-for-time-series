import torch 
from fmfts.utils.models.time_series_model import TimeSeriesModel

class DirectDistillationModel(TimeSeriesModel):
    def __init__(self, v):
        super().__init__()
        self.v = v

    def phi(self, x0, y):
        raise NotImplementedError()

    def compute_loss(self, y1, x1, steps=10, method="midpoint"):
        x0 = self.v.p0.sample(x1.shape).to(x1.device)

        with torch.no_grad(): F_multistep = self.v.sample(y1, x0=x0, steps=steps, method=method)
        F_single = self(x0, y1)
        loss = ( F_multistep - F_single ).pow(2).mean()
        return loss
    
    def forward(self, x0, y):
        phi = self.phi(x0, y)
        return x0 + phi
    
    def sample(self, y1, x0=None):
        if x0 is None: x0 = self.p0.sample(y1.shape).to(y1.device)
        x = self(x0, y1)
        return x