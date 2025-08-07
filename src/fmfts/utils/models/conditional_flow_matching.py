import torch
from fmfts.utils.models.time_series_model import TimeSeriesModel

class ConditionalFlowMatching(TimeSeriesModel):
    def __init__(self, p0=torch.distributions.Normal(0,1)):
        super().__init__()
        self.p0 = p0 

    def forward(self, x, y, tx):
        raise NotImplementedError()

    def compute_loss(self, y1, x1):
        bs = y1.shape[0] 
        x0 = self.p0.sample(x1.shape).to(x1.device)
        tx = torch.rand(bs)
        tx_ = tx.view(-1, *[1]*(x1.dim()-1)) 
        x = (1 - tx_)*x0 + tx_*x1

        v = self.forward(x, y1, tx)
        loss = ( v - (x1 - x0) ).pow(2).mean()
        return loss