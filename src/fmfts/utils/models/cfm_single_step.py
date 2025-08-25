import torch 
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.loss_fn import sobolev

class SingleStepModel(TimeSeriesModel):
    def __init__(self, v, p0=torch.distributions.Normal(0,1), loss="l2"):
        super().__init__()
        self.v = v
        self.p0 = p0
        self.loss_fn = loss
        assert self.loss_fn in ["l2", "l1", "sobolev"], "loss must be either 'l2', 'l1' or 'sobolev'"

    def phi(self, x0, y, v):
        raise NotImplementedError()

    def compute_loss(self, y1, x1, ctr, steps=10, method="midpoint"):
        x0 = self.p0.sample(x1.shape).to(x1.device)

        F_multistep = torch.no_grad(self.v.sample)(y1, x0=x0, steps=steps, method=method)
        F_single = self(x0, y1)
        if self.loss_fn == "l2": loss = ( F_multistep - F_single ).pow(2).mean()
        elif self.loss_fn == "l1": loss = ( F_multistep - F_single ).abs().mean()
        elif self.loss_fn == "sobolev": loss = sobolev(F_multistep - F_single, alpha=1.0, beta=1.0, t=torch.ones(len(x1)))
        return loss
    
    def forward(self, x0, y):
        # v = torch.no_grad(self.v)(x0, y, torch.zeros(len(x0)))
        phi = self.phi(x0, y, None)
        return x0 + phi
    
    def sample(self, y1, x0=None):
        if x0 is None: x0 = self.p0.sample(y1.shape).to(y1.device)
        x = self(x0, y1)
        return x