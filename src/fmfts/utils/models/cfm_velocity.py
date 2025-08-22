import torch
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.loss_fn import sobolev

class VelocityModel(TimeSeriesModel):
    def __init__(self, p0=torch.distributions.Normal(0,1)):
        super().__init__()
        self.p0 = p0 

    def forward(self, x, y, tx):
        raise NotImplementedError()

    def compute_loss(self, y1, x1, ctr):
        bs = y1.shape[0] 
        x0 = self.p0.sample(x1.shape).to(x1.device)
        tx = torch.rand(bs)
        tx_ = tx.view(-1, *[1]*(x1.dim()-1)) 
        x = (1 - tx_)*x0 + tx_*x1

        v = self.forward(x, y1, tx)
        #loss = ( v - (x1 - x0) ).pow(2).mean()
        loss = sobolev(v - (x1 - x0), alpha=1.0, beta=1.0, t=tx)
        return loss

    def sample(self, y1, x0=None, steps=10, method="midpoint"):
        assert method in ["euler", "rk4", "midpoint"]
        if x0 is None: x0 = self.p0.sample(y1.shape).to(y1.device)
        
        x = x0
        dt = torch.tensor(1/steps).expand(len(y1))
        dt_ = dt.view(-1, *[1]*(y1.dim()-1))
        for i in range(steps):
            if method == "euler":
                x = x + dt_ * self(x, y1, i*dt)
            elif method == "midpoint":
                k0 = x + dt_/2 * self(x,  y1, i*dt)
                k1 = x + dt_   * self(k0, y1, (i+0.5)*dt)
                x = k1
            elif method == "rk4":
                k0 = self(x, y1, i*dt)
                k1 = self(x + dt_/2*k0, y1, (i+0.5)*dt)
                k2 = self(x + dt_/2*k1, y1, (i+0.5)*dt)
                k3 = self(x + dt_*k2, y1, i*dt)
                x = x + dt_/6*(k0+2*k1+2*k2+k3)

        return x