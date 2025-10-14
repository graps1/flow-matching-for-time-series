import torch
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.loss_fn import sobolev

class VelocityModel(TimeSeriesModel):
    def __init__(self, p0=torch.distributions.Normal(0,1), loss="l2"):
        super().__init__()
        self.p0 = p0 
        self.loss_fn = loss
        assert self.loss_fn in ["l2", "sobolev"], "loss must be either 'l2' or 'sobolev'"

    def forward(self, x, y, tx):
        raise NotImplementedError()

    def compute_loss(self, y1, x1, ctr, x0=None):
        bs = y1.shape[0] 
        if x0 is None: x0 = self.p0.sample(x1.shape).to(x1.device)
        tx = torch.rand(bs)
        tx_ = tx.view(-1, *[1]*(x1.dim()-1)) 
        x = (1 - tx_)*x0 + tx_*x1

        v = self.forward(x, y1, tx)
        if self.loss_fn == "l2":      loss = ( v - (x1 - x0) ).pow(2).mean()
        elif self.loss_fn == "sobolev": loss = sobolev(v - (x1 - x0), alpha=1.0, beta=1.0, t=tx)
        return loss
    
    def integrate(self, y1, x, t=0.0, dt=1.0, steps = 10, method="midpoint"):
        assert method in ["euler", "rk4", "midpoint"]
        if not isinstance(t, torch.Tensor):     t = torch.tensor(t)
        if not isinstance(dt, torch.Tensor):    dt = torch.tensor(dt)
        if not isinstance(steps, torch.Tensor): steps = torch.tensor(steps)

        dt_small = (dt / steps).expand(len(y1))
        dt_small_ = dt_small.view(-1, *[1]*(y1.dim()-1))

        for i in range(steps):
            if method == "euler":
                x = x + dt_small_ * self(x, y1, t + i*dt_small)
            elif method == "midpoint":
                k0 = x + dt_small_/2 * self(x,  y1, t + i*dt_small)
                k1 = x + dt_small_   * self(k0, y1, t + (i+0.5)*dt_small)
                x = k1
            elif method == "rk4":
                k0 = self(x, y1, i*dt_small)
                k1 = self(x + dt_small_/2*k0, y1, t + (i+0.5)*dt_small)
                k2 = self(x + dt_small_/2*k1, y1, t + (i+0.5)*dt_small)
                k3 = self(x + dt_small_*k2, y1, t + i*dt_small)
                x = x + dt_small_/6*(k0+2*k1+2*k2+k3)

        return x

    def sample(self, y1, x0=None, steps=10, method="midpoint"):
        if x0 is None: x0 = self.p0.sample(y1.shape).to(y1.device)
        x = self.integrate(y1, x0, t=0.0, dt=1.0, steps=steps, method=method)
        return x
