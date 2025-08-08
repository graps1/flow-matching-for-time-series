import torch 
from fmfts.utils.models.time_series_model import TimeSeriesModel

class SingleStepModel(TimeSeriesModel):
    def __init__(self, v, p0=torch.distributions.Normal(0,1)):
        super().__init__()
        self.v = v
        self.p0 = p0

    def phi(self, x0, y, v):
        raise NotImplementedError()

    def compute_loss(self, y1, x1, steps=2):
        x0 = self.p0.sample(x1.shape).to(x1.device)

        with torch.no_grad():
            F_multistep = x0
            dt = torch.ones(len(x0))*(1/steps)
            dt_ = dt.view(-1, *[1]*(x0.dim()-1))
            for k in range(steps): 
                k0 = F_multistep + dt_/2 * self.v(F_multistep, y1,  k     *dt)
                k1 = F_multistep + dt_   * self.v(k0,          y1, (k+0.5)*dt)
                F_multistep = k1
    
        F_single = self(x0, y1)
        loss = ( F_multistep - F_single ).pow(2).mean()
        return loss
    
    def forward(self, x0, y):
        v = torch.no_grad(self.v)(x0, y, torch.zeros(len(x0)))
        phi = self.phi(x0, y, v)
        return x0 + v + phi
    
    def sample(self, y1, x0=None):
        if x0 is None: x0 = self.p0.sample(y1.shape).to(y1.device)
        x = self(x0, y1)
        return x