import torch 
from fmfts.utils.models.time_series_model import TimeSeriesModel

class FlowModel(TimeSeriesModel):
    def __init__(self, v, p0=torch.distributions.Normal(0,1)):
        super().__init__()
        self.v = v
        self.p0 = p0
    
    def phi(self, x, y, tx, v, delta):
        raise NotImplementedError()

    def compute_loss(self, y1, x1, steps=2):
        bs = y1.shape[0] 
        x0 = self.p0.sample(x1.shape).to(x1.device)
        delta = ( 1e-2+torch.rand(bs)*(1-1e-2) )**(1/3)
        tx = torch.rand(bs)*(1-delta)
        tx_ = tx.view(-1, *[1]*(x1.dim()-1)) 
        xt = (1 - tx_)*x0 + tx_*x1

        with torch.no_grad():
            F_multistep = xt 
            for k in range(steps): 
                F_multistep = self(F_multistep, y1, tx+k*delta/steps, delta/steps)

        F_single = self(xt, y1, tx, delta)
        loss = ( F_multistep - F_single ).pow(2).mean()
        return loss
    
    def forward(self, x, y, tx, delta):
        if not isinstance(delta, torch.Tensor): delta = torch.tensor(delta)
        if not isinstance(tx, torch.Tensor):    tx    = torch.tensor(tx)

        v = torch.no_grad(self.v)(x, y, tx)
        delta_ = delta.view(-1, *[1]*(x.dim()-1))
        phi = self.phi(x, y, tx, v, delta)
        return x + delta_*v + delta_**2/2 * phi
    
    def sample(self, y1, x0=None, steps=1):
        if x0 is None: x0 = self.p0.sample(y1.shape).to(y1.device)
        
        x = x0
        dt = torch.tensor(1/steps).expand(len(y1))
        for i in range(steps): 
            x = self(x, y1, i*dt, dt)
            
        return x