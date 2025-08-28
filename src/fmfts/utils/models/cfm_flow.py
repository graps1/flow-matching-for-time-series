import torch 
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.loss_fn import sobolev

class FlowModel(TimeSeriesModel):
    def __init__(self, v, p0=torch.distributions.Normal(0,1), loss="l2"):
        super().__init__()
        assert loss in ["l1", "l2", "sobolev", "PINN"]
        self.v = v
        self.p0 = p0
        self.loss_fn = loss

    def phi(self, x, y, tx, delta):
        raise NotImplementedError()
    
    def compute_loss(self, y1, x1, ctr, steps=2):
        bs = y1.shape[0] 
        x0 = self.p0.sample(x1.shape).to(x1.device)
        g = torch.distributions.Geometric(torch.tensor(0.5))
        delta = 2**(-g.sample((bs,)))
        delta_ = delta.view(-1, *[1]*(x1.dim()-1))
        tx = torch.rand(bs)*(1-delta)
        xt = x0 + tx.view(-1, *[1]*(x1.dim()-1))*(x1 - x0)
        
        if self.loss_fn == "PINN":
            F_single = self(xt, y1, tx, delta).detach()
            v = torch.no_grad(self.v)(F_single, y1, tx + delta)
            F_single_next = self(xt, y1, tx, delta + 1e-3)
            dFdt = (F_single_next - F_single)/1e-3
            loss = (dFdt - v).pow(2).mean()

        else:
            F_multistep = xt 
            for k in range(steps): 
                F_multistep = torch.no_grad(self)(F_multistep, y1, tx+k*delta/steps, delta/steps)

            F_single = self(xt, y1, tx, delta)
            if   self.loss_fn == "l1":      
                loss = ( F_multistep - F_single ).abs().mean()
                loss = loss / (1e-2+delta_.pow(2))
                loss = loss.mean()
            elif self.loss_fn == "l2":      
                loss = ( F_multistep - F_single ).pow(2)
                loss = loss / (1e-2+delta_.pow(4))
                loss = loss.mean() 
            elif self.loss_fn == "sobolev": 
                loss = sobolev(F_multistep - F_single, alpha=1.0, beta=20.0, t=tx, pointwise=True)
                loss = loss / (1e-2+delta_.pow(4))
                loss = loss.mean()

        return loss
    
    def forward(self, x, y, tx, delta):
        if not isinstance(delta, torch.Tensor): delta = torch.tensor(delta)
        if not isinstance(tx, torch.Tensor):    tx    = torch.tensor(tx)

        v = torch.no_grad(self.v)(x, y, tx)
        delta_ = delta.view(-1, *[1]*(x.dim()-1))
        phi = self.phi(x, y, tx, delta)
        return x + delta_*v + delta_**2 * (phi - v)
        # return x + delta_*phi
    
    def sample(self, y1, x0=None, steps=1):
        if x0 is None: x0 = self.p0.sample(y1.shape).to(y1.device)
        
        x = x0
        dt = torch.tensor(1/steps).expand(len(y1))
        for i in range(steps): 
            x = self(x, y1, i*dt, dt)
            
        return x