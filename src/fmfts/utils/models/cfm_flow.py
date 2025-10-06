import torch 
import copy
from torch.distributions.categorical import Categorical
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.loss_fn import sobolev

torch.set_printoptions(precision=4, sci_mode=True)

class FlowModel(TimeSeriesModel):
    def __init__(self, v, p0=torch.distributions.Normal(0,1), loss="l2"):
        super().__init__()
        assert loss in ["l1", "l2", "sobolev", "PINN"]
        self.p0 = p0
        self.loss_fn = loss
        self.errors_by_step = torch.nn.Parameter(torch.ones(20), requires_grad=False)

        self.v = v
        for param in self.v.parameters(): 
            param.requires_grad = False

    def phi(self, x, y, tx, delta):
        raise NotImplementedError()
    
    def compute_loss(self, y1, x1, ctr, p_geometric=0.5, steps=2):

        # cumsum = self.errors_by_step.cumsum(dim=0)
        # rcumsum = self.errors_by_step - cumsum + cumsum[-1:None]

        # weights = -(rcumsum - self.errors_by_step) + self.errors_by_step
        # weights = weights - weights.min()
        # weights = weights/weights.sum()
        # weights = weights + p_min_weight

        bs = y1.shape[0] 
        x0 = self.p0.sample(x1.shape).to(x1.device)

        # weights = torch.ones(self.max_log2_steps) # self.errors_by_step/self.errors_by_step.sum() + 1e-4
        print(self.errors_by_step)

        # log2_steps_distribution = torch.distributions.categorical.Categorical(weights)
        # log2_samples_steps = log2_steps_distribution.sample((bs,))
        
        # g = torch.distributions.Geometric(probs=p_geometric)
        # samples_steps = g.sample((bs,)).to(int)
        # delta = 1/(1.+samples_steps)
        # delta = float(steps)**(-samples_steps)

        delta = torch.rand((bs,))**(1-p_geometric)
    
        delta_ = delta.view(-1, *[1]*(x1.dim()-1))
        tx = torch.rand(bs)*(1-delta)
        xt = x0 + tx.view(-1, *[1]*(x1.dim()-1))*(x1 - x0)
        
        # print(delta)
        # F_single = self(xt, y1, tx, log2_samples_steps)
        F_single = self(xt, y1, tx, delta)

        with torch.no_grad():
            # recursive = samples_steps < len(self.errors_by_step) - 1
            F_multistep = xt
            for k in range(steps): 
                # F_multistep[~recursive] = F_multistep[~recursive] + delta_[~recursive]/steps * self.v(
                #     F_multistep[~recursive], 
                #     y1[~recursive], 
                #     tx[~recursive]+k*delta[~recursive]/steps)
                # F_multistep[recursive] = self(
                #     F_multistep[recursive], 
                #     y1[recursive], 
                #     tx[recursive]+k*delta[recursive]/steps, 
                #     delta[recursive]/steps)
                # F_multistep = self(F_multistep, y1, tx+k*delta/steps, delta/steps)
                F_multistep = F_multistep + delta_/steps * self.v(F_multistep, y1, tx+k*delta/steps)


        loss = (F_multistep - F_single).pow(2)
        # loss = loss / (1e-3 + delta_.pow(4))
        loss = loss.mean( dim=tuple(range(1,loss.dim())) )

        # print((F_multistep - F_single).abs().mean().detach())
        # print(samples_steps)

        samples_steps = (1/delta).to(int)
        for s in torch.unique(samples_steps):
            if s < len(self.errors_by_step):    
                self.errors_by_step[s] = 0.99 * self.errors_by_step[s] + 0.01 * loss[samples_steps == s].mean().detach()

        loss = loss.mean()
        # print(loss.item())

        if loss.item() > 100: 
            print(xt.abs().max()) 
            print(F_multistep.abs().max()) 
            print(F_single.abs().max()) 
            raise Exception()
        return loss
    
    def forward(self, x, y, tx, delta):
        if not isinstance(delta, torch.Tensor): delta = torch.tensor(delta)
        if not isinstance(tx, torch.Tensor):    tx    = torch.tensor(tx)

        v = self.v(x, y, tx)
        delta_ = delta.view(-1, *[1]*(x.dim()-1))
        phi = self.phi(x, y, tx, delta)
        return x + delta_*v + delta_**2 * (phi - v)
    
    def sample(self, y1, x0=None, steps=1):
        if x0 is None: x0 = self.p0.sample(y1.shape).to(y1.device)
        
        x = x0
        delta = 1./steps
        for s in range(steps): 
            x = self(x, y1, s * delta, delta)
            
        return x