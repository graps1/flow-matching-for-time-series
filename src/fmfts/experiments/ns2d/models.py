import torch
from fmfts.utils.unet import UNet
from fmfts.utils.models.cfm_velocity import VelocityModel
from fmfts.utils.models.cfm_flow import FlowModel
from fmfts.utils.models.cfm_single_step import SingleStepModel


class VelocityModelNS2D(VelocityModel):
    def __init__(self, p0=torch.distributions.Normal(0, 1), features=(64, 96, 128)):
        super().__init__(p0=p0)

        self.unet = UNet(
                2*4+1, 4,
                features=features,
                padding=("circular", "circular"),
                nl=torch.nn.ReLU())
        
    def forward(self, x, y, tx):
        tx = tx.view(-1, 1, 1, 1).expand(-1, 1, *x.shape[2:])
        x = self.unet(torch.cat([x, y, tx], dim=1))
        return x 


class FlowModelNS2D(FlowModel):
    def __init__(self, velocity_model, p0=torch.distributions.Normal(0, 1), features=(64, 96, 128)):
        super().__init__(velocity_model, p0=p0)

        self.time_embedding_dim = 5
        self.phi_net = UNet(
                3*4 + 2*self.time_embedding_dim, 4,
                features=features,
                padding=("circular", "circular"),
                nl=torch.nn.ReLU())
    
        
    def phi(self, x, y, tx, v, delta):
        # computes the time embedding
        tx = torch.stack([torch.cos(2**k*torch.pi*tx) for k in range(self.time_embedding_dim)], dim=1)
        delta = torch.stack([torch.cos(2**k*torch.pi*delta) for k in range(self.time_embedding_dim)], dim=1)
        tx = tx.view(-1, self.time_embedding_dim, 1, 1).expand(-1, -1, *x.shape[2:])
        delta = delta.view(-1, self.time_embedding_dim, 1, 1).expand(-1, -1, *x.shape[2:])

        x = self.phi_net(torch.cat([x, y, tx, v, delta], dim=1))
        return x 

class SingleStepModelNS2D(SingleStepModel):
    def __init__(self, velocity_model, p0=torch.distributions.Normal(0, 1), features=(64, 96, 128)):
        super().__init__(velocity_model, p0=p0)

        self.phi_net = UNet(
                3*4, 4,
                features=features,
                padding=("circular", "circular"),
                nl=torch.nn.ReLU())
    
        
    def phi(self, x, y, v):
        x = self.phi_net(torch.cat([x, y, v], dim=1))
        return x 