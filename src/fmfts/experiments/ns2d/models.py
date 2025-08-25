import torch
import copy
from fmfts.utils.unet import UNet
from fmfts.utils.models.cfm_velocity import VelocityModel
from fmfts.utils.models.cfm_flow import FlowModel
from fmfts.utils.models.cfm_single_step import SingleStepModel


class VelocityModelNS2D(VelocityModel):
    def __init__(self, p0=torch.distributions.Normal(0, 1), features=(64, 96, 128), loss="l2"):
        super().__init__(p0=p0, loss=loss)
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
    def __init__(self, velocity_model, p0=torch.distributions.Normal(0, 1), loss="l2"):
        super().__init__(velocity_model, p0=p0, loss=loss)
        self.phi_net = velocity_model.unet.clone_and_adapt(additional_in_channels=1) 

    def phi(self, x, y, tx, delta):
        tx = tx.view(-1, 1, 1, 1).expand(-1, -1, *x.shape[2:])
        delta = delta.view(-1, 1, 1, 1).expand(-1, -1, *x.shape[2:])
        x = self.phi_net(torch.cat([x, y, tx, delta], dim=1))
        return x 

class SingleStepModelNS2D(SingleStepModel):
    def __init__(self, velocity_model, p0=torch.distributions.Normal(0, 1), features=(64, 96, 128), loss="l2"):
        super().__init__(velocity_model, p0=p0, loss=loss)
        self.phi_net = copy.deepcopy(velocity_model.unet)
        
    def phi(self, x, y):
        tx = torch.zeros(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.phi_net(torch.cat([x, y, tx], dim=1))
        return x 