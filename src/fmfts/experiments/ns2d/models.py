import torch
import copy
from fmfts.utils.unet import UNet
from fmfts.utils.models.cfm_velocity import VelocityModel
from fmfts.utils.models.cfm_dir_dist import DirectDistillationModel
from fmfts.utils.models.deterministic import DeterministicModel

class DeterministicModelNS2D(DeterministicModel):
    def __init__(self, features=(64, 96, 128)):
        super().__init__()
        self.n_channels = 4
        self.unet = UNet(
                self.n_channels, self.n_channels,
                features=features,
                padding=("circular", "circular"),
                nl=torch.nn.ReLU())
        
    def forward(self, y):
        x = self.unet(y)
        return x 

class VelocityModelNS2D(VelocityModel):
    def __init__(self, p0=torch.distributions.Normal(0, 1), features=(64, 96, 128)):
        super().__init__(p0=p0)
        self.n_channels = 4
        self.unet = UNet(
                2*self.n_channels+1, self.n_channels,
                features=features,
                padding=("circular", "circular"),
                nl=torch.nn.ReLU())
        
    def forward(self, x, y, tx):
        tx = tx.view(-1, 1, 1, 1).expand(-1, 1, *x.shape[2:])
        x = self.unet(torch.cat([x, y, tx], dim=1))
        return x 

class DirectDistillationModelNS2D(DirectDistillationModel):
    def __init__(self, velocity_model):
        super().__init__(velocity_model)
        self.phi_net = copy.deepcopy(velocity_model.unet)
        
    def phi(self, x, y):
        tx = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3])
        x = self.phi_net(torch.cat([x, y, tx], dim=1))
        return x 
