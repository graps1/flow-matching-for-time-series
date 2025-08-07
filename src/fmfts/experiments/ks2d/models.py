import torch
from fmfts.utils.unet import UNet
from fmfts.utils.models.cfm_velocity import VelocityModel

class VelocityModelKS2D(VelocityModel):
    def __init__(self, p0=torch.distributions.Normal(0, 1), features=(64, 128)):
        super().__init__(p0=p0)

        self.unet = UNet(
                2+1, 1, # 2*#channels + 1, where the +1 is for the time embedding and the 2* is for the current and previous state
                features=features,
                padding=("circular", "circular"),
                nl=torch.nn.ReLU())
        
    def forward(self, x, y, tx):
        tx = tx.view(-1, 1, 1, 1).expand(-1, 1, *x.shape[2:])
        x = self.unet(torch.cat([x, y, tx], dim=1))
        return x 