import torch
from fmfts.utils.unet import UNet
from fmfts.utils.models.cfm_velocity import VelocityModel


class VelocityModelFullRTI3D(VelocityModel):
    def __init__(self, 
                 p0=torch.distributions.Normal(0, 1), 
                 features=(64, 96, 128),
                 include_timestamp=True,
                 include_vertical_position=True):
        super().__init__(p0=p0)

        self.include_timestamp = include_timestamp
        self.include_vertical_position = include_vertical_position

        self.n_channels = 4 + self.include_timestamp
        self.in_dim = 2*self.n_channels + 1 + self.include_vertical_position

        self.v_net = UNet(
            in_channels=self.in_dim, out_channels=self.n_channels,
            padding=("circular", "circular", "zeros"),
            features=features,
            nl = torch.nn.ReLU()
        )

    def forward(self, x, y, tx):
        bs, _, width, depth, height = x.shape
        tx = tx.view(-1, 1, 1, 1, 1)
        tx = tx.expand(bs, 1, width, depth, height)
        z = torch.cat([x, y, tx], dim=1)

        if self.include_vertical_position:
            pos = torch.linspace(0, 1, height)
            pos = pos.view(1, 1, 1, 1, height)
            pos = pos.expand(bs, 1, width, depth, height)
            z = torch.cat([z, pos], dim=1)

        return self.v_net(z)