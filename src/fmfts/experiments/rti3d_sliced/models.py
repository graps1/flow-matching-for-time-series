import torch
from fmfts.utils.unet import UNet
from fmfts.utils.models.cfm_velocity import VelocityModel
from fmfts.utils.models.cfm_single_step import SingleStepModel
from fmfts.utils.models.cfm_flow import FlowModel



class VelocityModelSlicedRTI3D(VelocityModel):
    def __init__(self, 
                 p0=torch.distributions.Normal(0, 1), 
                 features=(128, 196, 196, 256),
                 include_timestamp=True,
                 include_vertical_position=True,
                 time_embedding_dim = 4):
        super().__init__(p0=p0)

        self.include_timestamp = include_timestamp
        self.include_vertical_position = include_vertical_position
        self.time_embedding_dim = time_embedding_dim

        self.n_channels = 4 + self.include_timestamp
        self.in_dim = 2*self.n_channels + self.time_embedding_dim + self.include_vertical_position

        self.v_net = UNet(
            in_channels=self.in_dim, out_channels=self.n_channels,
            padding=("circular", "zeros"),
            features=features,
            nl = torch.nn.ReLU()
        )

    def forward(self, x, y, tx):
        bs, _, width, height = x.shape
        tx = torch.stack([torch.cos(2**k*torch.pi*tx) for k in range(self.time_embedding_dim)], dim=1)
        tx = tx.view(bs, self.time_embedding_dim, 1, 1).expand(-1, -1, width, height)
        z = torch.cat([x, y, tx], dim=1)

        if self.include_vertical_position:
            pos = torch.linspace(0, 1, height)
            pos = pos.view(1, 1, 1, height)
            pos = pos.expand(bs, 1, width, height)
            z = torch.cat([z, pos], dim=1)

        return self.v_net(z)

class SingleStepModelSlicedRTI3D(SingleStepModel):
    def __init__(self, 
                 velocity_model,
                 p0=torch.distributions.Normal(0, 1), 
                 features=(128, 196, 256),
                 include_timestamp=True,
                 include_vertical_position=True):
        super().__init__(v=velocity_model, p0=p0)

        self.include_timestamp = include_timestamp
        self.include_vertical_position = include_vertical_position
        self.n_channels = 4 + self.include_timestamp
        self.in_dim = 3*self.n_channels + self.include_vertical_position

        self.phi_net = UNet(
            in_channels=self.in_dim, out_channels=self.n_channels,
            padding=("circular", "zeros"),
            features=features,
            nl = torch.nn.ReLU()
        )

    def phi(self, x0, y, v):
        bs, _, width, height = x0.shape
        z = torch.cat([x0, y, v], dim=1)

        if self.include_vertical_position:
            pos = torch.linspace(0, 1, height)
            pos = pos.view(1, 1, 1, height)
            pos = pos.expand(bs, 1, width, height)
            z = torch.cat([z, pos], dim=1)

        return self.phi_net(z)

class FlowModelSlicedRTI3D(FlowModel):
    def __init__(self, 
                 velocity_model,
                 p0=torch.distributions.Normal(0, 1), 
                 features=(128, 196, 256),
                 include_timestamp=True,
                 include_vertical_position=True,
                 time_embedding_dim = 4):
        super().__init__(v=velocity_model, p0=p0)

        self.include_timestamp = include_timestamp
        self.include_vertical_position = include_vertical_position
        self.time_embedding_dim = time_embedding_dim
        self.n_channels = 4 + self.include_timestamp
        self.in_dim = 3*self.n_channels + 2*self.time_embedding_dim + self.include_vertical_position

        self.phi_net = UNet(
            in_channels=self.in_dim, out_channels=self.n_channels,
            padding=("circular", "zeros"),
            features=features,
            nl = torch.nn.ReLU()
        )

    def phi(self, x, y, tx, v, delta):
        bs, _, width, height = x.shape
        tx = torch.stack([torch.cos(2**k*torch.pi*tx) for k in range(self.time_embedding_dim)], dim=1)
        tx = tx.view(bs, self.time_embedding_dim, 1, 1).expand(-1, -1, width, height)
        delta = torch.stack([torch.cos(2**k*torch.pi*delta) for k in range(self.time_embedding_dim)], dim=1)
        delta = delta.view(bs, self.time_embedding_dim, 1, 1).expand(-1, -1, width, height)
        z = torch.cat([x, y, tx, v, delta], dim=1)

        if self.include_vertical_position:
            pos = torch.linspace(0, 1, height)
            pos = pos.view(1, 1, 1, height)
            pos = pos.expand(bs, 1, width, height)
            z = torch.cat([z, pos], dim=1)

        return self.phi_net(z)