import copy
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
                 loss = "l2"):
        super().__init__(p0=p0, loss=loss)
        self.include_timestamp = include_timestamp
        self.include_vertical_position = include_vertical_position
        self.n_channels = 4 + self.include_timestamp
        self.in_dim = 2*self.n_channels + 1 + self.include_vertical_position
        self.v_net = UNet(
            in_channels=self.in_dim, out_channels=self.n_channels,
            padding=("zeros", "circular"),
            features=features,
            nl = torch.nn.ReLU()
        )

    def forward(self, x, y, tx):
        bs, _, width, height = x.shape
        tx = tx.view(bs, 1, 1, 1).expand(-1, -1, width, height)
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
                 loss = "l2"):
        super().__init__(v=velocity_model, p0=p0, loss=loss)
        self.include_vertical_position = velocity_model.include_vertical_position
        self.phi_net = copy.deepcopy(velocity_model.v_net)

    def phi(self, x0, y):
        bs, _, width, height = x0.shape
        tx = torch.zeros(x0.shape[0], 1, x0.shape[2], x0.shape[3])
        z = torch.cat([x0, y, tx], dim=1)

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
                 loss = "l2"):
        super().__init__(v=velocity_model, p0=p0, loss=loss)
        self.include_vertical_position = velocity_model.include_vertical_position
        self.phi_net = velocity_model.v_net.clone_and_adapt(additional_in_channels=1)

    def phi(self, x, y, tx, delta):
        bs, _, width, height = x.shape
        tx = tx.view(bs, 1, 1, 1).expand(-1, -1, width, height)
        delta = delta.view(bs, 1, 1, 1).expand(-1, -1, width, height)

        pos = torch.empty(bs, 0, width, height)
        if self.include_vertical_position:
            pos = torch.linspace(0, 1, height)
            pos = pos.view(1, 1, 1, height)
            pos = pos.expand(bs, 1, width, height)
            
        z = torch.cat([x, y, tx, pos, delta], dim=1)
        return self.phi_net(z)