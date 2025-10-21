import torch
import copy
from fmfts.utils.unet import UNet
from fmfts.utils.models.cfm_velocity import VelocityModel
from fmfts.utils.models.cfm_single_step import SingleStepModel
from fmfts.utils.models.cfm_flow import FlowModel
from fmfts.utils.models.cfm_velocity_pd import DistilledVelocityMixin

class VelocityModelFullRTI3D(VelocityModel):
    def __init__(self, 
                 p0=torch.distributions.Normal(0, 1), 
                 features=(64, 96, 128),
                 include_timestamp=True,
                 include_vertical_position=True,
                 loss="l2"):
        super().__init__(p0=p0, loss=loss)

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

class SingleStepModelFullRTI3D(SingleStepModel):
    def __init__(self, 
                 velocity_model,
                 p0=torch.distributions.Normal(0, 1), 
                 include_timestamp=True,
                 include_vertical_position=True,
                 loss="l2"):
        super().__init__(v=velocity_model, p0=p0, loss=loss)
        self.include_timestamp = include_timestamp
        self.include_vertical_position = include_vertical_position
        self.phi_net = copy.deepcopy(velocity_model.v_net)
    
    def phi(self, x0, y):
        bs, _, width, depth, height = x0.shape
        tx = torch.zeros(bs, 1, width, depth, height)
        z = torch.cat([x0, y, tx], dim=1)

        if self.include_vertical_position:
            pos = torch.linspace(0, 1, height)
            pos = pos.view(1, 1, 1, 1, height)
            pos = pos.expand(bs, 1, width, depth, height)
            z = torch.cat([z, pos], dim=1)

        return self.phi_net(z)

class FlowModelFullRTI3D(FlowModel):
    def __init__(self, 
                 velocity_model,
                 p0=torch.distributions.Normal(0, 1), 
                 include_timestamp=True,
                 include_vertical_position=True,
                 loss="l2"):
        super().__init__(v=velocity_model, p0=p0, loss=loss)

        self.include_timestamp = include_timestamp
        self.include_vertical_position = include_vertical_position
        self.phi_net = velocity_model.v_net.clone_and_adapt(additional_in_channels=1) 

    def phi(self, x, y, tx, delta):
        bs, _, width, depth, height = x.shape
        tx = tx.view(bs, 1, 1, 1, 1).expand(-1, -1, width, depth, height)
        delta = delta.view(bs, 1, 1, 1, 1).expand(-1, -1, width, depth, height)

        pos = torch.empty(bs, 0, width, depth, height)
        if self.include_vertical_position:
            pos = torch.linspace(0, 1, height)
            pos = pos.view(1, 1, 1, 1, height)
            pos = pos.expand(bs, 1, width, depth, height)
        
        z = torch.cat([x, y, tx, pos, delta], dim=1)
        return self.phi_net(z)


class VelocityPDFullRTI3D(DistilledVelocityMixin, VelocityModelFullRTI3D):
    def __init__(
        self,
        teacher,                         # a trained VelocityModelFullRTI3D (the teacher)
        K: int = 2,                      # how many teacher fine steps to distill into 1 student macro step
        method: str = "midpoint",        # 'euler' | 'midpoint' | 'rk4'
        p0: torch.distributions.Distribution = torch.distributions.Normal(0, 1),
        features=(128, 196),
        include_timestamp: bool = True,
        include_vertical_position: bool = True,
        loss: str = "l2",
        delta_sampler=None,              # optional custom sampler for (delta, t)
        log_delta_t: bool = False,
    ):
        # Initialize the velocity model (creates self.v_net)
        VelocityModelFullRTI3D.__init__(
            self,
            p0=p0,
            features=features,
            include_timestamp=include_timestamp,
            include_vertical_position=include_vertical_position,
            loss=loss
        )

        # Initialize the PD mixin (overrides compute_loss)
        DistilledVelocityMixin.__init__(
            self,
            teacher_velocity=teacher,
            K=K,
            method=method,
            delta_sampler=delta_sampler,
            log_delta_t=log_delta_t,
        )