import copy
import torch
# from fmfts.utils.loss_fn import sobolev
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.models.cfm_velocity import VelocityModel

# def _velocity_step(model, x, y, t, d, method="midpoint"):
#     """
#     Advance by one macro-step of size `d` using `model.forward(x,y,t)` as the velocity field.
#     """
#     assert method in ["euler", "midpoint", "rk4"]
#     if not isinstance(d, torch.Tensor): d = torch.tensor(d, device=x.device)
#     if d.dim() == 0: d = d.expand(len(y))
#     dt_ = d.view(-1, *[1]*(x.dim()-1))
# 
#     if method == "euler":
#         return x + dt_ * model(x, y, t)
# 
#     elif method == "midpoint":
#         k0 = model(x, y, t)
#         x_mid = x + dt_/2 * k0
#         return x + dt_ * model(x_mid, y, t + d/2)
# 
#     elif method == "rk4":
#         k0 = model(x, y, t)
#         k1 = model(x + dt_/2 * k0, y, t + d/2)
#         k2 = model(x + dt_/2 * k1, y, t + d/2)
#         k3 = model(x + dt_   * k2, y, t + d)
#         return x + dt_/6 * (k0 + 2*k1 + 2*k2 + k3)


# def _rollout_velocity(model, x, y, t, delta, steps: int, method="midpoint"):
#     """
#     Roll out K fine steps of size delta/K using `model` as the velocity field.
#     """
#     if not isinstance(delta, torch.Tensor): delta = torch.tensor(delta, device=x.device)
#     if delta.dim() == 0: delta = delta.expand(len(y))
#     d = delta / steps
# 
#     xk, tk = x, t
#     for k in range(steps):
#         xk = _velocity_step(model, xk, y, tk, d, method=method)
#         tk = tk + d
#     return xk


# def uniform_delta_sampler(bs, device, delta_min: float = 1e-3):
#     """Sample Δ ~ Uniform(delta_min, 1), and t ~ Uniform(0, 1-Δ)."""
#     u = torch.rand(bs, device=device)
#     delta = delta_min + u * (1 - delta_min)
#     t = torch.rand(bs, device=device) * (1 - delta)
#     return delta, t


# def fixed_macrostep_sampler(bs, device, S: int = 4, jitter: float = 0.05, delta_min: float = 1e-3):
# def fixed_macrostep_sampler(bs, device, S, jitter: float = 0.0, delta_min: float = 0.0):
#     """Sample Δ around 1/S with small relative jitter, clamped to [delta_min, 1]."""
#     S = max(int(S), 1)
#     base = torch.full((bs,), 1.0 / S, device=device)
#     eps = (torch.rand(bs, device=device) - 0.5) * 2 * jitter * base
#     delta = (base + eps).clamp(delta_min, 1.0)
#     t = torch.rand(bs, device=device) * (1 - delta)
#     return delta, t
# 
# 
# def beta_delta_sampler(bs, device, alpha: float = 0.5, beta: float = 1.0, delta_min: float = 1e-3):
#     """Sample Δ using a Beta(alpha, beta) mapped to [delta_min, 1]. alpha<beta biases small Δ; alpha=beta=1 is uniform."""
#     beta_rv = torch.distributions.Beta(alpha, beta).sample((bs,)).to(device)
#     delta = delta_min + beta_rv * (1 - delta_min)
#     t = torch.rand(bs, device=device) * (1 - delta)
#     return delta, t


class ProgressiveDistillation(TimeSeriesModel):
    """
    Mixin that replaces `compute_loss` with a progressive-distillation loss for a VelocityModel.
    Now supports teacher update, device management, parameterized delta sampling, and logging.
    """
    def __init__(self, 
                 velocity_model: VelocityModel,
                 stage=5, # the student learns to make steps for size K**(-stage)
                 K=2):
                 # teacher_method="midpoint"):
                 # delta_sampler=None, 
                 #log_delta_t=False):
        assert K >= 2
        super().__init__()
        # assert method in ["euler", "midpoint", "rk4"]

        # Store a reference to the teacher & student, not a deepcopy
        self.velocity = velocity_model
        self.teacher  = velocity_model
        self.student  = copy.deepcopy(velocity_model)
        # self._teacher_device = next(self.teacher.parameters()).device
        # self.set_teacher(self.teacher)
        

        self.K = K
        # self.method = teacher_method
        # self.stage = torch.nn.Parameter(torch.tensor(stage), requires_grad=False)
        self.register_buffer("stage", torch.tensor(stage, dtype=torch.int))

    def additional_info(self):
        return { "k": self.K, "stage": self.stage.item() }

    def sample(self, y, x0=None):
        return self.student.sample(y, x0=x0, steps=self.K**self.stage, method="euler")

    def advance(self):
        assert self.stage > 0, "Already at stage 0, cannot advance further."
        self.stage = self.stage - 1
        self.teacher = copy.deepcopy(self.student)
        self.student = copy.deepcopy(self.velocity)

        # Default to uniform sampler unless a custom sampler is provided
        # self._delta_sampler = delta_sampler if delta_sampler is not None else uniform_delta_sampler
        # self._log_delta_t = log_delta_t

    # def set_teacher(self, new_teacher):
    #     """Update the teacher model reference, ensure eval mode and no grad."""
    #     self.teacher = new_teacher
    #     self.teacher.eval()
    #     for p in self.teacher.parameters():
    #         p.requires_grad_(False)
    #     self._teacher_device = next(self.teacher.parameters()).device

    # def move_teacher_to_device(self, device):
    #     """Move teacher to the specified device if not already there."""
    #     if self._teacher_device != device:
    #         self.teacher = self.teacher.to(device)
    #         self._teacher_device = device

    # @staticmethod
    # def default_delta_sampler(bs, device):
    #     # Deprecated: previously biased small-Δ sampler. Kept for backward reference.
    #     delta = (1e-3 + torch.rand(bs, device=device) * (1 - 1e-3)) ** (1 / 2)
    #     t = torch.rand(bs, device=device) * (1 - delta)
    #     return delta, t

    def compute_loss(self, y1, x1):
        """
        Distill K teacher steps (fine) into 1 student macro-step over a random Δ and start time t.
        Now supports parameterized delta sampling and logging.
        """
        # bs, device = y1.shape[0], x1.device
        x0 = self.teacher.p0.sample(x1.shape).to(x1.device)

        # self.move_teacher_to_device(device)
        # self.teacher.eval()  # Always enforce eval mode

        # Sample t and Δ using the configured sampler
        # delta, t = self._delta_sampler(bs, device)
        # delta, t = fixed_macrostep_sampler(len(x1), device, S=self.K**self.stage)
        delta = torch.full((len(x1),), 1/(self.K**self.stage), device=x1.device)
        t = torch.rand_like(delta) * (1 - delta)

        # Optionally log delta and t
        #if self._log_delta_t:
        #    logger.info(f"Sampled delta: {delta.detach().cpu().numpy()}, t: {t.detach().cpu().numpy()}")

        # Choose a realistic on-bridge state x_t
        t_ = t.view(-1, *[1]*(x1.dim()-1))
        xt = (1 - t_) * x0 + t_ * x1

        # Teacher: K fine steps with frozen teacher velocity
        with torch.no_grad():
            # if we're using the Euler method here with steps = K, 
            # the teacher takes K equal steps of size delta/K = K**(-stage-1) = K**(-(stage+1)), 
            # i.e., the integrate method evaluates the velocity at deltas of K**(-(stage+1)).
            x_teacher = self.teacher.integrate(y1, xt, t=t, dt=delta, steps=self.K, method="euler")
            # self.teacher.sample(xt, y1, steps=self.K, method=self.method)
            # x_teacher = _rollout_velocity(self.teacher, xt, y1, t, delta,
            #                               steps=self.K, method=self.method)

        # Student: 1 macro-step with the student velocity
        # x_student = _velocity_step(self, xt, y1, t, delta, method=self.method)
        x_student = self.student.integrate(y1, xt, t=t, dt=delta, steps=1, method="euler")
        
        loss = (x_teacher - x_student).pow(2).mean()

        # if   self.loss_fn == "l2":      loss = (x_teacher - x_student).pow(2).mean()
        # elif self.loss_fn == "sobolev": loss = sobolev(x_teacher - x_student, alpha=1.0, beta=1.0, t=t)
        # else: raise ValueError(f"Unknown loss {self.loss_fn}")

        # Optionally return delta and t for monitoring
        # if self._log_delta_t:
        #     return loss, delta.detach().cpu(), t.detach().cpu()
        return loss
