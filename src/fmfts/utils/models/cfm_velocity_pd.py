import copy
import torch
from fmfts.utils.loss_fn import sobolev
from fmfts.utils.models.time_series_model import TimeSeriesModel

def _velocity_step(model, x, y, t, d, method="midpoint"):
    """
    Advance by one macro-step of size `d` using `model.forward(x,y,t)` as the velocity field.
    """
    assert method in ["euler", "midpoint", "rk4"]
    if not isinstance(d, torch.Tensor): d = torch.tensor(d, device=x.device)
    if d.dim() == 0: d = d.expand(len(y))
    dt_ = d.view(-1, *[1]*(x.dim()-1))

    if method == "euler":
        return x + dt_ * model(x, y, t)

    elif method == "midpoint":
        k0 = model(x, y, t)
        x_mid = x + dt_/2 * k0
        return x + dt_ * model(x_mid, y, t + d/2)

    elif method == "rk4":
        k0 = model(x, y, t)
        k1 = model(x + dt_/2 * k0, y, t + d/2)
        k2 = model(x + dt_/2 * k1, y, t + d/2)
        k3 = model(x + dt_   * k2, y, t + d)
        return x + dt_/6 * (k0 + 2*k1 + 2*k2 + k3)


def _rollout_velocity(model, x, y, t, delta, steps: int, method="midpoint"):
    """
    Roll out K fine steps of size delta/K using `model` as the velocity field.
    """
    if not isinstance(delta, torch.Tensor): delta = torch.tensor(delta, device=x.device)
    if delta.dim() == 0: delta = delta.expand(len(y))
    d = delta / steps

    xk, tk = x, t
    for k in range(steps):
        xk = _velocity_step(model, xk, y, tk, d, method=method)
        tk = tk + d
    return xk


class DistilledVelocityMixin(TimeSeriesModel):
    """
    Mixin that replaces `compute_loss` with a progressive-distillation loss for a VelocityModel.
    Now supports teacher update, device management, parameterized delta sampling, and logging.
    """
    def __init__(self, teacher_velocity, K=2, method="midpoint", delta_sampler=None, log_delta_t=False):
        # Store a reference to the teacher, not a deepcopy
        self.teacher_velocity = teacher_velocity
        self._teacher_device = next(self.teacher_velocity.parameters()).device
        self.set_teacher(self.teacher_velocity)
        assert K >= 1
        assert method in ["euler", "midpoint", "rk4"]
        self._pd_K = K
        self._pd_method = method
        self._delta_sampler = delta_sampler if delta_sampler is not None else self.default_delta_sampler
        self._log_delta_t = log_delta_t

    def set_teacher(self, new_teacher):
        """Update the teacher model reference, ensure eval mode and no grad."""
        self.teacher_velocity = new_teacher
        self.teacher_velocity.eval()
        for p in self.teacher_velocity.parameters():
            p.requires_grad_(False)
        self._teacher_device = next(self.teacher_velocity.parameters()).device

    def move_teacher_to_device(self, device):
        """Move teacher to the specified device if not already there."""
        if self._teacher_device != device:
            self.teacher_velocity = self.teacher_velocity.to(device)
            self._teacher_device = device

    @staticmethod
    def default_delta_sampler(bs, device):
        # Default: bias towards small delta as before
        delta = (1e-3 + torch.rand(bs, device=device)*(1-1e-3))**(1/2)
        t = torch.rand(bs, device=device) * (1 - delta)
        return delta, t

    def compute_loss(self, y1, x1, ctr, **kwargs):
        """
        Distill K teacher steps (fine) into 1 student macro-step over a random Δ and start time t.
        Now supports parameterized delta sampling and logging.
        """
        bs, device = y1.shape[0], x1.device
        x0 = self.p0.sample(x1.shape).to(device)
        self.move_teacher_to_device(device)
        self.teacher_velocity.eval()  # Always enforce eval mode

        # Sample t and Δ using the configured sampler
        delta, t = self._delta_sampler(bs, device)

        # Optionally log delta and t
        #if self._log_delta_t:
        #    logger.info(f"Sampled delta: {delta.detach().cpu().numpy()}, t: {t.detach().cpu().numpy()}")

        # Choose a realistic on-bridge state x_t
        t_ = t.view(-1, *[1]*(x1.dim()-1))
        xt = (1 - t_) * x0 + t_ * x1

        # Teacher: K fine steps with frozen teacher velocity
        with torch.no_grad():
            x_teacher = _rollout_velocity(self.teacher_velocity, xt, y1, t, delta,
                                          steps=self._pd_K, method=self._pd_method)

        # Student: 1 macro-step with the student velocity
        x_student = _velocity_step(self, xt, y1, t, delta, method=self._pd_method)

        if   self.loss_fn == "l2":      loss = (x_teacher - x_student).pow(2).mean()
        elif self.loss_fn == "sobolev": loss = sobolev(x_teacher - x_student, alpha=1.0, beta=1.0, t=t)
        else: raise ValueError(f"Unknown loss {self.loss_fn}")

        # Optionally return delta and t for monitoring
        if self._log_delta_t:
            return loss, delta.detach().cpu(), t.detach().cpu()
        return loss