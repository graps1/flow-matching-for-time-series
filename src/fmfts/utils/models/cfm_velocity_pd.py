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
    Requirements:
      - The class using this mixin must implement `forward(self, x, y, tx)` (the velocity field).
      - `self.p0` and `self.loss_fn` must exist (from VelocityModel base).
    """
    def __init__(self, teacher_velocity, K=2, method="midpoint"):
        # keep teacher as an external attribute (NOT tracked by nn.Module)
        object.__setattr__(self, "teacher_velocity", copy.deepcopy(teacher_velocity).eval())
        for p in self.teacher_velocity.parameters():
            p.requires_grad_(False)
        assert K >= 1
        assert method in ["euler", "midpoint", "rk4"]
        self._pd_K = K
        self._pd_method = method

    def compute_loss(self, y1, x1, ctr, **kwargs):
        """
        Distill K teacher steps (fine) into 1 student macro-step over a random Δ and start time t.
        """
        bs, device = y1.shape[0], x1.device
        x0 = self.p0.sample(x1.shape).to(device)
        self.teacher_velocity = self.teacher_velocity.to(device)
        # sample t and Δ (bias towards smaller Δ like in your flow code)
        delta = (1e-3 + torch.rand(bs, device=device)*(1-1e-3))**(1/2)
        t = torch.rand(bs, device=device) * (1 - delta)

        # choose a realistic on-bridge state x_t
        t_ = t.view(-1, *[1]*(x1.dim()-1))
        xt = (1 - t_) * x0 + t_ * x1

        # teacher: K fine steps with frozen teacher velocity
        with torch.no_grad():
            x_teacher = _rollout_velocity(self.teacher_velocity, xt, y1, t, delta,
                                          steps=self._pd_K, method=self._pd_method)

        # student: 1 macro-step with the student velocity
        x_student = _velocity_step(self, xt, y1, t, delta, method=self._pd_method)

        if   self.loss_fn == "l2":      loss = (x_teacher - x_student).pow(2).mean()
        elif self.loss_fn == "sobolev": loss = sobolev(x_teacher - x_student, alpha=1.0, beta=1.0, t=t)
        else: raise ValueError(f"Unknown loss {self.loss_fn}")
        return loss