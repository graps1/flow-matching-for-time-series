import copy
import torch
from fmfts.utils.models.time_series_model import TimeSeriesModel
from fmfts.utils.models.cfm_velocity import VelocityModel


class ProgressiveDistillation(TimeSeriesModel):
    
    def __init__(self, 
                 velocity_model: VelocityModel,
                 stage=5, # the student learns to make steps for size K**(-stage). At stage == 0, the student makes a full step of size 1.
                 K=2):
        assert K >= 2
        super().__init__()
        
        self.velocity = velocity_model
        self.teacher  = velocity_model
        self.student  = copy.deepcopy(velocity_model)
        
        self.K = K
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

    def compute_loss(self, y1, x1):

        delta = torch.full((len(x1),), 1/(self.K**self.stage), device=x1.device)
        t = torch.rand_like(delta) * (1 - delta)
        t_ = t.view(-1, *[1]*(x1.dim()-1))
        
        x0 = self.teacher.p0.sample(x1.shape).to(x1.device)
        xt = (1 - t_) * x0 + t_ * x1

        with torch.no_grad():
            # if we're using the Euler method here with steps = K, 
            # the teacher takes K equal steps of size delta/K = K**(-stage-1) = K**(-(stage+1)), 
            # i.e., the integrate method evaluates the teacher model at deltas of K**(-(stage+1)).
            x_teacher = self.teacher.integrate(y1, xt, t=t, dt=delta, steps=self.K, method="euler")

        # Student: 1 macro-step with the student velocity
        x_student = self.student.integrate(y1, xt, t=t, dt=delta, steps=1, method="euler")
        
        loss = (x_teacher - x_student).pow(2).mean()
        return loss
