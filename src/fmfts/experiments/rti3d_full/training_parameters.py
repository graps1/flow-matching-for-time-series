from fmfts.experiments.rti3d_full.models import VelocityModelFullRTI3D, \
                                                  SingleStepModelFullRTI3D, \
                                                  FlowModelFullRTI3D, \
                                                  VelocityPDFullRTI3D
from fmfts.dataloader.rti3d_full import DatasetFullRTI3D
from fmfts.utils.models.cfm_velocity_pd import uniform_delta_sampler, \
                                                fixed_macrostep_sampler, \
                                                beta_delta_sampler

params = {
    "flow": {
        "model_kwargs": { 
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 5,
        },
        "lr_max": 5e-6,
        "lr_min": 5e-6,
        "cls": FlowModelFullRTI3D,
    },
    "velocity": {
        "model_kwargs": {
            "features": (128, 196),
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
        },
        "lr_max": 1e-5,
        "lr_min": 5e-6,
        "cls": VelocityModelFullRTI3D
    },
    "single_step": {
        "model_kwargs": {
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 5,
            "method": "midpoint"
        },
        "lr_max": 1e-5,
        "lr_min": 5e-6,
        "cls": SingleStepModelFullRTI3D
    },
    "velocity_pd": {
        "cls": VelocityPDFullRTI3D,
        "model_kwargs": {
            "features": (128, 196),  # Match velocity model (3D requires less features)
            "include_timestamp": True,
            "include_vertical_position": True,
            "loss": "l2",
            "K": 2,  # Distill 2 fine steps → 1 macro step
            "delta_sampler": beta_delta_sampler,  # Curriculum learning
        },
        "lr_max": 1e-4,  # Lower for 3D (more memory intensive)
        "lr_min": 5e-6,
        "training_kwargs": {
            "max_iters": 10000,  # Per-stage budget
        },
    },

    # Multistage PD orchestration defaults (used by multistage_pd.py)
    "multistage_pd": {
        "modeltype": "velocity_pd",
        "stages": 3,
        "stage_iters": [30000, 30000, 30000],  # Smaller budget for 3D
        "initial_teacher": "state_velocity_teacher1.pt",
    },
    "dataset": {
        "cls": DatasetFullRTI3D,
        "kwargs": {
            "At" : 3/4,
            "dt" : 5,
            "dx" : 4,
            "dy" : 4,
            "dz" : 4,
            "include_timestamp" : True
        }
    }
}