from fmfts.experiments.rti3d_sliced.models import VelocityModelSlicedRTI3D, \
                                                  SingleStepModelSlicedRTI3D, \
                                                  FlowModelSlicedRTI3D, \
                                                  VelocityPDSlicedRTI3D
from fmfts.dataloader.rti3d_sliced import DatasetSlicedRTI3D
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
            "steps": 2,
        },
        "lr_max": 5e-5,
        "lr_min": 1e-5,
        "cls": FlowModelSlicedRTI3D,
    },
    "velocity": {
        "model_kwargs": {
            "features": (128, 196, 196, 256),
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
        },
        "lr_max": 1e-5,
        "lr_min": 5e-6,
        "cls": VelocityModelSlicedRTI3D
    },
    "single_step": {
        "model_kwargs": {
            "loss": "l2",
        },
        "training_kwargs": {
            "batch_size": 4,
            "steps": 10,
        },
        "lr_max": 5e-5,
        "lr_min": 1e-5,
        "cls": SingleStepModelSlicedRTI3D
    },
    "velocity_pd": {
        "cls": VelocityPDSlicedRTI3D,
        "model_kwargs": {
            "features": (128, 196, 196, 256),
            "include_timestamp": True,
            "include_vertical_position": True,
            "loss": "l2",
            "K": 2,  # Distill 2 fine steps → 1 macro step
            "delta_sampler": beta_delta_sampler,  # Curriculum learning
        },
        "lr_max": 2e-4,
        "lr_min": 1e-5,
        "training_kwargs": {
            "max_iters": 10000,  # Per-stage budget
        },
    },

    # Multistage PD orchestration defaults (used by multistage_pd.py)
    "multistage_pd": {
        "modeltype": "velocity_pd",
        "stages": 3,
        "stage_iters": [50000, 50000, 50000],
        "initial_teacher": "state_velocity_teacher1.pt",
    },
    "dataset": {
        "cls": DatasetSlicedRTI3D,
        "kwargs": {
            "At" : 3/4,
            "dt" : 5,
            "dy" : 1,
            "dz" : 1,
            "include_timestamp" : True
        }
    }
}