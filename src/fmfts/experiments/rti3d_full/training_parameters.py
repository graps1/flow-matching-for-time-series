import os
from dataclasses import dataclass, field

from fmfts.experiments.rti3d_full.models import VelocityModelFullRTI3D, \
                                                  SingleStepModelFullRTI3D, \
                                                  FlowModelFullRTI3D
from fmfts.dataloader.rti3d_full import DatasetFullRTI3D

fpath = os.path.dirname(os.path.abspath(__file__))

@dataclass
class TrainingParameters():
    features_velocity : tuple = (64, 96, 128)
    features_single_step : tuple = (128, 196, 256)
    features_flow : tuple = (128, 196, 256)
    lr_max : float = 1e-4
    lr_min : float = 1e-5
    batch_size : int = 16
    steps_single_step : int = 10
    steps_flow : int = 5

    kwargs_dataset_cls : dict = field(default_factory=lambda : dict(
        At = 3/4,
        dt = 5, 
        dx = 4,
        dy = 4, 
        dz = 4,
        include_timestamp = True
    ))

    VelocityCls : ... = VelocityModelFullRTI3D
    SingleStepCls : ...  = SingleStepModelFullRTI3D
    FlowCls : ...  = FlowModelFullRTI3D
    DatasetCls : ...  = DatasetFullRTI3D

    runs_dir : str = f"{fpath}/runs"
    model_dir : str = f"{fpath}/trained_models"
    checkpoints_dir : str = f"{fpath}/checkpoints"