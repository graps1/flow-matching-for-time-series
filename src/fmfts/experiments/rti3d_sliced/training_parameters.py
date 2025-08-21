import os
from dataclasses import dataclass, field

from fmfts.experiments.rti3d_sliced.models import VelocityModelSlicedRTI3D, \
                                                  SingleStepModelSlicedRTI3D, \
                                                  FlowModelSlicedRTI3D
from fmfts.dataloader.rti3d_sliced import DatasetSlicedRTI3D

fpath = os.path.dirname(os.path.abspath(__file__))

@dataclass
class TrainingParameters():
    features_velocity : tuple = (128, 196, 196, 256)
    features_single_step : tuple = (128, 196, 256)
    features_flow : tuple = (128, 196, 256)
    lr_max : float = 1e-4
    lr_min : float = 1e-5
    batch_size : int = 4
    steps_single_step : int= 10
    steps_flow : int = 5

    kwargs_dataset_cls : dict = field(default_factory=lambda : dict(
        At = 3/4,
        dt = 5, 
        dy = 1, 
        dz = 1,
        include_timestamp = True
    ))

    VelocityCls : ... = VelocityModelSlicedRTI3D
    SingleStepCls : ...  = SingleStepModelSlicedRTI3D
    FlowCls : ...  = FlowModelSlicedRTI3D
    DatasetCls : ...  = DatasetSlicedRTI3D

    runs_dir : str = f"{fpath}/runs"
    model_dir : str = f"{fpath}/trained_models"
    checkpoints_dir : str = f"{fpath}/checkpoints"