import os
from dataclasses import dataclass, field

from fmfts.experiments.ns2d.models import   VelocityModelNS2D, \
                                            SingleStepModelNS2D, \
                                            FlowModelNS2D
from fmfts.dataloader.ns2d import DatasetNS2D

fpath = os.path.dirname(os.path.abspath(__file__))

@dataclass
class TrainingParameters():
    features_velocity    : tuple = (64, 96, 96, 128)
    features_flow        : tuple = (128, 256)
    features_single_step : tuple = (128, 256)
    lr_max : float = 5e-5
    lr_min : float = 1e-5
    batch_size : int = 4
    steps_single_step : int = 10
    steps_flow : int = 5
    current_time : float = 0.0

    kwargs_dataset_cls : dict = field(default_factory=dict)

    VelocityCls : ... = VelocityModelNS2D
    SingleStepCls : ...  = SingleStepModelNS2D
    FlowCls : ...  = FlowModelNS2D
    DatasetCls : ...  = DatasetNS2D