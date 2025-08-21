import os
from dataclasses import dataclass, field

from fmfts.experiments.ks2d.models import   VelocityModelKS2D
from fmfts.dataloader.ks2d import DatasetKS2D

fpath = os.path.dirname(os.path.abspath(__file__))

@dataclass
class TrainingParameters():
    features_velocity    : tuple = (64, 128)
    # features_flow        : tuple = (128, 192)
    # features_single_step : tuple = (128, 256)
    lr_max : float = 1e-4
    lr_min : float = 1e-5
    batch_size : int = 4
    steps_single_step : int = 10
    steps_flow : int = 5

    kwargs_dataset_cls : dict = field(default_factory=dict)

    VelocityCls : ... = VelocityModelKS2D
    #  SingleStepCls : ...  = SingleStepModelKS2D
    #  FlowCls : ...  = FlowModelKS2D
    DatasetCls : ...  = DatasetKS2D

    runs_dir : str = f"{fpath}/runs"
    model_dir : str = f"{fpath}/trained_models"
    checkpoints_dir : str = f"{fpath}/checkpoints"