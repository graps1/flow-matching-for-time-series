import torch
from torch.utils.data import DataLoader
from fmfts.utils.models.time_series_model import TimeSeriesModel

class DeterministicModel(TimeSeriesModel):
    def forward(self, y):
        raise NotImplementedError()

    def compute_loss(self, y1, x1):
        loss = ( self(y1) - x1 ).pow(2).mean()
        return loss
