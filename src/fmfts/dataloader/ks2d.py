import torch 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

class DatasetKS2D(Dataset):
    def __init__(self, mode, history=1): 
        assert mode in ["train", "test"]

        datapath = f"../datasets/ks2d/ks2d_data_{mode}.pt"

        self.history = history
        self.data = torch.load(datapath, weights_only=True)
        self.data = self.data.to(torch.get_default_device())
        self.data = self.data - self.data.mean(dim=[2,3], keepdim=True)

        self.mean = self.data.mean(dim=[0,2,3]).view(1, -1, 1, 1)
        self.std  =  self.data.std(dim=[0,2,3]).view(1, -1, 1, 1)
        self.n_samples, _, self.height, self.width = self.data.shape
        self.data = self.normalize(self.data)

    def normalize(self, x):
        return (x-self.mean)/self.std

    def denormalize(self, x):
        return x*self.std + self.mean

    def __len__(self): 
        return self.n_samples-self.history-1

    def __getitem__(self, idx): 
        y = self.data[idx:idx+self.history]
        y = y.flatten(end_dim=1)
        x = self.data[idx+self.history]
        return y, x
    
    def plot(self, x, *ax):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x = self.denormalize(x)
        x = x.cpu().numpy()

        if len(ax) == 0: 
            _, ax = plt.subplots(1, len(x), figsize=(12, 2.5), sharey=True, sharex=True)
            if len(x) == 1: ax = [ ax ]

        for i in range(len(ax)):
            k = i * len(x) // len(ax)
            ax[i].imshow(x[k,0], extent=(0,1,0,1), vmin=-15, vmax=16)
            ax[i].set_xlim(0,1)
            ax[i].set_ylim(0,1)
            ax[i].set_aspect("equal")
