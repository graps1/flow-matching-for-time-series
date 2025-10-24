import os
import torch 
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from torch.utils.data import Dataset

class DatasetNS2D(Dataset):
    def __init__(self, mode, history=1): 
        assert mode in ["train", "test"]
        
        datapath = f"{os.path.dirname(os.path.abspath(__file__))}/../datasets/ns2d/ns2d_data_{mode}.pt"
        self.history = history
        # data has shape (n_samples, n_timesteps, n_channels, height, width)
        self.data = torch.load(datapath, weights_only=True)
        self.data = self.data.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.n_samples, self.total_sequence_len, self.n_channels, self.height, self.width = self.data.shape
        
        self.mean = torch.tensor([2.7828e-05, 8.4455e-05, 1.0000e+00, 6.0033e+01]).view(1,-1,1,1)
        self.std = torch.tensor([0.7757, 0.7830, 0.0219, 2.1954]).view(1,-1,1,1)

        self.data = self.normalize(self.data)

    def normalize(self, x):
        return (x - self.mean)/self.std

    def denormalize(self, x):
        return x*self.std + self.mean

    def __len__(self): 
        return self.n_samples*(self.total_sequence_len - self.history - 1)
    
    def get(self, i1, i2, sequence_len):
        return self.data[i1, i2:i2+sequence_len]

    def __getitem__(self, idx): 
        i1 = idx % self.n_samples
        i2 = idx // self.n_samples
        
        y = self.get(i1, i2, self.history)
        x = self.get(i1, i2+self.history, 1)
        y = y.flatten(end_dim=1)
        x = x.flatten(end_dim=1)
        return y, x
    
    def get_fields(self, x):
        squeezed = len(x.shape) == 3
        if squeezed: x = x.unsqueeze(0)
        x = self.denormalize(x)
        density, velocity, pressure = x[:,2], x[:,:2], x[:,3]
        if squeezed: 
            velocity = velocity.squeeze(0)
            density = density.squeeze(0)
            pressure = pressure.squeeze(0)
        return density, velocity, pressure

    def compute_momentum(self, x):
        density, velocity, _ = self.get_fields(x)
        momentum = density * velocity.pow(2).sum(dim=-3).pow(0.5)
        return momentum

    def compute_energy(self, x):
        density, velocity, _ = self.get_fields(x)
        energy = 0.5 * density * velocity.pow(2).sum(dim=-3)
        return energy
    
    def plot(self, x, *ax, visualization="momentum"):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # rho, u = self.get_density_and_velocity(x)
        # x = self.denormalize(x)

        Y, X = torch.meshgrid(
            torch.linspace(0, 1, 64), 
            torch.linspace(0, 1, 64), 
            indexing="ij")
        X, Y = X.cpu().numpy(), Y.cpu().numpy()

        if ax is None: _, ax = plt.subplots(1, 5, figsize=(12, 2.5), sharey=True, sharex=True)
        for i in range(n_plots := len(ax)):
            k = i * len(x) // n_plots
            if visualization == "density": 
                rho, _ = self.get_density_and_velocity(x[k])
                ax[i].imshow(rho.cpu().numpy(), extent=(0,1,0,1), cmap="coolwarm" ) #, vmin=0.0, vmax=1.0)
            if visualization == "energy": 
                energy = self.compute_energy(x[k])
                ax[i].imshow(energy.cpu().numpy(), extent=(0,1,0,1), vmin=0.0, vmax=1.7, cmap="viridis")
            if visualization == "momentum": 
                momentum = self.compute_momentum(x[k])
                ax[i].imshow(momentum.cpu().numpy(), extent=(0,1,0,1), vmin=0.0, vmax=3, cmap="viridis")
            # if visualization == "streamlines":
            #     speed = self.compute_momentum(x[k]).cpu().numpy()
            #     norm = Normalize(vmin=0, vmax=4.5, clip=True)
            #     ax[i].streamplot(X, Y, x[k,1].cpu().numpy(), x[k,0].cpu().numpy(), density=1.5, 
            #                     linewidth=1, 
            #                     color=speed,
            #                     norm=norm, 
            #                     cmap="nipy_spectral")

            ax[i].set_xlim(0,1)
            ax[i].set_ylim(0,1)
            ax[i].set_aspect("equal")
