import torch 
import h5py
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os

class DatasetSlicedRTI3D(Dataset):
    def __init__(self, 
                 At, 
                 mode,
                 history=1,
                 dt=1, dy=1, dz=1, 
                 include_timestamp=False):
        super().__init__()

        assert mode in ["train", "test"]
        datapath = f"{os.path.dirname(os.path.abspath(__file__))}/../datasets/rti3d/{mode}/rayleigh_taylor_instability_At_{str(At)[2:]}.hdf5"
        f = h5py.File(datapath, "r")

        self.history = history
        self.dt = dt
        self.dy = dy
        self.dz = dz
        self.max_time_idx = 119 - self.history*self.dt
        self.velocity_raw = f["t1_fields"]["velocity"] # [:, :, :, ::self.dy, ::self.dz]).cuda()
        self.density_raw = f["t0_fields"]["density"] # [:, :, :, ::self.dy, ::self.dz]).cuda()
        
        # taken from stats.yaml
        self.mean = torch.tensor([-6.0036E-06, -1.6880E-05, -4.4674E-06, 7.7363E-01])
        self.std = torch.tensor([8.2252E-03, 8.1858E-03, 1.3937E-02, 2.6884E-01])

        self.n_runs = 9 if mode == "train" else 2

        self.rho2 = (1-At)/(At+1)
        self.rho1 = 1

        self.include_timestamp = include_timestamp

    def __len__(self):
        # nr_simulation * time * x-coordinate
        return self.n_runs * self.max_time_idx * 128 
    
    def normalize(self, data):
        mean = self.mean.view(1, -1, 1, 1)
        std  = self.std.view(1, -1, 1, 1)
        return (data - mean)/(std+1e-5)
    
    def denormalize(self, data):
        mean = self.mean.view(1, -1, 1, 1)
        std  = self.std.view(1, -1, 1, 1)
        return data*(std+1e-5) + mean
    
    def get(self, i0, i1, i2, sequence_len=1):
        velocity = torch.tensor(self.velocity_raw[i0, i1:i1+sequence_len*self.dt:self.dt, i2, ::self.dy, ::self.dz])
        density =  torch.tensor( self.density_raw[i0, i1:i1+sequence_len*self.dt:self.dt, i2, ::self.dy, ::self.dz])
        velocity = velocity.permute(0, 3, 1, 2)
        density  = density.unsqueeze(1)

        velocity = velocity.transpose(2,3).flip(2) # <- I'm not sure if this is the correct transformation for the velocity components
        density  = density.transpose(2,3).flip(2)  # but it doesn't matter much if we don't need to have them in the correct order

        x = torch.cat((velocity, density), dim=1)
        x = self.normalize(x)

        if self.include_timestamp:
            _, _, width, height = x.shape
            timestamp = (i1 + torch.arange(len(velocity))*self.dt)/119
            timestamp = timestamp.view(-1, 1, 1, 1).expand(-1, 1, width, height)
            x = torch.cat((x, timestamp), dim=1)

        return x
    
    def __getitem__(self, idx):
        i0 = idx % self.n_runs
        idx = idx // self.n_runs
        i1 = idx % self.max_time_idx
        idx = idx // self.max_time_idx
        i2 = idx
        
        y = self.get(i0, i1, i2, self.history)
        y = y.flatten(end_dim=1)
        x = self.get(i0, i1+self.history*self.dt, i2, 1)
        x = x[0]

        return y, x
    
    
    def plot(self, x, *ax):
        if x.dim() == 3:
            x = x.unsqueeze(0)

        if self.include_timestamp: x = x[:,:-1]  # removes the timestamp
        x = self.denormalize(x)

        Y, X = torch.meshgrid(
            torch.linspace(0, 1, 64), 
            torch.linspace(0, 1, 64), 
            indexing="ij")
        X, Y = X.cpu().numpy(), Y.cpu().numpy()

        if len(ax) == 0: 
            _, ax = plt.subplots(1, len(x), figsize=(len(x)*2, 2.5), sharey=True, sharex=True)
            if len(x) == 1: ax = [ ax ]

        for i in range(len(ax)):
            k = i * len(x) // len(ax)
            velocity = x[k,:3].cpu().numpy()
            density = x[k,-1].clip(min=0).cpu().numpy()
            # sqrt_energy = density * (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)
            ax[i].imshow(density, extent=(0,1,0,1), cmap="coolwarm", vmin=self.rho2, vmax=self.rho1)

            ax[i].set_xlim(0,1)
            ax[i].set_ylim(0,1)
            ax[i].set_aspect("equal")

        return ax