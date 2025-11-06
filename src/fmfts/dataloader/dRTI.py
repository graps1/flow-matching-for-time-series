import os
import torch 
import h5py
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

class DatasetFullRTI3D(Dataset):
    def __init__(self, 
                 At, 
                 mode,
                 history=1,
                 dt=1, dx=1, dy=1, dz=1, 
                 include_timestamp=False):
        super().__init__()
        assert mode in ["train", "test"]
        prefix = f"{os.path.dirname(os.path.abspath(__file__))}/../datasets/rti3d/{mode}" 
        datapath = f"{prefix}/rayleigh_taylor_instability_At_{str(At)[2:]}.hdf5"

        f = h5py.File(datapath, "r")
        self.velocity_raw = f["t1_fields"]["velocity"]
        self.density_raw = f["t0_fields"]["density"]

        self.history = history
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.max_time_idx = 119 - self.history*self.dt
        
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
        mean = self.mean.view(1, -1, 1, 1, 1)
        std  = self.std.view(1, -1, 1, 1, 1)
        return (data - mean)/(std+1e-5)
    
    def denormalize(self, data):
        mean = self.mean.view(1, -1, 1, 1, 1)
        std  = self.std.view(1, -1, 1, 1, 1)
        return data*(std+1e-5) + mean
    
    def get(self, i0, i1, sequence_len=1):
        velocity = torch.tensor(self.velocity_raw[i0, i1:i1+sequence_len*self.dt:self.dt, ::self.dx, ::self.dy, ::self.dz])
        density =  torch.tensor( self.density_raw[i0, i1:i1+sequence_len*self.dt:self.dt, ::self.dx, ::self.dy, ::self.dz])
        velocity = velocity.permute(0, 4, 1, 2, 3) # <- shifts the channel to dim=1
        density  = density.unsqueeze(1) # <- creates new channel dimension
        x = torch.cat((velocity, density), dim=1)
        x = self.normalize(x)

        if self.include_timestamp:
            _, _, width, depth, height = x.shape
            timestamp = (i1 + torch.arange(len(velocity))*self.dt)/119
            timestamp = timestamp.view(-1, 1, 1, 1, 1).expand(-1, 1, width, depth, height)
            x = torch.cat((x, timestamp), dim=1)

        return x
    
    def __getitem__(self, idx):
        i0 = idx % self.n_runs
        idx = idx // self.n_runs
        i1 = idx % self.max_time_idx
        
        y = self.get(i0, i1, self.history)
        y = y.flatten(end_dim=1)
        x = self.get(i0, i1+self.history*self.dt, 1)
        x = x[0]

        return y, x
    
    def __plot_density_3d(self, density, ax, colorbar=False):
        norm = Normalize(vmin=self.rho2, vmax=self.rho1)
        cmap = plt.get_cmap("coolwarm")
            
        x_plane = density[-1,:,:]
        y_plane = density[:,0,:]
        z_plane = density[:,:,-1]

        yi, zi = torch.meshgrid(
            torch.linspace(0, 1, x_plane.shape[0]), 
            torch.linspace(0, 1, x_plane.shape[1]),
            indexing="ij")
        xi = torch.ones_like(zi)

        colors = cmap(norm(x_plane.cpu().numpy()))
        ax.plot_surface(
            xi.cpu().numpy(), 
            yi.cpu().numpy(), 
            zi.cpu().numpy(), 
            facecolors=colors,
            cstride=1, rstride=1,
            shade=False)
        
        xi, yi = torch.meshgrid(
            torch.linspace(0, 1, z_plane.shape[0]), 
            torch.linspace(0, 1, z_plane.shape[1]),
            indexing="ij")
        zi = torch.ones_like(xi)

        colors = cmap(norm(z_plane.cpu().numpy()))
        ax.plot_surface(
            xi.cpu().numpy(), 
            yi.cpu().numpy(), 
            zi.cpu().numpy(), 
            facecolors=colors,
            cstride=1, rstride=1,
            shade=False)

        xi, zi = torch.meshgrid(
            torch.linspace(0, 1, y_plane.shape[0]), 
            torch.linspace(0, 1, y_plane.shape[1]),
            indexing="ij")
        yi = torch.zeros_like(xi)

        colors = cmap(norm(y_plane.cpu().numpy()))
        ax.plot_surface(
            xi.cpu().numpy(), 
            yi.cpu().numpy(), 
            zi.cpu().numpy(), 
            facecolors=colors,
            cstride=1, rstride=1,
            shade=False)

        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_zlim(0,1)
        ax.set_aspect("equal")
        ax.view_init(elev=30, azim=-45, roll=0)

        if colorbar:
            m = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            m.set_array([])
            plt.colorbar(m,  ax=ax, use_gridspec=False, location="left", anchor=(-1.0, 0.5), shrink=0.9, label="Density")
    
    def plot(self, x, *ax, colorbar=False):

        if x.dim() == 4:
            x = x.unsqueeze(0)

        if len(ax)==0: 
            _, ax = plt.subplots(1, len(x), 
                                 figsize=(len(x)*5, 5), 
                                 subplot_kw = { "projection": "3d" },
                                 sharey=True, sharex=True)
            ax = [ ax ]

        if x.shape[1] == 5: x = x[:,:-1] # strip timestamp
        x = self.denormalize(x)

        for i in range(len(ax)):
            k = i * len(x) // len(ax)
            density = x[k,3].clip(min=0)
            self.__plot_density_3d(density, ax[i], colorbar=colorbar and i==len(ax)-1)

        return ax
        

if __name__=="__main__":
    ds = DatasetFullRTI3D(At=1/8, dt=4, dx=1, dy=1, dz=1)
    y, x = ds[900]

    ds.plot(y)
    plt.savefig("density.png")