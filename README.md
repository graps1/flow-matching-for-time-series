## Installation

In the top level directory, run

    pip install -e .

this installs an editable version of this project.


## Directory structure

```
    |- src
        |- fmfts
            |- datasets
                (here the datasets are stored, see below)
            |- dataloader
                (contains code for loading and visualizing 
                the training/test data)
            |- experiments
                trainer.py                  // script for training
                |- <experiment name>
                    |- runs                 // contains data from tensorboard
                    |- checkpoints          // contains trained models w/ timestamp
                    |- trained_models
                        (contains the trained models)
                    models.py               // contains model classes (Velocity, Flow, SingleStep)
                    training_parameters.py  // parameters for training
                    visualization.ipynb     // notebook that visualizes trained models
            |- tests
                (some scripts that check if the provided utilities work)
            |- utils
                |- models
                    cfm_flow.py             // base class for flow model
                    cfm_single_step.py      // base class for single step model
                    cfm_velocity.py         // base class for velocity model
                loss_fn.py                  // defines a weighted Sobolev loss (not used)
                padding.py                  // defines better padding functionality
                unet.py                     // custom UNet class

```

## Simulation data

- Compressible Navier-Stokes & Kuramoto-Sivashinsky: https://drive.google.com/drive/folders/1IZ7tsLdnoQvchVx9RZq3__d5RBm7gSvc?usp=drive_link
    - both together < 0.5GB
- Rayleigh-Taylor instability: https://polymathic-ai.org/the_well/datasets/rayleigh_taylor_instability/
    - can be downloaded using the tutorial instructions https://polymathic-ai.org/the_well/tutorials/dataset/
    - relatively large, >200GB

---

Download the data & place the files such that the directory structure is as follows:

```
    |- src
        |- fmfts
            |- datasets
                |- ks2d
                    |- ks2d_data_test.pt
                    |- ks2d_data_train.pt
                |- ns2d
                    |- ns2d_data_test.pt
                    |- ns2d_data_train.pt
                |- rti3d
                    |- test
                        |- rayleigh_taylor_instability_At_25.hdf5
                        |- rayleigh_taylor_instability_At_50.hdf5
                        |- rayleigh_taylor_instability_At_75.hdf5
                        |- rayleigh_taylor_instability_At_125.hdf5
                        |- rayleigh_taylor_instability_At_0625.hdf5
                    |- train 
                        |- rayleigh_taylor_instability_At_25.hdf5
                        |- rayleigh_taylor_instability_At_50.hdf5
                        |- rayleigh_taylor_instability_At_75.hdf5
                        |- rayleigh_taylor_instability_At_125.hdf5
                        |- rayleigh_taylor_instability_At_0625.hdf5
```


## Models

For each of the four experiments (`ks2d`, `ns2d`, `rti3d_full` and `rti3d_sliced`), I have implemented a `velocity`, a `flow` and a `single_step` model. 

- The `velocity` models are trained using flow matching. They take as arguments the current state $x_t$, the current time $t$ and the previous state $y$. For a trained velocity model $v_t(x_t|y)$, one can sample the final state by sampling $x_0 \sim p_0$ and solving the ODE $\dot x_t = v_t(x_t|y)$.
- The `single_step` model is a simple distillation model that directly predicts $x_1$ from $x_0$. It is trained by solving the flow  matching ODE $\dot x_t = v_t(x_t|y)$ and using $x_1$ as the training target, given $x_0$ and $y$. It is initialized by using the pre-trained velocity model $\phi(x_0|y) = v_0(x_0|y)$ and setting $F_1^\xi(x_0|y) = x_0 + \phi\^xi(x_0|y)$, where $\xi$ are the parameters that are optimized.
- The `flow` model does something similar but is more intricate. It really tries to solve for the flow of the pre-trained velocity model $v_t$ for every increment $\delta$, i.e. the function $F_\delta(x_t, t) = x_{t+\delta}$ where $x_t$ solves the flow matching ODE. The flow model is defined as $$F_\delta^\xi(x_t,t) = x_t + \delta v_t(x_t) + \delta^2 (\phi_t^\xi(x_t,\delta) - v_t(x_t)),$$ where $\phi$ is a trainable neural network with parameters $\xi$. We initialize $\phi$ as a copy of $v$ with a slightly larger initial layer.

### Pretrained models

Download: https://drive.google.com/drive/folders/1UUHhh9m6asMiBOdKTth2c47Qvw2upUGw?usp=drive_link

Place the downloaded models such that the directory structure is as follows:

```
    |- src
        |- fmfts
            |- experiments 
                |- ks2d
                    |- trained_models
                        |- state_velocity.pt
                |- ns2d
                    |- trained_models
                        |- state_velocity.pt
                        |- state_flow.pt
                |- rti3d_full
                    |- trained_models
                        |- state_velocity.pt
                        |- state_flow.pt
                        |- state_single_step.pt
```

- Each of these `.pt` files stores the model, the optimizer, the time used for training, and the number of training iterations. 
- I have not yet uploaded all models since I have to do some retraining. In particular, models for the experiment `rti3d_sliced` are missing.


### Training models

This can be done by executing the script `trainer.py` in the `src/fmfts/experiments` section. 
Its first parameter is the experiment, i.e. either `ks2d`, `ns2d`, `rti3d_sliced` or `rti3d_full`. Its second argument is the model type that is to be trained, i.e. either `velocity`, `flow` or `single_step`. E.g.,

    python3 trainer.py ns2d flow

Continues to train the flow model. To train a new model, add `--new`.

Training parameters such as batch size, number of discretization steps for flow or single step model, or learning rates are stored in the `training_parameters.py` files that can be found in each experiment directory.

### Visualizing training

The `trainer.py` script provides some outputs. For better visualization, I have also implemented some tensorboard functionality (https://docs.pytorch.org/docs/stable//tensorboard.html) that tracks the training process. Visualize the training by running

    tensorboard --logdir <experiment name>/runs

