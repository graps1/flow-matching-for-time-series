Code for the paper https://arxiv.org/abs/2511.04641

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
                <experiment>.py             // loads & visualizes <experiment> dataset
            |- experiments
                trainer.py                      // script for training
                print_trained_model_stats.py    // prints properties of all trained models
                trainer.sh                      // shell script that calls trainer.py
                |- <experiment>
                    |- runs                 // contains data from tensorboard
                    |- checkpoints          // contains trained models w/ timestamp
                    |- trained_models       // contains the trained models
                    models.py               // contains model classes
                    training_parameters.py  // parameters for training
                    visualization.ipynb     // notebook that visualizes trained models
            |- tests                        // some scripts that check if the utilities work
            |- utils
                |- models
                    add.py                  // adversarial diffusion distillation
                    cfm_dir_dist.py         // direct distillation
                    cfm_prog_dist.py        // progressive distillation
                    cfm_rectifier.py        // rectified flows
                    cfm_velocity.py         // conditional flow matching velocity model
                    deterministic.py        // deterministic model
                    time_series_model.py    // abstract base class for time series prediction
                padding.py                  // defines flexible padding functionality
                unet.py                     // custom UNet class

```

## Simulation data

- [Compressible Navier-Stokes equations](https://drive.google.com/drive/folders/1IZ7tsLdnoQvchVx9RZq3__d5RBm7gSvc?usp=drive_link) (<0.5GB)
- [Rayleigh-Taylor instability](https://polymathic-ai.org/the_well/datasets/rayleigh_taylor_instability/) (>200GB). Can be downloaded using the [tutorial instructions](https://polymathic-ai.org/the_well/tutorials/dataset/).

Download the data & place the files such that the directory structure is as follows:

```
    |- src
        |- fmfts
            |- datasets
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



### Training models

This can be done by executing the script `trainer.py` in the `src/fmfts/experiments` section. 

The `trainer.py` script provides some outputs. For better visualization, I have also implemented some [tensorboard functionality](https://docs.pytorch.org/docs/stable//tensorboard.html) that tracks the training process. Visualize the training by running

    tensorboard --logdir <experiment>/runs

