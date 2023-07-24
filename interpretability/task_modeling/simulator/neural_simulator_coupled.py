import torch
import numpy as np
import os
import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
import h5py

# plt.switch_backend("Agg")
DATA_HOME = "/home/csverst/Github/InterpretabilityBenchmark/interpretability/data_modeling/datasets"

def sigmoidActivation(module, input):
    return 1 / (1 + module.exp(-1 * input))

def apply_data_warp_sigmoid(data):
    warp_functions = [sigmoidActivation, sigmoidActivation, sigmoidActivation]
    firingMax = [2, 2, 2, 2]
    numDims = data.shape[1]

    a = np.array(1)
    dataGen = type(a) == type(data)
    if dataGen:
        module = np
    else:
        module = torch

    for i in range(numDims):

        j = np.mod(i, len(warp_functions) * len(firingMax))
        # print(f'Max firing {firingMax[np.mod(j, len(firingMax))]}
        # warp {warp_functions[int(np.floor((j)/(len(warp_functions)+1)))]}')
        data[:, i] = firingMax[np.mod(j, len(firingMax))] * warp_functions[
            int(np.floor((j) / (len(warp_functions) + 1)))
        ](module, data[:, i])
    return data

class NeuralDataSimulatorCoupled():
    def __init__(
            self, 
            n_neurons = 50,
            nonlin_embed = False,
            ):
        self.n_neurons = n_neurons
        self.nonlin_embed = nonlin_embed
        self.obs_noise = "poisson"
        self.readout = None
        self.orig_mean = None
        self.orig_std = None


    def simulate_neural_data(self, task_trained_model, datamodule, seed):


        # Make a filename based on the system being modeled, the number of neurons, 
        # the nonlinearity, the observation noise, the epoch number, the model type,
        # and the seed

        filename = (f"{type(datamodule.data_env).__name__}_"
                    f"model_{type(task_trained_model).__name__}_"
                    f"n_neurons_{self.n_neurons}_"
                    f"nonlin_embed_{self.nonlin_embed}_"
                    f"obs_noise_{self.obs_noise}_"
                    f"seed_{seed}.h5")
        
        fpath = os.path.join(DATA_HOME, filename)

        # Get trajectories and model predictions
        train_data = datamodule.train_dataloader().dataset.tensors
        val_data = datamodule.val_dataloader().dataset.tensors

        state_train = train_data[0]
        goal_train = train_data[1]
        state_val = val_data[0]
        goal_val = val_data[1]
        state_data = torch.cat((state_train, state_val), dim=0)
        goal_data = torch.cat((goal_train, goal_val), dim=0)
        _,_, latents, actions = task_trained_model(state_data, goal_data)
        # combine the first tensor of train and val data
        latents= latents[:,1:,:]
        n_trials, n_times, n_lat_dim = latents.shape
        latents = latents.detach().numpy()


        if self.n_neurons is not None:
            rng = np.random.default_rng(seed)
            # Randomly sample, normalize, and sort readout
            readout = rng.uniform(
                -2, 2, (n_lat_dim, self.n_neurons)
            )
            if not self.nonlin_embed:
                readout = readout / np.linalg.norm(readout, ord=1, axis=0)

            readout = readout[:, np.argsort(readout[0])]
        else:
            # Use an identity readout
            readout = np.eye(n_lat_dim)
        self.readout = readout
        activity = latents @ readout

        # Standardize and record original mean and standard deviations
        orig_mean = np.mean(activity, axis=0, keepdims=True)
        orig_std = np.std(activity, axis=0, keepdims=True)
        activity = (activity - orig_mean) / orig_std

        self.orig_mean = orig_mean
        self.orig_std = orig_std

        if self.nonlin_embed:
            rng = np.random.default_rng(seed)
            scaling_matrix = np.logspace(0.2, 1, (self.n_neurons))
            activity = activity * scaling_matrix[None, :]
        # Add noise to the observations
        if self.obs_noise is not None:
            if self.nonlin_embed:
                activity = apply_data_warp_sigmoid(activity)
            elif self.obs_noise in ["poisson"]:
                activity = np.exp(activity)
            noise_fn = getattr(rng, self.obs_noise)
            data = noise_fn(activity).astype(float)
        else:
            if self.nonlin_embed:
                activity = apply_data_warp_sigmoid(activity)
            data = activity
            
        latents = latents.reshape(n_trials, n_times, n_lat_dim)
        activity = activity.reshape(n_trials, n_times, self.n_neurons)
        data = data.reshape(n_trials, n_times, self.n_neurons)

        # Perform data splits
        inds = np.arange(n_trials)
        train_inds, valid_inds = train_test_split(
            inds, test_size=0.2, random_state=seed
        )
        # Save the trajectories
        with h5py.File(fpath, "w") as h5file:
            h5file.create_dataset("train_encod_data", data=data[train_inds])
            h5file.create_dataset("valid_encod_data", data=data[valid_inds])
            # h5file.create_dataset("test_encod_data", data=data[test_inds])

            h5file.create_dataset("train_recon_data", data=data[train_inds])
            h5file.create_dataset("valid_recon_data", data=data[valid_inds])
            # h5file.create_dataset("test_recon_data", data=data[test_inds])

            # h5file.create_dataset("train_inputs", data=inputs[train_inds])
            # h5file.create_dataset("valid_inputs", data=inputs[valid_inds])
            # h5file.create_dataset("test_inputs", data=inputs[test_inds])

            h5file.create_dataset("train_activity", data=activity[train_inds])
            h5file.create_dataset("valid_activity", data=activity[valid_inds])
            # h5file.create_dataset("test_activity", data=activity[test_inds])

            h5file.create_dataset("train_latents", data=latents[train_inds])
            h5file.create_dataset("valid_latents", data=latents[valid_inds])
            # h5file.create_dataset("test_latents", data=latents[test_inds])

            h5file.create_dataset("train_inds", data=train_inds)
            h5file.create_dataset("valid_inds", data=valid_inds)
            # h5file.create_dataset("test_inds", data=test_inds)

            h5file.create_dataset("readout", data=readout)
            h5file.create_dataset("orig_mean", data=orig_mean)
            h5file.create_dataset("orig_std", data=orig_std)