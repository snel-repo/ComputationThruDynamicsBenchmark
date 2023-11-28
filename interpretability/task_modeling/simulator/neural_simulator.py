import os

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# plt.switch_backend("Agg")
DATA_HOME = (
    "/home/csverst/Github/InterpretabilityBenchmark/"
    "interpretability/data_modeling/datasets"
)


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


class NeuralDataSimulator:
    def __init__(
        self,
        n_neurons=50,
        nonlin_embed=False,
    ):
        self.n_neurons = n_neurons
        self.nonlin_embed = nonlin_embed
        self.obs_noise = "poisson"
        self.readout = None
        self.orig_mean = None
        self.orig_std = None
        self.use_neurons = True

    def simulate_neural_data(
        self, task_trained_model, datamodule, coupled=False, seed=0
    ):

        # Make a filename based on the system being modeled, the number of neurons,
        # the nonlinearity, the observation noise, the epoch number, the model type,
        # and the seed

        # Get trajectories and model predictions
        train_data = datamodule.train_dataloader().dataset.tensors
        val_data = datamodule.val_dataloader().dataset.tensors

        ics = torch.cat([train_data[0], val_data[0]])
        inputs = torch.cat([train_data[1], val_data[1]])
        targets = torch.cat([train_data[2], val_data[2]])

        controlled, latents, actions = task_trained_model(ics, inputs, targets)

        if self.n_neurons > latents.shape[-1]:
            self.n_neurons = latents.shape[-1]

        filename = (
            f"{datamodule.data_env.dataset_name}_"
            f"model_{type(task_trained_model.model).__name__}_"
            f"n_neurons_{self.n_neurons}_"
            f"nonlin_embed_{self.nonlin_embed}_"
            f"obs_noise_{self.obs_noise}_"
            f"seed_{seed}.h5"
        )

        fpath = os.path.join(DATA_HOME, filename)
        n_trials, n_times, n_lat_dim = latents.shape
        latents = latents.detach().numpy()
        if self.use_neurons:
            # Make random permutation of latents
            rng = np.random.default_rng(seed)
            # get random permutation indices
            perm_inds = rng.permutation(n_lat_dim)
            latents_perm = latents[:, :, perm_inds]
            activity = latents_perm[:, :, : self.n_neurons]
            perm_neurons = perm_inds[: self.n_neurons]
            # get the readout matrix
            # should have a shape of (n_lat_dim, n_neurons)
            readout = np.zeros((n_lat_dim, self.n_neurons))
            for i in range(self.n_neurons):
                readout[perm_inds[i], i] = 1
        else:
            if self.n_neurons is not None:
                rng = np.random.default_rng(seed)
                # Randomly sample, normalize, and sort readout
                readout = rng.uniform(-2, 2, (n_lat_dim, self.n_neurons))
                if not self.nonlin_embed:
                    readout = readout / np.linalg.norm(readout, ord=1, axis=0)

                readout = readout[:, np.argsort(readout[0])]
            else:
                # Use an identity readout
                readout = np.eye(n_lat_dim)
            self.readout = readout
            activity = latents @ readout

        # Standardize and record original mean and standard deviations
        orig_mean = np.mean(activity, keepdims=True)
        orig_std = np.std(activity, keepdims=True)
        activity = (activity - orig_mean) / (2 * orig_std)

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

            h5file.create_dataset("train_inputs", data=inputs[train_inds])
            h5file.create_dataset("valid_inputs", data=inputs[valid_inds])
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
            if self.use_neurons:
                h5file.create_dataset("perm_neurons", data=perm_neurons)

    def simulate_new_data(self, task_trained_model, datamodule, seed):

        if self.readout is None:
            ValueError("Must first generate data before simulating new data")

        # Get trajectories and model predictions
        train_data = datamodule.train_dataloader().dataset.tensors
        _, latents = task_trained_model(train_data[0])
        inputs = train_data[1].numpy()
        n_trials, n_times, n_lat_dim = latents.shape
        latents = latents.detach().numpy()

        readout = self.readout
        activity = latents @ readout

        activity = (activity - self.orig_mean) / self.orig_std

        rng = np.random.default_rng(seed)
        if self.nonlin_embed:
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

        return data, activity, latents, inputs
