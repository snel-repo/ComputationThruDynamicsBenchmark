import os

import dotenv
import h5py

dotenv.load_dotenv()

fname = (
    "20240130_RandomTargetDelay_Final_"
    "RandomTargetDelay_model_GRU_RNN_n_neurons_50_seed_0.h5"
)
fpath = os.path.join(os.environ["SIMULATED_HOME"], fname)

with h5py.File(fpath, "r") as h5file:
    train_enc_data = h5file["train_encod_data"][()]
    valid_enc_data = h5file["valid_encod_data"][()]
    train_inputs = h5file["train_inputs"][()]
    valid_inputs = h5file["valid_inputs"][()]

    train_activity = h5file["train_activity"][()]
    valid_activity = h5file["valid_activity"][()]

    train_latents = h5file["train_latents"][()]
    valid_latents = h5file["valid_latents"][()]

    train_inds = h5file["train_inds"][()]
    valid_inds = h5file["valid_inds"][()]

    readout = h5file["readout"][()]
    orig_mean = h5file["orig_mean"][()]
    orig_std = h5file["orig_std"][()]


temp = 1
