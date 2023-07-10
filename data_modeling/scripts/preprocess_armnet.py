import pickle

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Load .mat file
import scipy.io as sio


def plot_trial(trial_index):
    # Get trial data
    targetPos = targetPos_list[trial_index]
    pos = pos_list[trial_index]
    go_sig = go_signal_list[trial_index]
    force = force_list[trial_index]

    first_sig = np.where(go_sig == 1)[0][0]

    # Calculate velocity as difference between subsequent positions
    velocity = np.diff(pos, axis=0)

    # Create a new figure with 3 subplots
    fig, axs = plt.subplots(2, 1, sharex=False, figsize=(10, 10))

    # Plot target positions
    axs[0].scatter(targetPos[:, 0], targetPos[:, 1], c="r", label="Target Positions")

    # Plot positions
    axs[0].plot(pos[:, 0], pos[:, 1], c="b", label="Positions")

    # If any force is not 0, plot it
    if np.any(force != 0):
        # Find first point where force isnot 0
        first_force = np.where(force[:, 0] != 0)[0][0]
        # Plot arrow for force at first force point
        axs[0].arrow(
            pos[first_force, 0],
            pos[first_force, 1],
            force[first_force, 0] / 100,
            force[first_force, 1] / 100,
            color="k",
            width=0.006,
        )

    # Title and labels for positions
    axs[0].set_title(f"Trial {trial_index} Positions and Velocities")
    axs[0].set_ylabel("Y Position")

    axs[0].set_xlim([0.05, 0.35])
    axs[0].set_ylim([-0.5, -0.15])
    axs[0].set_aspect("equal", "box")
    # Plot x velocity from -150 to end of trial at 10 ms bins
    timevec = np.arange(-50, len(velocity) - 50) * 10
    axs[1].plot(timevec, velocity[:, 0], c="g", label="X Velocity")
    axs[1].set_ylabel("Velocity")
    axs[1].set_xlabel("Time (ms rel. go cue)")
    # Plot y velocity
    axs[1].plot(timevec, velocity[:, 1], c="m", label="Y Velocity")
    # Plot vertical line at go signal
    axs[1].axvline(x=timevec[first_sig], c="r", label="Go Signal")
    if np.any(force != 0):
        # Plot vertical line at first force
        axs[1].axvline(x=timevec[first_force], c="k", label="First Force")
    axs[1].legend()
    # Show the plot

    plt.tight_layout()
    plt.show()
    plt.savefig(f"trial_{trial_index}.png")


infer_inputs = True
path_to_mat = (
    "/snel/share/data/armnet/mat/" "hpRepeat_forDynFit_2dForceVis_longestDelay.mat"
)

path_to_pkl = (
    "/snel/share/data/armnet/pkl/" "hpRepeat_forDynFit_2dForceVis_longestDelay.pkl"
)

if infer_inputs:
    path_to_h5 = "/home/csverst/Github/lfads-torch/datasets/armnet_infer-10ms-val.h5"
else:
    path_to_h5 = "/home/csverst/Github/lfads-torch/datasets/armnet-10ms-val.h5"

mat_contents = sio.loadmat(path_to_mat)  # load .mat file
mat_keys = mat_contents.keys()  # show keys

trial_start_inds = mat_contents["trialStartIdx"]  # get trial start indices
rnnUnits = mat_contents["rnnUnits"]  # get RNN units
pos = mat_contents["cursorPos"]  # get cursor position
goSignal = mat_contents["goSignal"]  # get go signal
handPos = mat_contents["handPos"]  # get hand position
musLen = mat_contents["musLen"]  # get muscle length
musVel = mat_contents["musVel"]  # get muscle velocity
targetPos = mat_contents["targetPos"]  # get target position
force = mat_contents["appliedHandForce"]  # get force

trial_start_inds = np.squeeze(trial_start_inds)  # squeeze trial start indices
goSignal = np.squeeze(goSignal)  # squeeze go signal
num_samples = rnnUnits.shape[0]  # get number of samples
# add a -1 to the end of trial_start_inds
trial_start_inds_1 = np.append(trial_start_inds, num_samples + 1)

rnn_list = []
pos_list = []
handPos_list = []
musLen_list = []
musVel_list = []
force_list = []
targetPos_list = []
goSignal_inds = []
out_trial = []
go_signal_list = []

for i in range(len(trial_start_inds)):
    if not np.all(targetPos[trial_start_inds_1[i], :] == [0.2, -0.325]):
        rnn_list.append(
            rnnUnits[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1], :]
        )
        pos_list.append(pos[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1], :])
        handPos_list.append(
            handPos[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1], :]
        )
        musLen_list.append(
            musLen[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1], :]
        )
        musVel_list.append(
            musVel[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1], :]
        )
        force_list.append(
            force[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1], :]
        )
        targetPos_list.append(
            targetPos[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1], :]
        )
        go_signal_list.append(
            goSignal[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1]] == 1
        )
        # Find the index of the first go signal in the trial
        goSignal_inds.append(
            np.where(
                goSignal[trial_start_inds_1[i] - 15 : trial_start_inds_1[i + 1]] == 1
            )[0]
        )

for i in range(10):
    plot_trial(i)

# %%
# Trim the trials from the beginning to 100 bins after the go signal
# Get the length of each trial
trial_lens = [len(x) for x in rnn_list]
min_len = np.min(trial_lens)
max_len = np.max(trial_lens)
print(f"Minimum trial length: {min_len}")
# Trim the trials to the minimum length
rnn_arr = np.array([x[:min_len, :] for x in rnn_list])
pos_arr = np.array([x[:min_len, :] for x in pos_list])
handPos_arr = np.array([x[:min_len, :] for x in handPos_list])
musLen_arr = np.array([x[:min_len, :] for x in musLen_list])
musVel_arr = np.array([x[:min_len, :] for x in musVel_list])
force_arr = np.round(np.array([x[:min_len, :] for x in force_list]), 5)
targetPos_arr = np.array([x[:min_len, :] for x in targetPos_list])
go_signal_arr = np.array([x[:min_len] for x in go_signal_list])
# Add a dimension to the go signal array
go_signal_arr = np.expand_dims(go_signal_arr, axis=2)

# Pick 100 RNN units randomly
num_units = 100
unit_inds = np.random.permutation(rnn_arr.shape[2])[:num_units]
rnn_arr = rnn_arr[:, :, unit_inds]


# Create condition index based on the target position and the force
conds = np.concatenate((targetPos_arr, force_arr), axis=2)
conds = np.round(conds, 5)
unique_conds = np.unique(conds.reshape(-1, 4), axis=0)
unique_conds_theta = np.mod(
    np.arctan2(unique_conds[:, 1], unique_conds[:, 0]) + 2 * np.pi, 2 * np.pi
)
radsort = np.argsort(unique_conds_theta)
unique_conds_sort = unique_conds[radsort, :]
cond_inds = -1 * np.ones((len(targetPos_arr),))

for i in range(len(unique_conds_sort)):
    cond_inds[np.all(conds[:, -1, :] == unique_conds_sort[i, :], axis=1)] = int(i)

# External inputs are the go_signal and the force on the hand
ext_input_arr = np.concatenate((go_signal_arr, force_arr), axis=2)

# Train/val/test split the data
all_inds = np.arange(len(rnn_arr))
all_inds_rand = np.random.permutation(all_inds)
train_inds = all_inds_rand[: int(0.8 * len(all_inds))]
val_inds = all_inds_rand[int(0.8 * len(all_inds)) : int(0.9 * len(all_inds))]
test_inds = all_inds_rand[int(0.9 * len(all_inds)) :]

# Add noise to rnn_arr that scales with the variance of each channel
for i in range(rnn_arr.shape[2]):
    std_dev = np.std(rnn_arr[:, :, i])
    noise = 0.1 * np.random.randn(*rnn_arr.shape[:2]) * std_dev
    rnn_arr[:, :, i] += noise


# Split the data
rnn_train = rnn_arr[train_inds, :, :]
rnn_val = rnn_arr[val_inds, :, :]
rnn_test = rnn_arr[test_inds, :, :]
pos_train = pos_arr[train_inds, :, :]
pos_val = pos_arr[val_inds, :, :]
pos_test = pos_arr[test_inds, :, :]
handPos_train = handPos_arr[train_inds, :, :]
handPos_val = handPos_arr[val_inds, :, :]
handPos_test = handPos_arr[test_inds, :, :]
musLen_train = musLen_arr[train_inds, :, :]
musLen_val = musLen_arr[val_inds, :, :]
musLen_test = musLen_arr[test_inds, :, :]
musVel_train = musVel_arr[train_inds, :, :]
musVel_val = musVel_arr[val_inds, :, :]
musVel_test = musVel_arr[test_inds, :, :]
force_train = force_arr[train_inds, :, :]
force_val = force_arr[val_inds, :, :]
force_test = force_arr[test_inds, :, :]
targetPos_train = targetPos_arr[train_inds, :, :]
targetPos_val = targetPos_arr[val_inds, :, :]
targetPos_test = targetPos_arr[test_inds, :, :]
go_signal_train = go_signal_arr[train_inds, :]
go_signal_val = go_signal_arr[val_inds, :]
go_signal_test = go_signal_arr[test_inds, :]
cond_inds_train = cond_inds[train_inds]
cond_inds_val = cond_inds[val_inds]
cond_inds_test = cond_inds[test_inds]

ext_input_train = ext_input_arr[train_inds, :, :]
ext_input_val = ext_input_arr[val_inds, :, :]
ext_input_test = ext_input_arr[test_inds, :, :]

behavior_train = np.concatenate(
    (pos_train, handPos_train, musLen_train, musVel_train), axis=2
)
behavior_val = np.concatenate((pos_val, handPos_val, musLen_val, musVel_val), axis=2)
behavior_test = np.concatenate(
    (pos_test, handPos_test, musLen_test, musVel_test), axis=2
)

save_dict = {
    "rnn": rnn_arr,
    "pos": pos_arr,
    "handPos": handPos_arr,
    "musLen": musLen_arr,
    "musVel": musVel_arr,
    "force": force_arr,
    "targetPos": targetPos_arr,
    "go_signal": go_signal_arr,
}

# Save the dictionary as a pickle file
with open(path_to_pkl, "wb") as f:
    pickle.dump(save_dict, f)

if infer_inputs:
    # Save the dictionary as an h5 file
    with h5py.File(path_to_h5, "w") as f:
        f.create_dataset("train_encod_data", data=rnn_train)
        f.create_dataset("train_recon_data", data=rnn_train)

        f.create_dataset("valid_encod_data", data=rnn_val)
        f.create_dataset("valid_recon_data", data=rnn_val)

        f.create_dataset("test_encod_data", data=rnn_test)
        f.create_dataset("test_recon_data", data=rnn_test)

        f.create_dataset("train_behavior", data=behavior_train)
        f.create_dataset("valid_behavior", data=behavior_val)
        f.create_dataset("test_behavior", data=behavior_test)

        f.create_dataset("train_cond_idx", data=cond_inds_train)
        f.create_dataset("valid_cond_idx", data=cond_inds_val)
        f.create_dataset("test_cond_idx", data=cond_inds_test)

        f.create_dataset("condition_list", data=unique_conds_sort)

        f.create_dataset("train_decode_mask", data=train_inds)
        f.create_dataset("valid_decode_mask", data=val_inds)

else:
    # Save the dictionary as an h5 file
    with h5py.File(path_to_h5, "w") as f:
        f.create_dataset("train_encod_data", data=rnn_train)
        f.create_dataset("train_recon_data", data=rnn_train)

        f.create_dataset("valid_encod_data", data=rnn_val)
        f.create_dataset("valid_recon_data", data=rnn_val)

        f.create_dataset("test_encod_data", data=rnn_test)
        f.create_dataset("test_recon_data", data=rnn_test)

        f.create_dataset("train_ext_input", data=ext_input_train)
        f.create_dataset("valid_ext_input", data=ext_input_val)
        f.create_dataset("test_ext_input", data=ext_input_test)

        f.create_dataset("train_behavior", data=behavior_train)
        f.create_dataset("valid_behavior", data=behavior_val)
        f.create_dataset("test_behavior", data=behavior_test)

        f.create_dataset("train_cond_idx", data=cond_inds_train)
        f.create_dataset("valid_cond_idx", data=cond_inds_val)
        f.create_dataset("test_cond_idx", data=cond_inds_test)

        f.create_dataset("condition_list", data=unique_conds_sort)

        f.create_dataset("train_decode_mask", data=train_inds)
        f.create_dataset("valid_decode_mask", data=val_inds)
