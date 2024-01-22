# %%
import matplotlib.pyplot as plt
import numpy as np

from interpretability.task_modeling.task_env.multi_task_env import MultiTaskWrapper

task_list = [
    "DelayPro",
    "DelayAnti",
    "MemoryPro",
    "MemoryAnti",
    "ReactPro",
    "ReactAnti",
    # "IntMod1",
    # "IntMod2",
    # "ContextIntMod1",
    # "ContextIntMod2",
    # "ContextIntMultimodal",
    # "Match2Sample",
    # "NonMatch2Sample",
    # "MatchCatPro",
    # "MatchCatAnti",
]

bin_size = 20
n_timesteps = 640
num_targets = 32
noise = 0.31

# Create the task wrapper
task_wrapper = MultiTaskWrapper(
    task_list=task_list,
    bin_size=bin_size,
    n_timesteps=n_timesteps,
    num_targets=num_targets,
    noise=noise,
)
dataset = task_wrapper.generate_dataset(n_samples=1000)
targets = dataset["targets"]
phase_dict = dataset["phase_dict"]
# %%
trial_len = np.zeros(len(phase_dict))
targ_ang = np.zeros(len(phase_dict))
for i in range(len(phase_dict)):
    trial_len[i] = phase_dict[i]["response"][1]
    target_out = targets[i, int(trial_len[i] - 1), 1:]
    targ_ang[i] = np.arctan2(target_out[1], target_out[0])


# %%
unique_targs = np.unique(targ_ang)
print(unique_targs)
plt.hist(targ_ang, bins=320)

# %%
targ_nums = range(32)
targ_nums = np.array(targ_nums)
targ_angs = np.linspace(-np.pi, np.pi, 32, endpoint=False)
targ_num = np.random.randint(0, 32, 1000)
print(targ_num)
targ_rand = targ_angs[targ_num]
plt.hist(targ_rand, bins=320)
# %%
