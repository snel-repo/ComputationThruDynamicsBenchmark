# %%
from interpretability.task_modeling.task_env.multi_task_env import MultiTaskWrapper

# %%
task_list = [
    "DelayPro",
    "DelayAnti",
    "MemoryPro",
    "MemoryAnti",
    "ReactPro",
    "ReactAnti",
    "IntMod1",
    "IntMod2",
    "ContextIntMod1",
    "ContextIntMod2",
    "ContextIntMultimodal",
    "Match2Sample",
    "NonMatch2Sample",
    "MatchCatPro",
    "MatchCatAnti",
]

env = MultiTaskWrapper(
    task_list=task_list,
    bin_size=20,
    num_targets=32,
    n_timesteps=640,
    noise=0.15,
    grouped_sampler=False,
)
env.plot_tasks()


# %%
