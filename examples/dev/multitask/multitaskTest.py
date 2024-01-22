from interpretability.task_modeling.task_env.multi_task_env import MultiTaskWrapper

tasks = [
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
noise = 0.1
bin_size = 10
n_timesteps = 640

mtWrapper = MultiTaskWrapper(
    task_list=tasks,
    noise=noise,
    bin_size=bin_size,
    n_timesteps=n_timesteps,
)
mtWrapper.plot_tasks()
ics, inputs, outputs = mtWrapper.generate_dataset(100)
print(inputs.shape)
