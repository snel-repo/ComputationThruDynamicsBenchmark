import os
import pickle

from interpretability.task_modeling.datamodule.task_datamodule import TaskDataModule
from interpretability.task_modeling.task_env.multi_task_env import MultiTaskWrapper

SAVE_PATH = (
    "/home/csverst/Github/InterpretabilityBenchmark/pretrained/tt/MultiTaskSmall"
)
multitask = MultiTaskWrapper(
    task_list=[
        "MemoryPro",
        "MemoryAnti",
    ],
    noise=0.3,
    num_targets=32,
    bin_size=20,
    n_timesteps=640,
)

mt_dm = TaskDataModule(
    multitask,
    n_samples=50,
    batch_size=32,
    num_workers=0,
    seed=0,
)

path3 = os.path.join(SAVE_PATH, "datamodule_sim.pkl")
with open(path3, "wb") as f:
    pickle.dump(mt_dm, f)
