import logging
import shutil
from datetime import datetime
from pathlib import Path

import dotenv
import ray
from omegaconf import OmegaConf
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from utils import make_data_tag

# Load data directory and run directory as environment variables.
# These variables are used when the config is resolved.
dotenv.load_dotenv(override=True)

# Add custom resolver to create the data_tag so it can be used for run dir
OmegaConf.register_new_resolver("make_data_tag", make_data_tag)

log = logging.getLogger(__name__)
# torch.autograd.set_detect_anomaly(True)
# ---------------Options---------------
LOCAL_MODE = False
OVERWRITE = True
RUN_DESC = "TBFF_RNN_Inputs2"
NUM_SAMPLES = 1
TASK = "Nbff"
MODEL = "RNN"

# grid_search
# choice
# Learning Rates (Sim):
# Flow= 1.88e-4
# Linear = 2e-3
# MLP = 7.3e-3
# Correct = ?
# Invert = 1.88e-4

SEARCH_SPACE = dict(
    # -----------------Model Parameters -----------------------------------
    model=dict(
        # -----------------Model Parameters -----------------------------------
        latent_size = tune.grid_search([64]),
    ),
    params=dict(seed=tune.grid_search([0]),
    )    
)

# -----------------Default Parameter Sets -----------------------------------
path_dict = dict(
    model=Path(f"configs/model/task_trained_{MODEL}_{TASK}.yaml"),
    task_env=Path(f"configs/task_env/task_trained_{TASK}.yaml"),
    simulator = Path(f"configs/simulator/default.yaml"),
    callbacks=Path(f"configs/callbacks/default_{TASK}.yaml"),
    loggers=Path("configs/logger/default.yaml"),
    trainer=Path(f"configs/trainer/default.yaml"),
)

# ------------------Data Management Variables --------------------------------
DATE_STR = datetime.now().strftime("%Y%m%d")
RUN_TAG = f"{DATE_STR}_{RUN_DESC}"
RUNS_HOME = Path("/snel/share/runs/dysts-learning/")
RUN_DIR = RUNS_HOME / "MultiDatasets" / "NODE" / RUN_TAG


def trial_function(trial):
    return trial.experiment_tag


# -------------------Main Function----------------------------------
def main(
    run_tag_in: str,
    path_dict: dict,
):
    from interpretability.task_modeling.task_train_prep import train

    if LOCAL_MODE:
        ray.init(local_mode=True)
    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)

    RUN_DIR.mkdir()
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
    tune.run(
        tune.with_parameters(
            train,
            run_tag=run_tag_in,
            path_dict=path_dict,
        ),
        metric="loss",
        mode="min",
        config=SEARCH_SPACE,
        resources_per_trial=dict(cpu=3, gpu=0.4),
        num_samples=NUM_SAMPLES,
        local_dir=RUN_DIR,
        search_alg=BasicVariantGenerator(),
        scheduler=FIFOScheduler(),
        verbose=1,
        progress_reporter=CLIReporter(
            metric_columns=["loss", "training_iteration"],
            sort_by_metric=True,
        ),
        trial_dirname_creator=trial_function,
    )


if __name__ == "__main__":
    main(
        run_tag_in=RUN_TAG,
        path_dict=path_dict,
    )
