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
from ray.tune.search.basic_variant import BasicVariantGenerator

from utils import make_data_tag

# Load data directory and run directory as environment variables.
# These variables are used when the config is resolved.
dotenv.load_dotenv(override=True)

# Add custom resolver to create the data_tag so it can be used for run dir
OmegaConf.register_new_resolver("make_data_tag", make_data_tag)

log = logging.getLogger(__name__)
# torch.autograd.set_detect_anomaly(True)
# ---------------Options---------------
LOCAL_MODE = True
OVERWRITE = True
RUN_DESC = "NBFF_NODE"
NUM_SAMPLES = 1
TASK = "NBFF"
MODEL = "NODE"


SEARCH_SPACE = dict(
    # -----------------Model Parameters -----------------------------------
    model=dict(
        # -----------------Model Parameters -----------------------------------
        # latent_size = tune.grid_search([128]),
    ),
    task_wrapper=dict(
        # -----------------Task Wrapper Parameters -----------------------------------
        learning_rate = tune.grid_search([1e-3]),
    ),
    params=dict(seed=tune.grid_search([0]),
    )    
)

# -----------------Default Parameter Sets -----------------------------------
path_dict = dict(
    model=Path(f"configs/model/{MODEL}.yaml"),
    task_wrapper = Path(f"configs/task_wrapper/{TASK}.yaml"),
    datamodule=Path(f"configs/datamodule/task_trained_{TASK}.yaml"),
    simulator = Path(f"configs/simulator/default_{TASK}.yaml"),
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
        storage_path=str(RUN_DIR),
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
