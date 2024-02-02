import logging
import os
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

from interpretability.data_modeling.extensions.SAE.utils import make_data_tag
from interpretability.data_modeling.train_neural import train

dotenv.load_dotenv(override=True)
RUNS_HOME = os.environ.get("RUNS_HOME")
TRAINED_MODEL_PATH = os.environ.get("TRAINED_MODEL_PATH")

OmegaConf.register_new_resolver("make_data_tag", make_data_tag)

log = logging.getLogger(__name__)
# ---------------Options---------------
LOCAL_MODE = True
OVERWRITE = True
RUN_DESC = "NBFF_Comparison"
NUM_SAMPLES = 1
MODEL_CLASS = "SAE"
MODEL = "GRU_RNN"
DATA = "NBFF"
INFER_INPUTS = False

# -------------------------------------
SEARCH_SPACE = dict(
    # model=dict(
    #     latent_size=tune.grid_search([64]),
    # ),
    datamodule=dict(
        gen_model=tune.grid_search(["GRU_RNN"]),
        # Change the prefix to the correct path for your task-trained network
        prefix=tune.grid_search(["20240202_NBFF_Comparison"]),
    ),
    params=dict(
        seed=tune.grid_search([0]),
    ),
)

# -----------------Default Parameter Sets -----------------------------------
cpath = "../data_modeling/configs"

model_path = Path(
    (
        f"{cpath}/models/{MODEL_CLASS}/{DATA}/{DATA}_{MODEL}"
        f"{'_infer' if INFER_INPUTS else ''}.yaml"
    )
)
datamodule_path = Path(
    (
        f"{cpath}/datamodules/{MODEL_CLASS}/data_{DATA}"
        f"{'_infer' if INFER_INPUTS else ''}.yaml"
    )
)

callbacks_path = Path(f"{cpath}/callbacks/{MODEL_CLASS}/default_{DATA}.yaml")
loggers_path = Path(f"{cpath}/loggers/{MODEL_CLASS}/default.yaml")
trainer_path = Path(f"{cpath}/trainers/trainer_{DATA}.yaml")

path_dict = dict(
    model=model_path,
    datamodule=datamodule_path,
    callbacks=callbacks_path,
    loggers=loggers_path,
    trainer=trainer_path,
)

# ------------------Data Management Variables --------------------------------
DATE_STR = datetime.now().strftime("%Y%m%d")
RUN_TAG = f"{DATE_STR}_{RUN_DESC}"
RUNS_HOME = Path(RUNS_HOME)
RUN_DIR = RUNS_HOME / "data-trained" / RUN_TAG


def trial_function(trial):
    return trial.experiment_tag


# -------------------Main Function----------------------------------
def main(
    run_tag_in: str,
    path_dict: dict,
    trained_path: str,
):
    if LOCAL_MODE:
        ray.init(local_mode=True)
    if RUN_DIR.exists() and OVERWRITE:
        shutil.rmtree(RUN_DIR)

    RUN_DIR.mkdir(parents=True)
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
    run_dir = str(RUN_DIR)
    tune.run(
        tune.with_parameters(
            train,
            run_tag=run_tag_in,
            path_dict=path_dict,
        ),
        config=SEARCH_SPACE,
        resources_per_trial=dict(cpu=10, gpu=1),
        num_samples=NUM_SAMPLES,
        local_dir=run_dir,
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
        trained_path=TRAINED_MODEL_PATH,
    )
