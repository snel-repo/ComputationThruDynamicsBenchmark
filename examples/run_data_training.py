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

from ctd.data_modeling.extensions.SAE.utils import make_data_tag
from ctd.data_modeling.train_JAX import train_JAX
from ctd.data_modeling.train_PTL import train_PTL

dotenv.load_dotenv(override=True)
HOME_DIR = Path(os.environ.get("HOME_DIR"))

OmegaConf.register_new_resolver("make_data_tag", make_data_tag)

log = logging.getLogger(__name__)
# ---------------Options---------------
LOCAL_MODE = False
OVERWRITE = True
WANDB_LOGGING = True

RUN_DESC = "NBFF_ResLFADS_SuppInp"
NUM_SAMPLES = 1
MODEL_CLASS = "LFADS"  # "LFADS" or "SAE"
MODEL = "ResLFADS"  # "ResLFADS" or "LFADS"
DATA = "NBFF"
INFER_INPUTS = False

prefix = "20240229_3BFF_GRU_Tutorial"

# -------------------------------------
SEARCH_SPACE = dict(
    model=dict(
        lr_init=tune.grid_search([5e-3, 3e-3]),
        lr_stop=tune.grid_search([1e-7]),
    ),
    datamodule=dict(
        gen_model=tune.grid_search(["GRU_RNN"]),
        # Change the prefix to the correct path for your task-trained network
        prefix=tune.grid_search([prefix]),
    ),
    params=dict(
        seed=tune.grid_search([0]),
    ),
    trainer=dict(
        max_epochs=tune.grid_search([500]),
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
trainer_path = Path(f"{cpath}/trainers/{MODEL_CLASS}/trainer_{DATA}.yaml")

if not WANDB_LOGGING:
    loggers_path = Path(f"{cpath}/loggers/{MODEL_CLASS}/default_no_wandb.yaml")
    callbacks_path = Path(f"{cpath}/callbacks/{MODEL_CLASS}/default_no_wandb.yaml")

if MODEL_CLASS not in ["LDS"]:
    config_dict = dict(
        model=model_path,
        datamodule=datamodule_path,
        callbacks=callbacks_path,
        loggers=loggers_path,
        trainer=trainer_path,
    )
    train = train_PTL
else:
    config_dict = dict(
        model=model_path,
        datamodule=datamodule_path,
        trainer=trainer_path,
    )
    train = train_JAX

# ------------------Data Management Variables --------------------------------
DATE_STR = datetime.now().strftime("%Y%m%d")
RUN_TAG = f"{DATE_STR}_{RUN_DESC}"
RUNS_HOME = Path(HOME_DIR)
RUN_DIR = HOME_DIR / "content" / "runs" / "data-trained" / RUN_TAG
path_dict = dict(
    dt_datasets=HOME_DIR / "content" / "datasets" / "dt",
    trained_models=HOME_DIR / "content" / "trained_models" / "task-trained" / prefix,
)


def trial_function(trial):
    return trial.experiment_tag


# -------------------Main Function----------------------------------
def main(
    run_tag_in: str,
    path_dict: dict,
    config_dict: dict,
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
            train, run_tag=run_tag_in, config_dict=config_dict, path_dict=path_dict
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
        config_dict=config_dict,
        path_dict=path_dict,
    )
