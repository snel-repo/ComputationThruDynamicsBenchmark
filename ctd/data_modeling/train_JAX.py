import logging
import os
import pickle
from pathlib import Path

import dotenv
import hydra
import pytorch_lightning as pl

from ctd.data_modeling.extensions.SAE.utils import flatten

dotenv.load_dotenv(override=True)

log = logging.getLogger(__name__)


def train_JAX(
    overrides: dict = {},
    config_dict: dict = {},
    path_dict: str = "",
    run_tag: str = "",
):
    compose_list = config_dict.keys()
    # Format the overrides so they can be used by hydra
    override_keys = overrides.keys()
    overrides_flat = {}
    for key in override_keys:
        if type(overrides[key]) == dict:
            overrides_flat[key] = [
                f"{k}={v}" for k, v in flatten(overrides[key]).items()
            ]
        else:
            overrides_flat[key] = f"{key}={overrides[key]}"

    # Compose the configs for all components
    config_all = {}
    for field in compose_list:
        with hydra.initialize(
            config_path=str(config_dict[field].parent), job_name=field
        ):
            if field in overrides_flat.keys():
                config_all[field] = hydra.compose(
                    config_name=config_dict[field].name, overrides=overrides_flat[field]
                )
            else:
                config_all[field] = hydra.compose(config_name=config_dict[field].name)

    # Set seed for pytorch, numpy, and python.random
    if "params" in overrides:
        pl.seed_everything(overrides["params"]["seed"], workers=True)
        if "seed" in config_all["datamodule"]:
            config_all["datamodule"]["seed"] = overrides["params"]["seed"]
    else:
        pl.seed_everything(0, workers=True)

    # --------------------------Instantiate datamodule-------------------------------
    log.info("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(config_all["datamodule"], _convert_="all")

    # ------------------------------Instantiate model--------------------------------
    log.info(f"Instantiating model <{config_all['model']._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(
        config_all["model"], _convert_="all"
    )
    # -----------------------------Instantiate trainer---------------------------
    targ_string = config_all["trainer"]._target_
    log.info(f"Instantiating trainer <{targ_string}>")
    trainer = hydra.utils.instantiate(
        config_all["trainer"],
        _convert_="all",
    )
    trainer.set_model_and_data(model, datamodule)

    # -----------------------------Train the model-------------------------------
    log.info("Starting training")
    trainer.train()

    # -----------------------------Save the model-------------------------------
    # Save the model, datamodule, and simulator to the directory
    save_path = path_dict["trained_models"]
    save_path = os.path.join(save_path, run_tag)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(save_path, "model.pkl")
    datamodule_path = os.path.join(save_path, "datamodule.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(datamodule_path, "wb") as f:
        pickle.dump(datamodule, f)
