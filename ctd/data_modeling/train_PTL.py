import logging
import os
import pickle
from pathlib import Path
from typing import List

import dotenv
import hydra
import pytorch_lightning as pl

from ctd.data_modeling.extensions.SAE.utils import flatten

dotenv.load_dotenv(override=True)

log = logging.getLogger(__name__)


def train_PTL(
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
        if "obs_dim" in overrides["params"]:
            config_all["datamodule"]["obs_dim"] = overrides["params"]["obs_dim"]
            config_all["model"]["heldin_size"] = overrides["params"]["obs_dim"]
            config_all["model"]["heldout_size"] = overrides["params"]["obs_dim"]
        if "lr_all" in overrides["params"]:
            config_all["model"]["lr_readout"] = overrides["params"]["lr_all"]
            config_all["model"]["lr_encoder"] = overrides["params"]["lr_all"]
            config_all["model"]["lr_decoder"] = overrides["params"]["lr_all"]
        if "decay_all" in overrides["params"]:
            config_all["model"]["decay_readout"] = overrides["params"]["decay_all"]
            config_all["model"]["decay_encoder"] = overrides["params"]["decay_all"]
            config_all["model"]["decay_decoder"] = overrides["params"]["decay_all"]

    else:
        pl.seed_everything(0, workers=True)

    # --------------------------Instantiate datamodule-------------------------------
    log.info("Instantiating datamodule")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config_all["datamodule"], _convert_="all"
    )
    tt_name = datamodule.name[:-3]

    # ---------------------------Instantiate callbacks---------------------------
    callbacks: List[pl.Callback] = []
    if "callbacks" in config_all:
        for _, cb_conf in config_all["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="all"))

    # -----------------------------Instantiate loggers----------------------------
    flat_list = flatten(overrides).items()
    run_list = []
    for k, v in flat_list:
        if type(v) == float:
            v = "{:.2E}".format(v)
        k_list = k.split(".")
        run_list.append(f"{k_list[-1]}={v}")
    run_name = "_".join(run_list)

    logger: List[pl.LightningLoggerBase] = []
    if "loggers" in config_all:
        for _, lg_conf in config_all["loggers"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                if lg_conf._target_ == "pytorch_lightning.loggers.WandbLogger":
                    lg_conf["group"] = run_tag
                    lg_conf["name"] = run_name
                logger.append(hydra.utils.instantiate(lg_conf))

    # ------------------------------Instantiate model--------------------------------
    log.info(f"Instantiating model <{config_all['model']._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(
        config_all["model"], _convert_="all"
    )
    # -----------------------------Instantiate trainer---------------------------
    targ_string = config_all["trainer"]._target_
    log.info(f"Instantiating trainer <{targ_string}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config_all["trainer"],
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        _convert_="all",
    )
    # -----------------------------Train the model-------------------------------
    log.info("Starting training")
    trainer.fit(model=model, datamodule=datamodule)

    # -----------------------------Save the model-------------------------------
    # Save the model, datamodule, and simulator to the directory
    save_path = path_dict["trained_models"]
    save_path = os.path.join(save_path, tt_name, run_tag)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(save_path, "model.pkl")
    datamodule_path = os.path.join(save_path, "datamodule.pkl")

    model = model.to("cpu")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(datamodule_path, "wb") as f:
        pickle.dump(datamodule, f)
