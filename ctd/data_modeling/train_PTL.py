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
    # Convert the overrides dict into a list of override strings
    overrides_list = [f"{k}={v}" for k, v in overrides.items()]

    # Generate a run_name from the overrides
    run_list = []
    for k, v in overrides.items():
        if isinstance(v, float):
            v = "{:.2E}".format(v)
        k_list = k.split(".")
        run_list.append(f"{k_list[-1]}={v}")
    run_name = "_".join(run_list)

    # Compose the configs for all components
    config_all = {}
    for field in compose_list:
        with hydra.initialize(
            config_path=str(config_dict[field].parent), job_name=field
        ):
            # Filter overrides relevant to this field
            field_prefix = f"{field}."
            field_overrides = [
                override
                for override in overrides_list
                if override.startswith(field_prefix)
            ]
            # Remove the field prefix from the overrides
            field_overrides = [
                override[len(field_prefix) :]
                if override.startswith(field_prefix)
                else override
                for override in field_overrides
            ]
            config_all[field] = hydra.compose(
                config_name=config_dict[field].name, overrides=field_overrides
            )

    # Handle special parameters
    if "params.obs_dim" in overrides:
        obs_dim = overrides["params.obs_dim"]
        config_all["datamodule"]["obs_dim"] = obs_dim
        config_all["model"]["heldin_size"] = obs_dim
        config_all["model"]["heldout_size"] = obs_dim

    if "params.lr_all" in overrides:
        lr_all = overrides["params.lr_all"]
        config_all["model"]["lr_readout"] = lr_all
        config_all["model"]["lr_encoder"] = lr_all
        config_all["model"]["lr_decoder"] = lr_all

    if "params.decay_all" in overrides:
        decay_all = overrides["params.decay_all"]
        config_all["model"]["decay_readout"] = decay_all
        config_all["model"]["decay_encoder"] = decay_all
        config_all["model"]["decay_decoder"] = decay_all
    if "params.seed" in overrides:
        seed = overrides["params.seed"]
        pl.seed_everything(seed, workers=True)
    else:
        pl.seed_everything(0, workers=True)

    # --------------------------Instantiate datamodule-------------------------------
    log.info("Instantiating datamodule")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config_all["datamodule"], _convert_="all"
    )

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
    save_path = os.path.join(save_path, run_tag, run_name)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(save_path, "model.pkl")
    datamodule_path = os.path.join(save_path, "datamodule.pkl")

    model = model.to("cpu")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(datamodule_path, "wb") as f:
        pickle.dump(datamodule, f)
