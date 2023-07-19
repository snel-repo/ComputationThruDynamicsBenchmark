import logging
from typing import List

import hydra
import pytorch_lightning as pl
import torch
from interpretability.task_modeling.simulator.neural_simulator import NeuralDataSimulator
from gymnasium import Env

from utils import flatten

log = logging.getLogger(__name__)


def train(
    overrides: dict = {},
    path_dict: dict = {},
    run_tag: str = "",
):
    compose_list = path_dict.keys()
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
        with hydra.initialize(config_path=str(path_dict[field].parent), job_name=field):
            if field in overrides_flat.keys():
                config_all[field] = hydra.compose(
                    config_name=path_dict[field].name, overrides=overrides_flat[field]
                )
            else:
                config_all[field] = hydra.compose(config_name=path_dict[field].name)

    # Set seed for pytorch, numpy, and python.random
    if "params" in overrides:
        pl.seed_everything(overrides["params"]["seed"], workers=True)
        if "coupled" in overrides["params"]:
            coupled = overrides["params"]["coupled"]
        else:
            coupled = False
    else:
        pl.seed_everything(0, workers=True)
        coupled = True

    # --------------------------Instantiate datamodule----------------------------
    log.info("Instantiating datamodule")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config_all["datamodule"], _convert_="all"
    )

    # ---------------------------Instantiate simulator---------------------------
    log.info("Instantiating neural data simulator")
    simulator: NeuralDataSimulator = hydra.utils.instantiate(
        config_all["simulator"], _convert_="all"
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

    # -----------------------------Instantiate task-wrapper----------------------------
    log.info(f"Instantiating task-wrapper <{config_all['task_wrapper']._target_}")
    task_wrapper: Env = hydra.utils.instantiate(
        config_all["task_wrapper"], _convert_="all"
    )                

    # ------------------------------Instantiate model--------------------------------
    log.info(f"Instantiating model <{config_all['model']._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(
        config_all["model"], _convert_="all"
    )
    task_wrapper.set_model(model)

    # -----------------------------Instantiate trainer---------------------------
    targ_string = config_all["trainer"]._target_
    log.info(f"Instantiating trainer <{targ_string}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config_all["trainer"],
        logger=logger,
        callbacks=callbacks,
        accelerator= 'auto',
        _convert_="all",
    )

    # -----------------------------Train model---------------------------
    log.info("Training model")
    trainer.fit(model=task_wrapper, datamodule=datamodule)    
    simulator.simulate_neural_data(task_wrapper, datamodule, seed = 0)