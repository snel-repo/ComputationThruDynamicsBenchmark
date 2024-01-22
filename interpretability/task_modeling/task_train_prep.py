import logging
import os
import pickle
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from gymnasium import Env

from interpretability.task_modeling.simulator.neural_simulator import (
    NeuralDataSimulator,
)
from utils import flatten

log = logging.getLogger(__name__)
SAVE_PATH = (
    "/home/csverst/Github/InterpretabilityBenchmark/trained_models/task-trained/"
)


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
    else:
        pl.seed_everything(0, workers=True)

    # Order of operations:
    # 1. Instantiate environment
    # 2. Instantiate model
    #    - init_model with the correct input and output sizes
    # 3. Instantiate task-wrapper
    #    - Set wrapper environment and model
    # 4. Instantiate datamodule
    #    - Set datamodule environment
    # 5. Instantiate simulator
    # 6. Instantiate callbacks
    # 7. Instantiate loggers
    # 8. Instantiate trainer
    # 9. Train model

    # --------------------------Instantiate environment----------------------------
    log.info("Instantiating environment")
    task_env: Env = hydra.utils.instantiate(config_all["task_env"], _convert_="all")

    # ------------------------------Instantiate model--------------------------------
    log.info(f"Instantiating model <{config_all['model']._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(
        config_all["model"], _convert_="all"
    )
    n_outputs = task_env.action_space.shape[0]
    n_inputs = task_env.observation_space.shape[0] + task_env.goal_space.shape[0]
    model.init_model(n_inputs, n_outputs)

    # -----------------------------Instantiate task-wrapper----------------------------
    log.info(f"Instantiating task-wrapper <{config_all['task_wrapper']._target_}")
    task_wrapper: pl.LightningModule = hydra.utils.instantiate(
        config_all["task_wrapper"], _convert_="all"
    )
    task_wrapper.set_environment(task_env)
    task_wrapper.set_model(model)

    # --------------------------Instantiate datamodule----------------------------
    log.info("Instantiating datamodule")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config_all["datamodule"], _convert_="all"
    )
    datamodule.set_environment(task_env)

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
    print(len(os.sched_getaffinity(0)))
    # -----------------------------Train model---------------------------
    log.info("Training model")
    trainer.fit(model=task_wrapper, datamodule=datamodule)

    # Save the model, datamodule, and simulator to the directory
    log.info("Saving model, datamodule, and simulator")
    dir_path = SAVE_PATH + run_tag
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    path1 = SAVE_PATH + run_tag + "/model.pkl"
    with open(path1, "wb") as f:
        pickle.dump(task_wrapper, f)

    path2 = SAVE_PATH + run_tag + "/datamodule.pkl"
    with open(path2, "wb") as f:
        pickle.dump(datamodule, f)

    simulator.simulate_neural_data(task_wrapper, datamodule, run_tag, seed=0)
    path3 = SAVE_PATH + run_tag + "/simulator.pkl"
    with open(path3, "wb") as f:
        pickle.dump(simulator, f)
