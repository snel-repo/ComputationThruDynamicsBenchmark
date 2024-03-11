import logging
import os
import pickle
from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
from gymnasium import Env

from ctd.task_modeling.simulator.neural_simulator import NeuralDataSimulator
from utils import flatten

log = logging.getLogger(__name__)


def train(
    overrides: dict = {},
    config_dict: dict = {},
    run_tag: str = "",
    path_dict: dict = {},
):
    # Print the current working directory
    compose_list = config_dict.keys()
    # Format the overrides so they can be used by hydra
    override_keys = overrides.keys()
    overrides_flat = {}
    subfolder = ""
    for key in override_keys:
        if type(overrides[key]) == dict:
            overrides_flat[key] = [
                f"{k}={v}" for k, v in flatten(overrides[key]).items()
            ]
            temp = [f"{k}={v}" for k, v in flatten(overrides[key]).items()]
            # join the list of strings
            subfolder += " ".join(temp)
            subfolder += " "
        else:
            overrides_flat[key] = f"{key}={overrides[key]}"
            subfolder += f"_{key}={overrides[key]}_"

    # Compose the configs for all components
    subfolder = subfolder[:-1]
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
        if "seed" in overrides["params"]:
            pl.seed_everything(overrides["params"]["seed"], workers=True)
            config_all["datamodule_task"]["seed"] = overrides["params"]["seed"]
            config_all["datamodule_sim"]["seed"] += overrides["params"]["seed"]
        else:
            pl.seed_everything(0, workers=True)
    # Set shared params for both the task and sim envs
    if "env_params" in overrides:
        for k, v in overrides["env_params"].items():
            config_all["env_task"][k] = v
            config_all["env_sim"][k] = v

    # Order of operations:
    # 1. Instantiate environments
    #    - task_env is for training
    #    - sim_env is for simulating the neural data
    # 2. Instantiate model
    #    - init_model with the correct input and output sizes
    # 3. Instantiate task-wrapper
    #    - Set wrapper environment and model
    # 4. Instantiate training datamodule
    #    - Set datamodule environment
    # 5. Instantiate simulator
    # 6. Instantiate callbacks
    # 7. Instantiate loggers
    # 8. Instantiate trainer
    # 9. Train model
    # 10. Save model + datamodule
    # 11. Instantiate sim datamodule
    # 12. Simulate neural data
    # 13. Save simulator + sim datamodule

    # -----Step 1:----------------Instantiate environments----------------------------
    log.info("Instantiating environment")
    task_env: Env = hydra.utils.instantiate(config_all["env_task"], _convert_="all")

    log.info("Instantiating environment for neural simulation")
    sim_env: Env = hydra.utils.instantiate(config_all["env_sim"], _convert_="all")

    # -----Step 2:-----------------Instantiate model--------------------------------
    log.info(f"Instantiating model <{config_all['model']._target_}")
    model: pl.LightningModule = hydra.utils.instantiate(
        config_all["model"], _convert_="all"
    )
    n_outputs = task_env.action_space.shape[0]
    n_inputs = task_env.observation_space.shape[0] + task_env.context_inputs.shape[0]
    model.init_model(n_inputs, n_outputs)

    # -----Step 3:----------------Instantiate task-wrapper----------------------------
    log.info(f"Instantiating task-wrapper <{config_all['task_wrapper']._target_}")
    task_wrapper: pl.LightningModule = hydra.utils.instantiate(
        config_all["task_wrapper"], _convert_="all"
    )
    task_wrapper.set_environment(task_env)
    task_wrapper.set_model(model)

    # -----Step 4:----------------Instantiate datamodule----------------------------
    log.info("Instantiating datamodule for training")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config_all["datamodule_task"], _convert_="all"
    )
    datamodule.set_environment(data_env=task_env)

    # -----Step 5:----------------Instantiate simulator---------------------------
    log.info("Instantiating neural data simulator")
    simulator: NeuralDataSimulator = hydra.utils.instantiate(
        config_all["simulator"], _convert_="all"
    )

    # -----Step 6:-----------------Instantiate callbacks---------------------------
    callbacks: List[pl.Callback] = []
    if "callbacks" in config_all:
        for _, cb_conf in config_all["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf, _convert_="all"))

    # -----Step 7:------------------Instantiate loggers----------------------------
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

    # ------Step 8:--------------Instantiate trainer---------------------------
    targ_string = config_all["trainer"]._target_
    log.info(f"Instantiating trainer <{targ_string}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        config_all["trainer"],
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
        _convert_="all",
    )

    # -------Step 9:-----------------Train model---------------------------
    log.info("Training model")
    trainer.fit(model=task_wrapper, datamodule=datamodule)

    # -------Step 10:----------Save model and datamodule---------------------------
    log.info("Saving model and datamodule")
    SAVE_PATH = path_dict["trained_models"] / "task-trained"

    dir_path = os.path.join(SAVE_PATH, run_tag, subfolder, "")
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    path1 = os.path.join(SAVE_PATH, run_tag, subfolder, "model.pkl")
    with open(path1, "wb") as f:
        pickle.dump(task_wrapper, f)

    path2 = os.path.join(SAVE_PATH, run_tag, subfolder, "datamodule_train.pkl")
    with open(path2, "wb") as f:
        pickle.dump(datamodule, f)

    # ------Step 11:------------Instantiate sim datamodule---------------------------
    log.info("Instantiating datamodule for neural simulation")
    sim_datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        config_all["datamodule_sim"], _convert_="all"
    )
    sim_datamodule.set_environment(
        data_env=sim_env,
        for_sim=True,
    )

    sim_datamodule.prepare_data()
    sim_datamodule.setup()

    # ------Step 12:------------Simulate neural data---------------------------
    simulator.simulate_neural_data(
        task_trained_model=task_wrapper,
        datamodule=sim_datamodule,
        run_tag=run_tag,
        dataset_path=path_dict["dt_datasets"],
        subfolder=subfolder,
        seed=0,
    )

    # ------Step 13:--------Save simulator and sim datamodule------------
    path3 = os.path.join(SAVE_PATH, run_tag, subfolder, "simulator.pkl")
    with open(path3, "wb") as f:
        pickle.dump(simulator, f)

    path3 = os.path.join(SAVE_PATH, run_tag, subfolder, "datamodule_sim.pkl")
    with open(path3, "wb") as f:
        pickle.dump(sim_datamodule, f)
