import logging
import os
import warnings
from glob import glob
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import call, instantiate
from omegaconf import OmegaConf, open_dict
from ray import tune
from interpretability.data_modeling.lfads_torch.data_trained_wrapper.data_trained_wrapper import DataTrainWrapper

from .utils import flatten

OmegaConf.register_new_resolver("relpath", lambda p: Path(__file__).parent / ".." / p)


def run_model(
    project_str: str = None,
    overrides: dict = {},
    checkpoint_dir: str = None,
    config_path: str = "../configs/single.yaml",
    do_train: bool = True,
    do_posterior_sample: bool = True,
):
    """Adds overrides to the default config, instantiates all PyTorch Lightning
    objects from config, and runs the training pipeline.
    """

    # Compose the train config with properly formatted overrides
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
    with hydra.initialize(
        config_path=str(config_path.parent),
        job_name="run_model",
    ):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)

    # Avoid flooding the console with output during multi-model runs
    if config.ignore_warnings:
        logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
        warnings.filterwarnings("ignore")

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed") is not None:
        pl.seed_everything(config.seed, workers=True)
    # Check if there's a WandB logger in the config
    if "wandb_logger" in config.logger:
        config.logger.wandb_logger["group"] = project_str

    # Instantiate `LightningDataModule` and `LightningModule`
    datamodule = instantiate(config.datamodule, _convert_="all")
    model = instantiate(config.model)

    # If `checkpoint_dir` is passed, find the most recent checkpoint in the directory
    if checkpoint_dir:
        ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
        ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)

    # If both ray.tune and wandb are being used, ensure that loggers use same name
    if "single" not in str(config_path) and "wandb_logger" in config.logger:
        with open_dict(config):
            config.logger.wandb_logger.name = tune.get_trial_name()
            config.logger.wandb_logger.id = tune.get_trial_name()
    # Instantiate the pytorch_lightning `Trainer` and its callbacks and loggers
    trainer = instantiate(
        config.trainer,
        callbacks=[instantiate(c) for c in config.callbacks.values()],
        logger=[instantiate(lg) for lg in config.logger.values()],
        accelerator='auto',
    )
    data_trained_wrapper = DataTrainWrapper(model, datamodule, trainer)
    basePath = '/home/csverst/Github/InterpretabilityBenchmark/model_saves/data_trained/'
    path1 = basePath + project_str + '.pkl'
    data_trained_wrapper.train_model()

    data_trained_wrapper.save_wrapper(path1)



