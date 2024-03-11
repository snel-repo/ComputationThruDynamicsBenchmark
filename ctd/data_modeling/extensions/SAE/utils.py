import json
import logging
import os
import sys

import h5py
import hydra
import matplotlib
import numpy as np
import pandas as pd
import requests
import torch
import yaml

logger = logging.getLogger(__name__)


def softplusActivation(module, input):
    return module.log(1 + module.exp(input))


def tanhActivation(module, input):
    return (module.tanh(input) + 1) / 2


def sigmoidActivation(module, input):
    return 1 / (1 + module.exp(-1 * input))


def apply_data_warp(data):
    warp_functions = [tanhActivation, sigmoidActivation, softplusActivation]
    firingMax = [2, 4, 6, 8]
    numDims = data.shape[1]

    a = np.array(1)
    dataGen = type(a) == type(data)
    if dataGen:
        module = np
    else:
        module = torch

    for i in range(numDims):

        j = np.mod(i, len(warp_functions) * len(firingMax))
        # print(f'Max firing {firingMax[np.mod(j, len(firingMax))]}
        # warp {warp_functions[int(np.floor((j)/(len(warp_functions)+1)))]}')
        data[:, i] = firingMax[np.mod(j, len(firingMax))] * warp_functions[
            int(np.floor((j) / (len(warp_functions) + 1)))
        ](module, data[:, i])

    return data


def apply_data_warp_sigmoid(data):
    warp_functions = [sigmoidActivation, sigmoidActivation, sigmoidActivation]
    firingMax = [2, 2, 2, 2]
    numDims = data.shape[1]

    a = np.array(1)
    dataGen = type(a) == type(data)
    if dataGen:
        module = np
    else:
        module = torch

    for i in range(numDims):

        j = np.mod(i, len(warp_functions) * len(firingMax))
        # print(f'Max firing {firingMax[np.mod(j, len(firingMax))]}
        # warp {warp_functions[int(np.floor((j)/(len(warp_functions)+1)))]}')
        data[:, i] = firingMax[np.mod(j, len(firingMax))] * warp_functions[
            int(np.floor((j) / (len(warp_functions) + 1)))
        ](module, data[:, i])

    return data


def make_data_tag(dm_cfg):
    obs_dim = "" if "obs_dim" not in dm_cfg else dm_cfg.obs_dim
    obs_noise = "" if "obs_noise" not in dm_cfg else dm_cfg.obs_noise
    if "obs_noise_params" in dm_cfg:
        obs_noise_params = ",".join(
            [f"{k}={v}" for k, v in dm_cfg.obs_noise_params.items()]
        )
    else:
        obs_noise_params = ""
    data_tag = (
        f"{dm_cfg.system}{obs_dim}_"
        f"{dm_cfg.n_samples}S_"
        f"{dm_cfg.n_timesteps}T_"
        f"{dm_cfg.pts_per_period}P_"
        f"{dm_cfg.seed}seed"
    )
    if obs_noise:
        data_tag += f"_{obs_noise}{obs_noise_params}"
    return data_tag


def make_data_tag_multi_system(dm_cfg):
    if "obs_noise_params" in dm_cfg:
        obs_noise_params = ",".join(
            [f"{k}={v}" for k, v in dm_cfg.obs_noise_params.items()]
        )
    else:
        obs_noise_params = ""
    data_tag = (
        "MultiSystem_"
        f"{dm_cfg.n_samples}S_"
        f"{dm_cfg.n_timesteps}T_"
        f"{dm_cfg.pts_per_period}P_"
        f"{dm_cfg.seed}seed_"
        f"{dm_cfg.obs_noise}{obs_noise_params}"
    )
    return data_tag


def flatten(dictionary, level=[]):
    """Flattens a dictionary by placing '.' between levels.
    This function flattens a hierarchical dictionary by placing '.'
    between keys at various levels to create a single key for each
    value. It is used internally for converting the configuration
    dictionary to more convenient formats. Implementation was
    inspired by `this StackOverflow post
    <https://stackoverflow.com/questions/6037503/python-unflatten-dict>`_.
    Parameters
    ----------
    dictionary : dict
        The hierarchical dictionary to be flattened.
    level : str, optional
        The string to append to the beginning of this dictionary,
        enabling recursive calls. By default, an empty string.
    Returns
    -------
    dict
        The flattened dictionary.
    See Also
    --------
    lfads_tf2.utils.unflatten : Performs the opposite of this operation.
    """

    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten(val, level + [key]))
        else:
            tmp_dict[".".join(level + [key])] = val
    return tmp_dict


def send_completion_message(run_name, server):
    url = """https://hooks.slack.com/services/T3H0891U6/
    B03F69NEG4A/ishLtKN1050EQM60k19eAq6r"""
    title = "Great work! You're a real scientist now!"
    message = f"""*{os.path.split(run_name)}{chr(10)}* has encountered an error on
                {chr(10)} *{server}* {chr(10)}But it's a good error!"""
    slack_data = {
        "username": "ClusterBot",
        "icon_emoji": ":matrix:",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": title,
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message,
                },
            },
        ],
    }

    byte_length = str(sys.getsizeof(slack_data))
    headers = {"Content-Type": "application/json", "Content-Length": byte_length}
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)


def send_error_message(run_name, server):
    url = """https://hooks.slack.com/services/
    T3H0891U6/B03F69NEG4A/ishLtKN1050EQM60k19eAq6r"""
    title = "Get to work you lazy bum!"
    message = f"""*{os.path.split(run_name)}{chr(10)}*
    has encountered an error on{chr(10)} *{server}* {chr(10)}Bad job!"""
    slack_data = {
        "username": "ClusterBot",
        "icon_emoji": ":matrix:",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": title,
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message,
                },
            },
        ],
    }

    byte_length = str(sys.getsizeof(slack_data))
    headers = {"Content-Type": "application/json", "Content-Length": byte_length}
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)


matplotlib.rcParams.update(
    {
        "font.size": 8,
        "pdf.use14corefonts": True,
        "axes.unicode_minus": False,
    }
)


def get_results(multirun_dir):
    dirs = os.listdir(multirun_dir)
    sub_dir = dirs[1]
    multirun_dir = multirun_dir / sub_dir
    run_folders = multirun_dir.glob("*seed=*")

    def get_result(dirpath):
        try:
            # Load the metrics
            metrics = pd.read_csv(dirpath / "metrics.csv")
            # Get last 500 epochs of validation metrics and remove "valid" from name
            metrics = metrics[metrics.epoch > metrics.epoch.max() - 500]
            metrics = metrics[[col for col in metrics if "valid" in col]].dropna()
            metrics = metrics.rename(
                {col: col.replace("valid/", "") for col in metrics}, axis=1
            )
            # Compute medians and return as a dictionary
            metrics = metrics.median().to_dict()
            # Load the hyperparameters
            with open(dirpath / "params.json", "r") as file:
                hps = flatten(json.load(file))
            with open(dirpath / "hparams.yaml", "r") as file:
                hps = yaml.safe_load(file)
            return {**metrics, **hps}
        except FileNotFoundError:
            return {}

    results = pd.DataFrame([get_result(dirpath) for dirpath in run_folders])
    dirpaths = os.listdir(multirun_dir)
    return results, dirpaths, sub_dir


def instantiate_models(config_default, results_list):
    results_keys = results_list.keys()
    for key in results_keys:
        if key in config_default.keys():
            if (
                type(results_list[key]) == np.float64
                or type(results_list[key]) == np.int64
            ):
                if np.mod(results_list[key], 1) == 0:
                    config_default[key] = int(results_list[key])
                else:
                    config_default[key] = float(results_list[key])
            elif type(results_list[key]) == np.bool_:
                config_default[key] = bool(results_list[key])
            else:
                config_default[key] = results_list[key]
    model = hydra.utils.instantiate(config_default, _convert_="all")
    return model


def save_to_h5(data_dict, save_path, overwrite=False, dlen=32):
    """Function that saves dict as .h5 file while preserving
    nested dict structure

    Parameters
    ----------
    data_dict : dict
        Dict containing data to be saved in HDF5 format
    save_path : str
        Path to location where data should be saved
    overwrite : bool, optional
        Whether to overwrite duplicate data found
        at `save_path` if file already exists, by
        default False
    dlen : int, optional
        Byte length of data format to save numerical data,
        by default 32.
    """
    h5file = h5py.File(save_path, "a")
    good, dup_list = _check_h5_r(data_dict, h5file, overwrite)
    if good:
        if len(dup_list) > 0:
            logger.warning(f"{dup_list} already found in {save_path}. Overwriting...")
        _save_h5_r(data_dict, h5file, dlen)
        logger.info(f"Saved data to {save_path}")
    else:
        logger.warning(
            f"{dup_list} already found in {save_path}. Save to file canceled. "
            "Please set `overwrite=True` or specify a different file path."
        )
    h5file.close()


def _check_h5_r(data_dict, h5obj, overwrite):
    """Recursive helper function that finds duplicate keys
          and deletes them if `overwrite == True`

    Parameters
    ----------
    data_dict : dict
        Dict containing data to be saved in HDF5 format
    h5obj : h5py.File or h5py.Group
        h5py object to check for duplicates
    overwrite : bool, optional
        Whether to overwrite duplicate data found
        at `save_path` if file already exists, by
        default False

    Returns
    -------
    tuple
        Tuple containing bool of whether `h5obj` passes
        checks and list of duplicate keys found
    """
    dup_list = []
    good = True
    for key in data_dict.keys():
        if key in h5obj.keys():
            if isinstance(h5obj[key], h5py.Group) and isinstance(data_dict[key], dict):
                rgood, rdup_list = _check_h5_r(data_dict[key], h5obj[key], overwrite)
                good = good and rgood
                dup_list += list(zip([key] * len(rdup_list), rdup_list))
            else:
                dup_list.append(key)
                if overwrite:
                    del h5obj[key]
                else:
                    good = False
    return good, dup_list


def _save_h5_r(data_dict, h5obj, dlen):
    """Recursive function that adds all the items in a dict
          to an h5py.File or h5py.Group object

    Parameters
    ----------
    data_dict : dict
        Dict containing data to be saved in HDF5 format
    h5obj : h5py.File or h5py.Group
        h5py object to save data to
    dlen : int, optional
        Byte length of data format to save numerical data,
        by default 32.
    """
    for key, val in data_dict.items():
        if isinstance(val, dict):
            h5group = h5obj[key] if key in h5obj.keys() else h5obj.create_group(key)
            _save_h5_r(val, h5group, dlen)
        else:
            if val.dtype == "object":
                sub_dtype = (
                    f"float{dlen}"
                    if val[0].dtype == np.float
                    else f"int{dlen}"
                    if val[0].dtype == np.int
                    else val[0].dtype
                )
                dtype = h5py.vlen_dtype(sub_dtype)
            else:
                dtype = (
                    f"float{dlen}"
                    if val.dtype == np.float
                    else f"int{dlen}"
                    if val.dtype == np.int
                    else val.dtype
                )
            h5obj.create_dataset(key, data=val, dtype=dtype)


def h5_to_dict(h5obj):
    """Recursive function that reads HDF5 file to dict

    Parameters
    ----------
    h5obj : h5py.File or h5py.Group
        File or group object to load into a dict

    Returns
    -------
    dict of np.array
        Dict mapping h5obj keys to arrays
        or other dicts
    """
    data_dict = {}
    for key in h5obj.keys():
        if isinstance(h5obj[key], h5py.Group):
            data_dict[key] = h5_to_dict(h5obj[key])
        else:
            data_dict[key] = h5obj[key][()]
    return data_dict


def combine_h5(file_paths, save_path=None):
    """Function that takes multiple .h5 files and combines them into one.
    May be particularly useful for combining MC_Maze scaling results for submission

    Parameters
    ----------
    file_paths : list
        List of paths to h5 files to combine
    save_path : str, optional
        Path to save combined results to. By
        default None saves to first path in
        `file_paths`.
    """
    assert len(file_paths) > 1, "Must provide at least 2 files to combine"
    if save_path is None:
        save_path = file_paths[0]
    for fpath in file_paths:
        if fpath == save_path:
            continue
        with h5py.File(fpath, "r") as h5file:
            data_dict = h5_to_dict(h5file)
        save_to_h5(data_dict, save_path)
