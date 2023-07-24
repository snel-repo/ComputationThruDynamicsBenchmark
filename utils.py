import json
import os
import sys

import numpy as np
import requests
import torch


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
