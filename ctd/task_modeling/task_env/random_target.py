# Class to generate training data for task-trained RNN that does 3 bit memory task
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from motornet.environment import Environment
from numpy import ndarray
from torch._tensor import Tensor

from ctd.task_modeling.task_env.loss_func import RandomTargetLoss


def causal_smooth_vector(vector, window_size):
    smoothed_vector = np.zeros_like(vector)
    for i in range(len(vector)):
        if i < window_size:
            smoothed_vector[i] = np.mean(vector[: i + 1])
        else:
            smoothed_vector[i] = np.mean(vector[i - window_size + 1 : i + 1])
    return smoothed_vector


class RandomTarget(Environment):
    """A reach to a random target from a random starting position with a delay period.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass.
        This is the network that will perform the task.

        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative
        contribution to the default muscle L2 loss.

        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(
        self,
        pos_weight=1.0,
        act_weight=1.0,
        context_input_noise=0.05,
        bump_mag_low=5,
        bump_mag_high=10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.obs_noise[: self.skeleton.space_dim] = [
            0.0
        ] * self.skeleton.space_dim  # target info is noiseless

        self.dataset_name = "RandomTarget"
        self.n_timesteps = np.floor(self.max_ep_duration / self.effector.dt).astype(int)
        self.input_labels = ["TargetX", "TargetY", "GoCue"]
        self.output_labels = ["Pec", "Delt", "Brad", "TriLong", "Biceps", "TriLat"]
        self.context_inputs = spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32)
        self.coupled_env = True
        self.state_label = "fingertip"

        pos_weight = pos_weight
        act_weight = act_weight

        self.bump_mag_low = bump_mag_low
        self.bump_mag_high = bump_mag_high

        self.context_input_noise = context_input_noise

        self.loss_func = RandomTargetLoss(
            position_loss=nn.MSELoss(), pos_weight=pos_weight, act_weight=act_weight
        )

    def generate_dataset(self, n_samples):
        # Make target circular, change loss function to be pinned at zero
        initial_state = []
        inputs = np.zeros((n_samples, self.n_timesteps, 3))
        goal_list = []
        go_cue_list = []
        target_on_list = []
        catch_trials = []
        ext_inputs_list = []
        if self.dataset_name == "MoveBump":
            bump_cond = True
        else:
            bump_cond = False

        for i in range(n_samples):
            if not bump_cond:
                catch_trial = np.random.choice([0, 1], p=[0.8, 0.2])
                bump_trial = np.random.choice([0, 1], p=[0.5, 0.5])
                move_bump_trial = np.random.choice([0, 1], p=[0.5, 0.5])

                target_on = np.random.randint(10, 30)
                go_cue = np.random.randint(target_on + 5, self.n_timesteps)
                if move_bump_trial:
                    bump_time = np.random.randint(go_cue, go_cue + 40)
                else:
                    bump_time = np.random.randint(0, self.n_timesteps - 30)
                bump_duration = np.random.randint(15, 30)
                bump_theta = np.random.uniform(0, 2 * np.pi)
                bump_mag = np.random.uniform(self.bump_mag_low, self.bump_mag_high)
                info = self.generate_trial_info()

            else:
                catch_trial = False
                bump_trial = True
                move_bump_trial = True
                target_on = 10
                go_cue = 50
                bump_time = 60
                bump_duration = 10
                bump_theta = np.random.choice([0])
                if np.random.random() < 0.5:
                    bump_mag = np.random.uniform(-20, -5)
                else:
                    bump_mag = np.random.uniform(5, 20)
                info = self.generate_bump_trial_info()

            target_on_list.append(target_on)

            initial_state.append(info["ics_joint"])
            initial_state_xy = info["ics_xy"]

            env_inputs_mat = np.zeros((self.n_timesteps, 2))
            if bump_trial:
                bump_end = min(bump_time + bump_duration, self.n_timesteps)
                env_inputs_mat[bump_time:bump_end, :] = np.array(
                    [bump_mag * np.cos(bump_theta), bump_mag * np.sin(bump_theta)]
                )

            goal_matrix = torch.zeros((self.n_timesteps, self.skeleton.space_dim))
            if catch_trial:
                go_cue = -1
                goal_matrix[:, :] = initial_state_xy
            else:
                inputs[i, go_cue:, 2] = 1
                # Smoothly interpolate to the goal position
                inputs[i, :, 2] = causal_smooth_vector(inputs[i, :, 2], 5)

                goal_matrix[:go_cue, :] = initial_state_xy
                goal_matrix[go_cue:, :] = torch.squeeze(info["goal"])

            go_cue_list.append(go_cue)
            inputs[i, target_on:go_cue, 0:2] = info["goal"]

            catch_trials.append(catch_trial)
            goal_list.append(goal_matrix)
            ext_inputs_list.append(env_inputs_mat)

        go_cue_list = np.array(go_cue_list)
        target_on_list = np.array(target_on_list)
        env_inputs = np.stack(ext_inputs_list, axis=0)
        extra = np.stack((target_on_list, go_cue_list), axis=1)
        conds = np.array(catch_trials)
        true_inputs = inputs.copy()
        inputs = inputs + np.random.normal(
            0, self.context_input_noise, (n_samples, self.n_timesteps, 3)
        )
        initial_state = torch.stack(initial_state, axis=0)
        goal_list = torch.stack(goal_list, axis=0)
        dataset_dict = {
            "ics": initial_state,
            "inputs": inputs,
            "inputs_to_env": env_inputs,
            "targets": goal_list,
            "conds": conds,
            "extra": extra,
            "true_inputs": true_inputs,
        }
        extra_dict = {}
        return dataset_dict, extra_dict

    def generate_trial_info(self):
        """
        Generate a trial for the task.
        This is a reach to a random target from a random starting
        position with a delay period.
        """
        sho_limit = [0, 135]  # mechanical constraints - used to be -90 180
        elb_limit = [0, 155]
        sho_ang = np.deg2rad(np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30))
        elb_ang = np.deg2rad(np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30))

        sho_ang_targ = np.deg2rad(
            np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30)
        )
        elb_ang_targ = np.deg2rad(
            np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30)
        )

        angs = torch.tensor(np.array([sho_ang, elb_ang, 0, 0]))
        ang_targ = torch.tensor(np.array([sho_ang_targ, elb_ang_targ, 0, 0]))

        target_pos = self.joint2cartesian(
            torch.tensor(ang_targ, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]

        start_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]

        info = dict(
            ics_joint=angs,
            ics_xy=start_xy,
            goal=target_pos,
        )
        return info

    def generate_bump_trial_info(self):
        """
        Generate a co trial for the task.
        This is a reach to a random target from a random starting
        position with a delay period.
        """

        sho_ang = np.deg2rad(100)
        elb_ang = np.deg2rad(120)

        angs = torch.tensor(np.array([sho_ang, elb_ang, 0, 0]))

        start_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]
        target_pos = start_xy.clone()
        target_pos[0, 1] = target_pos[0, 1] + 0.4

        info = dict(
            ics_joint=angs,
            ics_xy=start_xy,
            goal=target_pos,
        )
        return info

    def set_goal(
        self,
        goal: torch.Tensor,
    ):
        """
        Sets the goal of the task. This is the target position of the effector.
        """
        self.goal = goal

    def get_obs(self, action=None, deterministic: bool = False) -> Tensor | ndarray:
        self.update_obs_buffer(action=action)

        obs_as_list = [
            self.obs_buffer["vision"][0],
            self.obs_buffer["proprioception"][0],
        ] + self.obs_buffer["action"][: self.action_frame_stacking]

        obs = torch.cat(obs_as_list, dim=-1)

        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)

        return obs if self.differentiable else self.detach(obs)

    def reset(
        self,
        batch_size: int = 1,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:

        """
        Uses the :meth:`Environment.reset()` method of the parent class
        :class:`Environment` that can be overwritten to change the returned data.
        Here the goals (`i.e.`, the targets) are drawn from a random uniform
        distribution across the full joint space.
        """
        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])
        # Make self.obs_noise a list
        self._set_generator(seed=seed)
        # if ic_state is in options, use that
        if options is not None and "deterministic" in options.keys():
            deterministic = options["deterministic"]
        else:
            deterministic = False
        if options is not None and "ic_state" in options.keys():
            ic_state_shape = np.shape(self.detach(options["ic_state"]))
            if ic_state_shape[0] > 1:
                batch_size = ic_state_shape[0]
            ic_state = options["ic_state"]
        else:
            ic_state = self.q_init

        if options is not None and "target_state" in options.keys():
            self.goal = options["target_state"]
        else:
            sho_ang = np.random.uniform(
                sho_limit[0] + 20, sho_limit[1] - 20, size=batch_size
            )
            elb_ang = np.random.uniform(
                elb_limit[0] + 20, elb_limit[1] - 20, size=batch_size
            )
            sho_vel = np.zeros(batch_size)
            elb_vel = np.zeros(batch_size)
            angs = np.stack((sho_ang, elb_ang, sho_vel, elb_vel), axis=1)
            self.goal = self.joint2cartesian(
                torch.tensor(angs, dtype=torch.float32, device=self.device)
            ).chunk(2, dim=-1)[0]

        options = {
            "batch_size": batch_size,
            "joint_state": ic_state,
        }
        self.effector.reset(options=options)

        self.elapsed = 0.0

        action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.goal if self.differentiable else self.detach(self.goal),
        }
        return obs, info


class RandomTargetAligned(Environment):
    """A reach to a random target from a random starting position with a delay period.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass.
        This is the network that will perform the task.

        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative
        contribution to the default muscle L2 loss.

        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(
        self,
        pos_weight=1.0,
        act_weight=1.0,
        context_input_noise=0.05,
        bump_mag_low=5,
        bump_mag_high=10,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.obs_noise[: self.skeleton.space_dim] = [
            0.0
        ] * self.skeleton.space_dim  # target info is noiseless

        self.dataset_name = "RandomTargetAligned"
        self.n_timesteps = np.floor(self.max_ep_duration / self.effector.dt).astype(int)
        self.input_labels = ["TargetX", "TargetY", "GoCue"]
        self.output_labels = ["Pec", "Delt", "Brad", "TriLong", "Biceps", "TriLat"]
        self.context_inputs = spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32)
        self.coupled_env = True
        self.state_label = "fingertip"

        pos_weight = pos_weight
        act_weight = act_weight

        self.bump_mag_low = bump_mag_low
        self.bump_mag_high = bump_mag_high

        self.context_input_noise = context_input_noise

        self.loss_func = RandomTargetLoss(
            position_loss=nn.MSELoss(), pos_weight=pos_weight, act_weight=act_weight
        )

    def generate_dataset(self, n_samples):
        # Make target circular, change loss function to be pinned at zero
        initial_state = []
        inputs = np.zeros((n_samples, self.n_timesteps, 3))
        goal_list = []
        go_cue_list = []
        target_on_list = []
        catch_trials = []
        ext_inputs_list = []

        for i in range(n_samples):
            catch_trial = 0
            bump_trial = np.random.choice([0, 1], p=[0.5, 0.5])
            move_bump_trial = np.random.choice([0, 1], p=[0.5, 0.5])

            target_on = 30
            go_cue = 70
            if move_bump_trial:
                bump_time = np.random.randint(go_cue, go_cue + 40)
            else:
                bump_time = np.random.randint(0, go_cue - 3)
            bump_duration = np.random.randint(15, 30)
            bump_theta = np.random.uniform(0, 2 * np.pi)
            bump_mag = np.random.uniform(self.bump_mag_low, self.bump_mag_high)
            info = self.generate_trial_info()

            target_on_list.append(target_on)

            initial_state.append(info["ics_joint"])
            initial_state_xy = info["ics_xy"]

            env_inputs_mat = np.zeros((self.n_timesteps, 2))
            if bump_trial:
                bump_end = min(bump_time + bump_duration, self.n_timesteps)
                env_inputs_mat[bump_time:bump_end, :] = np.array(
                    [bump_mag * np.cos(bump_theta), bump_mag * np.sin(bump_theta)]
                )

            goal_matrix = torch.zeros((self.n_timesteps, self.skeleton.space_dim))

            inputs[i, go_cue:, 2] = 1
            # Smoothly interpolate to the goal position
            inputs[i, :, 2] = causal_smooth_vector(inputs[i, :, 2], 5)

            goal_matrix[:go_cue, :] = initial_state_xy
            goal_matrix[go_cue:, :] = torch.squeeze(info["goal"])

            go_cue_list.append(go_cue)
            inputs[i, target_on:go_cue, 0:2] = info["goal"]

            catch_trials.append(catch_trial)
            goal_list.append(goal_matrix)
            ext_inputs_list.append(env_inputs_mat)

        go_cue_list = np.array(go_cue_list)
        target_on_list = np.array(target_on_list)
        env_inputs = np.stack(ext_inputs_list, axis=0)
        extra = np.stack((target_on_list, go_cue_list), axis=1)
        conds = np.array(catch_trials)
        true_inputs = inputs.copy()
        inputs = inputs + np.random.normal(
            0, self.context_input_noise, (n_samples, self.n_timesteps, 3)
        )
        initial_state = torch.stack(initial_state, axis=0)
        goal_list = torch.stack(goal_list, axis=0)
        dataset_dict = {
            "ics": initial_state,
            "inputs": inputs,
            "inputs_to_env": env_inputs,
            "targets": goal_list,
            "conds": conds,
            "extra": extra,
            "true_inputs": true_inputs,
        }
        extra_dict = {}
        return dataset_dict, extra_dict

    def generate_trial_info(self):
        """
        Generate a trial for the task.
        This is a reach to a random target from a random starting
        position with a delay period.
        """
        sho_limit = [0, 135]  # mechanical constraints - used to be -90 180
        elb_limit = [0, 155]
        sho_ang = np.deg2rad(np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30))
        elb_ang = np.deg2rad(np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30))

        sho_ang_targ = np.deg2rad(
            np.random.uniform(sho_limit[0] + 30, sho_limit[1] - 30)
        )
        elb_ang_targ = np.deg2rad(
            np.random.uniform(elb_limit[0] + 30, elb_limit[1] - 30)
        )

        angs = torch.tensor(np.array([sho_ang, elb_ang, 0, 0]))
        ang_targ = torch.tensor(np.array([sho_ang_targ, elb_ang_targ, 0, 0]))

        target_pos = self.joint2cartesian(
            torch.tensor(ang_targ, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]

        start_xy = self.joint2cartesian(
            torch.tensor(angs, dtype=torch.float32, device=self.device)
        ).chunk(2, dim=-1)[0]

        info = dict(
            ics_joint=angs,
            ics_xy=start_xy,
            goal=target_pos,
        )
        return info

    def set_goal(
        self,
        goal: torch.Tensor,
    ):
        """
        Sets the goal of the task. This is the target position of the effector.
        """
        self.goal = goal

    def get_obs(self, action=None, deterministic: bool = False) -> Tensor | ndarray:
        self.update_obs_buffer(action=action)

        obs_as_list = [
            self.obs_buffer["vision"][0],
            self.obs_buffer["proprioception"][0],
        ] + self.obs_buffer["action"][: self.action_frame_stacking]

        obs = torch.cat(obs_as_list, dim=-1)

        if deterministic is False:
            obs = self.apply_noise(obs, noise=self.obs_noise)

        return obs if self.differentiable else self.detach(obs)

    def reset(
        self,
        batch_size: int = 1,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:

        """
        Uses the :meth:`Environment.reset()` method of the parent class
        :class:`Environment` that can be overwritten to change the returned data.
        Here the goals (`i.e.`, the targets) are drawn from a random uniform
        distribution across the full joint space.
        """
        sho_limit = np.deg2rad([0, 135])  # mechanical constraints - used to be -90 180
        elb_limit = np.deg2rad([0, 155])
        # Make self.obs_noise a list
        self._set_generator(seed=seed)
        # if ic_state is in options, use that
        if options is not None and "deterministic" in options.keys():
            deterministic = options["deterministic"]
        else:
            deterministic = False
        if options is not None and "ic_state" in options.keys():
            ic_state_shape = np.shape(self.detach(options["ic_state"]))
            if ic_state_shape[0] > 1:
                batch_size = ic_state_shape[0]
            ic_state = options["ic_state"]
        else:
            ic_state = self.q_init

        if options is not None and "target_state" in options.keys():
            self.goal = options["target_state"]
        else:
            sho_ang = np.random.uniform(
                sho_limit[0] + 20, sho_limit[1] - 20, size=batch_size
            )
            elb_ang = np.random.uniform(
                elb_limit[0] + 20, elb_limit[1] - 20, size=batch_size
            )
            sho_vel = np.zeros(batch_size)
            elb_vel = np.zeros(batch_size)
            angs = np.stack((sho_ang, elb_ang, sho_vel, elb_vel), axis=1)
            self.goal = self.joint2cartesian(
                torch.tensor(angs, dtype=torch.float32, device=self.device)
            ).chunk(2, dim=-1)[0]

        options = {
            "batch_size": batch_size,
            "joint_state": ic_state,
        }
        self.effector.reset(options=options)

        self.elapsed = 0.0

        action = torch.zeros((batch_size, self.action_space.shape[0])).to(self.device)

        self.obs_buffer["proprioception"] = [self.get_proprioception()] * len(
            self.obs_buffer["proprioception"]
        )
        self.obs_buffer["vision"] = [self.get_vision()] * len(self.obs_buffer["vision"])
        self.obs_buffer["action"] = [action] * self.action_frame_stacking

        action = action if self.differentiable else self.detach(action)

        obs = self.get_obs(deterministic=deterministic)
        info = {
            "states": self._maybe_detach_states(),
            "action": action,
            "noisy action": action,
            "goal": self.goal if self.differentiable else self.detach(self.goal),
        }
        return obs, info
