# Class to generate training data for task-trained RNN that does 3 bit memory task
from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from motornet.environment import Environment


class DecoupledEnvironment(gym.Env, ABC):
    """
    Abstract class representing a decoupled environment.
    This class is abstract and cannot be instantiated.

    """

    # All decoupled environments should have
    # a number of timesteps and a noise parameter

    @abstractmethod
    def __init__(self, n_timesteps: int, noise: float):
        super().__init__()
        self.dataset_name = "DecoupledEnvironment"
        self.n_timesteps = n_timesteps
        self.noise = noise

    # All decoupled environments should have
    # functions to reset, step, and generate trials
    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def generate_dataset(self, n_samples):
        """Must return a dictionary with the following keys:
        #----------Mandatory keys----------
        ics: initial conditions
        inputs: inputs to the environment
        targets: targets of the environment
        conds: conditions information (if applicable)
        extra: extra information
        #----------Optional keys----------
        true_inputs: true inputs to the environment (if applicable)
        true_targets: true targets of the environment (if applicable)
        phase_dict: phase information (if applicable)
        """

        pass


class NBitFlipFlop(DecoupledEnvironment):
    """
    An environment for an N-bit flip flop.
    This is a simple toy environment where the goal is to flip the required bit.
    """

    def __init__(self, n_timesteps: int, noise: float, n=1, switch_prob=0.01):
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        self.dataset_name = f"{n}BFF"
        self.action_space = spaces.Box(low=-0.5, high=1.5, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.5, high=1.5, shape=(n,), dtype=np.float32
        )
        self.goal_space = spaces.Box(low=-1.5, high=1.5, shape=(0,), dtype=np.float32)
        self.n = n
        self.state = np.zeros(n)
        self.input_labels = [f"Input {i}" for i in range(n)]
        self.output_labels = [f"Output {i}" for i in range(n)]
        self.noise = noise
        self.coupled_env = False
        self.switch_prob = switch_prob

    def step(self, action):
        # Generates state update given an input to the flip-flop
        for i in range(self.n):
            if action[i] == 1:
                self.state[i] = 1
            elif action[i] == -1:
                self.state[i] = 0

    def generate_trial(self):
        # Make one trial of flip-flop
        self.reset()

        # Generate the times when the bit should flip
        inputRand = np.random.random(size=(self.n_timesteps, self.n))
        inputs = np.zeros((self.n_timesteps, self.n))
        inputs[
            inputRand > (1 - self.switch_prob)
        ] = 1  # 2% chance of flipping up or down
        inputs[inputRand < (self.switch_prob)] = -1

        # Set the first 3 inputs to 0 to make sure no inputs come in immediately
        inputs[0:3, :] = 0

        # Generate the desired outputs given the inputs
        outputs = np.zeros((self.n_timesteps, self.n))
        for i in range(self.n_timesteps):
            self.step(inputs[i, :])
            outputs[i, :] = self.state

        # Add noise to the inputs for the trial
        true_inputs = inputs
        inputs = inputs + np.random.normal(loc=0.0, scale=self.noise, size=inputs.shape)
        return inputs, outputs, true_inputs

    def reset(self):
        self.state = np.zeros(self.n)
        return self.state

    def generate_dataset(self, n_samples):
        # Generates a dataset for the NBFF task
        n_timesteps = self.n_timesteps
        ics_ds = np.zeros(shape=(n_samples, self.n))
        outputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        true_inputs_ds = np.zeros(shape=(n_samples, n_timesteps, self.n))
        for i in range(n_samples):
            inputs, outputs, true_inputs = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            true_inputs_ds[i, :, :] = true_inputs

        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "targets": outputs_ds,
            "true_inputs": true_inputs_ds,
            "conds": np.zeros(shape=(n_samples, 1)),
            # No extra info for this task, so just fill with zeros
            "extra": np.zeros(shape=(n_samples, 1)),
        }
        return dataset_dict

    def render(self):
        states, inputs = self.generate_trial()
        fig1, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax1 = axes[0]
        ax1.plot(states)
        ax1.set_ylabel("Flip-flop state")
        ax2 = axes[1]
        ax2.plot(inputs)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Inputs")
        plt.savefig("sampleTrial.png", dpi=300)


class RandomTargetDelay(Environment):
    """A reach to a random target from a random starting position with a delay period.

    Args:
        network: :class:`motornet.nets.layers.Network` object class or subclass.
        This is the network that will perform the task.

        name: `String`, the name of the task object instance.
        deriv_weight: `Float`, the weight of the muscle activation's derivative
        contribution to the default muscle L2 loss.

        **kwargs: This is passed as-is to the parent :class:`Task` class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.obs_noise[: self.skeleton.space_dim] = [
            0.0
        ] * self.skeleton.space_dim  # target info is noiseless

        self.dataset_name = "RandomTargetDelay"
        self.n_timesteps = np.floor(self.max_ep_duration / self.effector.dt).astype(int)
        self.input_labels = ["ShoAng", "ElbAng", "ShoVel", "ElbVel"]
        self.output_labels = ["M1", "M2", "M3", "M4"]
        self.goal_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.coupled_env = True

    def generate_dataset(self, n_samples):
        # Make target circular, change loss function to be pinned at zero
        initial_state = []
        inputs = np.zeros((n_samples, self.n_timesteps, 2))

        goal_list = []
        go_cue_list = []
        target_on_list = []

        for i in range(n_samples):
            target_on = np.random.randint(10, 30)
            go_cue = np.random.randint(target_on, self.n_timesteps)

            go_cue_list.append(go_cue)
            target_on_list.append(target_on)

            obs, info = self.reset()
            initial_state.append(torch.squeeze(info["states"]["joint"]))

            # TODO: States["fingertip"]
            initial_state_xy = self.joint2cartesian(
                torch.squeeze(info["states"]["joint"])
            ).chunk(2, dim=-1)[0]

            goal_matrix = torch.zeros((self.n_timesteps, self.skeleton.space_dim))
            goal_matrix[:go_cue, :] = initial_state_xy
            goal_matrix[go_cue:, :] = torch.squeeze(info["goal"])
            inputs[i, target_on:, :] = info["goal"]
            goal_list.append(goal_matrix)

        go_cue_list = np.array(go_cue_list)
        target_on_list = np.array(target_on_list)
        extra = np.stack((target_on_list, go_cue_list), axis=1)

        initial_state = torch.stack(initial_state, axis=0)
        goal_list = torch.stack(goal_list, axis=0)
        dataset_dict = {
            "ics": initial_state,
            "inputs": inputs,
            "targets": goal_list,
            "conds": torch.zeros((n_samples, 1)),
            "extra": extra,
        }
        return dataset_dict

    def set_goal(
        self,
        goal: torch.Tensor,
    ):
        """
        Sets the goal of the task. This is the target position of the effector.
        """
        self.goal = goal

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
            self.goal = self.joint2cartesian(
                self.effector.draw_random_uniform_states(batch_size)
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
