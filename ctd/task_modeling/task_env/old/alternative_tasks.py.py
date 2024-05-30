from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from motornet.environment import Environment

from ctd.task_modeling.task_env.task_env import DecoupledEnvironment


class SimonSays(DecoupledEnvironment):
    def __init__(
        self,
        n_timesteps,
        n_samples,
        noise,
        n_pres_low=2,
        n_pres_high=4,
        n_targs=4,
        knownFIFO=True,
    ):
        super(SimonSays, self).__init__(n_timesteps=n_timesteps, noise=noise)

        self.dataset_name = "SimonSays"
        self.n_pres_low = n_pres_low
        self.n_pres_high = n_pres_high + 1
        self.n_targs = n_targs
        self.action_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-2, high=2, shape=(5,), dtype=np.float32
        )
        self.goal_space = spaces.Box(low=-1.5, high=1.5, shape=(0,), dtype=np.float32)
        self.input_labels = ["targX", "targY", "delay", "go", "isFIFO"]
        self.output_labels = ["Reach0", "Reach1"]
        self.noise = noise
        self.knownFIFO = knownFIFO
        self.coupled_env = False
        if knownFIFO:
            self.dataset_name += "_KnownFIFO"
        else:
            self.dataset_name += "_UnknownFIFO"

    def generate_targets(self, num_pres):
        targets = np.zeros((num_pres, 2))
        for i in range(num_pres):
            # draw target location from 8 different targets on unit circle
            target = np.random.randint(0, self.n_targs)
            targets[i, 0] = np.round(np.cos(target * np.pi / (self.n_targs / 2)), 2)
            targets[i, 1] = np.round(np.sin(target * np.pi / (self.n_targs / 2)), 2)
        return targets

    def step(self):
        pass

    def generate_target_vec(self, targets):
        # Make a time varying signal of what the shown target is,
        # with each target seperated by 5 timesteps
        num_targs = targets.shape[0]
        target_vec = np.zeros((15 * num_targs + 5, 2))
        for i in range(len(targets)):
            target_vec[5 + i * 15 : i * 15 + 15, :] = targets[i, :]
        return target_vec

    def reset(self):
        self.state = np.zeros(5)
        return self.state

    def render(self):
        inputs, outputs = self.generate_trial()
        fig1, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax1 = axes[0]
        ax1.plot(inputs, label=self.input_labels)
        ax1.set_ylabel("Inputs")
        ax1.set_ylim([-1.5, 1.5])
        ax1.legend()
        ax2 = axes[1]
        ax2.plot(outputs, label=self.output_labels)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Outputs")
        ax2.set_ylim([-1.5, 1.5])
        ax2.legend()

        plt.savefig("sampleTrial.png", dpi=300)

    def generate_dataset(self, n_samples, targets=None, isFIFO=None):
        # TODO: Maybe batch this?
        n_timesteps = self.n_timesteps
        n_outputs = self.action_space.shape[0]
        n_inputs = self.observation_space.shape[0]
        ics_ds = np.zeros(shape=(n_samples, n_outputs))
        outputs_ds = np.zeros(shape=(n_samples, n_timesteps, n_outputs))
        inputs_ds = np.zeros(shape=(n_samples, n_timesteps, n_inputs))
        isFIFO_ds = np.zeros(shape=(n_samples, 1))
        for i in range(n_samples):
            inputs, outputs = self.generate_trial(targets, isFIFO)
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            isFIFO_ds[i] = isFIFO
        inputs_ds += np.random.normal(loc=0.0, scale=self.noise, size=inputs_ds.shape)
        outputs_ds += np.random.normal(loc=0.0, scale=self.noise, size=outputs_ds.shape)
        dataset_dict = {
            "ics": ics_ds,
            "inputs": inputs_ds,
            "targets": outputs_ds,
            "conds": isFIFO_ds,
        }
        return dataset_dict

    def generate_trial(self, target_inds=None, isFIFO=None):
        self.reset()
        # if targets are not none
        if target_inds is None:
            n_pres = np.random.randint(self.n_pres_low, self.n_pres_high)
            targets = self.generate_targets(n_pres)
        else:
            targets = self.generate_targets_testing(target_inds)

        target_vec = self.generate_target_vec(targets)
        targ_len = target_vec.shape[0]
        if isFIFO is None:
            isFIFO = np.random.randint(0, 2)
            isFIFO = isFIFO * 2 - 1  # Convert from 0,1 to -1,1
        delay = np.random.randint(20, 40)

        inputs = np.zeros((self.n_timesteps, 5))
        outputs = np.zeros((self.n_timesteps, 2))
        inputs[0:targ_len, 0:2] = target_vec
        inputs[targ_len : targ_len + delay, 2] = 1
        inputs[targ_len + delay :, 3] = 1
        if not self.knownFIFO:
            inputs[targ_len + delay - 5 :, 4] = isFIFO
        else:
            inputs[:, 4] = isFIFO

        if isFIFO == 1:
            outputs[targ_len + delay : 2 * targ_len + delay, 0] = target_vec[:, 0]
            outputs[targ_len + delay : 2 * targ_len + delay, 1] = target_vec[:, 1]
        else:  # Stack outputs in reverse
            outputs[targ_len + delay : 2 * targ_len + delay, 0] = target_vec[::-1, 0]
            outputs[targ_len + delay : 2 * targ_len + delay, 1] = target_vec[::-1, 1]
        return inputs, outputs, isFIFO


class ReadySetGoTask(DecoupledEnvironment):
    def __init__(
        self,
        n_timesteps,
        noise,
        n_samples,
    ):
        super().__init__(n_timesteps=n_timesteps, noise=noise)

        self.dataset_name = "ReadySetGo"
        self.action_space = spaces.Box(low=-0.5, high=1.5, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-0.5, high=1.5, shape=(4,), dtype=np.float32
        )
        self.goal_space = spaces.Box(low=-1.5, high=1.5, shape=(0,), dtype=np.float32)
        self.input_labels = ["ShortPrior", "EyeTrial", "ReachDir", "PulseLine"]
        self.output_labels = ["Reach0", "Reach1"]
        self.noise = noise
        self.coupled_env = False

    def reset(self):
        self.short_prior = np.random.randint(0, 2)
        self.eye_trial = np.random.randint(0, 2)
        self.theta_trial = np.random.randint(0, 2)
        if self.short_prior:
            prior = np.random.choice(a=[24, 28, 32, 36, 40])
        else:
            prior = np.random.choice(a=[40, 45, 50, 55, 60])
        self.prior = prior

    def step(self, i):
        input_row = np.zeros((1, 4))
        input_row[0][0] = self.short_prior
        input_row[0][1] = self.eye_trial
        output_row = np.zeros((1, 2))
        input_row[0][2] = self.theta_trial
        if i == 5:
            input_row[0][3] = 1
        if i == 5 + self.prior:
            input_row[0][3] = 1
        if i >= 5 + (2 * self.prior):
            output_row[0][self.eye_trial] = (-1) ** self.theta_trial
        return input_row, output_row

    def generate_trial(self):
        self.reset()
        inputs = np.zeros((self.n_timesteps, 4))
        outputs = np.zeros((self.n_timesteps, 2))

        for i in range(self.n_timesteps):
            input_row, output_row = self.step(i)
            inputs[i, :] = input_row + np.random.normal(
                loc=0.0, scale=self.noise, size=input_row.shape
            )
            outputs[i, :] = output_row + np.random.normal(
                loc=0.0, scale=self.noise, size=output_row.shape
            )
        return inputs, outputs

    def generate_dataset(self, n_samples):
        n_timesteps = self.n_timesteps

        self.n_timesteps = n_timesteps
        ics_ds = np.zeros(shape=(n_samples, 2))
        input_ds = np.zeros(shape=(n_samples, self.n_timesteps, 4))
        output_ds = np.zeros(shape=(n_samples, self.n_timesteps, 2))
        for i in range(n_samples):
            inputs, outputs = self.generate_trial()
            input_ds[i, :, :] = inputs
            output_ds[i, :, :] = outputs
        dataset_dict = {
            "ics": ics_ds,
            "inputs": input_ds,
            "targets": output_ds,
        }
        return dataset_dict

    def plot_trial(self):
        inputs, outputs = self.generate_trial()
        fig1, axes = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax1 = axes[0]
        ax1.plot(inputs[:, 0], label="short_prior")
        ax1.plot(inputs[:, 1], label="eye_trial")
        ax1.plot(inputs[:, 2], label="theta")
        ax1.plot(inputs[:, 3], label="pulse_line")

        ax1.set_ylabel("Input fields")
        ax1.legend(loc="right")

        ax2 = axes[1]
        ax2.plot(outputs[:, 0], label="Hand output")
        ax2.plot(outputs[:, 1], label="Eye output")

        ax2.set_xlabel("Time (bins)")
        ax2.set_ylabel("Outputs")
        ax1.set_title(
            f"Example Trial:\n Short Prior: {self.short_prior}, Pulse Timing:"
            f"{self.prior}, Eye: {self.eye_trial}, Theta: {self.theta_trial}"
        )

        ax2.legend(loc="right")
        plt.savefig("sampleTrialRSG.png", dpi=300)


class RandomTargetReach(Environment):
    """A reach to a random target from a random starting position.

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
        self.dataset_name = "RandomTargetReach"
        self.n_timesteps = np.floor(self.max_ep_duration / self.effector.dt).astype(int)
        self.input_labels = ["ShoAng", "ElbAng", "ShoVel", "ElbVel"]
        self.output_labels = ["M1", "M2", "M3", "M4"]
        self.coupled_env = True
        self.goal_space = spaces.Box(low=-1.5, high=1.5, shape=(0,), dtype=np.float32)

    def generate_dataset(self, n_samples):
        initial_state = []
        inputs = np.zeros((n_samples, self.n_timesteps, 0))
        goal_list = []
        for i in range(n_samples):
            obs, info = self.reset()
            initial_state.append(torch.squeeze(info["states"]["joint"]))
            goal_matrix = torch.zeros((self.n_timesteps, self.skeleton.space_dim))
            goal_matrix[:, :2] = torch.squeeze(info["goal"])
            goal_list.append(goal_matrix)

        initial_state = torch.stack(initial_state, axis=0)
        goal_list = torch.stack(goal_list, axis=0)
        dataset_dict = {
            "ics": initial_state,
            "inputs": inputs,
            "targets": goal_list,
            "conds": torch.zeros((n_samples, 1)),
            "extra": torch.zeros((n_samples, 1)),
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
        ic_state: Any | None = None,
        target_state: Any | None = None,
        deterministic: bool = False,
        seed: int | None = None,
    ) -> tuple[Any, dict[str, Any]]:

        """
        Uses the :meth:`Environment.reset()` method of the parent class
        :class:`Environment` that can be overwritten to change the returned data.
        Here the goals (`i.e.`, the targets) are drawn from a random uniform
        distribution across the full joint space.
        """

        self._set_generator(seed=seed)

        if ic_state is not None:
            ic_state_shape = np.shape(self.detach(ic_state))
            if ic_state_shape[0] > 1:
                batch_size = ic_state_shape[0]
        else:
            ic_state = self.q_init

        self.effector.reset(batch_size, ic_state)
        if target_state is None:
            self.goal = self.joint2cartesian(
                self.effector.draw_random_uniform_states(batch_size)
            ).chunk(2, dim=-1)[0]
        else:
            self.goal = target_state
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
