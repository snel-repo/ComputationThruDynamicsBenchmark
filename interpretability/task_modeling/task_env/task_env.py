# Class to generate training data for task-trained RNN that does 3 bit memory task
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod
# TODO: Add abstract wrapper class for task environments

class DecoupledEnvironment(gym.Env, ABC):
    """
    Abstract class representing a decoupled environment.
    This class is abstract and cannot be instantiated.
    """
    @abstractmethod
    def __init__(self, n_timesteps: int, noise: float):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.noise = noise

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def reset(self):
        pass


class NBitFlipFlop(DecoupledEnvironment):
    """
    An environment for an N-bit flip flop.
    This is a simple toy text environment where the goal is to flip the required bit.
    """
    def __init__(self, n_timesteps: int, noise: float, n=1):
        super().__init__(n_timesteps=n_timesteps, noise=noise)
        # TODO: Seed environment
        self.dataset_name = f'{n}BFF'
        self.action_space = spaces.Box(low=-0.5, high=1.5, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-0.5, high=1.5, shape=(n,), dtype=np.float32)
        self.n = n
        self.state = np.zeros(n)
        self.input_labels = [f'Input {i}' for i in range(n)]
        self.output_labels = [f'Output {i}' for i in range(n)]
        self.noise = noise

    def step(self, action):
        for i in range(self.n):
            if action[i] ==1:
                self.state[i] = 1
            elif action[i]==-1:
                self.state[i] = 0

    def generate_trial(self):
        self.reset()
        inputRand = np.random.random(size = (self.n_timesteps, self.n))
        inputs = np.zeros((self.n_timesteps, self.n))
        inputs[inputRand > 0.98] = 1
        inputs[inputRand <0.02] = -1
        inputs = inputs 
        outputs = np.zeros((self.n_timesteps, self.n))
        for i in range(self.n_timesteps):
            self.step(inputs[i,:])
            outputs[i,:] = self.state + np.random.normal(loc = 0.0, scale = self.noise, size= (outputs[i,:].shape))
        inputs = inputs + np.random.normal(loc = 0.0, scale = self.noise, size= inputs.shape)
        return inputs, outputs
    
    def reset(self):
        self.state = np.zeros(self.n)
        return self.state

    def generate_dataset(self, n_samples):
        # TODO: Maybe batch this?
        # TODO: Code formatter
        # TODO: Inputs then outputs
        n_timesteps = self.n_timesteps
        outputs_ds = np.zeros(shape= (n_samples, n_timesteps, self.n))
        inputs_ds = np.zeros(shape= (n_samples, n_timesteps, self.n))
        for i in range(n_samples):
            inputs, outputs = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
        return  inputs_ds, outputs_ds

    def render(self):
        states, inputs = self.generate_trial()
        fig1, axes = plt.subplots(nrows=2, ncols= 1, sharex= True)
        ax1 = axes[0]
        ax1.plot(states)
        ax1.set_ylabel("Flip-flop state")
        ax2 = axes[1]
        ax2.plot(inputs)
        ax2.set_xlabel('Time')
        ax2.set_ylabel("Inputs")
        plt.savefig("sampleTrial.png", dpi=300)

        

class ReadySetGoTask(DecoupledEnvironment):
    def __init__(self,n_timesteps, n_samples, noise):
        super(ReadySetGoTask, self).__init__()

        self.dataset_name = "ReadySetGo"
        self.action_space = spaces.Box(low=-0.5, high=1.5, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-0.5, high=1.5, shape=(6,), dtype=np.float32)
        self.input_labels = ['ShortPrior', 'EyeTrial', 'ReachDir', 'PulseLine']
        self.output_labels = ['Reach0', 'Reach1']
        self.noise = noise

    def generate_trial(self):
        self.short_prior = np.random.randint(0,2) 
        self.eye_trial = np.random.randint(0,2)
        self.theta_trial = np.random.randint(0,2)
        if self.short_prior:
            prior = np.random.choice(a = [24, 28,32,36,40])
        else:
            prior = np.random.choice(a= [40, 45,50,55,60])
        self.prior= prior
        inputs = np.zeros((self.n_timesteps, 4))
        outputs = np.zeros((self.n_timesteps, 2))

        for i in range(self.n_timesteps):
            input_row = np.zeros((1,4))
            input_row[0][0] = self.short_prior
            input_row[0][1] = self.eye_trial
            output_row = np.zeros((1,2))
            input_row[0][2] = self.theta_trial
            if i == 5:
                input_row[0][3] = 1
            if i == 5 + prior:
                input_row[0][3] = 1
            if i >= 5 + (2 * prior):
                output_row[0][self.eye_trial] = (-1)**self.theta_trial 
            inputs[i,:] = input_row + np.random.normal(loc = 0.0, scale = self.noise, size= input_row.shape)
            outputs[i,:] = output_row + np.random.normal(loc = 0.0, scale = self.noise, size= output_row.shape)
        return inputs, outputs

    def generate_dataset(self):
        n_samples = self.n_samples
        n_timesteps = self.n_timesteps

        self.n_timesteps = n_timesteps
        input_ds = np.zeros(shape= (n_samples, self.n_timesteps, 4))
        output_ds = np.zeros(shape = (n_samples, self.n_timesteps, 2))
        comb_ds = np.zeros(shape = (n_samples, self.n_timesteps, 6))
        for i in range(n_samples):
            inputs, outputs = self.generate_trial()
            input_ds[i, :, :] = inputs
            output_ds[i, :, :] = outputs
        return input_ds, output_ds

    def plot_trial(self):
        inputs, outputs = self.generate_trial()
        fig1, axes = plt.subplots(nrows=2, ncols= 1, sharex= True)
        ax1 = axes[0]
        ax1.plot(inputs[:,0], label = "short_prior")
        ax1.plot(inputs[:,1], label = "eye_trial")
        ax1.plot(inputs[:,2], label = "theta")
        ax1.plot(inputs[:,3], label = "pulse_line")
        

        ax1.set_ylabel("Input fields")
        ax1.legend(loc = 'right')

        ax2 = axes[1]
        ax2.plot(outputs[:,0], label = "Hand output")
        ax2.plot(outputs[:,1], label = "Eye output")

        ax2.set_xlabel('Time (bins)')
        ax2.set_ylabel("Outputs")
        ax1.set_title(f"Example Trial:\n Short Prior: {self.short_prior}, Pulse Timing: {self.prior}, Eye: {self.eye_trial}, Theta: {self.theta_trial}")
        
        ax2.legend(loc = 'right')
        plt.savefig("sampleTrialRSG.png", dpi=300)

