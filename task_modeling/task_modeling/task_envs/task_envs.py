# Class to generate training data for task-trained RNN that does 3 bit memory task
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import gymnasium as gym
from gymnasium import spaces

class NBitFlipFlop(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, N, n_timesteps, n_samples, noise):
        super(NBitFlipFlop, self).__init__()
        self.dataset_name = f'{N}BFF'
        self.action_space = spaces.Box(low=-0.5, high=1.5, shape=(N,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-0.5, high=1.5, shape=(N,), dtype=np.float32)
        self.N = N
        self.state = np.zeros(N)
        self.input_labels = [f'Input {i}' for i in range(N)]
        self.output_labels = [f'Output {i}' for i in range(N)]
        self.n_samples = n_samples
        self.n_timesteps = n_timesteps
        self.noise = noise

    def step(self, action):
        for i in range(self.N):
            if action[i] ==1:
                self.state[i] = 1
            elif action[i]==-1:
                self.state[i] = 0

    def generate_trial(self):
        self.reset()
        inputRand = np.random.random(size = (self.n_timesteps, self.N))
        inputs = np.zeros((self.n_timesteps, self.N))
        inputs[inputRand > 0.98] = 1
        inputs[inputRand <0.02] = -1
        inputs = inputs 
        outputs = np.zeros((self.n_timesteps, self.N))
        for i in range(self.n_timesteps):
            self.step(inputs[i,:])
            outputs[i,:] = self.state + np.random.normal(loc = 0.0, scale = self.noise, size= (outputs[i,:].shape))
        inputs = inputs + np.random.normal(loc = 0.0, scale = self.noise, size= inputs.shape)
        return inputs, outputs
    
    def reset(self):
        self.state = np.zeros(self.N)
        return self.state

    def generate_dataset(self):
        n_samples = self.n_samples
        n_timesteps = self.n_timesteps
        outputs_ds = np.zeros(shape= (n_samples, n_timesteps, self.N))
        inputs_ds = np.zeros(shape= (n_samples, n_timesteps, self.N))
        comb_ds = np.zeros(shape = (n_samples, n_timesteps, 2*self.N))
        for i in range(n_samples):
            inputs, outputs = self.generate_trial()
            outputs_ds[i, :, :] = outputs
            inputs_ds[i, :, :] = inputs
            comb_ds[i,:,0:self.N] = outputs
            comb_ds[i,:,self.N:2*self.N] = inputs
        return  outputs_ds, inputs_ds, comb_ds,

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

        

class ReadySetGoTask(gym.Env):
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
            comb_ds[i, :, :4] = inputs
            comb_ds[i,:, 4:6] = outputs
        return output_ds, input_ds, comb_ds

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

