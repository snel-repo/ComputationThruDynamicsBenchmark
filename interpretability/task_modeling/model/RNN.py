
from torch import nn
from torch.nn import GRUCell
import torch
class RNN(nn.Module):
    def __init__(self,input_size, latent_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, latent_size, output_size)
        
    def forward(self, input):
        n_samples, n_times, n_inputs = input.shape
        dev = input.device
        hidden = torch.zeros((n_samples, self.latent_size), device= dev)
        states = []
        for input_step in input.transpose(0, 1):
            hidden = self.cell(input_step, hidden)
            states.append(hidden)
        states = torch.stack(states, dim=1)
        return states, hidden