
from torch import nn
from torch.nn import GRUCell
import torch
class RNN(nn.Module):
    def __init__(self,latent_size, input_size= None,  output_size=None):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.cell = None
        self.readout = None
        
    def init_model(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.cell = GRUCell(input_size, self.latent_size)
        self.readout = nn.Linear(self.latent_size, output_size)
    
    def forward(self, inputs, hidden = None):
        n_samples, n_times, n_inputs = inputs.shape
        dev = inputs.device
        if hidden == None:
            hidden = torch.zeros((n_samples,1, self.latent_size), device= dev)
        latents = []
        for input_step in range(inputs.shape[1]):
            hidden = self.cell(inputs[:,input_step,:], hidden[:,0,:])
            latents.append(hidden)
        latents = torch.stack(latents, dim=1)
        output = self.readout(latents)
        return output, latents