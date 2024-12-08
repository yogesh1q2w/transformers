import numpy as np
import torch
import abc

class Sampler(abc.ABC):
    def __init__(self, sampler_name):
        self.sampler_name = sampler_name

    @abc.abstractmethod
    def sample(self, hidden_states):
        pass

    def __call__(self, hidden_states):
        return self.sample(hidden_states)
    

class UniformSampler(Sampler):
    def __init__(self, config):
        super().__init__("uniform")
        self.retain_proportion = config.retain_proportion
    
    def sample(self, hidden_states):
        # modified_states = []
        # for tensor in hidden_states:
        #     # if not isinstance(tensor, torch.FloatTensor):
        #     #     raise ValueError("All elements in hidden_states must be of type torch.FloatTensor.")
            
        #     mask = torch.rand(tensor.shape) < self.retain_proportion
            
        #     # Apply the mask: zeros out some positions randomly
        #     modified_tensor = tensor * mask.float()
        #     modified_states.append(modified_tensor)
        batch_size, seq_len, embed_dim = hidden_states.shape

        # Create a mask for the sequence length
        mask = torch.rand(seq_len, device=hidden_states.device) < self.retain_proportion
        # Expand the mask to match the shape of hidden_states
        mask_expanded = mask.unsqueeze(0).unsqueeze(2).expand(batch_size, seq_len, embed_dim)

        # Apply the mask: set random tokens to zero
        hidden_states = hidden_states * mask_expanded.float()

        return hidden_states
