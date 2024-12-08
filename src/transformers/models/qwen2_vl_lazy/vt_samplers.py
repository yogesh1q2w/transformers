import numpy as np
import abc

class Sampler(abc.ABC):
    def __init__(self, sampler_name):
        self.sampler_name = sampler_name
    
    def sample(self):
        raise NotImplementedError
    

class UniformSampler(Sampler):
    def __init__(self, retain_proportion):
        super().__init__("uniform")
        self.retain_proportion = retain_proportion
    
    def sample(self, inputs):
        indices = np.random.choice(inputs.shape[0], size=int(inputs.shape[0] * self.retain_proportion), replace=False)
        return inputs[indices]
    
def a():
    pass