import torch.nn as nn
from model.config import ModelConfig
from model.model import CLSADecoder

"""
Backward Decoder: Forms the approximation to the posterior p_phi(z|x, y)
- x: Today's Frame
- y: Current Frame

- Overall structure is the the same as the CLSADecoder but 
    - excludes inter-attention mechanism
    - takes in the reverse-order of target day frames

*Only used during training
"""

class BackwardDecoder(CLSADecoder):
    def __init__(self, config: ModelConfig):
        super().__init__(ModelConfig)
        pass
    
    def forward(self, enc_outputs, ):
        super().forward()
        
        pass


'''
CLVSA Model:
'''
class CLVSAModel():
    def __init(self, config: ModelConfig):
        pass