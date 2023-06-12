from block import *
import torch.nn as nn
import torch.nn.functional as F



class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.head = None


    def forward(self, x):
        
        x = self.head.forward(x)
        return x