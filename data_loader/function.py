import torch
from torch import nn
import torch.nn as nn
import torch.nn.functional as F



#construction of function classes: fully connected layers for nouns and verbs
class Model(nn.Module):
    def __init__(self,input_size,output_size):
        super(Model,self).__init__()
        self.layer1=nn.Linear(input_size,output_size)
        self.layer2=nn.Linear(output_size,1)
    
    def forward(self,x):   
        x=self.layer1(x)
        x=self.layer2(x)
        return x[0][0]