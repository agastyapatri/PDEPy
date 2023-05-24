import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from numba import jit   
torch.manual_seed(0)
dtype = torch.float32 
device = "cuda" if torch.cuda.is_available() else "cpu"


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Sequential(
            nn.Linear(in_features = self.in_features, out_features = self.out_features, dtype=dtype, device=device),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.linear(x)
    

class DeepONet(nn.Module):
    """
        Defining a Deep Operator Network 
    """
    def __init__(self, branch_arch:list, trunk_arch:list) -> None:
        super().__init__()
        self.branch_arch = branch_arch
        self.trunk_arch = trunk_arch 

        self.branch_net = nn.Sequential()
        for i in range(len(branch_arch)-1):        
            self.branch_net.append(LinearBlock(in_features=branch_arch[i], out_features=branch_arch[i+1]))

        self.trunk_net = nn.Sequential()
        for i in range(len(branch_arch)-1):        
            self.trunk_net.append(LinearBlock(in_features=trunk_arch[i], out_features=trunk_arch[i+1]))
    
    def forward(self, branch_data, trunk_data):
        output = torch.matmul(self.branch_net(branch_data), torch.transpose(self.trunk_net(trunk_data), dim0=0, dim1=1))
        return output 


if __name__ == "__main__":
    BRANCH_ARCH = [100, 80, 60, 40]
    TRUNK_ARCH = [1, 80, 60, 40]

    branch_data = torch.randn(size=(150, 100), dtype = dtype, device = device)
    trunk_data = torch.randn(size=(100, 1), dtype = dtype, device = device)
    deeponet = DeepONet(branch_arch=BRANCH_ARCH, trunk_arch=TRUNK_ARCH) 
    print(deeponet)