import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from numba import jit   
torch.manual_seed(0)
dtype = torch.float32 
device = "cuda" if torch.cuda.is_available() else "cpu"


class DeepONet(nn.Module):
    """
        Defining a Deep Operator Network 
    """
    def __init__(self, branch_arch:list, trunk_arch:list) -> None:
        super().__init__()
        self.branch_net = nn.Sequential(
            nn.Linear(in_features=branch_arch[0], out_features=branch_arch[1], dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(in_features=branch_arch[1], out_features=branch_arch[2], dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(in_features=branch_arch[2], out_features=branch_arch[3], dtype=dtype, device=device),
            nn.ReLU(),
        )
        self.trunk_net = nn.Sequential(
            nn.Linear(in_features=trunk_arch[0], out_features=trunk_arch[1], dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(in_features=trunk_arch[1], out_features=trunk_arch[2], dtype=dtype, device=device),
            nn.ReLU(),
            nn.Linear(in_features=trunk_arch[2], out_features=trunk_arch[3], dtype=dtype, device=device),
            nn.ReLU(),
        )
    
    def forward(self, branch_data, trunk_data):
        output = torch.matmul(self.branch_net(branch_data), torch.transpose(self.trunk_net(trunk_data), dim0=0, dim1=1))
        return output 

if __name__ == "__main__":
    BRANCH_ARCH = [100, 80, 60, 40]
    TRUNK_ARCH = [1, 80, 60, 40]

    branch_data = torch.randn(size=(150, 100), dtype = dtype, device = device)
    trunk_data = torch.randn(size=(100, 1), dtype = dtype, device = device)
    
    deeponet = DeepONet(branch_arch=BRANCH_ARCH, trunk_arch=TRUNK_ARCH) 
    print(deeponet(branch_data, trunk_data).shape)