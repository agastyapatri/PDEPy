#   Global Imports
import numpy as np
import torch
import torch.nn as nn
import argparse 
import numba 
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

#   Custom Imports
from pdepy.models import deeponet
BRANCH_ARCH = [100, 80, 60, 40]
TRUNK_ARCH = [1, 80, 60, 40]


model = deeponet.DeepONet()




