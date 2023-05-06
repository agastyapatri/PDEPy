#   Global Imports
import numpy as np
import torch
import torch.nn as nn
import argparse 
import numba 
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

#   Custom Imports
import pdepy.models 






