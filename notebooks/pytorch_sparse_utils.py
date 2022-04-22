import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import scipy


from utils import *


import torch
from torch import nn
import torch.nn.functional as F

def scipy_sparse_to_pytorch_sparse(X):
    
    values = X.data
    indices = np.vstack((X.row, X.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = X.shape

    sparse_tensor = torch.sparse_coo_tensor(i, v, torch.Size(shape))
    
    return sparse_tensor

def bilinear_mult(dense, sparse):
    
    return ((sparse @ dense).T @ dense).T