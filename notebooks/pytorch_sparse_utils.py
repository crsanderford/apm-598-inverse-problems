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


def sparse_bilinear_mult(dense, sparse):
    
    return ((sparse @ dense).T @ dense).T


def data_adjacency_batcher(data, adjacency, batch_size):

    rows = torch.arange(0,data.shape[0])
    p = np.random.permutation(len(rows))
    rows = rows[p]

    iterations = 0

    while (batch_size * iterations <= len(rows)):

        data_batch = (data[ rows[batch_size*iterations : batch_size*(iterations+1) ] , : ])

        mask_batch = (data_batch != 0).bool()

        A_batch = adjacency[ rows[batch_size*iterations : batch_size*(iterations+1)], : ][ :, rows[batch_size*iterations : batch_size*(iterations+1)] ]

        yield (data_batch, mask_batch, A_batch, rows[batch_size*iterations : batch_size*(iterations+1) ])

        iterations = iterations + 1

def adjacency_to_laplacian(adjacency):

    degree = torch.diag(adjacency.sum(axis=0))

    return (degree - adjacency)

def adjacency_to_L1_adjacency(data_estimate, adjacency, epsilon, device):

    epsilon = torch.Tensor([epsilon]).to(device=device)

    L1_adjacency = torch.zeros_like(adjacency).to(device=device)

    for ii in range(adjacency.shape[0]): # for each row in the adjacency matrix:

        indices = adjacency[:,ii].bool() # get the nonzero elements of the adjacency matrix (which will be updated in the calculation)

        neighbors_data = data_estimate[ indices, :] # find the corresponding rows of data

        diffs = torch.stack([data_estimate[ii,:]]*neighbors_data.shape[0], dim=0) - neighbors_data # calculate the difference between the row and its neighbors

        norms = torch.linalg.norm(diffs, ord=1, dim=1) # calculate the L1 norm of those differences

        divisors = torch.where(norms > epsilon, norms, epsilon).to(device=device) # substitute the norm value with epsilon when the norm value is smaller

        L1_adjacency[ii, indices] = torch.div( adjacency[ii, indices].to(device=device), divisors) # perform the division on nonzero elements

    return L1_adjacency


class MatrixFactorization(torch.nn.Module):
    def __init__(self, rows, cols, n_factors=10):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.row_embedding = torch.nn.parameter.Parameter(torch.ones(rows, n_factors)*(1/n_factors**2), requires_grad=True)
        self.col_embedding = torch.nn.parameter.Parameter(torch.ones(n_factors, cols)*(1/n_factors**2), requires_grad=True)

    def forward(self, row_indices, col_indices):
        return torch.matmul( self.row_embedding[row_indices, :], self.col_embedding[:, col_indices] )