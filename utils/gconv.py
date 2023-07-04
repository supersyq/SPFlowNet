"""
References:
FLOT: https://github.com/valeoai/FLOT
"""

import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class SetConv(torch.nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        """
        Module that performs PointNet++-like convolution on point clouds.

        Parameters
        ----------
        nb_feat_in : int
            Number of input channels.
        nb_feat_out : int
            Number of ouput channels.

        Returns
        -------
        None.

        """

        super(SetConv, self).__init__()

        self.fc1 = torch.nn.Conv2d(nb_feat_in + 3, nb_feat_out, 1, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc2 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn2 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc3 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn3 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.pool = lambda x: torch.max(x, 2)[0]
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, signal, graph):
        """        Parameters
        ----------
        signal : torch.Tensor
            Input features of size B x N x nb_feat_in.
        torch.Tensor
            Ouput features of size B x N x nb_feat_out.

        """

        # Input features dimension
        b, n, c = signal.shape
        n_out = graph.size[0] // b

        # Concatenate input features with edge features
        signal = signal.reshape(b * n, c)
        signal = torch.cat((signal[graph.edges], graph.edge_feats), -1)#[b*n,k,3+c]
        signal = signal.view(b, n_out, graph.k_neighbors, c + 3)
        signal = signal.transpose(1, -1)#[b,c+3,k,n_out]

        # Pointnet++-like convolution
        for func in [
            self.fc1,
            self.bn1,
            self.lrelu,
            self.fc2,
            self.bn2,
            self.lrelu,
            self.fc3,
            self.bn3,
            self.lrelu,
            self.pool,
        ]:
            signal = func(signal)#[b,c',n_out]

        return signal.transpose(1, -1)#[b,n_out,c']
    
class SetConv3(torch.nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        super(SetConv3, self).__init__()

        self.fc1 = torch.nn.Conv2d(nb_feat_in, nb_feat_out, 1, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc2 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn2 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc3 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn3 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.pool = lambda x: torch.max(x, 2)[0]
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, signal):

        # Input features dimension
        b, c, n, k = signal.shape
        # Pointnet++-like convolution
        for func in [
            self.fc1,
            self.bn1,
            self.lrelu,
            self.fc2,
            self.bn2,
            self.lrelu,
            self.fc3,
            self.bn3,
            self.lrelu,
        ]:
            signal = func(signal)#[b,c',k, n_out]

        return signal#[b,c',k, n_out]
