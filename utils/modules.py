"""
References:
SPNet: https://github.com/fpthink/SPNet
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .gconv import SetConv3
from lib.pointops.functions import pointops 

class learn_SLIC_calc_mutual(nn.Module):
    """
    calculate association between superpoints and points
    superpoint updating
    """
    def __init__(self, xyz_in, xyz_out, fea_in, fea_out, flow_in, flow_out, 
                 bn=True, use_xyz=True, use_softmax=True, use_norm=True, last=False):
        super().__init__()
        self.bn = bn
        self.use_xyz = use_xyz
        self.use_softmax = use_softmax
        self.use_norm = use_norm
        self.last = last

        self.w_c2p_fea = SetConv3(fea_in, fea_out)
        self.w_c2p_xyz = SetConv3(xyz_in, xyz_out)
        self.w_c2p_flow = SetConv3(flow_in, flow_out)
        self.mlp = torch.nn.Linear(fea_in, fea_out)

    def _l1norm(self, inp, dim):
        return inp / (1e-6 + inp.sum(dim=dim, keepdim=True))

    def forward(self, sp_fea, sp_xyz, sp_flow, o_p_fea, p_xyz, p_flow, c2p_idx_abs, c2p_idx, cluster_idx, prototypes_src_fused_xyz, prototypes_src_fused_feat, fused_xyz_0, fused_feats_0):
        # sp_fea: b x m x c
        # sp_xyz: b x m x 3
        # sp_flow: b x m x 3
        # o_p_fea: b x n x c
        # p_xyz: b x n x 3
        # p_flow: b x n x 3
        # c2p_idx_abs: b x n x nc2p
        bs, n, nc2p = c2p_idx_abs.size()

        c2p_fea = pointops.grouping(prototypes_src_fused_feat.transpose(1, 2).contiguous(), c2p_idx_abs) - fused_feats_0.transpose(1, 2).contiguous().unsqueeze(-1).repeat(1, 1, 1, nc2p)
        # c2p_fea: b x c x n x nc2p
        
        c2p_xyz = pointops.grouping(prototypes_src_fused_xyz.transpose(1, 2).contiguous(), c2p_idx_abs) - fused_xyz_0.transpose(1, 2).contiguous().unsqueeze(-1).repeat(1, 1, 1, nc2p)
        # c2p_xyz: b x 3 x n x nc2p

        c2p_fea = self.w_c2p_fea(c2p_fea)   # b x 16 x n x nc2p
        c2p_xyz = self.w_c2p_xyz(c2p_xyz)   # b x 16 x n x nc2p

        diff = c2p_fea + c2p_xyz # b x 16 x n x nc2p
        
        bi_w = torch.sum(diff, 1).view(bs, n, nc2p) # b x n x nc2p

        if self.use_softmax:
            bi_w = F.softmax(bi_w, dim=-1)  # b x n x nc2p

        f, sp_nei_cnt = pointops.assomatrixfloat(nc2p, bi_w, c2p_idx, cluster_idx.unsqueeze(-1))
        # f: b x m x n
        # sp_nei_cnt: b x m x 1

        z = self._l1norm(f, dim=2)

        sp_fea = torch.matmul(z, o_p_fea)   # (b, m, n) X (b, n, c) -> (b, m, c)
        
        sp_xyz = torch.matmul(z, p_xyz)     # (b, m, n) X (b, n, 3) -> (b, m, 3)

        sp_flow = torch.matmul(z, p_flow)     # (b, m, n) X (b, n, 3) -> (b, m, 3)
        
        if self.last:
            return bi_w, f, z, sp_fea, sp_xyz, sp_flow, sp_nei_cnt
        return sp_fea, sp_xyz, sp_flow