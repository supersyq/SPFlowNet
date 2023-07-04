"""
References:
PointPWC-Net: https://github.com/DylanWusee/PointPWC
FLOT: https://github.com/valeoai/FLOT
FlowStep3D: https://github.com/yairkit/flowstep3d
"""

import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import ot
from utils.graph import Graph
from utils.gconv import SetConv
from utils.modules import learn_SLIC_calc_mutual
from lib.pointops.functions import pointops 

from utils.pointconv_util import UpsampleFlow, FlowEmbedding, PointWarping, index_points_gather, PointWarping_feat

class GRU(nn.Module):
    def __init__(self,hidden_dim, input_dim):
        super(GRU, self).__init__()
        in_ch = hidden_dim + input_dim
        self.convz = SetConv(in_ch, hidden_dim)
        self.convr = SetConv(in_ch, hidden_dim)
        self.convq = SetConv(in_ch, hidden_dim)

    def forward(self, h, x, c, graph):
        hx = torch.cat([h, x], dim=2) 
        z = torch.sigmoid(self.convz(hx, graph))
        r = torch.sigmoid(self.convr(hx, graph))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=2), graph))
        h = (1 - z) * h + z * q
        return h

class SPFlowNet(torch.nn.Module):
    def __init__(self, args):

        super(SPFlowNet, self).__init__()

        n = 32
        self.k_decay_fact = 1.0
        self.nc2p = args.nc2p # 2
        self.numc = args.num_sp #128 for non_occluded data; 30 for occluded data.
        self.down_scale = args.down_scale #4 for non_occluded data; 2 for occluded data.
        self.weight_const = args.weight_const # weight of consistency loss
        self.distance_margin = args.distance_margin #100 for non_occluded data; 12 for occluded data.
        self.num_iters = 3 # iteration number

        self.use_xyz = True               # True
        self.use_softmax = True       # True
        self.use_norm = True            # True

        # Feature extraction
        self.feat_conv1 = SetConv(3, n)
        self.feat_conv2 = SetConv(n, 2 * n)
        self.feat_conv3 = SetConv(2 * n, 4 * n)

        # flow_regressor
        self.flow_conv1 = SetConv(4 * n,4 * n)
        self.flow_conv2 = SetConv(4 * n,4 * n)
        self.fc = torch.nn.Linear(4 * n, 3)

        # get_x
        self.delta_flow_conv_x = SetConv(4 * n,4 * n)
        self.flow_conv_x = SetConv(4 * n,4 * n)
        self.flow_encoder = torch.nn.Linear(3,4 * n)

        # get_h
        # self.h_conv1 = torch.nn.Linear(4 * n, 4 * n)
        # self.h_conv2 = torch.nn.Linear(4 * n, 4 * n)
        self.h_conv1 = SetConv(4 * n, 4 * n)
        self.h_conv2 = SetConv(4 * n, 4 * n)

        self.learn_SLIC_calc_1 = learn_SLIC_calc_mutual(xyz_in = 2*3, xyz_out = n, fea_in = 2 * 4 * n, fea_out = n, flow_in = 3, flow_out = n,
                            bn=True, use_xyz=self.use_xyz, use_softmax=self.use_softmax, use_norm=self.use_norm, last=True)

        self.upsample = UpsampleFlow()

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

        self.gru = GRU(hidden_dim = 4 * n, input_dim = 4 * n)
        self.local_corr_layer = FlowEmbedding(nsample=16, in_channel=4 * n, mlp=[4 * n, 4 * n, 4 * n])

        self.warping = PointWarping()
        self.obtain_feat = PointWarping_feat()
        self.var_encoding = nn.Sequential(nn.Linear(3, n),nn.Linear(n, 1), nn.Sigmoid())

    def _l1norm(self, inp, dim):
        return inp / (1e-6 + inp.sum(dim=dim, keepdim=True))

    def get_features(self, pcloud, nb_neighbors, graph):
        if graph == None:
            graph = Graph.construct_graph(pcloud, nb_neighbors)
        x = self.feat_conv1(pcloud, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)
        return x, graph

    def get_flow(self, pc0, pc1, feats_0, feats_1):
        sim_matrix = torch.einsum("nlc,nsc->nls", feats_0, feats_1)
        log_assign_matrix = ot.log_optimal_transport(sim_matrix, self.bin_score, 2)
        assign_matrix = log_assign_matrix.exp()
        support = (ot.pairwise_distance(pc0, pc1, normalized=False) < self.distance_margin).float()
        conf_matrix = assign_matrix[:, :-1, :-1] * support
        row_sum = conf_matrix.sum(-1, keepdim=True) 
        pred_correspondence = (conf_matrix @ pc1) / (row_sum + 1e-8) 
        sa_feats_0 = (conf_matrix @ feats_1) / (row_sum + 1e-8) 
        ot_flow0 = pred_correspondence - pc0 #[b,n,3]

        conf_matrix = conf_matrix.permute(0, 2, 1) #[b,m,n]
        row_sum2 = conf_matrix.sum(-1, keepdim=True) #[b,m,1]
        pred_correspondence2 = (conf_matrix @ pc0) / (row_sum2 + 1e-8)
        sa_feats_1 = (conf_matrix @ feats_0) / (row_sum2 + 1e-8)  
        ot_flow_back = pred_correspondence2 - pc1 #[b,n,3]

        return sa_feats_0, ot_flow0, sa_feats_1, ot_flow_back

    def down(self, pc0, feat0, fused_feat0, fused_xyz0, flow0, npoint):
        fps_pc0_l1 = pointops.furthestsampling(pc0, npoint)  # [B, N]
        pc0_l1 = index_points_gather(pc0, fps_pc0_l1)  # [B, N, 3]
        feats_pc0_l1 = index_points_gather(feat0, fps_pc0_l1)  # [B, N, C]
        flow0_l1 = index_points_gather(flow0, fps_pc0_l1)  # [B, N, 3]
        xyz_pc0_fused = index_points_gather(fused_xyz0, fps_pc0_l1)  # [B, N, C]
        feats_pc0_fused = index_points_gather(fused_feat0, fps_pc0_l1)  # [B, N, C]
        return fps_pc0_l1, pc0_l1, feats_pc0_l1, flow0_l1, xyz_pc0_fused, feats_pc0_fused


    def get_x(self, feats1_loc_new, corr_feats, flow, graph, confidence): #feature,flow_feature,flow,cordi
        corr_feats = self.delta_flow_conv_x(corr_feats*confidence, graph) 
        flow_feats = self.flow_encoder(flow)
        flow_feats = self.flow_conv_x(flow_feats*confidence, graph)
        x = corr_feats + flow_feats
        return x

    def flow_regressor(self, h, graph):
        flow_feats = self.flow_conv1(h, graph)
        flow_feats = self.flow_conv2(flow_feats,graph)
        x = self.fc(flow_feats)
        return x

    def calc_h0(self, flow, feats1_loc, pc, graph):
        # h = self.h_conv1(feats1_loc)
        # h = self.h_conv2(h)
        h = self.h_conv1(feats1_loc,graph)
        h = self.h_conv2(h,graph)
        h = torch.tanh(h)
        return h
    

    def feature_coordinate_space(self, pc0, pc1, ot_flow0, ot_flow1, feat0, feat1, new_feat0, new_feat1):
        xyz_out0 = torch.cat((pc0, pc0 + ot_flow0),dim=-1)
        xyz_out1 = torch.cat((pc1, pc1 + ot_flow1),dim=-1)

        feat_out0 = torch.cat((feat0, new_feat0),dim=-1)
        feat_out1 = torch.cat((feat1, new_feat1),dim=-1)

        return xyz_out0, xyz_out1, feat_out0, feat_out1
    
    def GRU_based_refinement(self, sp2p_flow, C, pc0, pc1, feat0, feat1, graph, pc0_ori, h, iter):
        pc_warp = pc0 + sp2p_flow
        corr_feats = self.local_corr_layer(pc_warp, pc1, feat0, feat1)
        x = self.get_x(feat0, corr_feats, sp2p_flow, graph, C)
        h = self.gru(h=h, x=x, c= C, graph=graph) 

        delta_flow = self.flow_regressor(h, graph) #get residual flow
        delta_flow = delta_flow / (self.k_decay_fact*iter + 1)
        sp2p_flow = sp2p_flow + delta_flow

        flow0_up = self.upsample(pc0_ori.permute(0, 2, 1), pc0.permute(0, 2, 1), sp2p_flow.permute(0, 2, 1)).permute(0, 2, 1)
        return sp2p_flow, flow0_up, h

    def confidence_encoding(self, prototypes_src_flow, prototypes_tgt_flow, c2p_idx_abs0, c2p_idx_abs1, a0, a1, pc0, pc1):
        bs,num_c,_ = prototypes_src_flow.size()
        c2p_flow0 = pointops.grouping(prototypes_src_flow.view(bs,num_c,3).transpose(1, 2).contiguous(), c2p_idx_abs0)
        # (b, 12, m), (b, n, n2cp) -> b x 12 x n x n2cp
        sp2p_flow0 = torch.sum(c2p_flow0 * a0.unsqueeze(1), dim=-1, keepdim=False).transpose(1, 2).contiguous() #[b,n,3]

        bs,num_c,_ = prototypes_tgt_flow.size()
        c2p_flow1 = pointops.grouping(prototypes_tgt_flow.view(bs,num_c,3).transpose(1, 2).contiguous(), c2p_idx_abs1)
        # (b, 12, m), (b, n, n2cp) -> b x 12 x n x n2cp
        sp2p_flow1 = torch.sum(c2p_flow1 * a1.unsqueeze(1), dim=-1, keepdim=False).transpose(1, 2).contiguous() #[b,n,3]

        new_ot_flow1 = self.warping(pc0, pc1, sp2p_flow0)
        new_ot_flow0 = self.warping(pc1, pc0, sp2p_flow1)

        C0 = self.var_encoding(sp2p_flow0 - new_ot_flow0)
        C1 = self.var_encoding(sp2p_flow1 - new_ot_flow1)
        return sp2p_flow0, sp2p_flow1, C0, C1

    def obtain_clustering(self, pc0, pc1, feat0, feat1, ot_flow, ot_flow_back, flows0, flows1, pc0_ori, pc1_ori, graph0, graph1, \
        fused_xyz_0, fused_xyz_1, fused_feats_0, fused_feats_1, num_c):

        # obtain sp for source point cloud
        bs, npoint,_ = pc0.size()
        cluster_idx0, prototypes_src_xyz, prototypes_src_feat, prototypes_src_flow, prototypes_src_fused_xyz, prototypes_src_fused_feat = \
            self.down(pc0, feat0, fused_feats_0, fused_xyz_0, ot_flow, num_c)
        c2p_idx0, c2p_idx_abs0 = pointops.knnquerycluster(self.nc2p, prototypes_src_xyz, cluster_idx0, pc0)
        # c2p_idx: b x n x 6
        # c2p_idx_abs: b x n x 6

        ## obtain sp for target point cloud
        cluster_idx1, prototypes_tgt_xyz, prototypes_tgt_feat, prototypes_tgt_flow, prototypes_tgt_fused_xyz, prototypes_tgt_fused_feat = \
            self.down(pc1, feat1, fused_feats_1, fused_xyz_1, ot_flow_back, num_c)
        c2p_idx1, c2p_idx_abs1 = pointops.knnquerycluster(self.nc2p, prototypes_tgt_xyz, cluster_idx1, pc1)

        h0 = self.calc_h0(ot_flow, feat0, pc0, graph0) 
        h1 = self.calc_h0(ot_flow_back, feat1, pc1, graph1) 
        
        for iter in range(self.num_iters):
            a0, f0, z0, prototypes_src_feat, prototypes_src_xyz, prototypes_src_flow, sp_numc0 = self.learn_SLIC_calc_1(prototypes_src_feat, prototypes_src_xyz, prototypes_src_flow, \
                feat0, pc0, ot_flow, c2p_idx_abs0, c2p_idx0, cluster_idx0, prototypes_src_fused_xyz, prototypes_src_fused_feat, fused_xyz_0, fused_feats_0) # a [b,n,nc2p]# z [b,m,n] l1_norm/dim2

            a1, f1, z1, prototypes_tgt_feat, prototypes_tgt_xyz, prototypes_tgt_flow, sp_numc1 = self.learn_SLIC_calc_1(prototypes_tgt_feat, prototypes_tgt_xyz, prototypes_tgt_flow, \
                feat1, pc1, ot_flow_back, c2p_idx_abs1, c2p_idx1, cluster_idx1, prototypes_tgt_fused_xyz, prototypes_tgt_fused_feat, fused_xyz_1, fused_feats_1) # a [b,n,nc2p]# z [b,m,n] l1_norm/dim2
            
            sp2p_flow0, sp2p_flow1, C0, C1 = self.confidence_encoding(prototypes_src_flow, prototypes_tgt_flow, c2p_idx_abs0, c2p_idx_abs1, a0, a1, pc0, pc1)
            
            ot_flow, flow0_up, h0 = self.GRU_based_refinement(sp2p_flow0, C0, pc0, pc1, feat0, feat1, graph0, pc0_ori, h0, iter)
            flows0.append(flow0_up)
            ot_flow_back, flow1_up, h1 = self.GRU_based_refinement(sp2p_flow1, C1, pc1, pc0, feat1, feat0, graph1, pc1_ori, h1, iter)
            flows1.append(flow1_up)

            # at superpoint-level
            prototypes_sa_feats_0 = self.obtain_feat(prototypes_src_xyz, pc1, feat1, prototypes_src_flow)
            prototypes_sa_feats_1 = self.obtain_feat(prototypes_tgt_xyz, pc0, feat0, prototypes_tgt_flow)

            prototypes_src_fused_xyz, prototypes_tgt_fused_xyz, prototypes_src_fused_feat, prototypes_tgt_fused_feat = \
                self.feature_coordinate_space(prototypes_src_xyz, prototypes_tgt_xyz, prototypes_src_flow, prototypes_tgt_flow, \
                prototypes_src_feat, prototypes_tgt_feat, prototypes_sa_feats_0, prototypes_sa_feats_1)

            # at point-level
            sa_feats_0 = self.obtain_feat(pc0, pc1, feat1, ot_flow)
            sa_feats_1 = self.obtain_feat(pc1, pc0, feat0, ot_flow_back)
            fused_xyz_0, fused_xyz_1, fused_feats_0, fused_feats_1 = self.feature_coordinate_space(pc0, pc1, ot_flow, ot_flow_back, \
                feat0, feat1, sa_feats_0, sa_feats_1)

        return flows0, flows1
            
    def process(self, pc0, pc1, npoint, nsample):
        flows0 = [] 
        flows1 = []
        fps_idx_l0 =pointops.furthestsampling(pc0, npoint)
        pc0_d = index_points_gather(pc0, fps_idx_l0)
        fps_idx = pointops.furthestsampling(pc1, npoint)
        pc1_d = index_points_gather(pc1, fps_idx)
         
        feats_0, graph0 = self.get_features(pc0_d, nsample, None)#[b,n,c]
        feats_1, graph1 = self.get_features(pc1_d, nsample, None) #[b,m,c]
        
        # initial flow 
        sa_feats_0, ot_flow_forward, sa_feats_1, ot_flow_backward = self.get_flow(pc0_d, pc1_d, feats_0, feats_1)

        # obtain intial correspondences
        fused_xyz_0, fused_xyz_1, fused_feats_0, fused_feats_1 = self.feature_coordinate_space(pc0_d, pc1_d, ot_flow_forward, ot_flow_backward, \
            feats_0, feats_1, sa_feats_0, sa_feats_1)

        up_flow0 = self.upsample(pc0.permute(0, 2, 1), pc0_d.permute(0, 2, 1), ot_flow_forward.permute(0, 2, 1)).permute(0, 2, 1)
        flows0.append(up_flow0)
        up_flow1 = self.upsample(pc1.permute(0, 2, 1), pc1_d.permute(0, 2, 1), ot_flow_backward.permute(0, 2, 1)).permute(0, 2, 1)
        flows1.append(up_flow1)

        flows0, flows1 = self.obtain_clustering(pc0_d, pc1_d, feats_0, feats_1, ot_flow_forward, ot_flow_backward, \
            flows0, flows1, pc0, pc1, graph0, graph1, fused_xyz_0, fused_xyz_1, fused_feats_0, fused_feats_1, num_c = self.numc)
        return flows0, flows1
    
    def cal_consistency_loss(self, pc0, pc1, flows0, flows1, weight_const):
        loss_consistency_forward = 0.0
        loss_consistency_backward = 0.0
        for i in range(len(flows0)):
            ot_flow, ot_flow_back = flows0[i], flows1[i]
            new_ot_flow1 = self.warping(pc0, pc1, ot_flow)
            new_ot_flow0 = self.warping(pc1, pc0, ot_flow_back)
            loss_consistency_forward += (ot_flow - new_ot_flow0).norm(p=1, dim=-1).mean()
            loss_consistency_backward += (ot_flow_back - new_ot_flow1).norm(p=1, dim=-1).mean()
        loss = loss_consistency_forward + loss_consistency_backward
        return weight_const*loss

    def forward(self, pc0, pc1, feature0, feature1):
        (B, N_s, _) = pc0.size()
        npoint = N_s//self.down_scale
        nsample = 16
        flows0, flows1 = self.process(pc0, pc1, npoint, nsample)
        loss = self.cal_consistency_loss(pc0, pc1, flows0, flows1, self.weight_const)
        return flows0, loss
