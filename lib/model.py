from math import gamma
from re import L
from .layers import *
import torch
import torch.nn as nn
import torch.nn.functional as Func
import numpy as np

class TrustGeo(nn.Module):
    def __init__(self, dim_in):
        super(TrustGeo, self).__init__()
        self.dim_in = dim_in
        self.dim_z = dim_in + 2

        # TrustGeo
        self.att_attribute = SimpleAttention(temperature=self.dim_z ** 0.5,
                                             d_q_in=self.dim_in,
                                             d_k_in=self.dim_in,
                                             d_v_in=self.dim_in + 2,
                                             d_q_out=self.dim_z,
                                             d_k_out=self.dim_z,
                                             d_v_out=self.dim_z)


        # calculate A
        self.gamma_1 = nn.Parameter(torch.ones(1, 1))
        self.gamma_2 = nn.Parameter(torch.ones(1, 1))
        self.gamma_3 = nn.Parameter(torch.ones(1, 1))
        self.alpha = nn.Parameter(torch.ones(1, 1))
        self.beta = nn.Parameter(torch.zeros(1, 1))

        # transform in Graph
        self.w_1 = nn.Linear(self.dim_in + 2, self.dim_in + 2)
        self.w_2 = nn.Linear(self.dim_in + 2, self.dim_in + 2)


        # higher-order evidence
        # graph view 
        self.out_layer_graph_view = nn.Linear(self.dim_z*2, 5)
        # attribute view 
        self.out_layer_attri_view = nn.Linear(self.dim_in, 5)
    

    # for output mu, v, alpha, beta
    def evidence(self, x):
        return Func.softplus(x)

    def trans(self, gamma1, gamma2, logv, logalpha, logbeta):
        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)
        return gamma1, gamma2, v, alpha, beta
    

    def forward(self, lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, add_noise=0):
        """
        :param lm_X: feature of landmarks [..., 30]: 14 attribute + 16 measurement
        :param lm_Y: location of landmarks [..., 2]: longitude + latitude
        :param tg_X: feature of targets [..., 30]
        :param tg_Y: location of targets [..., 2]
        :param lm_delay: delay from landmark to the common router [..., 1]
        :param tg_delay: delay from target to the common router [..., 1]
        :return:
        """

  

        N1 = lm_Y.size(0)
        N2 = tg_Y.size(0)
        ones = torch.ones(N1 + N2 + 1).cuda()
        lm_feature = torch.cat((lm_X, lm_Y), dim=1)
        tg_feature_0 = torch.cat((tg_X, torch.zeros(N2, 2).cuda()), dim=1)
        router_0 = torch.mean(lm_feature, dim=0, keepdim=True)
        all_feature_0 = torch.cat((lm_feature, tg_feature_0, router_0), dim=0)

        '''
        star-GNN
        properties:
        1. single directed graph: feature of <landmarks> will never be updated.
        2. the target IP will receive from surrounding landmarks from two ways: 
            (1) attribute similarity-based one-hop propagation;
            (2) delay measurement-based two-hop propagation via the common router;
        '''
        # GNN-step 1
        adj_matrix_0 = torch.diag(ones)

        # star connections (measurement)
        delay_score = torch.exp(-self.gamma_1 * (self.alpha * lm_delay + self.beta))

        rou2tar_score_0 = torch.exp(-self.gamma_2 * (self.alpha * tg_delay + self.beta)).reshape(N2)

        # satellite connections (feature)
        _, attribute_score = self.att_attribute(tg_X, lm_X, lm_feature)
        attribute_score = torch.exp(attribute_score)

        adj_matrix_0[N1:N1 + N2, :N1] = attribute_score
        adj_matrix_0[-1, :N1] = delay_score
        adj_matrix_0[N1:N1 + N2:, -1] = rou2tar_score_0

        degree_0 = torch.sum(adj_matrix_0, dim=1)
        degree_reverse_0 = 1.0 / (degree_0 + 1e-12)
        degree_matrix_reverse_0 = torch.diag(degree_reverse_0)

        degree_mul_adj_0 = degree_matrix_reverse_0 @ adj_matrix_0
        step_1_all_feature = self.w_1(degree_mul_adj_0 @ all_feature_0)

        tg_feature_1 = step_1_all_feature[N1:N1 + N2, :]
        router_1 = step_1_all_feature[-1, :].reshape(1, -1)

        # GNN-step 2
        adj_matrix_1 = torch.diag(ones)
        rou2tar_score_1 = torch.exp(-self.gamma_3 * (self.alpha * tg_delay + self.beta)).reshape(N2)
        adj_matrix_1[N1:N1 + N2:, -1] = rou2tar_score_1

        all_feature_1 = torch.cat((lm_feature, tg_feature_1, router_1), dim=0)

        degree_1 = torch.sum(adj_matrix_1, dim=1)
        degree_reverse_1 = 1.0 / (degree_1 + 1e-12)
        degree_matrix_reverse_1 = torch.diag(degree_reverse_1)

        degree_mul_adj_1 = degree_matrix_reverse_1 @ adj_matrix_1
        step_2_all_feature = self.w_2(degree_mul_adj_1 @ all_feature_1)
        tg_feature_2 = step_2_all_feature[N1:N1 + N2, :]

        # graph view
        tg_feature_graph_view = torch.cat((
                                      tg_feature_1,
                                      tg_feature_2), dim=-1)
        # attribute view (for shanghai dim=51) 
        tg_feature_attribute_view = tg_X
        
        '''
        predict
        '''
        output1 = self.out_layer_graph_view(tg_feature_graph_view)
        gamma1_g, gamma2_g, v_g, alpha_g, beta_g = torch.split(output1, 1, dim=-1)
        # attribute
        output2 = self.out_layer_attri_view(tg_feature_attribute_view)
        gamma1_a, gamma2_a, v_a, alpha_a, beta_a = torch.split(output2, 1, dim=-1)
    
        # transform, let v>0, aplha>1, beta>0 
        gamma1_g, gamma2_g, v_g, alpha_g, beta_g = self.trans(gamma1_g, gamma2_g, v_g, alpha_g, beta_g)
        gamma1_a, gamma2_a, v_a, alpha_a, beta_a = self.trans(gamma1_a, gamma2_a, v_a, alpha_a, beta_a)
        
        two_gamma_g = torch.cat((gamma1_g, gamma2_g), dim=1)
        two_gamma_a = torch.cat((gamma1_a, gamma2_a), dim=1)
        
        return two_gamma_g, v_g, alpha_g, beta_g, \
               two_gamma_a, v_a, alpha_a, beta_a
