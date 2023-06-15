import torch.nn as nn
import torch
from .sublayers import MultiHeadAttention, PositionwiseFeedForward
import numpy as np
import torch.functional as F


class SimpleAttention(nn.Module):
    ''' Just follow GraphGeo '''

    def __init__(self, temperature, attn_dropout=0.1, d_q_in=32, d_q_out=32, d_k_in=32, d_k_out=32, d_v_in=32,
                 d_v_out=32, dropout=0.1, drop_last_layer=False):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.q_w = nn.Linear(d_q_in, d_q_out)
        self.k_w = nn.Linear(d_k_in, d_k_out)
        if not drop_last_layer:
            self.v_w = nn.Linear(d_v_in, d_v_out)
        self.drop_last_layer = drop_last_layer

        # self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        q = self.q_w(q)
        k = self.k_w(k)
        if not self.drop_last_layer:
            v = self.v_w(v)
        att_score = (q / self.temperature) @ k.transpose(0, 1)
        att_weight = torch.softmax(att_score, dim=-1)
        output = att_weight @ v
        return output, att_score


class VanillaAttention(nn.Module):
    ''' Just follow GraphGeo '''

    def __init__(self, temperature, attn_dropout=0.1, d_q_in=32, d_q_out=32, d_k_in=32, d_k_out=32, d_v_in=32,
                 d_v_out=32, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        assert d_q_out == d_k_out
        self.dropout = nn.Dropout(attn_dropout)
        self.q_w = nn.Linear(d_q_in, d_q_out)
        self.k_w = nn.Linear(d_k_in, d_k_out)
        # self.v_w = nn.Linear(d_v_in, d_v_out)
        self.w = nn.Linear(d_q_out, 1)
        self.sigma = nn.Sigmoid()
        # self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        '''
        x' = w3*sigma(w1x1+w2x2)
        q:[N1, d]
        k:[N2, d]
        v:[N2, d]
        '''
        N1 = q.size(0)
        N2 = k.size(0)
        q = self.q_w(q).reshape(N1, 1, -1)
        k = self.k_w(k).reshape(1, N2, -1)
        att_score = self.w(self.sigma(q + k)).reshape(N1, N2)
        att_weight = torch.softmax(att_score, dim=-1)
        output = att_weight @ v

        return output, att_score


class Similarity(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, query, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            query, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class MultiLayerPerceptron(torch.nn.Module):
    """
    Class to instantiate a Multilayer Perceptron model
    """

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        self.output_layer = output_layer
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        shape0 = x.shape[0]
        shape1 = x.shape[1]
        x = x.reshape((-1, x.shape[-1]))
        x = self.mlp(x)
        if self.output_layer:
            x = x.reshape((shape0, shape1))
        else:
            x = x.reshape((shape0, shape1, -1))
        return x
