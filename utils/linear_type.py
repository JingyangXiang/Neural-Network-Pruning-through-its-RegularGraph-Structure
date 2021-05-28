import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from args import args as parser_args


DenseLinear = nn.Linear


"""
Graph subnets 
"""

class GraphLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 确保节点的数量是小于等于min(in_channels,out_channels)
        # print(parser_args.nodes,self.in_channels,self.out_channels)
        self.in_channels = self.in_features
        self.out_channels = self.out_features
        if parser_args.nodes <= min(self.in_channels, self.out_channels):
            self.dense = True
        else:
            self.dense = False
        self.scores = np.random.randn(parser_args.nodes, parser_args.nodes)
        self.scores = self.compute_densemask(self.scores)


    def forward(self, x):

        # print(self.weight.shape, self.scores.shape)
        if not self.dense:
            x = F.linear(
                x, self.weight, self.bias
            )
            return x

        w = self.weight * self.scores
        x = F.linear(
            x, w, self.bias
        )
        return x

    def compute_densemask(self, scores):

        nodes = scores.shape[0]
        repeat_in = self.compute_size(self.in_channels, nodes)
        repeat_out = self.compute_size(self.out_channels, nodes)
        scores = np.repeat(scores, repeat_in, axis=1)
        scores = np.repeat(scores, repeat_out, axis=0)

        return nn.Parameter(torch.Tensor(scores))

    def compute_size(self, channel, nodes, seed=1):

        # 计算分配的节点数量
        np.random.seed(seed)
        divide = channel // nodes
        remain = channel % nodes
        out = np.zeros(nodes, dtype=int)
        out[:remain] = divide + 1
        out[remain:] = divide
        out = np.random.permutation(out)

        return out

    def set_connection(self, scores):
        # 确保原来的矩阵和现在的矩阵的节点数量是一致的
        # 不知道加了有没有用, 反正先限制一下再说
        # assert matrix.shape[0] == self.matrix.shape[0]
        self.scores = self.compute_densemask(scores)
