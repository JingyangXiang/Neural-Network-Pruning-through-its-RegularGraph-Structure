import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from args import args as parser_args


DenseConv = nn.Conv2d


class GraphConv2D(nn.Conv2d):
    """GraphMap Conv2D"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # make use min(in_channels, out_channels) <= nodes
        assert parser_args.nodes <= min(self.in_channels, self.out_channels)
        self.scores = np.random.randn(parser_args.nodes, parser_args.nodes, 1, 1)
        self.scores = self.compute_densemask(self.scores)

    def forward(self, x):
        w = self.weight * self.scores
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
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
        np.random.seed(seed)
        divide = channel // nodes
        remain = channel % nodes
        out = np.zeros(nodes, dtype=int)
        out[:remain] = divide + 1
        out[remain:] = divide
        out = np.random.permutation(out)

        return out

    def set_connection(self, scores):
        scores = scores[...,np.newaxis, np.newaxis]
        self.scores = self.compute_densemask(scores)
