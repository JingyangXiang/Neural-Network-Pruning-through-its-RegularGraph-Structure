import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from args import args as parser_args


DenseLinear = nn.Linear


class GraphLinear(nn.Linear):
    """GraphMap Linear"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert parser_args.nodes <= min(self.in_channels, self.out_channels)
        self.scores = np.random.randn(parser_args.nodes, parser_args.nodes)
        self.scores = self.compute_densemask(self.scores)

    def forward(self, x):
        w = self.weight * self.scores
        x = F.linear(
            x, w, self.bias
        )

        return x

    def compute_densemask(self, scores):
        nodes = scores.shape[0]
        repeat_in = self.compute_size(self.in_features, nodes)
        repeat_out = self.compute_size(self.out_features, nodes)
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
        self.scores = self.compute_densemask(scores)
