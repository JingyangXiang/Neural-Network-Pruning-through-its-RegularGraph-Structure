import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

from args import args as parser_args


DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int(k * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def forward(self, x):
        subnet = GetSubnet.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Sample Based Sparsification
"""


class StraightThroughBinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None


class BinomialSample(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        ctx.save_for_backward(output)

        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        subnet, = ctx.saved_variables

        grad_inputs = grad_outputs.clone()
        grad_inputs[subnet == 0.0] = 0.0

        return grad_inputs, None


# Not learning weights, finding subnet
class SampleSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        return torch.sigmoid(self.scores)

    def forward(self, x):
        subnet = StraightThroughBinomialSample.apply(self.clamped_scores)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x


"""
Fixed subnets 
"""


class FixedSubnetConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate
        print("prune_rate_{}".format(self.prune_rate))

    def set_subnet(self):
        output = self.clamped_scores().clone()
        _, idx = self.clamped_scores().flatten().abs().sort()
        p = int(self.prune_rate * self.clamped_scores().numel())
        flat_oup = output.flatten()
        flat_oup[idx[:p]] = 0
        flat_oup[idx[p:]] = 1
        self.scores = torch.nn.Parameter(output)
        self.scores.requires_grad = False

    def clamped_scores(self):
        return self.scores.abs()

    def get_subnet(self):
        return self.weight * self.scores

    def forward(self, x):
        w = self.get_subnet()
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


"""
Graph subnets 
"""


class GraphConv2D(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 确保节点的数量是小于等于min(in_channels,out_channels)
        # print(parser_args.nodes,self.in_channels,self.out_channels)
        assert parser_args.nodes <= min(self.in_channels, self.out_channels)
        self.scores = np.random.randn(parser_args.nodes, parser_args.nodes, 1, 1)
        self.scores = self.compute_densemask(self.scores)


    def forward(self, x):

        # print(self.weight.shape, self.scores.shape)
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
        scores = scores[...,np.newaxis, np.newaxis]
        self.scores = self.compute_densemask(scores)
