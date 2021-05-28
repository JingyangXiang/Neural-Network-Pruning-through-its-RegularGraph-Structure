import math
import torch
import torch.nn as nn
import torch.nn.init as init
from utils.builder import get_builder
from args import args
import models

class MLP(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, builder=get_builder()):
        super(MLP, self).__init__()

        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(32*32*3, 512, 1, bias=False)
        else:
            self.conv1 = builder.conv1x1(32*32*3, 512)
        self.features = nn.Sequential(
            builder.norm(512),
            builder.activation(),
            builder.conv1x1(512, 512),
            builder.norm(512),
            builder.activation(),
            builder.conv1x1(512, 512),
            builder.norm(512),
            builder.activation(),
            builder.conv1x1(512, 512),
            builder.norm(512),
            builder.activation(),
        )
        if args.last_layer_dense:
            self.features.add_module('classifier', nn.Conv2d(512, args.num_classes, 1))
        else:
            self.features.add_module('classifier', builder.conv1x1(512, args.num_classes))

         # Initialize weights

    def forward(self, x):
        x = x.view(x.size(0), -1, 1, 1)
        x = self.conv1(x)
        x = self.features(x).view(x.size(0), -1)
        return x

