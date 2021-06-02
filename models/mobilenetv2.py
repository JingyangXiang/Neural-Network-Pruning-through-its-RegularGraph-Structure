import torch.nn as nn
import torch.nn.functional as F

from args import args
from utils.builder import get_builder


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, builder, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = builder.conv1x1(in_planes, planes)
        self.bn1 = builder.norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, groups=planes,
                               bias=False)
        self.bn2 = builder.norm(planes)
        self.conv3 = builder.conv1x1(planes, out_planes)
        self.bn3 = builder.norm(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, out_planes),
                builder.norm(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNet(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, builder):
        super(MobileNet, self).__init__()
        self.builder = builder
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        else:
            self.conv1 = builder.conv1x1(3, 32)
        self.bn1 = builder.norm(32)
        self.relu1 = builder.activation()
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = builder.conv1x1(320, 1280)
        self.bn2 = builder.norm(1280)
        self.relu2 = builder.activation()
        if args.last_layer_dense:
            self.classifier = nn.Conv2d(1280, args.num_classes, 1)
        else:
            self.classifier = builder.conv1x1(1280, args.num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(
                    Block(self.builder, in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.relu2(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1, 1, 1)
        out = self.classifier(out)
        return out.squeeze()


def MobileNetV2():
    return MobileNet(get_builder())
