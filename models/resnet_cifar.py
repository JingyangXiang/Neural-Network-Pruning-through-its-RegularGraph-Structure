"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F

from args import args
from utils.builder import get_builder


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, builder, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = builder.conv3x3(in_planes, planes, stride=stride)
        self.bn1 = builder.norm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=1)
        self.bn2 = builder.norm(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.norm(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, builder, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = builder.conv1x1(in_planes, planes)
        self.bn1 = builder.norm(planes)
        self.conv2 = builder.conv3x3(planes, planes, stride=stride)
        self.bn2 = builder.norm(planes)
        self.conv3 = builder.conv1x1(planes, self.expansion * planes)
        self.bn3 = builder.norm(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                builder.conv1x1(in_planes, self.expansion * planes, stride=stride),
                builder.norm(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, builder, block, num_blocks, base_width=64):
        super(ResNet, self).__init__()
        self.in_planes = base_width
        self.builder = builder
        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(3, base_width, 3, 1, 1, bias=False)
        else:
            self.conv1 = builder.conv3x3(3, base_width, stride=1)
        self.bn1 = builder.norm(base_width)
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if args.last_layer_dense:
            self.fc = nn.Conv2d(base_width*8 * block.expansion, args.num_classes, 1)
        else:
            self.fc = builder.conv1x1(base_width*8 * block.expansion, args.num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.builder, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.fc(out)

        return out.flatten(1)


def cResNet18():
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2])


def cResNet34():
    return ResNet(get_builder(), BasicBlock, [3, 4, 6, 3])


def cResNet50():
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3])


def cResNet101():
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3])


def cResNet152():
    return ResNet(get_builder(), Bottleneck, [3, 8, 36, 3])
