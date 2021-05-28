"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.builder import get_builder
from args import args
import models

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
    def __init__(self, builder, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.builder = builder
        if args.first_layer_dense:
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        else:
            self.conv1 = builder.conv3x3(3, 64, stride=1, first_layer=True)
        self.bn1 = builder.norm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if args.last_layer_dense:
            self.fc1 = nn.Conv2d(512 * block.expansion, 2, 1)
            self.fc = nn.Conv2d(2, args.num_classes, 1)
        else:
            self.fc = builder.conv1x1(512 * block.expansion, args.num_classes)

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
        out = self.fc1(out)
        out = self.fc(out)
        return out.flatten(1)


def cResNet18v():
    return ResNet(get_builder(), BasicBlock, [2, 2, 2, 2])


def cResNet34v():
    return ResNet(get_builder(), BasicBlock, [3, 4, 6, 3])


def cResNet50v():
    return ResNet(get_builder(), Bottleneck, [3, 4, 6, 3])


def cResNet101v():
    return ResNet(get_builder(), Bottleneck, [3, 4, 23, 3])


def cResNet152v():
    return ResNet(get_builder(), Bottleneck, [3, 8, 36, 3])

# if __name__ == '__main__':
#     args.conv_type = "GraphConv2D"
#     args.bn_type = "NonAffineBatchNorm"
#     args.nodes = 16
#     args.first_layer_dense = True
#     args.last_layer_dense = True
#     model = cResNet18()
#     data = torch.randn(1,3,224,224)
#     out = model(data)
#     print(out.shape)