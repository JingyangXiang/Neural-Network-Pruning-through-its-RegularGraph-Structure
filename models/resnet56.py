"""
https://github.com/lmbxmu/HRank/blob/master/models/resnet_cifar.py, we change its name to resnet56.py
"""
import torch.nn as nn
import torch.nn.functional as F

from args import args
from utils.builder import get_builder


def adapt_channel(num_layers):
    if num_layers==56:
        stage_repeat = [9, 9, 9]
        stage_out_channel = [16] + [16] * 9 + [32] * 9 + [64] * 9
    elif num_layers==110:
        stage_repeat = [18, 18, 18]
        stage_out_channel = [16] + [16] * 18 + [32] * 18 + [64] * 18

    stage_oup_cprate = []
    stage_oup_cprate += [0]
    for i in range(len(stage_repeat)-1):
        stage_oup_cprate += [0] * stage_repeat[i]
    stage_oup_cprate +=[0.] * stage_repeat[-1]

    overall_channel = []
    mid_channel = []
    for i in range(len(stage_out_channel)):
        if i == 0 :
            overall_channel += [int(stage_out_channel[i])]
        else:
            overall_channel += [int(stage_out_channel[i])]
            mid_channel += [int(stage_out_channel[i])]

    return overall_channel, mid_channel


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, midplanes, inplanes, planes, stride=1, builder=None):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = builder.conv3x3(inplanes, midplanes, stride)
        self.bn1 = builder.norm(midplanes)
        self.relu1 = builder.activation()

        self.conv2 = builder.conv3x3(midplanes, planes)
        self.bn2 = builder.norm(planes)
        self.relu2 = builder.activation()
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if stride!=1:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))
            else:
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, :, :],
                                    (0, 0, 0, 0, (planes-inplanes)//2, planes-inplanes-(planes-inplanes)//2), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_layers, builder):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6

        self.num_layer = num_layers
        self.overall_channel, self.mid_channel = adapt_channel(num_layers)

        self.layer_num = 0
        self.conv1 = nn.Conv2d(3, self.overall_channel[self.layer_num], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = builder.norm(self.overall_channel[self.layer_num])
        self.relu = builder.activation()
        self.layers = nn.ModuleList()
        self.layer_num += 1

        self.layer1 = self._make_layer(block, blocks_num=n, stride=1, builder=builder)
        self.layer2 = self._make_layer(block, blocks_num=n, stride=2, builder=builder)
        self.layer3 = self._make_layer(block, blocks_num=n, stride=2, builder=builder)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.num_layer == 56:
            self.fc = nn.Linear(64 * BasicBlock.expansion, args.num_classes)
        else:
            self.linear = nn.Linear(64 * BasicBlock.expansion, args.num_classes)


    def _make_layer(self, block, blocks_num, stride, builder):
        layers = []
        layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                 self.overall_channel[self.layer_num], stride, builder=builder))
        self.layer_num += 1

        for i in range(1, blocks_num):
            layers.append(block(self.mid_channel[self.layer_num - 1], self.overall_channel[self.layer_num - 1],
                                     self.overall_channel[self.layer_num],builder=builder))
            self.layer_num += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for i, block in enumerate(self.layer1):
            x = block(x)
        for i, block in enumerate(self.layer2):
            x = block(x)
        for i, block in enumerate(self.layer3):
            x = block(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.num_layer == 56:
            x = self.fc(x)
        else:
            x = self.linear(x)

        return x


def resnet56():
    return ResNet(BasicBlock, 56, builder=get_builder())


def resnet110():
    return ResNet(BasicBlock, 110, builder=get_builder())
