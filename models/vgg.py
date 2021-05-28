import torch.nn as nn
from utils.builder import get_builder
from args import args


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, builder, features):
        super(VGG, self).__init__()

        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            builder.conv1x1(512, 512),
            builder.activation(),
            nn.Dropout(),
            builder.conv1x1(512, 512),
            builder.activation(),
        )
        if args.last_layer_dense:
            self.classifier.add_module('classifier', nn.Conv2d(512, args.num_classes, 1))
        else:
            self.classifier.add_module('classifier', builder.conv1x1(512, args.num_classes))

         # Initialize weights

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x).view(x.size(0), -1)
        return x


def make_layers(cfg, builder, batch_norm=False):
    layers = []
    in_channels = 3
    for index, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if args.first_layer_dense and index==0:
                conv2d = nn.Conv2d(in_channels, v, 3, 1, 1, bias=False)
            else:
                conv2d = builder.conv3x3(in_channels, v)
            if batch_norm:
                layers += [conv2d, builder.norm(v), builder.activation()]
            else:
                layers += [conv2d, builder.activation()]
            in_channels = v
    layers += [nn.AdaptiveAvgPool2d(1)]
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(get_builder(), make_layers(cfg['A'], get_builder()))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(get_builder(), make_layers(cfg['A'], get_builder(), batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(get_builder(), make_layers(cfg['B'], get_builder()))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(get_builder(), make_layers(cfg['B'], get_builder(), batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(get_builder(), make_layers(cfg['D'], get_builder()))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(get_builder(),make_layers(cfg['D'], get_builder(), batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(get_builder(), make_layers(cfg['E'], get_builder()))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(get_builder(), make_layers(cfg['E'], get_builder(), batch_norm=True))


# if __name__ == '__main__':
#     args.conv_type = "GraphConv2D"
#     args.bn_type = "NonAffineBatchNorm"
#     args.nodes = 16
#     args.first_layer_dense = True
#     args.last_layer_dense = True
#     model = vgg19()
#     data = torch.randn(1, 3, 224, 224)
#     out = model(data)
#     print(out.shape)