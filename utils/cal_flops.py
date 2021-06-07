# -*- coding:utf8 -*-
import os
import time
import pathlib
import numpy as np

import torch

from args import args
args.conv_type = "DenseConv"
args.norm_type = "NonAffineBatchNorm"


import models



def print_model_param_flops(model=None, input_res=224, multiply_adds=True,nodes=64,degree=64):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))
    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)


    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size

        list_conv.append(flops*degree/nodes)


    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = torch.rand(1, 3, input_res, input_res).cuda()
    out = model(input)


    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    print('  + Number of FLOPs: %.2fG' % (total_flops / 1e9))
    return total_flops


def write_to_csv(**kwargs):
    if not os.path.exists("Flops_Params"):
        os.makedirs("Flops_Params")
    results = pathlib.Path("Flops_Params") / "Flops_and_Params.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "data, "
            "num_classes, "
            "model_name, "
            "degree, "
            "nodes, "
            "parameter, "
            "flops, "
            "size\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (   "{now}, "
                "{data}, "
                "{num_classes}, "
                "{model_name}, "
                "{degree}, "
                "{nodes}, "
                "{parameter}M, "
                "{flops}M, "
                "{size}\n"
            ).format(now=now, **kwargs)
        )


def get_class(name):

    if name in ['CIFAR10', 'SVHN']:
        classes = 10
    elif name in ['CIFAR100', ]:
        classes = 100
    elif name in ['TinyImageNet', ]:
        classes = 200
    elif name in ['ImageNet', ]:
        classes = 1000
    else:
        raise ValueError('true name needed')

    return classes


def get_size(name):
    if name in ['CIFAR10', 'SVHN', 'CIFAR100']:
        size = 32
    elif name in ['TinyImageNet','ImageNet']:
        size = 224
    return size


datas = ['CIFAR10', 'SVHN', 'CIFAR100']
models_name = ["cResNet18", "vgg16_bn"]
degrees = [4, 6, 8, 10, 12, 14, 16, 18, 20, 64]
nodes = 64

for data in datas:
    args.num_classes = get_class(data)
    for model_name in models_name:
        model = models.__dict__[model_name]()
        model.cuda()
        for degree in degrees:
            size = get_size(data)
            flops = print_model_param_flops(model, size, degree=degree, nodes=nodes)
            model_parameters = sum(
                int(p.numel() * degree / nodes) for n, p in model.named_parameters() if not n.endswith('scores'))
            write_to_csv(
                data=data,
                num_classes=args.num_classes,
                model_name=model_name,
                degree=degree,
                nodes=nodes,
                parameter=round(model_parameters * 1e-6, 2),
                flops=round(flops * 1e-6, 2),
                size=size
            )


data2s = ['TinyImageNet','ImageNet']
models_name = ["ResNet18", "vgg16_bn", "ResNet50"]
for data2 in data2s:
    args.num_classes = get_class(data2)
    for model_name in models_name:
        model = models.__dict__[model_name]()
        model.cuda()
        for degree in degrees:
            size = get_size(data2)
            flops = print_model_param_flops(model, size, degree=degree, nodes=nodes)
            model_parameters = sum(
                int(p.numel() * degree / nodes) for n, p in model.named_parameters() if not n.endswith('scores'))
            write_to_csv(
                data=data2,
                num_classes=args.num_classes,
                model_name=model_name,
                degree=degree,
                nodes=nodes,
                parameter=round(model_parameters*1e-6,2),
                flops=round(flops*1e-6,2),
                size=size
            )


datas = ['CIFAR10', 'SVHN', 'CIFAR100']
models_name = ["resnet56", ]
degrees = [4,6,8,10,12,16]

nodes = 16

for data in datas:
    args.num_classes = get_class(data)
    for model_name in models_name:
        model = models.__dict__[model_name]()
        model.cuda()
        for degree in degrees:
            size = get_size(data)
            flops = print_model_param_flops(model, size, degree=degree, nodes=nodes)
            model_parameters = sum(
                int(p.numel() * degree / nodes) for n, p in model.named_parameters() if not n.endswith('scores'))
            write_to_csv(
                data=data,
                num_classes=args.num_classes,
                model_name=model_name,
                degree=degree,
                nodes=nodes,
                parameter=round(model_parameters * 1e-6, 2),
                flops=round(flops * 1e-6, 2),
                size=size
            )
