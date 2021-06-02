import os
import pathlib
import random
import time
import importlib
import networkx as nx
import copy
import numpy as np
from ruamel import yaml

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import data
import models
from args import args
from utils.conv_type import FixedSubnetConv, SampleSubnetConv
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    freeze_model_subnet,
    set_model_connection
)
from utils.schedulers import get_policy
from graph.build_graph  import first_regular_graph
from graph.optimizer import PathOptimizer


def main():
    print(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    args.gpu = None
    train, validate, modifier = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model and optimizer
    model = get_model(args)
    model = set_gpu(args, model)

    if args.pretrained:
        pretrained(args, model)

    optimizer = get_optimizer(args, model)
    data = get_dataset(args)
    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )

        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir

    # save a yaml file to read to record parameters
    args_text = copy.copy(args.__dict__)
    del args_text['ckpt_base_dir']
    del args_text['matrix']
    with open(run_base_dir/'args.yaml', 'w', encoding="utf-8") as f:
        yaml.dump(
            args_text,
            f,
            Dumper=yaml.RoundTripDumper,
            default_flow_style=False,
            allow_unicode=True,
            indent=4
        )

    # create recorder
    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state
    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # Start training
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        lr_policy(epoch, iteration=None)
        modifier(args, epoch, model)

        cur_lr = get_lr(optimizer)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()
        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)
        validation_time.update((time.time() - start_validation) / 60)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0
        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")
                best_time = time.time() - start_time
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

        epoch_time.update((time.time() - end_epoch) / 60)
        progress_overall.display(epoch)
        progress_overall.write_to_tensorboard(
            writer, prefix="diagnostics", global_step=epoch
        )

        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()
    if args.conv_type == 'DenseConv':
        args.matrix = np.ones([args.nodes,args.nodes],dtype=np.int32)
    try:
        graph = nx.from_numpy_matrix(args.matrix)
        # 记录单次结束的平均最短路径
        path = nx.average_shortest_path_length(graph)
        clustering = nx.average_clustering(graph)
        tranvity = nx.transitivity(graph)
    except:
        print(f'=========warning=============')
        path = 0.
        clustering = 0.
        tranvity = 0.



    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
        conv_type=args.conv_type,
        norm_type=args.norm_type,
        degree=np.sum(args.matrix)/args.matrix.shape[0],
        path=path,
        clustering=clustering,
        tranvity=tranvity,
        arch=args.arch,
        freeze_subnet=args.freeze_subnet,
        freeze_weights=args.freeze_weights,
        trainer=args.trainer,
        model_parameters=model_parameters,
        best_time=best_time,
        batch_size=args.batch_size,
        model_optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        set=args.set,
        run_base_dir=run_base_dir,
        linear_type=args.linear_type
    )


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier


def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    elif args.multigpu is None:
        device = torch.device("cpu")
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        print(f"=> Parallelizing on {args.multigpu} gpus")
        torch.cuda.set_device(args.multigpu[0])
        args.gpu = args.multigpu[0]
        model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
            args.multigpu[0]
        )

    cudnn.benchmark = True

    return model


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume, map_location=f"cuda:{args.multigpu[0]}")
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]

        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))

    for n, m in model.named_modules():
        if isinstance(m, FixedSubnetConv):
            m.set_subnet()


def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"

    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # applying sparsity to the network

    # freezing the weights if we are only doing subnet training
    if args.freeze_weights:
        freeze_model_weights(model)

    if os.path.isfile(args.matrix):
        print('==> load the matrix')
        args.matrix = np.load(args.matrix)
    elif args.conv_type == 'DenseConv':
        pass
    else:
        raise ValueError("Matrix is need to be given")

    if args.set_connection:
        set_model_connection(model, args.matrix)

    if args.freeze_subnet:
        freeze_model_subnet(model)

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    elif args.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    elif args.optimizer == "Adagrad":
        optimizer = torch.optim.Adagrad(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            weight_decay = args.weight_decay
        )
    elif args.optimizer == "SparseAdam":
        optimizer = torch.optim.SparseAdam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        )
    elif args.optimizer == "Adamax":
        optimizer = torch.optim.Adamax(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            weight_decay=args.weight_decay
        )

    return optimizer


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Set, "
            "Batch Size, "
            "Base Config, "
            "Name, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5, "
            "Best Time, "
            "Freeze Weights, "
            "Freeze Subnet, "
            "Conv Type, "
            "Bn Type, "
            "Degree, "
            "Path Long, "
            "Cluster, "
            "Tranvity, "
            "Arch, "
            "Trainer, "
            "Parameters, "
            "Optimizers, "
            "WeightDecay, "
            "Run Base Dir, "
            "Arch, , "
            "Linear Type\n"

        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (   "{now}, "
                "{set}, "
                "{batch_size}, "
                "{base_config}, "
                "{name}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}, "
                "{best_time:.02f}, "
                "{freeze_weights}, "
                "{freeze_subnet}, "
                "{conv_type}, "
                "{norm_type}, "
                "{degree}, "
                "{path}, "
                "{clustering}, "
                "{tranvity}, "
                "{arch}, "
                "{trainer}, "
                "{model_parameters}, "
                "{model_optimizer}, "
                "{weight_decay}, "
                "{run_base_dir},"
                "{linear_type}\n"
            ).format(now=now, **kwargs)
        )

if __name__ == "__main__":
    main()
