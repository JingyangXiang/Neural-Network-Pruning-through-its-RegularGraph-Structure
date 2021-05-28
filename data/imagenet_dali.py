import time
import torch.utils.data
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision.datasets as datasets
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
from args import args

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.readers.File(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True)
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.random.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.readers.File(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size,
                                    random_shuffle=False)
        self.decode = ops.decoders.Image(device="mixed")
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                           world_size=1, local_rank=0):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                    data_dir=image_dir + '/imagenet/train',
                                    crop=crop, world_size=world_size, local_rank=local_rank)
        pip_train.build()
        print(f'pip_train.epoch_size("Reader"):{pip_train.epoch_size("Reader")}')
        dali_iter_train = DALIClassificationIterator(
            pip_train,
            size=pip_train.epoch_size("Reader") // world_size,
            # reader_name='train'
        )
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                                data_dir=image_dir + '/imagenet/val',
                                crop=crop, size=val_size, world_size=world_size, local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(
            pip_val,
            size=pip_val.epoch_size("Reader") // world_size,
            # reader_name='val'
        )
        return dali_iter_val


def get_imagenet_iter_torch(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                            world_size=1, local_rank=0):
    if type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/imagenet/train', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
                                                 pin_memory=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/imagenet/val', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads,
                                                 pin_memory=True)
    return dataloader

class ImageNetDali:
    def __init__(self, args):
        super(ImageNetDali, self).__init__()
        self.train_loader = get_imagenet_iter_dali(
            type='train',
            image_dir=args.data,
            batch_size=args.batch_size,
            num_threads=args.workers,
            crop=224,
            device_id=0,
            num_gpus=1
        )
        self.val_loader = get_imagenet_iter_dali(
            type='val',
            image_dir=args.data,
            batch_size=args.batch_size,
            num_threads=args.workers,
            crop=224,
            device_id=0,
            num_gpus=1
        )


# if __name__ == '__main__':
#     train_loader = get_imagenet_iter_dali(type='train', image_dir='/public/xjy2/ImageProject/data/ImageNet', batch_size=256,
#                                           num_threads=4, crop=224, device_id=0, num_gpus=1)
#     print('start iterate')
#     start = time.time()
#     for i, data in enumerate(train_loader):
#         images = data[0]["data"].cuda(non_blocking=True)
#         labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
#     end = time.time()
#     print('end iterate')
#     print('dali iterate time: %fs' % (end - start))
#
#     train_loader = get_imagenet_iter_torch(type='train', image_dir='/public/xjy2/ImageProject/data/ImageNet', batch_size=256,
#                                            num_threads=4, crop=224, device_id=0, num_gpus=1)
#     print('start iterate')
#     start = time.time()
#     for i, data in enumerate(train_loader):
#         images = data[0].cuda(non_blocking=True)
#         labels = data[1].cuda(non_blocking=True)
#     end = time.time()
#     print('end iterate')
#     print('torch iterate time: %fs' % (end - start))