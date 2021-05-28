import torch.nn as nn

LearnedBatchNorm = nn.BatchNorm2d
LearnedLayerNorm = nn.LayerNorm

class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)

class NonAffineLayerNorm(nn.LayerNorm):
    def __init__(self, dim):
        super(NonAffineLayerNorm, self).__init__(dim, elementwise_affine=False)