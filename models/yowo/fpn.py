"""
定义了FPN,用于2D、3D融合后的特征，不同空间尺寸之间的信息传递
"""
import torch
import torch.nn as nn
import math

from models.backbone.backbone_2d.cnn_2d.yolov8.modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP,
                                                               C2f, C3Ghost, C3x, Concat, Conv, ConvTranspose, Head,
                                                               DWConv, DWConvTranspose2d, Focus, GhostBottleneck,
                                                               GhostConv)

# YOLOv8.0l head
cfg = {
    'fpn':
    [[-1, 1, nn.Upsample, [None, 2, 'nearest']],
     [[-1, 6], 1, 'Concat', [1]],  # cat backbone P4
     [-1, 3, 'C2f', [512]],  # 12

     [-1, 1, nn.Upsample, [None, 2, 'nearest']],
     [[-1, 4], 1, 'Concat', [1]],  # cat backbone P3
     [-1, 3, 'C2f', [256]],  # 15 (P3/8-small)

     [-1, 1, 'Conv', [256, 3, 2]],
     [[-1, 12], 1, 'Concat', [1]],  # cat head P4
     [-1, 3, 'C2f', [512]],  # 18 (P4/16-medium)

     [-1, 1, 'Conv', [512, 3, 2]],
     [[-1, 9], 1, 'Concat', [1]],  # cat head P5
     [-1, 3, 'C2f', [512]],  # 21 (P5/32-large)
     [[15, 18, 21], 1, 'Head', [256]]],  # Head(P3, P4, P5)
}


# 使x能被divisor整除，往上找
def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


# FPN Module  FPN模块，返回(B，C，H，W)
class fpn(nn.Module):
    """ Channel attention module """
    def __init__(self, in_dim=256, out_dim=256):
        super(fpn, self).__init__()
        self.cfg = cfg
        ch = [in_dim]
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(self.cfg['fpn']):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings  将字符串转化为对应的类别名称
            n = n_ = max(round(n), 1) if n > 1 else n  # gd=depth gain深度增益   n=重复次数
            if m in (Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF,
                     DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d,
                     DWConvTranspose2d, C3x):
                c1, c2 = ch[f], args[0]  # c1是输入通道数，c2是输出通道数
                c2 = make_divisible(c2, 8)  # 使c2*gw可以被8整除，往上面找
                args = [c1, c2, *args[1:]]
                if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x):
                    args.insert(2, n)  # number of repeats
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]

            elif m is Concat:
                c2 = sum(ch[x] for x in f)

            elif m is Head:
                args.append([ch[x] for x in f])
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
            t = str(m)[8:-2].replace('__main__.', '')  # module type
            m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)

        self.backbone = nn.Sequential(*layers)
        save.append(len(self.backbone) - 1)
        self.save = sorted(save)

    def forward(self, x):
        """
            inputs :
                x : input feature map list [( B x C x H x W )x3]
            returns :
                out : output feature map list [( B x C x H x W )x3]
        """
        x1, x2, x3 = x
        y, dt = [], []  # outputs
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        pyramid_feats = y[-1]
        return pyramid_feats


# fpn
def build_fpn(dim):
    encoder = fpn(
        in_dim=dim,
        out_dim=dim
        )

    return encoder
