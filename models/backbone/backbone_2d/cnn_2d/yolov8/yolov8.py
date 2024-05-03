import torch
import torch.nn as nn
import math
try:
    from .modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                          Concat, Conv, ConvTranspose, Head, DWConv, DWConvTranspose2d, Ensemble, Focus,
                          GhostBottleneck, GhostConv)
except:
    from modules import (C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                         Concat, Conv, ConvTranspose, Head, DWConv, DWConvTranspose2d, Ensemble, Focus,
                         GhostBottleneck, GhostConv)

__all__ = ['build_yolov8']  # 指明能被其他文件import*时的变量或函数，引入其他函数要写明from xxx import xxx


weight_pth = {
    'yolov8l': '/home/su/YOWOv3/weights/yolov8l.pth',
}

yolov8_config = {
    'yolov8l': {
        'depth_multiple': 1.00,  # scales module repeats
        'width_multiple': 1.00,  # scales convolution channels

        # YOLOv8.0l backbone
        'backbone':
        # [from, repeats, module, args]
            [[-1, 1, 'Conv', [64, 3, 2]],  # 0-P1/2
             [-1, 1, 'Conv', [128, 3, 2]],  # 1-P2/4
             [-1, 3, 'C2f', [128, True]],
             [-1, 1, 'Conv', [256, 3, 2]],  # 3-P3/8
             [-1, 6, 'C2f', [256, True]],
             [-1, 1, 'Conv', [512, 3, 2]],  # 5-P4/16
             [-1, 6, 'C2f', [512, True]],
             [-1, 1, 'Conv', [512, 3, 2]],  # 7-P5/32
             [-1, 3, 'C2f', [512, True]],
             [-1, 1, 'SPPF', [512, 5]]],  # 9

        # YOLOv8.0l head
        'head':
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

        # coupled head
        'fpn_dim': [256, 515, 512],
        'head_dim': 256,
        },


}


# 使x能被divisor整除，往上找
def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor


class YOLOv8(nn.Module):
    def __init__(self, cfg, ch=3):
        super(YOLOv8, self).__init__()
        # --------- Basic Config ----------
        self.cfg = cfg
        gd, gw = self.cfg['depth_multiple'], self.cfg['width_multiple']
        ch = [ch]
        layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
        for i, (f, n, m, args) in enumerate(self.cfg['backbone'] + self.cfg['head']):  # from, number, module, args
            m = eval(m) if isinstance(m, str) else m  # eval strings  将字符串转化为对应的类别名称
            n = n_ = max(round(n * gd), 1) if n > 1 else n  # gd=depth gain深度增益   n=重复次数
            if m in (Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF,
                     DWConv, Focus, BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d,
                     DWConvTranspose2d, C3x):
                c1, c2 = ch[f], args[0]  # c1是输入通道数，c2是输出通道数
                c2 = make_divisible(c2 * gw, 8)  # 使c2*gw可以被8整除，往上面找
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
        y, dt = [], []  # outputs
        for m in self.backbone:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output

        pyramid_feats = y[-1]
        return pyramid_feats


# build YOLOv8 构建YOLOv8
def build_yolov8(model_name='yolov8l', pretrained=False):
    # model config
    cfg = yolov8_config[model_name]

    # FreeYOLO
    model = YOLOv8(cfg)
    # feat_dims = [model.cfg['head_dim']] * 3  # 三个不同层级特征图的通道数一致
    feat_dims = [model.cfg['head_dim']] * 3  # 三个不同层级特征图的通道数一致
    decoupled = False

    # Load COCO pretrained weight
    if pretrained:
        print('Loading 2D backbone pretrained weight: {}'.format(model_name.upper()))
        # checkpoint state dict
        checkpoint_state_dict = torch.load(weight_pth[model_name], map_location=torch.device('cpu'))

        # model state dict
        model_state_dict = model.state_dict()
        # check
        checkpoint_state_dict_modified = {}
        for k in list(checkpoint_state_dict.keys()):
            k_modified = k.replace('model', 'backbone')
            checkpoint_state_dict_modified[k_modified] = checkpoint_state_dict[k]
            if k_modified in model_state_dict:
                shape_model = tuple(model_state_dict[k_modified].shape)
                shape_checkpoint = tuple(checkpoint_state_dict_modified[k_modified].shape)
                if shape_model != shape_checkpoint:
                    # print(k)
                    checkpoint_state_dict_modified.pop(k_modified)
            else:
                checkpoint_state_dict_modified.pop(k_modified)
                # print(k)

        model.load_state_dict(checkpoint_state_dict_modified, strict=False)

    return model, feat_dims


if __name__ == '__main__':
    model, feat_dims = build_yolov8(model_name='yolov8l', pretrained=True)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    feats = model(x)
