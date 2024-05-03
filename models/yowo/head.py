import torch.nn as nn
import torch
import math
from models.basic import Conv2d


class DecoupledHead(nn.Module):  # YOWOv2所用的头部(预测层之前)
    def __init__(self, m_cfg):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        self.num_cls_heads = m_cfg['num_cls_heads']
        self.num_reg_heads = m_cfg['num_reg_heads']
        self.act_type = m_cfg['head_act']
        self.norm_type = m_cfg['head_norm']
        self.head_dim = m_cfg['head_dim']
        self.depthwise = m_cfg['head_depthwise']

        self.cls_head = nn.Sequential(*[
            Conv2d(self.head_dim, 
                   self.head_dim, 
                   k=3, p=1, s=1, 
                   act_type=self.act_type, 
                   norm_type=self.norm_type,
                   depthwise=self.depthwise)
                   for _ in range(self.num_cls_heads)])
        self.reg_head = nn.Sequential(*[
            Conv2d(self.head_dim, 
                   self.head_dim, 
                   k=3, p=1, s=1, 
                   act_type=self.act_type, 
                   norm_type=self.norm_type,
                   depthwise=self.depthwise)
                   for _ in range(self.num_reg_heads)])

    def forward(self, cls_feat, reg_feat):
        cls_feats = self.cls_head(cls_feat)
        reg_feats = self.reg_head(reg_feat)

        return cls_feats, reg_feats


# heads  YOLOv8的头部 不同层级互不影响
class Detect(nn.Module):
    # YOLOv8 Detect head for detection models
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, m_cfg, nc=24, ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # 输入的层级数量
        self.reg_max = m_cfg['reg_max']  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + 4 * self.reg_max  # number of outputs per anchor
        self.stride = torch.tensor(m_cfg['stride'], dtype=torch.float32)  # strides computed during build

        c2, c3 = max((16, ch[0] // 4, 4 * self.reg_max)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv2d(x, c2, 3, p=1, norm_type='BN', act_type='silu'),
                          Conv2d(c2, c2, 3, p=1, norm_type='BN', act_type='silu'),
                          nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)  # 定位分支
        self.cv3 = nn.ModuleList(
            nn.Sequential(Conv2d(x, c3, 3, p=1, norm_type='BN', act_type='silu'),
                          Conv2d(c3, c3, 3, p=1, norm_type='BN', act_type='silu'),
                          nn.Conv2d(c3, self.nc, 1)) for x in ch)  # 分类分支，没经过sigmoid

    def forward(self, x, decoupled_in=False):  # 输入必须是列表，每一项是一个tensor对应一个层级
        assert isinstance(x, list)
        y = []
        if decoupled_in:  # x=[cls_feats,reg_feats]
            for i in range(self.nl):  # 层级数量
                y.append(torch.cat((self.cv2[i](x[1][i]), self.cv3[i](x[0][i])), 1))
        else:  # x=feats
            for i in range(self.nl):  # 层级数量
                y.append(torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1))
        return y  # 返回一个列表，其中每一项对应一层输出，每层的输出是reg和cls在通道维度上拼接起来

    def bias_init(self):
        # Initialize Detect() biases, WARNING: requires stride availability
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


def build_head(m_cfg, num_classes=24, decoupled_early=True, ch=256):
    if decoupled_early:
        return DecoupledHead(m_cfg)
    else:
        return Detect(m_cfg, num_classes, ch=ch)
    