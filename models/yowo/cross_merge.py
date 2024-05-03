import torch
import torch.nn as nn
import torch.nn.functional as F
from ..basic.conv import Conv2d


class cross_merge(nn.Module):
    def __init__(self, chin2d, chin3d):
        # 降低维度可以用C2F做？
        # 可以用其他方式替换拼接？
        super(cross_merge, self).__init__()
        self.chin2d = chin2d
        self.chin3d = chin3d
        self.cv_channel = nn.ModuleList(Conv2d(k=1, c1=chin3d[i], c2=2048, act_type='silu', norm_type='BN') for i in range(len(self.chin3d)))
        self.cv_spatial = Conv2d(c1=2048, c2=2048, k=3, p=1, s=2, act_type='silu', norm_type='BN')
        self.chout = 256+2048  # 先排列同一个列表的

    def forward(self, feats_2d, feats_3d):
        feats_merge = []
        for i, feat_3d in enumerate(feats_3d):  # 28 14 7
            feat_3d = feat_3d.mean(dim=-3)  # [B,C,H,W]  取平均破坏了时序，可能可以改为拼接？拼接的话就不需要1x1卷积了
            feat_3d = self.cv_channel[i](feat_3d)  # [B,2048,H,W]
            for j, feat_2d in enumerate(feats_2d):  # 28 14 7
                # 上采样和下采样 调节尺寸
                if i > j:
                    feat_3d_ = F.interpolate(feat_3d, scale_factor=2 ** (i-j))
                elif i < j:
                    feat_3d_ = self.cv_spatial(feat_3d)
                    if i == j-2:
                        feat_3d_ = self.cv_spatial(feat_3d_)
                else:
                    feat_3d_ = feat_3d
                assert feat_3d_.shape[-1] == feat_2d.shape[-1]
                assert feat_3d_.shape[-3] == 2048
                feat_merge = [feat_2d, feat_3d_]
                feats_merge.append(feat_merge)
        return feats_merge
