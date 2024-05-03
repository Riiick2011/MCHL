import torch.nn as nn
from .cnn_3d import build_3d_cnn
    

class Backbone3D(nn.Module):
    def __init__(self, m_cfg, pretrained=False):
        super().__init__()
        self.m_cfg = m_cfg

        # 3D CNN
        self.backbone, self.feat_dim = build_3d_cnn(m_cfg, pretrained)

    def forward(self, x):
        """
            Input:
                x: (Tensor) -> [B, C, T, H, W]
            Output:
                y: (List) -> [
                    (Tensor) -> [B, C1, H1, W1],
                    (Tensor) -> [B, C2, H2, W2],
                    (Tensor) -> [B, C3, H3, W3]
                    ]
        """
        feat = self.backbone(x)

        return feat
