import torch
import torch.nn as nn

try:
    from .yolo_free_backbone import build_backbone
    from .yolo_free_neck import build_neck
    from .yolo_free_fpn import build_fpn
    from .yolo_free_head import build_head
except:
    from yolo_free_backbone import build_backbone
    from yolo_free_neck import build_neck
    from yolo_free_fpn import build_fpn
    from yolo_free_head import build_head


__all__ = ['build_yolo_free']  # 指明能被其他文件import*时的变量或函数，引入其他函数要写明from xxx import xxx


weight_pth = {
    'yolo_free_nano': '/home/su/YOWOv3/weights/yolo_free_nano_coco.pth',
    'yolo_free_tiny': '/home/su/YOWOv3/weights/yolo_free_tiny_coco.pth',
    'yolo_free_large': '/home/su/YOWOv3/weights/yolo_free_large_coco.pth',
}


yolo_free_config = {
    'yolo_free_nano': {
        # model
        'backbone': 'shufflenetv2_1.0x',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        'anchor_size': None,
        # neck
        'neck': 'sppf',
        'neck_dim': 232,
        'expand_ratio': 0.5,
        'pooling_size': 5,
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': True,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'nano',
        'fpn_dim': [116, 232, 232],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': True,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': True,
        },

    'yolo_free_tiny': {
        # model
        'backbone': 'elannet_tiny',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 256,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'lrelu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'tiny', # 'tiny', 'large', 'huge
        'fpn_dim': [128, 256, 256],
        'fpn_norm': 'BN',
        'fpn_act': 'lrelu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 64,
        'head_norm': 'BN',
        'head_act': 'lrelu',
        'num_cls_head': 2,
        'num_reg_head': 2,
        'head_depthwise': False,
        },

    'yolo_free_large': {
        # model
        'backbone': 'elannet_large',
        'pretrained': True,
        'stride': [8, 16, 32],  # P3, P4, P5
        # neck
        'neck': 'spp_block_csp',
        'neck_dim': 512,
        'expand_ratio': 0.5,
        'pooling_size': [5, 9, 13],
        'neck_act': 'silu',
        'neck_norm': 'BN',
        'neck_depthwise': False,
        # fpn
        'fpn': 'pafpn_elan',
        'fpn_size': 'large',  # 'tiny', 'large', 'huge
        'fpn_dim': [512, 1024, 512],
        'fpn_norm': 'BN',
        'fpn_act': 'silu',
        'fpn_depthwise': False,
        # head
        'head': 'decoupled_head',
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        'num_cls_head': 2,   # head的cls支路的卷积次数
        'num_reg_head': 2,
        'head_depthwise': False,
        },

}


# Anchor-free YOLO
class FreeYOLO(nn.Module):
    def __init__(self, cfg):
        super(FreeYOLO, self).__init__()
        # --------- Basic Config -----------
        self.cfg = cfg

        # --------- Network Parameters ----------
        # backbone  默认采用YOLOv7的骨架ELANNet(其中的ELANBlock也带有CSP设计)，空间尺寸逐层减半，通道数逐层增加
        self.backbone, bk_dim = build_backbone(self.cfg['backbone'])  # bk_dim是backbone最后3层的通道数

        # neck 采用久经YOLO系列考验过的SPP，空间金字塔池化(可以额外带有CSP设计降低计算量，Cross Stage Partial)
        # 对一个特征图采用不同尺寸的池化来关注不同尺寸的特征，输出的尺寸是一样的
        self.neck = build_neck(cfg=self.cfg, in_dim=bk_dim[-1], out_dim=self.cfg['neck_dim'])
        
        # fpn 采用YOLOv7的PaFPN结构   不同层级上下融合   输出是一个列表，每一项对应一个层级的特征图，均为256通道，尺寸分别为28、14、7
        self.fpn = build_fpn(cfg=self.cfg, in_dims=self.cfg['fpn_dim'], out_dim=self.cfg['head_dim'])

        # non-shared heads  不同层级特征之间独立的头部，每个头部都解耦输出分类特征和回归特征    通道数不变，尺寸也不变
        self.non_shared_heads = nn.ModuleList(
            [build_head(cfg) for _ in range(len(cfg['stride']))])

    def forward(self, x):
        # backbone
        feats = self.backbone(x)  # 一个字典，不同层的特征图，尺寸逐层减半，通道数逐层增加 28、14、7    512、1024、1024

        # neck
        feats['layer4'] = self.neck(feats['layer4'])  # 只对最后一层特征图7x7x1024(空间尺寸最小，通道数最多)进行SPP空间金字塔池化 neck设计

        # fpn
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_cls_feats = []
        all_reg_feats = []
        for feat, head in zip(pyramid_feats, self.non_shared_heads):
            # [B, C, H, W]
            cls_feat, reg_feat = head(feat)  # 输入一个层级的特征图，解耦出该层级的2种特征图   通道数不变，尺寸也不变

            all_cls_feats.append(cls_feat)
            all_reg_feats.append(reg_feat)

        return all_cls_feats, all_reg_feats  # 均为列表，含有3项，每一项对应一个层级的特征图     通道数均为256，尺寸为28、14、7


# build FreeYOLO 构建FreeYOLO
def build_yolo_free(model_name='yolo_free_large', pretrained=False):
    # model config
    cfg = yolo_free_config[model_name]

    # FreeYOLO
    model = FreeYOLO(cfg)
    feat_dims = [model.cfg['head_dim']] * 3  # 三个不同层级特征图的通道数一致
    decoupled = True

    # Load COCO pretrained weight
    if pretrained:
        print('Loading 2D backbone pretrained weight: {}'.format(model_name.upper()))
        # checkpoint state dict
        checkpoint = torch.load(weight_pth[model_name], map_location=torch.device('cpu'))
        checkpoint_state_dict = checkpoint.pop('model')

        # model state dict
        model_state_dict = model.state_dict()
        # check
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    # print(k)
                    checkpoint_state_dict.pop(k)
            else:
                checkpoint_state_dict.pop(k)
                # print(k)

        model.load_state_dict(checkpoint_state_dict, strict=False)

    return model, feat_dims


if __name__ == '__main__':
    model, fpn_dim, decoupled = build_yolo_free(model_name='yolo_free_large', pretrained=True)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    cls_feats, reg_feats = model(x)

    for cls_feat, reg_feat in zip(cls_feats, reg_feats):
        print(cls_feat.shape, reg_feat.shape)
