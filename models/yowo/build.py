"""
本文件根据配置参数创建一个YOWO类模型实例和一个损失函数实例并返回   包含了训练恢复机制
"""
import torch
from .yowo import YOWO
from .loss import build_criterion


# build YOWO detector  该函数用于构建一个YOWO检测器，并根据是否处于训练中返回损失函数
def build_yowo(args,
               d_cfg,
               m_cfg,
               device,
               num_classes=80,
               trainable=False,
               resume=None):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    # build YOWO
    model = YOWO(
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes,
        conf_thresh=args.conf_thresh,
        nms_thresh=m_cfg['nms_thresh'],
        nms_iou_type=m_cfg['nms_iou_type'],
        totaltopk=args.totaltopk,
        trainable=trainable,
        multi_hot=d_cfg['multi_hot'],
        clstopk=args.clstopk,
        det_save_type=args.det_save_type,
        bbox_with_feat=args.bbox_with_feat
    )

    if trainable:
        # Freeze backbone
        if args.freeze_backbone_2d:
            print('Freeze 2D Backbone ...')
            for m in model.backbone_2d.parameters():
                m.requires_grad = False
        if args.freeze_backbone_3d:
            print('Freeze 3D Backbone ...')
            for m in model.backbone_3d.parameters():
                m.requires_grad = False
            
        # keep training       
        if resume is not None:
            print('keep training: ', resume)
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            model.load_state_dict(checkpoint_state_dict)

        # build criterion
        criterion = build_criterion(
            m_cfg, d_cfg['train_size'], num_classes, d_cfg['multi_hot'], noconf=m_cfg['noconf'])
    
    else:
        criterion = None
                        
    return model, criterion
