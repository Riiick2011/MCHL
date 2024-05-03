# 本文件记录网络的配置参数
# Model configuration


yowo_config = {
    'yowo_v2_large': {
        ## Backbone
        # 2D    输出的通道数都是256，空间尺寸都是3个层级
        'decoupled_early': True,
        'backbone_2d': 'yolov8l',  # 'yolov8l', # 'yolo_free_large',  # 'yolov8l', #  'yolo_free_large', # 'yolov8l',   # ,  # decoupled_early=False
        'pretrained_2d': True,
        'stride': [8, 16, 32],  # 空间上的stride层级
        # 3D  输出的通道数不一样，空间尺寸是三个层级，时间尺寸是len_clip/4 /8 /16
        'backbone_3d': 'resnext101',
        'pretrained_3d': True,
        'memory_momentum': 0.9,
        'multilevel_3d': False,  # 开启后则3D骨架输出3个层级的特征图
        ## Neck
        'fpn': False,
        'fpn_after_encoder': True,  # 跟在编码器后面 只交换信息 不改变空间尺寸和通道数量
        ## Head  Head、Matcher、Loss的类别一般是一起换的
        'noconf': False,
        'head_dim': 256,
        'head_norm': 'BN',
        'head_act': 'silu',
        # ConfHead
        'num_cls_heads': 2,
        'num_reg_heads': 2,
        'head_depthwise': False,
        # NoConfHead
        'reg_max': 16,  # 大于1则表示开启DFL分布式focal损失    表示预测框的lt 和 rb到 锚点中心的横向距离、纵向距离的上限-1(没乘以stride)
        ## Matcher
        'topk_candidate': 10,  # Max Positive Sample Number of one gt bbox 一个目标框最多得到10个正样本
        # SimOTA
        'center_sampling_radius': 2.5,  # Positive Sample Radius of Grid Cell
        ## Loss
        'nwd': False,
        'loss_iou_type': 'giou',  # Conf default: 'giou',NoConf default：'ciou'
        'matcher_iou_type': 'iou',   # Conf default:'iou',NoConf default：'ciou'
        # ConfCriterion
        'CCloss_conf_weight': 1,
        'CCloss_cls_weight': 1,
        'CCloss_box_weight': 5,
        'conf_iou_aware': False,   # conf分支是否关注iou，开启后三个分支均使用同一种iou_type  取False时cls_ori_iou_type才有效
        'cls_ori_iou_type': True,  # 默认是False，代表不取原始类型    原始类型是iou
        # NoConfCriterion
        'VFL': False,  # cls_loss  VFL或者BCE   应该进行实验
        'NCCloss_box_weight': 7.5,
        'NCCloss_cls_weight': 0.5,
        'NCCloss_dfl_weight': 1.5,
        ## NMS
        'nms_iou_type': 'iou',  # default：'iou'  如果修改要连同nms_thresh一起修改
        'nms_thresh': 0.5,  # 只对评估时有用
    },
}
