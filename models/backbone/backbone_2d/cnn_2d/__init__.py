# import 2D backbone
from models.backbone.backbone_2d.cnn_2d.yolo_free.yolo_free import build_yolo_free
from models.backbone.backbone_2d.cnn_2d.yolov8.yolov8 import build_yolov8


def build_2d_cnn(m_cfg, pretrained=False):
    print('==============================')
    print('2D Backbone: {}'.format(m_cfg['backbone_2d'].upper()))
    print('--pretrained: {}'.format(pretrained))

    if m_cfg['backbone_2d'] in ['yolo_free_nano', 'yolo_free_tiny',
                                'yolo_free_large', 'yolo_free_huge']:
        model, feat_dims = build_yolo_free(m_cfg['backbone_2d'], pretrained)
    elif m_cfg['backbone_2d'] in ['yolov8n', 'yolov8m', 'yolov8l']:
        model, feat_dims = build_yolov8(m_cfg['backbone_2d'], pretrained)

    else:
        print('Unknown 2D Backbone ...')
        exit()

    return model, feat_dims
