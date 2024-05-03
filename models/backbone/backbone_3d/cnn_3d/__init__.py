from .resnet import build_resnet_3d
from .resnext import build_resnext_3d
from .shufflnetv2 import build_shufflenetv2_3d


def build_3d_cnn(m_cfg, pretrained=False):
    print('==============================')
    print('3D Backbone: {}'.format(m_cfg['backbone_3d'].upper()))
    print('--pretrained: {}'.format(pretrained))
    multilevel = m_cfg['multilevel_3d']

    if 'resnet' in m_cfg['backbone_3d']:
        model, feat_dims = build_resnet_3d(
            model_name=m_cfg['backbone_3d'],
            pretrained=pretrained
            )
    elif 'resnext' in m_cfg['backbone_3d']:
        model, feat_dims = build_resnext_3d(
            model_name=m_cfg['backbone_3d'],
            pretrained=pretrained,
            multilevel=multilevel
            )
    elif 'shufflenetv2' in m_cfg['backbone_3d']:
        model, feat_dims = build_shufflenetv2_3d(
            model_size=m_cfg['model_size'],
            pretrained=pretrained
            )
    else:
        print('Unknown Backbone ...')
        exit()

    return model, feat_dims
