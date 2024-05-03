from .dataset_config import dataset_config
from .yowo_config import yowo_config


# 根据网络型号是yowo的哪个型号，返回对应网络配置参数的字典
def build_model_config(args):
    print('==============================')
    print('Model Config: {} '.format(args.version.upper()))
    
    if 'yowo_' in args.version:
        m_cfg = yowo_config[args.version]

    return m_cfg


# 根据数据集名称，返回一个字典，字典内包含对应该数据集的配置参数
def build_dataset_config(args):
    print('==============================')
    print('Dataset Config: {} '.format(args.dataset.upper()))
    
    d_cfg = dataset_config[args.dataset]

    return d_cfg
