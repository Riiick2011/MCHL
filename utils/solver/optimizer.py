"""
该文件构建优化器
"""
import torch
from torch import optim


# 返回优化器和周期  可以恢复训练
def build_optimizer(d_cfg, model, base_lr=0.0, resume=None):
    print('==============================')
    print('Optimizer: {}'.format(d_cfg['optimizer']))
    print('--momentum: {}'.format(d_cfg['momentum']))
    print('--weight_decay: {}'.format(d_cfg['weight_decay']))

    if d_cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=base_lr,
            momentum=d_cfg['momentum'],
            weight_decay=d_cfg['weight_decay'])

    elif d_cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=base_lr,
            eight_decay=d_cfg['weight_decay'])
                                
    elif d_cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=base_lr,
            weight_decay=d_cfg['weight_decay'])
          
    last_epoch = -1
    if resume is not None:
        print('keep training: ', resume)
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        # checkpoint state dict
        checkpoint_state_dict = checkpoint.pop("optimizer")
        optimizer.load_state_dict(checkpoint_state_dict)  # 恢复后不关心base_lr
        print('optimizer lr: ', optimizer.param_groups[0]['lr'])
        last_epoch = checkpoint.pop("epoch")  # 表示已经训练完成了该epoch
    return optimizer, last_epoch
