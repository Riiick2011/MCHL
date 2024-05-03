"""
本文件计算模型的flop和参数量
"""

import torch
from thop import profile


# 计算Flop和参数数量
def FLOPs_and_Params(model, img_size, len_clip, device):
    # generate init video clip
    video_clip = torch.randn(1, 3, len_clip, img_size, img_size).to(device)

    # set eval mode  模型设置为评估模式
    model.trainable = False
    model.eval()

    print('==============================')
    flops, params = profile(model, inputs=(video_clip, ))
    print('==============================')
    print('FLOPs : {:.2f} G'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))
    
    # set train mode.  模型设置回训练模式
    model.trainable = True
    model.train()


if __name__ == "__main__":
    pass
