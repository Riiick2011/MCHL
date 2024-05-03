"""
本文件定义了数据增强的类别和测试时所进行的基础变换类别
"""
import random
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


# Augmentation for Training
class Augmentation(object):
    def __init__(self, img_size=224, jitter=0.2, hue=0.1, saturation=1.5, exposure=1.5):
        self.img_size = img_size
        self.jitter = jitter
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    # 随机尺度
    def rand_scale(self, s):
        scale = random.uniform(1, s)

        if random.randint(0, 1): 
            return scale

        return 1./scale

    # 随机扰动
    def random_distort_image(self, video_clip):
        dhue = random.uniform(-self.hue, self.hue)
        dsat = self.rand_scale(self.saturation)
        dexp = self.rand_scale(self.exposure)
        
        video_clip_ = []
        for image in video_clip:
            image = image.convert('HSV')
            cs = list(image.split())  # PIL.Image.Image.split方法，将图像分割成HSV三个通道部分
            cs[1] = cs[1].point(lambda i: i * dsat)  # PIL.Image.Image.point方法，映射一个函数到图片上
            cs[2] = cs[2].point(lambda i: i * dexp)
            
            def change_hue(x):
                x += dhue * 255
                if x > 255:
                    x -= 255
                if x < 0:
                    x += 255
                return x

            cs[0] = cs[0].point(change_hue)
            image = Image.merge(image.mode, tuple(cs))  # PIL.Image.merge函数，多个单通道图像拼接成多通道图像

            image = image.convert('RGB')

            video_clip_.append(image)

        return video_clip_

    # 随机裁剪，返回裁剪后的视频片段 和 裁剪比例
    def random_crop(self, video_clip, width, height):
        dw = int(width * self.jitter)
        dh = int(height * self.jitter)

        pleft = random.randint(-dw, dw)
        pright = random.randint(-dw, dw)
        ptop = random.randint(-dh, dh)
        pbot = random.randint(-dh, dh)

        swidth = width - pleft - pright
        sheight = height - ptop - pbot

        sx = float(swidth) / width
        sy = float(sheight) / height
        
        dx = (float(pleft) / width)/sx
        dy = (float(ptop) / height)/sy

        # random crop
        cropped_clip = [img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1)) for img in video_clip]

        return cropped_clip, dx, dy, sx, sy

    # 将该视频片段的关键帧对应的所有真实标注框(原始尺寸下的绝对坐标两点式)根据裁剪进行调整变换为百分比两点式
    def apply_bbox(self, target, ow, oh, dx, dy, sx, sy):
        sx, sy = 1./sx, 1./sy
        # apply deltas on bbox  先根据裁剪比例和偏移调整真实标注框，输出是百分比两点式
        target[..., 0] = np.minimum(0.999, np.maximum(0, target[..., 0] / ow * sx - dx)) 
        target[..., 1] = np.minimum(0.999, np.maximum(0, target[..., 1] / oh * sy - dy)) 
        target[..., 2] = np.minimum(0.999, np.maximum(0, target[..., 2] / ow * sx - dx)) 
        target[..., 3] = np.minimum(0.999, np.maximum(0, target[..., 3] / oh * sy - dy)) 

        # refine target
        refine_target = []
        for i in range(target.shape[0]):  # 逐个真实标注框进行
            tgt = target[i]
            bw = (tgt[2] - tgt[0]) * ow  # 裁剪后的真实标注框，绝对数值表示的宽和高，如果过小则抛弃
            bh = (tgt[3] - tgt[1]) * oh

            if bw < 1. or bh < 1.:
                continue
            
            refine_target.append(tgt)  # 裁剪后真实标注框过小的抛弃，其余的保留，百分比两点式

        refine_target = np.array(refine_target).reshape(-1, target.shape[-1])

        return refine_target  # 返回裁剪后的真实标注框，数组格式，第0维是剩余真实标注框数量，第一维是一个真实标注框的元素

    # F.to_tensor对视频片段进行变换，RGB的值都除以了255变成了百分比表示
    # 将视频切片列表中的每一项均转换为tensor，乘以255因为RGB都是0-255表示，依旧返回一个列表，每一项是一个tensor
    def to_tensor(self, video_clip):
        return [F.to_tensor(image) * 255. for image in video_clip]

    # 输入视频片段和真实标注框，进行数据增强，返回增强后的视频片段和增强后的真实标注框
    def __call__(self, video_clip, target):
        # Initialize Random Variables 图像的原始尺寸高宽
        oh = video_clip[0].height  
        ow = video_clip[0].width
        
        # random crop
        video_clip, dx, dy, sx, sy = self.random_crop(video_clip, ow, oh)

        # resize 变为正方形
        video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]

        # random flip 随机对该视频片段进行左右翻转
        flip = random.randint(0, 1)
        if flip:
            video_clip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in video_clip]

        # distort 随机扰动
        video_clip = self.random_distort_image(video_clip)

        # process target 处理真实标注框，变为百分比两点式
        if target is not None:
            target = self.apply_bbox(target, ow, oh, dx, dy, sx, sy)
            if flip:
                target[..., [0, 2]] = 1.0 - target[..., [2, 0]]
        else:
            target = np.array([])
            
        # to tensor 将视频片段中的每一帧图像都转换为一个经过归一化的tensor，一帧对应列表中的一个tensor
        video_clip = self.to_tensor(video_clip)
        target = torch.as_tensor(target).float()

        # video_clip是列表表示的视频片段，每一项是一帧增强后图像对应的tensor；target是增强后的真实标注框(丢弃了增强过程中不合理的真实标注框图)，百分比两点式
        return video_clip, target 


# Transform for Testing 用于在测试时进行图像变换，改变尺寸和用像素均值、像素标准差进行归一化，并变换为tensor形式
class BaseTransform(object):
    def __init__(self, img_size=224):
        self.img_size = img_size

    # F.to_tensor对视频片段进行变换，RGB的值都除以了255变成了百分比表示
    # 将视频切片列表中的每一项均转换为tensor，乘以255因为RGB都是0-255表示，依旧返回一个列表，每一项是一个tensor
    def to_tensor(self, video_clip):
        return [F.to_tensor(image) * 255. for image in video_clip]

    def __call__(self, video_clip, target=None, normalize=True):
        # video_clip是视频采样后的一个列表，每一项是一个Image对象，target是数组表示的标注

        oh = video_clip[0].height
        ow = video_clip[0].width

        # resize 尺寸变为正方形
        video_clip = [img.resize([self.img_size, self.img_size]) for img in video_clip]

        # normalize target
        if target is not None:
            if normalize:  # 如果要对真实标注进行归一化，则将标注归一化为小数表示，target还是数组   默认进行归一化
                target[..., [0, 2]] /= ow
                target[..., [1, 3]] /= oh

        else:  # 如果没提供标注，则返回空数组 针对的是video mAP的任务，无需对真实标注变换为小数
            target = np.array([])

        # to tensor  将视频切片列表中的每一项均进行归一化并转换为tensor，依旧返回一个列表，每一项是一个tensor
        video_clip = self.to_tensor(video_clip)
        target = torch.as_tensor(target).float()

        return video_clip, target  # video_clip是一个列表，每一项对应一帧图像的tensor，target是一个tensor对应唯一关键帧的标注

