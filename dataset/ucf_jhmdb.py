#!/usr/bin/python
# encoding: utf-8
"""
本文件包含数据集类ucf_jhmdb的定义
"""

import os
import random
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from scipy.io import loadmat


# Dataset for UCF24 & JHMDB  UCF24 & JHMDB数据集类，用于训练和frame级别的评估
class UCF_JHMDB_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='ucf24',
                 img_size=224,
                 transform=None,
                 is_train=False,
                 len_clip=16,
                 sampling_rate=1,
                 multi_sampling_rate=1,
                 untrimmed=False):
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate  # 默认
        self.multi_sampling_rate = multi_sampling_rate
        self.untrimmed = untrimmed if self.dataset == 'ucf24' else False

        if self.is_train:  # 根据需要可以切换为训练集和测试集   里面存有对应每张图片的标注文件名
            self.split_list = 'trainlist.txt'
            self.untrimmed_list = 'trainlist_untrimmed.txt'
        else:
            self.split_list = 'testlist.txt'
            self.untrimmed_list = 'testlist_untrimmed.txt'

        # load data
        with open(os.path.join(data_root, self.split_list), 'r') as file:
            self.file_names = file.readlines()
        self.num_samples = len(self.file_names)  # 样本数量=有标注的图片的总数
        # 用于untrimmed数据集的video mAP任务进行训练
        if self.untrimmed:
            with open(os.path.join(data_root, self.untrimmed_list), 'r') as file:
                self.file_names_untrimmed = file.readlines()
            self.num_samples = len(self.file_names_untrimmed)  # 样本数量=所有的图片的总数

        if dataset == 'ucf24':
            self.num_classes = 24
        elif dataset == 'jhmdb21':
            self.num_classes = 21

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        # load a data
        img_name, video_clip, target = self.pull_item(index)

        return img_name, video_clip, target  # 返回 该帧图片的标注的完整名称，视频片段-是方形的，真实标注（字典形式)

    def pull_item(self, index):  # 返回 该帧图片的完整名称，视频片段，真实标注（字典形式)
        """ load a data """
        assert index <= len(self), 'index range error'
        if self.untrimmed:
            img_name = self.file_names_untrimmed[index].rstrip()
            label_name = img_name.replace('jpg', 'txt')
            label_path = os.path.join(self.data_root, 'labels', label_name)
        else:
            label_name = self.file_names[index].rstrip()  # 该帧图片的标注的完整名称
            if self.dataset == 'ucf24':
                img_name = label_name.replace('txt', 'jpg')  # 该帧图片的完整名称
            elif self.dataset == 'jhmdb21':
                img_name = label_name.replace('txt', 'jpg')  # 该帧图片的完整名称
            # path to label 该帧图片的标注的完整路径
            label_path = os.path.join(self.data_root, 'labels', label_name)

        img_split = img_name.split('/')  # ex. ['Basketball', 'v_Basketball_g08_c01', '00070.jpg']
        video_name = os.path.join(img_split[0], img_split[1])
        # image name 该帧图片的名称ID
        frame_id = int(img_split[-1][:5])

        # image folder 该帧图片的文件夹的完整路径
        img_folder = os.path.join(self.data_root, 'rgb-images', img_split[0], img_split[1])

        # frame numbers  文件夹中的总帧数，其中jhmdb21需要-1 为什么?
        if self.dataset == 'ucf24':
            max_num = len(os.listdir(img_folder))
        elif self.dataset == 'jhmdb21':
            max_num = len(os.listdir(img_folder)) - 1

        if self.multi_sampling_rate > 1:  # 如果采用多采样率模块
            # load images  video_clip是一个列表，每一项是一个Image对象
            video_clip = []
            for i in reversed(range(self.len_clip)):  # 15~0
                # make it as a loop

                len_mini_clip = self.len_clip//self.multi_sampling_rate  # 一个clip由几部分组成，每一部分包含几帧
                mini_clip_idx = i//len_mini_clip  # 处于第几个mini_clip
                # 1 9 17 25 29 33 37 41 43 45 47 49 50 51 52 53
                frame_id_temp = frame_id - sum([len_mini_clip * (2**_) for _ in range(0, mini_clip_idx)]) - (
                        i - len_mini_clip*mini_clip_idx) * (2**mini_clip_idx)

                # 限位
                if frame_id_temp < 1:
                    frame_id_temp = 1
                elif frame_id_temp > max_num:
                    frame_id_temp = max_num

                # load a frame
                if self.dataset == 'ucf24':
                    path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[0], img_split[1],
                                            '{:05d}.jpg'.format(frame_id_temp))
                elif self.dataset == 'jhmdb21':
                    path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[0], img_split[1],
                                            '{:05d}.jpg'.format(frame_id_temp))
                frame = Image.open(path_tmp).convert('RGB')  # 一个Image对象
                ow, oh = frame.width, frame.height  # 图片的原始宽度和原始高度

                video_clip.append(frame)

        else:
            # sampling rate  如果训练中则采样率从1和2两个整数中随机取
            if self.is_train:
                d = random.randint(1, 2)
            else:
                d = self.sampling_rate  # 1

            # load images  video_clip是一个列表，每一项是一个Image对象
            video_clip = []
            for i in reversed(range(self.len_clip)):
                # make it as a loop
                frame_id_temp = frame_id - i * d
                if frame_id_temp < 1:
                    frame_id_temp = 1
                elif frame_id_temp > max_num:
                    frame_id_temp = max_num

                # load a frame
                if self.dataset == 'ucf24':
                    path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[0], img_split[1], '{:05d}.jpg'.format(frame_id_temp))
                elif self.dataset == 'jhmdb21':
                    path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[0], img_split[1], '{:05d}.jpg'.format(frame_id_temp))
                frame = Image.open(path_tmp).convert('RGB')  # 一个Image对象
                ow, oh = frame.width, frame.height  # 图片的原始宽度和原始高度

                video_clip.append(frame)

        # load an annotation  如果存在，则载入该帧图像对应的标注文件，数组表示
        if os.path.exists(label_path):
            target = np.loadtxt(label_path)
            # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]  调整标注的排列方式，数组表示
            label = target[..., :1]
            boxes = target[..., 1:]
            target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)

            # transform后尺寸变为方形，video_clip是一个列表，每一项对应一帧图像的tensor，
            # transform对视频片段进行变换，RGB的值都除以了255变成了百分比表示
            # target是一个tensor对应唯一关键帧的标注（默认是两点小数格式)
            video_clip, target = self.transform(video_clip, target)
            # List [T, 3, H, W] -> [3, T, H, W]  将视频片段列表转换维度顺序，成为一个[3, T, H, W]维度的tensor
            video_clip = torch.stack(video_clip, dim=1)

            targets = {
                'boxes': target[:, :4].float(),  # [N, 4]
                'labels': target[:, -1].long() - 1,  # [N,]  #  训练的时候类别要从0开始
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_name': video_name,
                'frame_id': frame_id
            }
        else:
            target = None

            # transform后尺寸变为方形，video_clip是一个列表，每一项对应一帧图像的tensor，
            # transform对视频片段进行变换，RGB的值都除以了255变成了百分比表示
            # target是一个tensor对应唯一关键帧的标注（默认是两点小数格式)
            video_clip, target = self.transform(video_clip, target)
            # List [T, 3, H, W] -> [3, T, H, W]  将视频片段列表转换维度顺序，成为一个[3, T, H, W]维度的tensor
            video_clip = torch.stack(video_clip, dim=1)

            targets = {
                'boxes': target,  # 没有的时候是空tensor
                'labels': target,  # 没有的时候是空tensor
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_name': video_name,
                'frame_id': frame_id
            }

        # img_name是该帧图片的名称带后缀
        # video_clip是一个[3, T, H, W]维度的tensor，表示该帧图片对应的视频片段，是方形的
        # target是一个字典，表示该帧图片的标注，边界框是默认为两点式百分比的tensor，类别是tensor
        return img_name, video_clip, targets


# Video Dataset for UCF24 & JHMDB UCF24 & JHMDB数据集类，用于video级别的评估，每个视频作为一个独立的数据集实例
class UCF_JHMDB_VIDEO_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='ucf24',
                 img_size=224,
                 transform=None,
                 len_clip=16,
                 multi_sampling_rate=1,
                 untrimmed=False):  # 没有is_train一说，因为video级别的评估都是测试阶段
        self.data_root = data_root
        self.dataset = dataset
        self.transform = transform
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.multi_sampling_rate = multi_sampling_rate
        self.untrimmed = untrimmed if self.dataset == 'ucf24' else False
            
        if dataset == 'ucf24':
            self.num_classes = 24
            if not self.untrimmed:
                with open(os.path.join(self.data_root, 'testlist.txt'), 'r') as file:
                    self.file_names = file.readlines()
        elif dataset == 'jhmdb21':
            self.num_classes = 21

    def set_video_data(self, video_name):  # 输入一个视频的名称
        self.video_name = video_name
        # load a video该视频的完整路径
        self.img_folder = os.path.join(self.data_root, 'rgb-images', self.video_name)
        if self.dataset == 'ucf24':
            self.img_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))  # 一个列表，存放该视频路径下的所有图片的完整路径
            if not self.untrimmed:  # 如果是trimmed
                img_paths = []
                for img_path in self.img_paths:
                    img_path_split = img_path.replace('jpg', 'txt').split('/')
                    if os.path.join(img_path_split[-3], img_path_split[-2], img_path_split[-1]) + '\n' \
                            in self.file_names:
                        img_paths.append(img_path)
                self.img_paths = img_paths
        elif self.dataset == 'jhmdb21':
            self.img_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        return self.pull_item(index)

    def pull_item(self, index):
        image_path = self.img_paths[index]
        video_name = self.video_name
        # for windows:
        # img_split = image_path.split('\\')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']
        # for linux
        img_split = image_path.split('/')  # ex. [..., 'Basketball', 'v_Basketball_g08_c01', '00070.txt']

        # image name
        frame_id = int(img_split[-1][:5])
        file_name = img_split[-2]
        class_name = img_split[-3]
        max_num = len(os.listdir(self.img_folder))

        if self.dataset == 'ucf24':
            img_name = os.path.join(video_name, '{:05d}.jpg'.format(frame_id))
            label_path = os.path.join(self.data_root, 'labels', class_name, file_name,
                                      '{:05d}.txt'.format(frame_id))  # 起始帧的标注文件
        elif self.dataset == 'jhmdb21':
            img_name = os.path.join(video_name, '{:05d}.jpg'.format(frame_id))
            label_path = os.path.join(self.data_root, 'labels', class_name, file_name,
                                      '{:05d}.txt'.format(frame_id))  # 起始帧的标注文件

        # load video clip载入该图片对应的视频片段
        if self.multi_sampling_rate > 1:  # 如果采用多采样率模块
            # load images  video_clip是一个列表，每一项是一个Image对象
            video_clip = []
            for i in reversed(range(self.len_clip)):  # 15~0
                # make it as a loop

                len_mini_clip = self.len_clip//self.multi_sampling_rate  # 一个clip由几部分组成，每一部分包含几帧
                mini_clip_idx = i//len_mini_clip  # 处于第几个mini_clip
                # 1 9 17 25 29 33 37 41 43 45 47 49 50 51 52 53
                frame_id_temp = frame_id - sum([len_mini_clip * (2**_) for _ in range(0, mini_clip_idx)]) - (
                        i - len_mini_clip*mini_clip_idx) * (2**mini_clip_idx)

                # 限位
                if frame_id_temp < 1:
                    frame_id_temp = 1
                elif frame_id_temp > max_num:
                    frame_id_temp = max_num

                # load a frame
                if self.dataset == 'ucf24':
                    path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[0], img_split[1],
                                            '{:05d}.jpg'.format(frame_id_temp))
                elif self.dataset == 'jhmdb21':
                    path_tmp = os.path.join(self.data_root, 'rgb-images', img_split[0], img_split[1],
                                            '{:05d}.jpg'.format(frame_id_temp))
                frame = Image.open(path_tmp).convert('RGB')  # 一个Image对象
                ow, oh = frame.width, frame.height  # 图片的原始宽度和原始高度

                video_clip.append(frame)
        else:
            video_clip = []
            for i in reversed(range(self.len_clip)):
                # make it as a loop
                frame_id_temp = frame_id - i
                if frame_id_temp < 1:
                    frame_id_temp = 1
                elif frame_id_temp > max_num:
                    frame_id_temp = max_num

                # load a frame
                if self.dataset == 'ucf24':
                    path_tmp = os.path.join(
                        self.data_root, 'rgb-images', video_name, '{:05d}.jpg'.format(frame_id_temp))
                elif self.dataset == 'jhmdb21':
                    path_tmp = os.path.join(
                        self.data_root, 'rgb-images', video_name, '{:05d}.jpg'.format(frame_id_temp))
                frame = Image.open(path_tmp).convert('RGB')
                ow, oh = frame.width, frame.height

                video_clip.append(frame)

        # 载入该图片对应的真实标注  因为video推断，也需要推断不含标注的图片，因此该图片可能没有对应的标注文件
        # load an annotation 如果存在，则载入该帧图像对应的标注文件，数组表示
        if os.path.exists(label_path):
            target = np.loadtxt(label_path)
            # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label]  调整标注的排列方式，数组表示
            label = target[..., :1]
            boxes = target[..., 1:]
            target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)

            # transform对视频片段进行变换，RGB的值都除以了255变成了百分比表示
            video_clip, _ = self.transform(video_clip, normalize=False)
            # List [T, 3, H, W] -> [3, T, H, W]
            video_clip = torch.stack(video_clip, dim=1)

            targets = {
                'boxes': target[:, :4],  # [N, 4]
                'labels': target[:, -1] - 1,  # [N,]     -1是背景类别，即无标注，其他类别从0开始
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_name': video_name,
                'frame_id': frame_id
            }
        else:
            target = None

            # transform对视频片段进行变换，RGB的值都除以了255变成了百分比表示
            video_clip, _ = self.transform(video_clip, normalize=False)
            # List [T, 3, H, W] -> [3, T, H, W]
            video_clip = torch.stack(video_clip, dim=1)

            targets = {
                'boxes': target,  # 没有的时候是空tensor
                'labels': target,  # 没有的时候是空tensor
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_name': video_name,
                'frame_id': frame_id
            }

        return img_name, video_clip, targets


if __name__ == '__main__':
    import cv2
    from dataset.transforms import Augmentation, BaseTransform

    data_root = '/media/su/d/datasets/UCF24-YOWO'
    dataset = 'ucf24'
    is_train = True
    img_size = 224
    len_clip = 16
    trans_config = {
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5
    }
    train_transform = Augmentation(
        img_size=img_size,
        jitter=trans_config['jitter'],
        saturation=trans_config['saturation'],
        exposure=trans_config['exposure'])
    val_transform = BaseTransform(img_size=img_size)

    train_dataset = UCF_JHMDB_Dataset(
        data_root=data_root,
        dataset=dataset,
        img_size=img_size,
        transform=train_transform,
        is_train=is_train,
        len_clip=len_clip)

    print(len(train_dataset))
    for i in range(len(train_dataset)):
        frame_id, video_clip, target = train_dataset[i]
        key_frame = video_clip[:, -1, :, :]

        # to numpy
        key_frame = key_frame.permute(1, 2, 0).numpy()
        key_frame = key_frame.astype(np.uint8)

        # to BGR
        key_frame = key_frame[..., (2, 1, 0)]
        H, W, C = key_frame.shape

        key_frame = key_frame.copy()
        bboxes = target['boxes']
        labels = target['labels']

        for box, cls_id in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            key_frame = cv2.rectangle(key_frame, (x1, y1), (x2, y2), (255, 0, 0))

        # cv2 show
        cv2.imshow('key frame', key_frame)
        cv2.waitKey(5)
        
