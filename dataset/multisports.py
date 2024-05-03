#!/usr/bin/python
# encoding: utf-8
"""
本文件包含数据集类multisports的定义
"""

import os
import random
import numpy as np
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle


# Dataset for MultiSports  MultiSports数据集类，用于frame级别的评估
class MultiSports_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='multisports',
                 img_size=224,
                 transform=None,
                 is_train=False,
                 len_clip=16,
                 sampling_rate=1,
                 multi_sampling_rate=1,
                 untrimmed=False):
        self.data_root = data_root  # pk文件完整路径
        self.dataset = dataset
        self.transform = transform
        self.is_train = is_train
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate  # 默认
        self.multi_sampling_rate = multi_sampling_rate
        self.untrimmed = untrimmed

        GT_path = os.path.join(self.data_root, "trainval/multisports_GT.pkl")
        GT = pickle.load(open(GT_path, 'rb'))
        person_proposal_dir = os.path.join(
            self.data_root, '/media/su/d/datasets/MultiSports/MultiSports_box')  # Faster R-CNN给出的人员提议
        if self.is_train:
            self.videolist = GT['train_videos'][0]  # 训练集视频名称
            person_proposal = pickle.load(open(os.path.join(person_proposal_dir, 'train.recall_96.13.pkl'), 'rb'))
        else:
            self.videolist = GT['test_videos'][0]  # 测试集视频名称
            person_proposal = pickle.load(open(os.path.join(person_proposal_dir, 'val.recall_96.13.pkl'), 'rb'))

        # 去除不要的类别
        for error_class in ['aerobic kick jump', 'aerobic off axis jump', 'aerobic butterfly jump',
                            'aerobic balance turn', 'basketball save', 'basketball jump ball']:
            GT['labels'].remove(error_class)
        self.classes = GT['labels']

        self.nframes = GT['nframes']

        gt = {}
        viddict = {}  # 便于通过视频名称查询视频编号  给person proposal使用
        for iv, video_name in enumerate(self.videolist):

            # 视频名称与iv的对应关系字典
            viddict[video_name.split('/')[1]] = iv

            for label_id in GT['gttubes'][video_name].keys():  # 该视频的管道类别 类别从0开始
                if label_id in [15, 16, 17, 20, 64, 65]:
                    continue
                tubes_of_one_class = GT['gttubes'][video_name][label_id]
                label_id = label_id - sum(label_id > np.array([15, 16, 17, 20, 64, 65]))  # 修正到只有60个类别
                for tube in tubes_of_one_class:
                    tube = np.insert(tube, 1, label_id, axis=1)
                    for i in range(tube.shape[0]):  # tube长度内的帧索引
                        k = (iv, int(tube[i, 0]))  # 元组（视频序号(从0开始)，帧序号（从1开始完全对应图片文件名））
                        if k not in gt:
                            gt[k] = []
                        gt[k].append(tube[i, 1:6].tolist())  # 在该帧的位置增加一个gt框列表，列表长度是5:label+xyxy
        for k in gt:  # 将同一个位置的所有gt框合成一个数组
            gt[k] = np.array(gt[k])

        # 将person proposal的格式调整为按照(vid, frame_id)查询的字典  有动作实例出现的帧数占所有有人出现的帧数的1/3
        self.person_proposal = {}
        for _ in person_proposal:
            iv = viddict[_.split(',')[0]]
            self.person_proposal[iv, int(_.split(',')[1])] = person_proposal[_]

        self.num_classes = 60
        self.gt = gt  # 存放标注的字典 用 元组(video_id,frame_id)索引
        self.gt_ids = sorted(self.gt.keys())
        self.num_samples = len(self.gt_ids)  # 样本数量=有标注的图片的总数

        if self.untrimmed:  # 需要训练集视频的所有帧进行训练，需要测试集视频的所有帧进行测试
            self.gt_ids_untrimmed = []
            for iv, video_name in enumerate(self.videolist):
                nframes = GT['nframes'][video_name]
                self.gt_ids_untrimmed.extend([(iv, frame_id) for frame_id in range(1, nframes+1)])
            self.num_samples = len(self.gt_ids_untrimmed)  # 样本数量=所有图片数量

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
            gt_id = self.gt_ids_untrimmed[index]
        else:
            gt_id = self.gt_ids[index]
        if gt_id in self.gt_ids:
            target = self.gt[gt_id]
        else:
            target = None

        # 模仿一个faster rcnn人员检测器，如果该图片上有人员检出则返回人员位置
        if gt_id in self.person_proposal:
            person_proposal = self.person_proposal[gt_id]
        else:
            person_proposal = None

        video_id = gt_id[0]
        frame_id = gt_id[1]
        img_name = os.path.join(self.videolist[video_id], '{:05d}.jpg'.format(frame_id))  # 该帧图片的完整名称

        # frame numbers  该视频的总帧数
        max_num = self.nframes[self.videolist[video_id]]

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
                path_tmp = os.path.join(self.data_root, 'trainval', 'rawframes', self.videolist[video_id],
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
                path_tmp = os.path.join(self.data_root, 'trainval', 'rawframes', self.videolist[video_id],
                                        '{:05d}.jpg'.format(frame_id_temp))
                frame = Image.open(path_tmp).convert('RGB')  # 一个Image对象
                ow, oh = frame.width, frame.height  # 图片的原始宽度和原始高度

                video_clip.append(frame)

        if target is not None:
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

            # reformat target
            targets = {
                'boxes': target[:, :4].float(),      # [N, 4]
                'labels': target[:, -1].long(),    # [N,]
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_id_and_frame_id': gt_id,   # （video_id，frame_id)
                'person_proposal': person_proposal  # ndarray(N,5)  bbox+conf
            }

        else:

            # transform后尺寸变为方形，video_clip是一个列表，每一项对应一帧图像的tensor，
            # transform对视频片段进行变换，RGB的值都除以了255变成了百分比表示
            # target是一个tensor对应唯一关键帧的标注（默认是两点小数格式)
            video_clip, target = self.transform(video_clip, target)
            # List [T, 3, H, W] -> [3, T, H, W]  将视频片段列表转换维度顺序，成为一个[3, T, H, W]维度的tensor
            video_clip = torch.stack(video_clip, dim=1)

            # reformat target
            targets = {
                'boxes': target,  # [N, 4]
                'labels': target,  # [N,]
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_id_and_frame_id': gt_id,  # （video_id，frame_id)
                'person_proposal': person_proposal  # ndarray(N,5)  bbox+conf
            }

        # img_name是该帧图片的名称带后缀
        # video_clip是一个[3, T, H, W]维度的tensor，表示该帧图片对应的视频片段，是方形的
        # target是一个字典，表示该帧图片的标注，边界框是默认为两点式百分比的tensor，类别是tensor
        return img_name, video_clip, targets
        

# Video Dataset for MultiSports数据集类，用于video级别的评估，每个视频作为一个独立的数据集实例
class MultiSports_VIDEO_Dataset(Dataset):
    def __init__(self,
                 data_root,
                 dataset='multisports',
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
        self.untrimmed = untrimmed

        GT_path = os.path.join(self.data_root, "trainval/multisports_GT.pkl")
        GT = pickle.load(open(GT_path, 'rb'))
        self.videolist = GT['test_videos'][0]  # 测试集视频名称

        person_proposal_dir = os.path.join(
            self.data_root, '/media/su/d/datasets/MultiSports/MultiSports_box')  # Faster R-CNN给出的人员提议
        person_proposal = pickle.load(open(os.path.join(person_proposal_dir, 'val.recall_96.13.pkl'), 'rb'))

        gt = {}
        viddict = {}  # 便于通过视频名称查询视频编号  给person proposal使用
        for iv, video_name in enumerate(self.videolist):

            # 视频名称与iv的对应关系字典
            viddict[video_name.split('/')[1]] = iv

            for label_id in GT['gttubes'][video_name].keys():  # 该视频的管道类别 类别从0开始
                if label_id in [15, 16, 17, 20, 64, 65]:
                    continue
                tubes_of_one_class = GT['gttubes'][video_name][label_id]
                label_id = label_id - sum(label_id > np.array([15, 16, 17, 20, 64, 65]))  # 修正到只有60个类别
                for tube in tubes_of_one_class:
                    tube = np.insert(tube, 1, label_id, axis=1)
                    for i in range(tube.shape[0]):  # tube长度内的帧索引
                        k = (iv, int(tube[i, 0]))  # 元组（视频序号(从0开始)，帧序号（从1开始完全对应图片文件名））
                        if k not in gt:
                            gt[k] = []
                        gt[k].append(tube[i, 1:6].tolist())  # 在该帧的位置增加一个gt框列表，列表长度是5:label+xyxy
        for k in gt:  # 将同一个位置的所有gt框合成一个数组
            gt[k] = np.array(gt[k])

        # 将person proposal的格式调整为按照(vid, frame_id)查询的字典  有动作实例出现的帧数占所有有人出现的帧数的1/3
        self.person_proposal = {}
        for _ in person_proposal:
            iv = viddict[_.split(',')[0]]
            self.person_proposal[iv, int(_.split(',')[1])] = person_proposal[_]

        self.gt = gt  # 存放标注的字典 用 元组(video_id,frame_id)索引
        self.gt_ids = sorted(self.gt.keys())
        self.nframes = GT['nframes']
        self.num_classes = 60

    def set_video_data(self, video_id):  # 输入一个测试集的视频序号
        self.video_id = video_id
        self.video_name = self.videolist[video_id]
        # frame numbers  该视频的总帧数
        self.max_num = self.nframes[self.video_name]
        # load a video该视频的完整路径
        self.img_folder = os.path.join(self.data_root, 'trainval', 'rawframes', self.video_name)
        self.img_paths = sorted(glob.glob(os.path.join(self.img_folder, '*.jpg')))  # 一个列表，存放该视频路径下的所有图片的完整路径
        if not self.untrimmed:
            self.gt_ids_trimmed = [(self.video_id, frame_id) for frame_id in range(1, self.max_num+1)
                                   if (self.video_id, frame_id) in self.gt_ids]
            self.max_num = len(self.gt_ids_trimmed)

    def __len__(self):
        return self.max_num

    def __getitem__(self, index):
        return self.pull_item(index)

    def pull_item(self, index):
        if self.untrimmed:
            frame_id = index + 1
            gt_id = (self.video_id, frame_id)  # 该图片的索引元组， 不一定有gt标注
        else:
            gt_id = self.gt_ids_trimmed[index]
            frame_id = gt_id[1]

        # 模仿一个faster rcnn人员检测器，如果该图片上有人员检出则返回人员位置
        if gt_id in self.person_proposal:
            person_proposal = self.person_proposal[gt_id]
        else:
            person_proposal = None

        # image name
        img_name = os.path.join(self.videolist[self.video_id], '{:05d}.jpg'.format(frame_id))  # 该帧图片的完整名称

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
                elif frame_id_temp > self.max_num:
                    frame_id_temp = self.max_num

                # load a frame
                path_tmp = os.path.join(self.data_root, 'trainval', 'rawframes', self.videolist[self.video_id],
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
                elif frame_id_temp > self.max_num:
                    frame_id_temp = self.max_num

                # load a frame
                path_tmp = os.path.join(self.data_root, 'trainval', 'rawframes', self.videolist[self.video_id],
                                        '{:05d}.jpg'.format(frame_id_temp))
                frame = Image.open(path_tmp).convert('RGB')
                ow, oh = frame.width, frame.height

                video_clip.append(frame)

        # 载入该图片对应的真实标注  因为video推断，也需要推断不含标注的图片，因此该图片可能没有对应的标注文件
        # load an annotation 如果存在，则载入该帧图像对应的标注文件，数组表示
        if gt_id in self.gt_ids:
            target = self.gt[gt_id]
            # [label, x1, y1, x2, y2] -> [x1, y1, x2, y2, label] 调整标注的排列方式，数组表示
            label = target[..., :1]
            boxes = target[..., 1:]
            target = np.concatenate([boxes, label], axis=-1).reshape(-1, 5)

            # transform对视频片段进行变换，RGB的值都除以了255变成了百分比表示
            video_clip, _ = self.transform(video_clip, normalize=False)
            # List [T, 3, H, W] -> [3, T, H, W]
            video_clip = torch.stack(video_clip, dim=1)

            # reformat target  默认是原始尺寸下的绝对坐标两点式
            targets = {
                'boxes': target[:, :4],  # [N, 4]
                'labels': target[:, -1],  # [N,]    类别从0开始
                'orig_size': [ow, oh],  # 图片的原始宽度和原始高度
                'video_id_and_frame_id': gt_id,  # （video_id，frame_id)
                'person_proposal': person_proposal  # ndarray(N,5)  bbox+conf
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
                'video_id_and_frame_id': gt_id,  # （video_id，frame_id)
                'person_proposal': person_proposal  # ndarray(N,5)  bbox+conf
            }

        return img_name, video_clip, targets


if __name__ == '__main__':
    import cv2
    from dataset.transforms import Augmentation, BaseTransform

    data_root = '/media/su/d/datasets/MultiSports'
    dataset = 'multisports'
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

    train_dataset = MultiSports_Dataset(
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
        
