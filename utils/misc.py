import os

import torch
import torch.nn as nn

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.ava import AVA_Dataset
from dataset.multisports import MultiSports_Dataset
from dataset.transforms import Augmentation, BaseTransform

from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from evaluator.ava_evaluator import AVA_Evaluator
from evaluator.multisports_evaluator import MultiSports_Evaluator


# 构建训练集实例、frame mAP评估器实例(包含测试集实例，如果要求进行评估)并返回    目前只用于训练中
def build_dataset(d_cfg, args, is_train=False):
    """
        d_cfg: dataset config
    """
    # transform
    augmentation = Augmentation(
        img_size=d_cfg['train_size'],
        jitter=d_cfg['jitter'],
        hue=d_cfg['hue'],
        saturation=d_cfg['saturation'],
        exposure=d_cfg['exposure']
        )  # 训练时用，该数据增强实例，输入视频片段和真实标注框，进行数据增强，返回增强后(正方形)的视频片段和增强后的真实标注框(百分比两点式)
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        )  # 测试时用，基本变换，包括尺寸变为正方形，真实标注变为百分比模式

    # dataset
    if args.dataset in ['ucf24', 'jhmdb21']:
        # dataset  构建训练集实例
        dataset = UCF_JHMDB_Dataset(
            data_root=args.data_root,
            dataset=args.dataset,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            is_train=is_train,
            len_clip=args.len_clip,
            multi_sampling_rate=args.multi_sampling_rate,
            untrimmed=args.untrimmed    # 训练和frame级别的评估
            )
        num_classes = dataset.num_classes

        # evaluator  评估器只用评估frame mAP，建立一个frame mAP评估器，包含构建测试集实例（一定是trimmed）
        evaluator = UCF_JHMDB_Evaluator(
            args,
            metric='fmap',
            img_size=d_cfg['test_size'],
            iou_thresh=0.5,
            transform=basetransform,
            collate_fn=CollateFunc(),
            gt_folder=d_cfg['gt_folder'],
        )

    elif args.dataset == 'ava_v2.2':
        data_dir = os.path.join(args.data_root, 'AVA_Dataset')
        
        # dataset
        dataset = AVA_Dataset(
            cfg=d_cfg,
            data_root=data_dir,
            is_train=True,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
        )
        num_classes = 80

        # evaluator
        evaluator = AVA_Evaluator(
            d_cfg=d_cfg,
            data_root=data_dir,
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate'],
            batch_size=args.test_batch_size,
            transform=basetransform,
            collate_fn=CollateFunc(),
            full_test_on_val=False,
            version='v2.2'
            )

    elif args.dataset == 'multisports':
        # dataset  构建训练集实例
        dataset = MultiSports_Dataset(
            data_root=args.data_root,
            dataset=args.dataset,
            img_size=d_cfg['train_size'],
            transform=augmentation,
            is_train=is_train,
            len_clip=args.len_clip,
            multi_sampling_rate=args.multi_sampling_rate,
            untrimmed=args.untrimmed
        )
        num_classes = dataset.num_classes

        # evaluator  评估器只用评估frame mAP，建立一个frame mAP评估器，包含构建测试集实例（一定是trimmed）
        evaluator = MultiSports_Evaluator(
            args,
            metric='fmap',
            img_size=d_cfg['test_size'],
            iou_thresh=0.5,
            transform=basetransform,
            collate_fn=CollateFunc(),
        )

    else:
        print('unknown dataset !! Only support ucf24 & jhmdb21 & ava_v2.2 !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    if not args.eval:  # 是否进行评估
        # no evaluator during training stage
        evaluator = None

    return dataset, evaluator, num_classes


# 该函数创建一个dataloader，目前只用于训练中
def build_dataloader(args, dataset, batch_size, collate_fn=None):
    # distributed 如果采用分布式训练
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    # 丢弃最后不足以构成一个batch的样本  batch size是每一个进程的batch size
    batch_sampler_train = torch.utils.data.BatchSampler(sampler,
                                                        batch_size,
                                                        drop_last=True)
    # train dataloader  训练用的dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
        )
    return dataloader


# 评估和测试时用来载入训练好的模型权重
def load_weight(model, path_to_ckpt=None):
    if path_to_ckpt is None:
        print('No trained weight ..')
        return model
        
    checkpoint = torch.load(path_to_ckpt, map_location=torch.device('cpu'))
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check  检查不匹配的权重参数
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                print(k)
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    epoch = checkpoint.pop("epoch")
    print('Finished loading model!')

    return model, epoch


def is_parallel(model):  # 判断模型是否是并行
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


# 该类将一个批次的输入进行整理，分别返回批次内的关键帧id列表、批次内的视频片段tensor、批次内的关键帧真实标注列表  训练和评估的dataloader中都会用到
class CollateFunc(object):
    def __call__(self, batch):
        batch_frame_id = []
        batch_key_target = []
        batch_video_clips = []

        for sample in batch:
            key_frame_id = sample[0]
            video_clip = sample[1]
            key_target = sample[2]
            
            batch_frame_id.append(key_frame_id)
            batch_video_clips.append(video_clip)
            batch_key_target.append(key_target)

        # List [B, 3, T, H, W] -> [B, 3, T, H, W]
        batch_video_clips = torch.stack(batch_video_clips)
        
        return batch_frame_id, batch_video_clips, batch_key_target
