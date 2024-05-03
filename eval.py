"""
该文件用于读取训练好的模型文件，然后评估和计算mAP
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import argparse
import torch


from evaluator.ucf_jhmdb_evaluator import UCF_JHMDB_Evaluator
from evaluator.ava_evaluator import AVA_Evaluator
from evaluator.multisports_evaluator import MultiSports_Evaluator

from dataset.transforms import BaseTransform

from utils.misc import load_weight, CollateFunc

from config import build_dataset_config, build_model_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, jhmdb21, ava_v2.2.')
    parser.add_argument('--data_root', default='/media/su/d/datasets/UCF24-YOWO/',
                        help='data root')

    # Batchsize
    parser.add_argument('-tbs', '--test_batch_size', default=8, type=int,
                        help='total test batch size')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')  # 训练好的模型文件完整路径
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')

    # Evaluation
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold. We suggest 0.005 for UCF24 and 0.1 for AVA.')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold. We suggest 0.5 for UCF24 and AVA.')
    parser.add_argument('--totaltopk', default=40, type=int,
                        help='topk prediction candidates.')  # 跨类别的所有预测框保留修正分类得分最高的topk个
    parser.add_argument('--infer_dir', default='./results/',
                        type=str, help='save inference results.')  # 评估的推断结果保存地址
    parser.add_argument('--map_path', default='./evaluator/eval_results/',
                        type=str, help='path to save mAP results')  # frame mAP结果的保存路径
    parser.add_argument('--cal_frame_mAP', action='store_true', default=False, 
                        help='calculate frame mAP.')  # 是否计算frame mAP
    parser.add_argument('--cal_video_mAP', action='store_true', default=False, 
                        help='calculate video mAP.')  # 是否计算video mAP
    parser.add_argument('--frame_det_dir', default=None,
                        type=str, help='path of frame_det_pkl')  # (包括video mAP计算中的)帧级检测结果的pkl文件，给定则直接读取
    parser.add_argument('--video_det_dir', default=None,
                        type=str, help='path to video_det_pkl')  # 视频级检测结果(管道)的pkl文件，给定则直接读取
    parser.add_argument('--link_method', default='viterbi',
                        type=str, help='tube link method')  # 关联算法
    parser.add_argument('--det_save_type', default='one_class',
                        type=str, help='')  # 默认one_class表示一个检测框只对应一个类别得分，multi_class表示一个检测框对应所有类别得分
    # 只有进行video mAP计算并且关联算法为ojla或者MCCLA时需要改为multi_class

    # Experiment
    parser.add_argument('--multi_sampling_rate', default=1, type=int,
                        help='number of sampling rates')  # 是否开启多采样率模块，几种采样率，1代表不开启，2代表2种（1、2），4代表4种（1、2、4、8）
    parser.add_argument('--nms_iou_type', default='iou', type=str,
                        help='iou type for calculating iou in nms.')  # 评估时nms计算中用到的IoU类型，可选'iou'、'giou'等
    parser.add_argument('--clstopk', default=0, type=int,
                        help='topk per cls after nms.')  # nms  默认0或者3，0表示不开启，3表示每个类别只保留最高的3个
    parser.add_argument('--untrimmed', action='store_true', default=False,
                        help='use untrimmed frames to train')  # 是否使用untrimmed frames进行训练以获得鉴定动作起始点的能力
    # 训练出的模型只用于评估untrimmed数据集上的video mAP
    parser.add_argument('--bbox_with_feat', action='store_true', default=False,
                        help='bbox_with_feat for MCCLA link')  # 是否保存bbox_feat用于MCCLA关联算法计算video mAP


    return parser.parse_args()


# 该函数首先建立一个评估器，然后进行评估
def ucf_jhmdb_eval(args, d_cfg, model, transform, collate_fn, epoch=1):
    if args.cal_frame_mAP:
        # Frame mAP evaluator 建立一个frame mAP评估器
        evaluator = UCF_JHMDB_Evaluator(
            args,
            metric='fmap',
            img_size=d_cfg['test_size'],
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            gt_folder=d_cfg['gt_folder'],
            )
        # evaluate
        evaluator.evaluate_frame_map(model, epoch=epoch, show_pr_curve=True)

    elif args.cal_video_mAP:
        # Video mAP evaluator  建立一个video mAP评估器
        evaluator = UCF_JHMDB_Evaluator(
            args,
            metric='vmap',
            img_size=d_cfg['test_size'],
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            frame_det_dir=args.frame_det_dir,
            video_det_dir=args.video_det_dir,
            link_method=args.link_method,
            bbox_with_feat=args.bbox_with_feat
            )
        # evaluate 评估video mAP
        evaluator.evaluate_video_map(model, epoch=epoch)


def ava_eval(args, d_cfg, model, transform, collate_fn):
    evaluator = AVA_Evaluator(
        d_cfg=d_cfg,
        data_root=args.data_root,
        img_size=d_cfg['test_size'],
        len_clip=args.len_clip,
        sampling_rate=d_cfg['sampling_rate'],
        batch_size=args.test_batch_size,
        transform=transform,
        collate_fn=collate_fn,
        full_test_on_val=False,
        version='v2.2')

    mAP = evaluator.evaluate_frame_map(model)


def multisports_eval(args, d_cfg, model, transform, collate_fn, epoch=1):
    if args.cal_frame_mAP:
        # Frame mAP evaluator 建立一个frame mAP评估器
        evaluator = MultiSports_Evaluator(
            args,
            metric='fmap',
            img_size=d_cfg['test_size'],
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            frame_det_dir=args.frame_det_dir
            )
        # evaluate
        evaluator.evaluate_frame_map(model, epoch=epoch, show_pr_curve=True)

    elif args.cal_video_mAP:
        # Video mAP evaluator  建立一个video mAP评估器
        evaluator = MultiSports_Evaluator(
            args,
            metric='vmap',
            img_size=d_cfg['test_size'],
            iou_thresh=0.5,
            transform=transform,
            collate_fn=collate_fn,
            frame_det_dir=args.frame_det_dir,
            video_det_dir=args.video_det_dir,
            link_method=args.link_method,
            bbox_with_feat=args.bbox_with_feat
        )
        # evaluate 评估video mAP
        evaluator.evaluate_video_map(model, epoch=epoch)


if __name__ == '__main__':
    args = parse_args()
    # dataset
    if args.dataset == 'ucf24':
        num_classes = 24

    elif args.dataset == 'jhmdb21':
        num_classes = 21

    elif args.dataset == 'ava_v2.2':
        num_classes = 80

    elif args.dataset == 'multisports':
        num_classes = 60

    else:
        print('unknown dataset.')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    if args.cal_video_mAP and args.link_method in ['ojla', 'MCCLA']:  # 当计算video mAP并且采用ojla关联算法时，det保存模式要更改，并且不能单类别topk
        args.det_save_type = 'multi_class'
        args.clstopk = 0  # 从而保留totaltopk的作用

    # build model
    model, _ = build_model(
        args=args, 
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )  # 构建完其实大多数参数还是在cpu中，只有损失函数、先验锚点框在指定device上

    # load trained weight
    model, epoch = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval  将batch normalization层和dropout层调整到评估模式    模型全部挪到指定device上
    model = model.to(device).eval()

    # transform
    # BaseTransform类的实例，用于进行图像变换，改变尺寸，能返回变换后的视频片段tensor和target tensor
    basetransform = BaseTransform(img_size=d_cfg['test_size'])

    # run
    if args.dataset in ['ucf24', 'jhmdb21']:
        ucf_jhmdb_eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc(),
            epoch=epoch
            )  # epoch是为了直接使用帧级推断好的结果文件夹，CollateFunc类将一个批次的输入进行整理，分别返回批次内的关键帧id列表、批次内的视频片段tensor、批次内的关键帧标注列表
    elif args.dataset == 'ava_v2.2':
        ava_eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc()
            )
    elif args.dataset == 'multisports':
        multisports_eval(
            args=args,
            d_cfg=d_cfg,
            model=model,
            transform=basetransform,
            collate_fn=CollateFunc(),
            epoch=epoch
            )  # epoch是为了直接使用帧级推断好的结果文件夹，CollateFunc类将一个批次的输入进行整理，分别返回批次内的关键帧id列表、批次内的视频片段tensor、批次内的关键帧标注列表
