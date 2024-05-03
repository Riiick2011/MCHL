"""
该文件用于读取训练好的模型文件，然后对一个样本进行推断
该文件定义了命令行解析函数、帧级推断函数
"""

import argparse
import cv2
import os
import time
import numpy as np
import torch

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.ava import AVA_Dataset
from dataset.transforms import BaseTransform

from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import convert_tensor_to_cv2img, vis_detection

from config import build_dataset_config, build_model_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24,jhdmb21,ava.')
    parser.add_argument('--data_root', default='/media/su/d/datasets/UCF24-YOWO/',
                        help='data root')

    # model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('-ct', '--conf_thresh', default=0.1, type=float,
                        help='confidence threshold')
    parser.add_argument('-nt', '--nms_thresh', default=0.5, type=float,
                        help='NMS threshold')
    parser.add_argument('--totaltopk', default=40, type=int,
                        help='save topk dets')  # 跨类别的所有预测框保留修正分类得分最高的topk个

    # basic
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--save', action='store_true', default=False, 
                        help='save detection results.')
    parser.add_argument('-vs', '--vis_thresh', default=0.4, type=float,
                        help='threshold for visualization')  # 可视化的阈值，得分高于该阈值的检测框才进行可视化展示
    parser.add_argument('-sid', '--start_index', default=0, type=int,
                        help='start index to test.')  # 从第几个样本开始测试
    parser.add_argument('--infer_dir', default='./results/', type=str,
                        help='Dir to save inference results')  # 用于保存绘制了检测框的图片的文件夹，最好改为绝对路径

    return parser.parse_args()

    
@torch.no_grad()  # 该函数不跟踪梯度，该函数用于进行推断
def inference_ucf24_jhmdb21(args, model, device, dataset, class_names=None, class_colors=None):
    # 创建保存检测结果的路径
    if args.save:
        save_path = os.path.join(
            args.infer_dir, args.dataset[:-2] + '_detections',
            args.version, 'vis_det')
        os.makedirs(save_path, exist_ok=True)

    # inference  没有batch size的考虑，所以更不需要dataloader
    for index in range(args.start_index, len(dataset)):  # 从指定的起始样本开始到整个数据集结束，一个样本对应一帧有标注的图片也对应包含该图片的一个视频片段
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        frame_id, video_clip, target = dataset[index]  # 返回 该帧图片的完整名称，视频片段，真实标注（字典形式)

        orig_size = tuple(target['orig_size'])  # width, height # 一个长度为2的列表，表示图片的原始宽度和原始高度

        # prepare  视频片段放到指定设备上
        video_clip = video_clip.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

        t0 = time.time()
        # inference 推断得到得分、标签、检测框   均为列表，每一项对应一个样本的结果
        batch_scores, batch_labels, batch_bboxes = model(video_clip)
        print("inference time ", time.time() - t0, "s")

        # batch size = 1 因为测试时批次大小为1，所以去掉批次维度
        scores = batch_scores[0]
        labels = batch_labels[0]
        bboxes = batch_bboxes[0]
        
        # rescale  将小数表示的检测框变换为原始绝对坐标表示    原来的bbox是两点式 全局小数表示
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame  关键帧可视化
        key_frame_tensor = video_clip[0, :, -1, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor)  # 变换回图像格式(H,W,RGB)

        # resize key_frame to orig size 恢复到原始尺寸
        key_frame = cv2.resize(key_frame, orig_size)

        # visualize detection 检测结果可视化，显示和保存
        # 本函数将检测结果加到原始图片上，生成包含检测框的图片并返回
        vis_results = vis_detection(
            frame=key_frame,
            scores=scores,
            labels=labels,
            bboxes=bboxes,
            vis_thresh=args.vis_thresh,
            class_names=class_names,
            class_colors=class_colors
            )

        if args.show:
            cv2.imshow('key-frame detection', vis_results)
            cv2.waitKey(10)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path, '{:0>5}.jpg'.format(index)), vis_results)
            print('finish')


@torch.no_grad()
def inference_ava(args, model, device, dataset, class_names=None, class_colors=None):
    # path to save 
    if args.save:
        save_path = os.path.join(
            args.save_folder, args.dataset, 
            args.version, 'video_clips')
        os.makedirs(save_path, exist_ok=True)

    # inference
    for index in range(args.start_index, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        frame_id, video_clip, target = dataset[index]
        orig_size = target['orig_size']  # width, height

        # prepare
        video_clip = video_clip.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

        t0 = time.time()
        # inference
        batch_bboxes = model(video_clip)
        print("inference time ", time.time() - t0, "s")

        # vis results of key-frame
        key_frame_tensor = video_clip[0, :, -1, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor)
        
        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)

        # batch size = 1
        bboxes = batch_bboxes[0]

        # visualize detection results
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            det_conf = float(bbox[4])
            cls_scores = np.sqrt(det_conf * bbox[5:])
        
            # rescale bbox
            x1, x2 = int(x1 * orig_size[0]), int(x2 * orig_size[0])
            y1, y2 = int(y1 * orig_size[1]), int(y2 * orig_size[1])

            # thresh
            indices = np.where(cls_scores > args.vis_thresh)
            scores = cls_scores[indices]
            indices = list(indices[0])
            scores = list(scores)

            if len(scores) > 0:
                # draw bbox
                cv2.rectangle(key_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # draw text
                blk   = np.zeros(key_frame.shape, np.uint8)
                font  = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text  = []
                text_size = []
                # scores, indices  = [list(a) for a in zip(*sorted(zip(scores,indices), reverse=True))] # if you want, you can sort according to confidence level
                for _, cls_ind in enumerate(indices):
                    text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                    text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                    coord.append((x1+3, y1+7+10*_))
                    cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                key_frame = cv2.addWeighted(key_frame, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(key_frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)
        
        if args.show:
            cv2.imshow('key-frame detection', key_frame)
            cv2.waitKey(0)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path,
            '{:0>5}.jpg'.format(index)), key_frame)
        

if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")  # 使用的是可见GPU的第0个，这里可以指定一下
    else:
        device = torch.device("cpu")

    # config 两个字典，包含对应该数据集的配置文件和对应该网络类型的配置文件
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # BaseTransform类的实例，用于进行图像变换，改变尺寸，能返回变换后的视频片段tensor和target tensor
    basetransform = BaseTransform(img_size=d_cfg['test_size'])

    # dataset 创建数据集实例
    if args.dataset in ['ucf24', 'jhmdb21']:
        dataset = UCF_JHMDB_Dataset(
            data_root=args.data_root,
            dataset=args.dataset,
            img_size=d_cfg['test_size'],
            transform=basetransform,
            is_train=False,
            len_clip=args.len_clip,
            )  # 通过is_train来选择是训练集还是测试集
        class_names = d_cfg['label_map']
        num_classes = dataset.num_classes

    elif args.dataset == 'ava_v2.2':
        data_dir = os.path.join(args.data_root, 'AVA_Dataset')
        dataset = AVA_Dataset(
            cfg=d_cfg,
            data_root=data_dir,
            is_train=False,
            img_size=d_cfg['test_size'],
            transform=basetransform,
            len_clip=args.len_clip,
            sampling_rate=d_cfg['sampling_rate']
        )
        class_names = d_cfg['label_map']
        num_classes = dataset.num_classes

    else:
        print('unknow dataset !! Only support ucf24 & jhmdb21 & ava_v2.2 and coco !!')
        exit(0)

    np.random.seed(100)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]   # 画图用，每一个类别对应一个随机颜色

    # build model  默认是训练模式   trainable是一个表明处于训练模式的参数
    model, _ = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
        )  # 构建完其实大多数参数还是在cpu中，只有损失函数、先验锚点框在指定device上

    # load trained weight  读取训练好的模型权重
    model, epoch = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval 模型挪动到指定设备上并切换为eval模式  将batch normalization层和dropout层调整到评估模式
    model = model.to(device).eval()

    # run
    if args.dataset in ['ucf24', 'jhmdb21']:
        inference_ucf24_jhmdb21(
            args=args,
            model=model,
            device=device,
            dataset=dataset,
            class_names=class_names,
            class_colors=class_colors
            )
    elif args.dataset in ['ava_v2.2']:
        inference_ava(
            args=args,
            model=model,
            device=device,
            dataset=dataset,
            class_names=class_names,
            class_colors=class_colors
            )
