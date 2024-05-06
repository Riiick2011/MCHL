"""
该文件定义了评估器类，包含计算frame mAP的类方法和计算video mAP的类方法
"""
import os
import pickle
import torch
import numpy as np
from scipy.io import loadmat

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset, UCF_JHMDB_VIDEO_Dataset
from utils.box_ops import rescale_bboxes

from .cal_frame_mAP import evaluate_frameAP
from evaluator.link_method import tube_link
from utils.utils import iou3dt, pr_to_ap_voc


class UCF_JHMDB_Evaluator(object):  # 评估器类
    def __init__(self,
                 args,
                 metric='fmap',
                 img_size=224,
                 iou_thresh=0.5,
                 transform=None,
                 collate_fn=None,
                 gt_folder=None,
                 frame_det_dir=None,
                 video_det_dir=None,
                 link_method='viterbi',
                 bbox_with_feat=False
                 ):
        self.data_root = args.data_root
        self.dataset = args.dataset
        self.model_name = args.version
        self.img_size = img_size
        self.len_clip = args.len_clip
        self.test_batch_size = args.test_batch_size
        self.iou_thresh = iou_thresh  # frame_mAP的iou_thresh
        self.collate_fn = collate_fn
        self.infer_dir = args.infer_dir

        # 只用于frame mAP计算
        self.gt_folder = gt_folder
        self.map_path = args.map_path  # 保存评估结果的路径

        # 只用于video mAP计算
        self.gt_dir = os.path.join(self.data_root, 'splitfiles/finalAnnots.mat')  # ucf24测试集的真实管道标注 jhmdb需要自己关联
        # 可以直接读取的结果文件路径，如果有的话
        self.frame_det_dir = frame_det_dir     # 视频数据集的帧级检测结果   只对ucf有效
        self.video_det_dir = video_det_dir     # 视频级检测结果，关联好的tube  只对ucf有效
        self.link_method = link_method
        self.videolist = []  # 存放测试集的视频文件名
        self.det_save_type = args.det_save_type  # 当计算v-mAP并且关联算法是多类别算法时 det_save_type会变为'multi_class'
        # 只用于video mAP的MCHL算法
        self.video_root = None
        if not bbox_with_feat:
            self.video_root = os.path.join(self.data_root, 'rgb-images')  # 视频的根目录

        with open(os.path.join(self.data_root, 'testlist_video.txt'), 'r') as file:
            lines = file.readlines()
            for line in lines:
                line = line.rstrip()
                self.videolist.append(line)
        self.videolist.sort()


        # dataset 构建测试集对象
        if metric == 'fmap':
            self.testset = UCF_JHMDB_Dataset(
                data_root=self.data_root,
                dataset=self.dataset,
                img_size=img_size,
                transform=transform,
                is_train=False,
                len_clip=self.len_clip,
                multi_sampling_rate=args.multi_sampling_rate)  # 构建专用的测试集，通过is_train来选择是训练集还是测试集
            self.num_classes = self.testset.num_classes
        elif metric == 'vmap':
            self.testset = UCF_JHMDB_VIDEO_Dataset(
                data_root=self.data_root,
                dataset=self.dataset,
                img_size=img_size,
                transform=transform,
                len_clip=self.len_clip,
                multi_sampling_rate=args.multi_sampling_rate,
                untrimmed=args.untrimmed)   # 根据需要和数据集类型创建untrimmed数据集
            self.num_classes = self.testset.num_classes

    def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):  # 评估frame mAP
        print("Metric: Frame mAP")
        # dataloader
        self.testloader = torch.utils.data.DataLoader(
            dataset=self.testset, 
            batch_size=self.test_batch_size,
            shuffle=False,
            collate_fn=self.collate_fn, 
            num_workers=4,
            drop_last=False,
            pin_memory=True
            )  # collate_fn函数将一个批次的输入进行整理，分别返回批次内的关键帧id列表、批次内的视频片段tensor、批次内的关键帧标注列表
        
        epoch_size = len(self.testloader)

        # inference
        # 检查和创建推断结果文件夹
        inference_dir = os.path.join(self.infer_dir, '{}'.format(self.dataset), self.model_name, 'epoch_' + str(epoch))
        if not os.path.exists(inference_dir):
            print('Creating Inference Dir：' + inference_dir)
            os.makedirs(inference_dir, exist_ok=True)
            # 如果不存在推断结果则开始推断
            for batch_idx, (batch_img_name, batch_video_clip, batch_target) in enumerate(self.testloader):
                # to device 把视频片段tensor移动到指定设备上
                batch_video_clip = batch_video_clip.to(model.device)

                with torch.no_grad():  # 表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建
                    # inference  模型中的inference方法  返回的3项均为列表，列表中每一项对应一个样本的结果，bbox是两点式归一化
                    batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

                    # process batch
                    for bi in range(len(batch_scores)):
                        img_name = batch_img_name[bi]
                        scores = batch_scores[bi]
                        labels = batch_labels[bi]
                        bboxes = batch_bboxes[bi]
                        target = batch_target[bi]

                        # rescale bbox  将两点式归一化小数表示的预测框变换为原始图片尺寸下的两点式绝对坐标表示  以便于跟真实标注比较
                        orig_size = target['orig_size']
                        bboxes = rescale_bboxes(bboxes, orig_size)

                        infer_name = img_name.replace('/', '_')[:-3] + 'txt'  # 用于保存该帧推断结果的文件名
                        detection_path = os.path.join(inference_dir, infer_name)  # 用于保存该帧推断结果的完整文件路径

                        with open(detection_path, 'w+') as f_detect:
                            for score, label, bbox in zip(scores, labels, bboxes):
                                x1 = round(bbox[0])
                                y1 = round(bbox[1])
                                x2 = round(bbox[2])
                                y2 = round(bbox[3])
                                cls_id = int(label) + 1  # 原标注的类别从1开始，0是背景类别

                                f_detect.write(
                                    str(cls_id) + ' ' + str(score) + ' '
                                    + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

                    if batch_idx % 100 == 0:
                        log_info = "[%d / %d]" % (batch_idx, epoch_size)
                        print(log_info, flush=True)

        print('calculating Frame mAP ...')
        metric_list = evaluate_frameAP(self.gt_folder, inference_dir, self.iou_thresh, self.map_path, self.dataset,
                                       show_pr_curve)  # 本函数根据测试集中每一帧的单帧推断结果 来计算整个测试集上的frame mAP
        for metric in metric_list:
            print(metric)

        # 在一个txt中记录fmap
        fmaptxt = inference_dir[:inference_dir.rfind('/')] + '/fmap.txt'  # 存放fmap结果的txt文件
        with open(fmaptxt, 'a+') as f:
            f.write('epoch:{}\n'.format(epoch))
            for metric in metric_list:
                f.write(metric)
            f.write('\n')

    def evaluate_video_map(self, model, epoch=1):
        print("Metric: Video mAP")
        all_gts = {}  # jhmdb不存在管道标注，只有帧级检测框标注，需要将同一个视频的所有真实帧级检测框标注合并成真实管道标注再计算video_mAP
        all_dets = {}

        if self.dataset == 'ucf24':

            # 读取真实管道标注
            gt_data = loadmat(self.gt_dir)['annot']  # ucf的管道标注存放在mat文件中
            n_videos = gt_data.shape[1]
            print('loading gt tubes ...')  # 读取真实管道标注
            for i in range(n_videos):
                video_name = gt_data[0][i][1][0]
                if video_name in self.videolist:
                    n_tubes = len(gt_data[0][i][2][0])
                    v_annotation = {}
                    all_gt_boxes = []
                    for j in range(n_tubes):  # 利用了同一个视频中只有一种类别的管道的先验知识
                        gt_one_tube = []
                        tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                        tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                        tube_class = gt_data[0][i][2][0][j][2][0][0] - 1   # 类别改为从0开始计数
                        tube_data = gt_data[0][i][2][0][j][3]
                        tube_length = tube_end_frame - tube_start_frame + 1

                        for k in range(tube_length):  # 每一个管道
                            # gt_boxes是一个列表，包含5项，对应帧序号、一帧上的一个检测框的点模式坐标
                            gt_boxes = [int(tube_start_frame+k),
                                        float(tube_data[k][0]),
                                        float(tube_data[k][1]),
                                        float(tube_data[k][0]) + float(tube_data[k][2]),
                                        float(tube_data[k][1]) + float(tube_data[k][3])]
                            gt_one_tube.append(gt_boxes)
                        all_gt_boxes.append(np.array(gt_one_tube))  # 包含管道个数项，每一项是一个数组

                    v_annotation[tube_class] = all_gt_boxes
                    all_gts[video_name] = v_annotation  # {video_name:{tube_class:[tube1(array),tube2(array),..]}}

            # 判断是否需要关联或者推断
            if not self.video_det_dir:
                self.video_det_dir = os.path.join(
                    self.infer_dir, 'ucf24', self.model_name, 'video_det_' + str(epoch) + '.pkl')
                if not self.frame_det_dir:  # 如果不存在视频数据集的帧级检测结果，则重新进行推断
                    frame_det_dir = os.path.join(
                        self.infer_dir, 'ucf24', self.model_name, 'frame_det_in_vmap_' + str(epoch) + '.pkl')
                    os.makedirs(os.path.dirname(frame_det_dir), exist_ok=True)
                    # inference  对frame级检测结果进行关联
                    print('inference ...')
                    for video_id, video_name in enumerate(self.videolist):
                        if video_id % 50 == 0:
                            print('Video: [%d / %d] - %s' % (video_id, len(self.videolist), video_name))
                        # set video  根据该视频设置视频测试集实例的索引长度和索引路径
                        self.testset.set_video_data(video_name)
                        # dataloader
                        self.testloader = torch.utils.data.DataLoader(
                            dataset=self.testset,
                            batch_size=self.test_batch_size,
                            shuffle=False,
                            collate_fn=self.collate_fn,
                            num_workers=4,
                            drop_last=False,
                            pin_memory=True
                            )

                        dets_video = {}
                        for batch_idx, (batch_img_name, batch_video_clip, batch_target) in enumerate(self.testloader):
                            # to device
                            batch_video_clip = batch_video_clip.to(model.device)

                            with torch.no_grad():
                                # model.inference
                                batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

                                # process batch
                                for bi in range(len(batch_scores)):
                                    scores = batch_scores[bi]
                                    labels = batch_labels[bi]
                                    bboxes = batch_bboxes[bi]
                                    target = batch_target[bi]

                                    # rescale bbox  将两点式归一化小数表示的预测框变换为原始图片尺寸下的两点式绝对坐标表示
                                    orig_size = target['orig_size']
                                    bboxes = rescale_bboxes(bboxes, orig_size)

                                    # generate detected tubes for all classes 保存格式
                                    frame_id = target['frame_id']
                                    if self.det_save_type == 'multi_class':  # ojla不分类别
                                        if len(scores) == 0:  # 如果该样本没有检测则直接跳到下一个样本
                                            continue
                                        c_bboxes = bboxes
                                        c_scores = scores
                                        c_labels = labels
                                        # [n_box, 6]
                                        boxes = np.concatenate([np.repeat(frame_id, c_scores.shape[0])[..., None],
                                                                c_bboxes, c_scores[..., None], c_labels], axis=-1)
                                        if self.det_save_type not in dets_video.keys():
                                            dets_video[self.det_save_type] = []  # 该视频关于该类别新建一个列表，每一项是一个数组对应一帧
                                        dets_video[self.det_save_type].append(boxes)
                                    else:
                                        for cls_idx in range(self.num_classes):
                                            inds = np.where(labels == cls_idx)[0]
                                            if len(inds) == 0:  # 如果该样本没有该类别的检测则直接跳下一类别
                                                continue
                                            c_bboxes = bboxes[inds]
                                            c_scores = scores[inds]
                                            # [n_box, 6]
                                            boxes = np.concatenate([np.repeat(frame_id, c_scores.shape[0])[..., None],
                                                                    c_bboxes, c_scores[..., None]], axis=-1)
                                            if cls_idx not in dets_video.keys():
                                                dets_video[cls_idx] = []  # 该视频关于该类别新建一个列表，每一项是一个数组对应一帧
                                            dets_video[cls_idx].append(boxes)

                        all_dets[video_id] = dets_video  # 用video_id索引
                        # delete testloader
                        del self.testloader
                    print('saving frame_dets to ' + frame_det_dir)
                    pickle.dump(all_dets, open(frame_det_dir, 'wb'))
                else:  # 存在视频集上的帧级别检测结果，则直接读取    嵌套字典，第一层是video_id,第二层是cls_idx
                    frame_det_dir = self.frame_det_dir
                    print('loading frame_dets from ' + frame_det_dir)
                    all_dets = pickle.load(open(frame_det_dir, 'rb'))
                # 管道关联 并保存
                all_tubes = tube_link(all_dets, num_classes=self.num_classes, videolist=self.videolist,
                                      link_method=self.link_method, video_root=self.video_root)
                print('saving video_dets to ' + self.video_det_dir)
                pickle.dump(all_tubes, open(self.video_det_dir, 'wb'))
            else:
                print('loading video_dets from ' + self.video_det_dir)
                all_tubes = pickle.load(open(self.video_det_dir, 'rb'))

        elif self.dataset == 'jhmdb21':
            # inference  对frame级检测结果进行关联 不管如何都得重新推断和关联
            print('inference ...')
            for video_id, video_name in enumerate(self.videolist):
                if video_id % 50 == 0:
                    print('Video: [%d / %d] - %s' % (video_id, len(self.videolist), video_name))
                # set video 根据该视频设置视频测试集实例的索引长度和索引路径
                self.testset.set_video_data(video_name)
                # dataloader
                self.testloader = torch.utils.data.DataLoader(
                    dataset=self.testset,
                    batch_size=self.test_batch_size,
                    shuffle=False,
                    collate_fn=self.collate_fn,
                    num_workers=4,
                    drop_last=False,
                    pin_memory=True
                    )

                v_annotation = {}
                all_gt_boxes = []
                gt_one_tube = []
                dets_video = {}

                for batch_idx, (batch_img_name, batch_video_clip, batch_target) in enumerate(self.testloader):
                    # to device
                    batch_video_clip = batch_video_clip.to(model.device)

                    with torch.no_grad():
                        # model.inference
                        batch_scores, batch_labels, batch_bboxes = model(batch_video_clip)

                        # process batch
                        for bi in range(len(batch_scores)):
                            img_name = batch_img_name[bi]
                            scores = batch_scores[bi]
                            labels = batch_labels[bi]
                            bboxes = batch_bboxes[bi]
                            target = batch_target[bi]

                            # rescale bbox 将两点式归一化小数表示的预测框变换为原始图片尺寸下的两点式绝对坐标表示
                            orig_size = target['orig_size']
                            bboxes = rescale_bboxes(bboxes, orig_size)
                            frame_id = target['frame_id']

                            # generate corresponding gts
                            # save format: {v_name: {tubes: [[frame_index, x1,y1,x2,y2]], gt_classes: vlabel}}
                            # 重做gt
                            if target['labels'] is None:  # jhmdb测试集上有少数帧是无标注的 直接跳过不保存gt和det
                                # print('no ground-truth label at %s' % img_name)
                                continue
                            else:
                                tube_class = int(target['labels'][0])  # 建立在jhmdb数据集中一个视频只有一种类别标注的认识上
                                num_gts = len(target['labels'])  # 该帧图片的真实标注边界框数量
                                # print('%d ground-truth label at %s' % (num_gts, img_name))
                                for g in range(num_gts):
                                    gt_boxes = [frame_id]
                                    gt_boxes.extend(target['boxes'][g])
                                    gt_one_tube.append(gt_boxes)  # 建立在jhmdb数据集中一个视频只有一条管道的认识上

                            # generate detected tubes for all classes 保存格式
                            if self.det_save_type == 'multi_class':  # ojla不分类别
                                if len(scores) == 0:  # 如果该样本没有检测则直接跳到下一个样本
                                    continue
                                c_bboxes = bboxes
                                c_scores = scores
                                c_labels = labels
                                # [n_box, 6]
                                boxes = np.concatenate([np.repeat(frame_id, c_scores.shape[0])[..., None],
                                                        c_bboxes, c_scores[..., None], c_labels], axis=-1)
                                if self.det_save_type not in dets_video.keys():
                                    dets_video[self.det_save_type] = []  # 该视频关于该类别新建一个列表，每一项是一个数组对应一帧
                                dets_video[self.det_save_type].append(boxes)
                            else:
                                for cls_idx in range(self.num_classes):
                                    inds = np.where(labels == cls_idx)[0]
                                    if len(inds) == 0:  # 如果该样本没有该类别的检测则直接跳下一类别
                                        continue
                                    c_bboxes = bboxes[inds]
                                    c_scores = scores[inds]
                                    # [n_box, 6]
                                    boxes = np.concatenate([np.repeat(frame_id, c_scores.shape[0])[..., None],
                                                            c_bboxes, c_scores[..., None]], axis=-1)
                                    if cls_idx not in dets_video.keys():
                                        dets_video[cls_idx] = []  # 该视频关于该类别新建一个列表，每一项是一个数组对应一帧
                                    dets_video[cls_idx].append(boxes)

                all_gt_boxes.append(np.array(gt_one_tube))
                v_annotation[tube_class] = all_gt_boxes
                all_gts[video_name] = v_annotation  # {video_name:{tube_class:[tube1(array),tube2(array),..]}}

                all_dets[video_id] = dets_video  # 用video_id索引
                # delete testloader
                del self.testloader

            # 管道关联
            all_tubes = tube_link(all_dets, num_classes=self.num_classes, videolist=self.videolist,
                                  link_method=self.link_method, video_root=self.video_root)

        iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
        print('calculating video mAP ...')
        for iou_thresh in iou_list:
            per_ap = self.calculate_videomAP(all_gts, all_tubes, iou_thresh)  # 计算video mAP
            print('-------------------------------')
            print('V-mAP @ {} IoU:'.format(iou_thresh))
            print('--Per AP: \n', per_ap)

    def calculate_videomAP(self, all_gts, all_tubes, iou_thresh):
        # all_gts是一个字典，索引是video_name，每一项是一个字典, 索引是该视频具有的管道类别，每一类对应一个列表，
        # 列表内包含该类别的管道数量个数组，每个数组对应一个管道

        # all_tubes[cls_ind]是个列表，每一项是一个元组(video_name, tube_score, tube_bboxes) 对应一个tube
        # return  [label, video_index, [[frame_index, x1,y1,x2,y2], [], []] ]
        pr_all = {}
        for cls_ind in range(self.num_classes):
            # 筛选该类别的gts和dets
            # gts_cls是面向该类别的字典，索引是video_name，值是列表，列表内每一项是一个(tube_length,5(frame_id,bbox))形状的数组对应一个管道
            gts_cls = {video_name: all_gts[video_name][cls_ind] for video_name in all_gts.keys()
                       if cls_ind in all_gts[video_name]}
            dets_cls = all_tubes[cls_ind]

            # 计算pr
            pr = np.empty((len(dets_cls) + 1, 2), dtype=np.float32)  # precision, recall
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0
            gt_num_cls = sum([len(gts) for gts in gts_cls.values()])
            fp = 0  # false positives
            tp = 0  # true positives
            is_gt_box_detected = {}
            # sort tubes according to scores (descending order) 对dets_cls中的管道按照管道得分排序
            argsort_scores = np.argsort(-np.array([tube_tuple[1] for tube_tuple in dets_cls]))
            for i, tube_id in enumerate(argsort_scores):
                video_name, tube_score, tube = dets_cls[tube_id]
                ispositive = False
                if video_name in gts_cls:  # 将该管道分配给该视频下3DIoU重合程度最高的gt管道
                    if video_name not in is_gt_box_detected:  # 该字典用于记录gt管道是否已经被分配出去
                        is_gt_box_detected[video_name] = np.zeros(len(gts_cls[video_name]), dtype=bool)
                    ious = [iou3dt(gt, tube) for gt in gts_cls[video_name]]
                    for k in reversed(np.argsort(ious)):  # 从高到低
                        # 如果该gt没有被分配出去并且iou高于阈值，则分配给该det
                        if (not is_gt_box_detected[video_name][k]) and (ious[k] >= iou_thresh):
                            ispositive = True
                            is_gt_box_detected[video_name][k] = True
                            break
                if ispositive:
                    tp += 1
                else:
                    fp += 1
                pr[i + 1, 0] = float(tp) / float(tp + fp)
                pr[i + 1, 1] = float(tp) / float(gt_num_cls + 0.00001)

            pr_all[cls_ind] = pr

        # display results
        AP_res = ''
        ap = 100 * np.array([pr_to_ap_voc(pr_all[cls_ind]) for cls_ind in pr_all])
        for cls_ind in range(len(ap)):  # 从0~23
            ap_str = "{0:.2f}%".format(ap[cls_ind])
            AP_res += ('AP: %s (%d) \n' % (ap_str, cls_ind))
        video_map = np.mean(ap)  # 有效类别取平均
        map_str = "{0:.2f}%".format(video_map)
        AP_res += ('mAP: %s \n' % map_str)
        return AP_res


if __name__ == "__main__":
    pass
