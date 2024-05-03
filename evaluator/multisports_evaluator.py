"""
该文件定义了评估器类，包含计算frame mAP的类方法和计算video mAP的类方法
"""
import os
import pickle
import torch
import numpy as np

from dataset.multisports import MultiSports_Dataset, MultiSports_VIDEO_Dataset
from utils.box_ops import rescale_bboxes

from evaluator.link_method import tube_link
from utils.utils import iou3dt, pr_to_ap_voc, bbox_iou


class MultiSports_Evaluator(object):  # 评估器类
    def __init__(self,
                 args,
                 metric='fmap',
                 img_size=224,
                 iou_thresh=0.5,
                 transform=None,
                 collate_fn=None,
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

        self.gt_dir = os.path.join(self.data_root, 'trainval/multisports_GT.pkl')  # multisports的真实管道标注
        # 只用于video mAP的MCCLA算法
        self.video_root = None
        if not bbox_with_feat:
            self.video_root = os.path.join(self.data_root, 'trainval/rawframes')  # 视频的根目录

        # 只用于frame mAP计算
        self.map_path = args.map_path  # 保存评估结果的路径

        # 可以直接读取的结果文件路径，如果有的话
        self.frame_det_dir = frame_det_dir     # 帧级检测结果,包括视频数据集的帧级检测结果  根据metric类型而定
        self.video_det_dir = video_det_dir     # 视频级检测结果，关联好的tube
        self.link_method = link_method
        self.det_save_type = args.det_save_type  # 当计算v-mAP并且关联算法是多类别算法时 det_save_type会变为'multi_class'

        # dataset 构建测试集对象
        if metric == 'fmap':
            self.testset = MultiSports_Dataset(
                data_root=self.data_root,
                dataset=self.dataset,
                img_size=img_size,
                transform=transform,
                is_train=False,
                len_clip=self.len_clip,
                multi_sampling_rate=args.multi_sampling_rate)  # 构建专用的测试集，通过is_train来选择是训练集还是测试集
            self.num_classes = self.testset.num_classes
        elif metric == 'vmap':
            self.testset = MultiSports_VIDEO_Dataset(
                data_root=self.data_root,
                dataset=self.dataset,
                img_size=img_size,
                transform=transform,
                len_clip=self.len_clip,
                multi_sampling_rate=args.multi_sampling_rate,
                untrimmed=args.untrimmed)
            self.num_classes = self.testset.num_classes

    def evaluate_frame_map(self, model, epoch=1, show_pr_curve=False):  # 评估frame mAP
        print("Metric: Frame mAP")

        # all_gts是一个字典，索引是video_name，每一项是一个字典, 索引是该视频具有的管道类别，每一类对应一个列表，
        # 列表内包含该类别的管道数量个数组，每个数组对应一个管道
        # 处理gt文件
        print('loading gts from ' + self.gt_dir)
        GT = pickle.load(open(self.gt_dir, 'rb'))
        all_gts = {}
        self.videolist = GT['test_videos'][0]  # 测试集视频名称
        for video_name in self.videolist:
            tubes = GT['gttubes'][video_name]
            tubes_filted = {}
            for ilabel in tubes:
                if ilabel not in [15, 16, 17, 20, 64, 65]:
                    ilabel_filted = ilabel - sum(ilabel > np.array([15, 16, 17, 20, 64, 65]))  # 修正换算到只有60个类别
                    tubes_filted[ilabel_filted] = tubes[ilabel]
            all_gts[video_name] = tubes_filted

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

        if not self.frame_det_dir:  # 如果不存在的帧级检测结果，则重新进行推断
            frame_det_dir = os.path.join(
                self.infer_dir, 'multisports', self.model_name, 'frame_det_' + str(epoch) + '.pkl')
            inference_dir = os.path.join(os.path.dirname(frame_det_dir), 'detections_' + str(epoch))  # 只用来便于查看
            print('Creating Inference Dir：' + inference_dir)
            os.makedirs(inference_dir, exist_ok=True)
            all_dets = []
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
                        video_id = target['video_id_and_frame_id'][0]
                        frame_id = target['video_id_and_frame_id'][1]
                        bboxes = rescale_bboxes(bboxes, orig_size)

                        infer_name = img_name.replace('/', '_')[:-3] + 'txt'  # 用于保存该帧推断结果的文件名
                        detection_path = os.path.join(inference_dir, infer_name)  # 用于保存该帧推断结果的完整文件路径
                        with open(detection_path, 'w+') as f_detect:
                            for score, label, bbox in zip(scores, labels, bboxes):
                                x1 = round(bbox[0])
                                y1 = round(bbox[1])
                                x2 = round(bbox[2])
                                y2 = round(bbox[3])
                                cls_id = int(label)  # 类别从0开始

                                all_dets.append([video_id, frame_id, cls_id, score, x1, y1, x2, y2])
                                f_detect.write(
                                    str(cls_id) + ' ' + str(score) + ' '
                                    + str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2) + '\n')

                    if batch_idx % 100 == 0:
                        log_info = "[%d / %d]" % (batch_idx, epoch_size)
                        print(log_info, flush=True)
            all_dets = np.array(all_dets)
            print('saving frame_dets to ' + frame_det_dir)
            pickle.dump(all_dets, open(frame_det_dir, 'wb'))
        else:  # 存在视频集上的帧级别检测结果，则直接读取    嵌套字典，第一层是video_id,第二层是cls_idx
            frame_det_dir = self.frame_det_dir
            print('loading frame_dets from ' + frame_det_dir)
            all_dets = pickle.load(open(frame_det_dir, 'rb'))

        print('calculating Frame mAP ...')
        metric_list = self.calculate_framemAP(all_gts, all_dets, self.iou_thresh)
        for metric in metric_list:
            print(metric)

        # 在一个txt中记录fmap
        fmaptxt = frame_det_dir[:frame_det_dir.rfind('/')] + '/fmap.txt'  # 存放fmap结果的txt文件
        with open(fmaptxt, 'a+') as f:
            f.write('epoch:{}\n'.format(epoch))
            for metric in metric_list:
                f.write(metric)
            f.write('\n')

    def calculate_framemAP(self, all_gts, all_dets, iou_thresh):
        # all_gts是一个字典，索引是video_name，每一项是一个字典, 索引是该视频具有的管道类别，每一类对应一个列表，
        # 列表内包含该类别的管道数量个数组，每个数组对应一个管道
        pr_all = {}
        for cls_ind in range(self.num_classes):  # 每个类别独立进行
            dets_cls = all_dets[all_dets[:, 2] == cls_ind, :]  # 该类别的检测
            # load ground-truth of this class  载入该类别的所有gt框
            gts_cls = {}
            for video_id, video_name in enumerate(self.videolist):
                tubes = all_gts[video_name]
                if cls_ind not in tubes:
                    continue
                for tube in tubes[cls_ind]:  # 该类别的tube
                    for i in range(tube.shape[0]):  # tube长度内的帧索引
                        frame_id = int(tube[i, 0])
                        k = (video_id, frame_id)  # 元组（视频序号，帧序号）
                        if k not in gts_cls:
                            gts_cls[k] = []
                        gts_cls[k].append(tube[i, 1:5].tolist())  # 在该帧的位置增加一个gt框列表，列表长度是4 xyxy
            for k in gts_cls:  # 将同一个位置的所有gt框合成一个数组
                gts_cls[k] = np.array(gts_cls[k])

            # pr will be an array containing precision-recall values
            pr = np.empty((len(dets_cls) + 1, 2), dtype=np.float32)  # precision, recall
            pr[0, 0] = 1.0
            pr[0, 1] = 0.0
            gt_num_cls = sum([len(gts) for gts in gts_cls.values()])
            fp = 0  # false positives
            tp = 0  # true positives
            is_gt_box_detected = {}
            for i, j in enumerate(np.argsort(-dets_cls[:, 3])):  # 按照得分降序排序
                k = (int(dets_cls[j, 0]), int(dets_cls[j, 1]))  # 位置元组
                box = dets_cls[j, 4:8]
                ispositive = False
                if k in gts_cls:  # 如果该位置在gt中存在
                    if k not in is_gt_box_detected:  # 如果该位置是第一次被检测
                        is_gt_box_detected[k] = np.zeros(gts_cls[k].shape[0], dtype=bool)  # 表明该帧上的gt框有几个被检测出来
                    ious = [bbox_iou(gt, box) for gt in gts_cls[k]]
                    for m in reversed(np.argsort(ious)):  # 从高到低
                        # 如果该gt没有被分配出去并且iou高于阈值，则分配给该det
                        if (not is_gt_box_detected[k][m]) and (ious[m] >= iou_thresh):
                            ispositive = True
                            is_gt_box_detected[k][m] = True
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

    def evaluate_video_map(self, model, epoch=1):
        print("Metric: Video mAP")
        all_gts = {}
        all_dets = {}
        # all_gts是一个字典，索引是video_name，每一项是一个字典, 索引是该视频具有的管道类别，每一类对应一个列表，
        # 列表内包含该类别的管道数量个数组，每个数组对应一个管道

        # 处理gt文件
        print('loading gts from ' + self.gt_dir)
        GT = pickle.load(open(self.gt_dir, 'rb'))
        self.videolist = GT['test_videos'][0]  # 测试集视频名称
        for video_name in self.videolist:
            tubes = GT['gttubes'][video_name]
            tubes_filted = {}
            for ilabel in tubes:
                if ilabel not in [15, 16, 17, 20, 64, 65]:
                    ilabel_filted = ilabel - sum(ilabel > np.array([15, 16, 17, 20, 64, 65]))  # 修正换算到只有60个类别
                    tubes_filted[ilabel_filted] = tubes[ilabel]
            all_gts[video_name] = tubes_filted

        # 如果不存在关联结果检测结果，则重新进行关联
        if not self.video_det_dir:
            self.video_det_dir = os.path.join(
                self.infer_dir, 'multisports', self.model_name, 'video_det_' + str(epoch) + '.pkl')
            if not self.frame_det_dir:  # 如果不存在视频数据集的帧级检测结果，则重新进行推断
                frame_det_dir = os.path.join(
                    self.infer_dir, 'multisports', self.model_name, 'frame_det_in_vmap_' + str(epoch) + '.pkl')
                os.makedirs(os.path.dirname(frame_det_dir), exist_ok=True)
                # inference  对frame级检测结果进行推断
                for video_id, video_name in enumerate(self.videolist):
                    if video_id % 10 == 0:
                        print('Video: [%d / %d] - %s' % (video_id, len(self.videolist), video_name))

                    # set video  根据该视频设置视频测试集实例的索引长度和索引路径
                    self.testset.set_video_data(video_id)

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

                    dets_video = {}  # 针对该视频,用类别作索引(只包含存在检测的类别)，每一项是一个列表，列表中每一项是一个数组，数组对应该视频该类别下对应某一帧的检测框数组
                    for iter_i, (batch_img_name, batch_video_clip, batch_target) in enumerate(self.testloader):
                        # to device
                        batch_video_clip = batch_video_clip.to(model.device)

                        with torch.no_grad():
                            # model.inference  video mAP计算中用人员检测器在nms阶段提供人员提议有一定帮助，用跟踪器在初始阶段提供的帮助更大
                            batch_scores, batch_labels, batch_bboxes = model(batch_video_clip, batch_target)

                            # process batch
                            for bi in range(len(batch_scores)):
                                img_name = batch_img_name[bi]
                                scores = batch_scores[bi]
                                labels = batch_labels[bi]
                                bboxes = batch_bboxes[bi]
                                target = batch_target[bi]

                                # rescale bbox  将两点式归一化小数表示的预测框变换为原始图片尺寸下的两点式绝对坐标表示
                                orig_size = target['orig_size']
                                bboxes = rescale_bboxes(bboxes, orig_size)

                                # generate detected tubes for all classes 保存格式
                                # save format: {img_name: {cls_ind: array[[x1,y1,x2,y2, cls_score], [], ...]}}
                                video_id = target['video_id_and_frame_id'][0]
                                frame_id = target['video_id_and_frame_id'][1]
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
                                    for cls_idx in range(self.num_classes):  # 真实标注的类别从0开始
                                        inds = np.where(labels == cls_idx)[0]
                                        if len(inds) == 0:  # 如果该样本没有该类别的检测则直接跳下一类别
                                            continue
                                        c_bboxes = bboxes[inds]  # np.array
                                        c_scores = scores[inds]  # np.array

                                        # [n_box, 6]  frame_id,bboxes,score
                                        boxes = \
                                            np.concatenate([np.repeat(frame_id, c_scores.shape[0])[..., None], c_bboxes, c_scores[..., None]], axis=-1)
                                        if cls_idx not in dets_video.keys():
                                            dets_video[cls_idx] = []  # 该视频关于该类别新建一个列表，每一项是一个数组对应一帧
                                        dets_video[cls_idx].append(boxes)  # 该图片、该类别下的所有预测框 每个预测框是4+1 原始尺寸下的绝对坐标两点式+该类别的得分
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

        iou_list = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75]
        print('calculating video mAP ...')
        for iou_thresh in iou_list:
            # 输入视频级检测结果(管道)
            per_ap = self.calculate_videomAP(all_gts, all_tubes, iou_thresh)  # 计算video mAP
            print('-------------------------------')
            print('V-mAP @ {} IoU:'.format(iou_thresh))
            print('--Per AP: ', per_ap)

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


def test():
    gt_dir = '/media/su/d/datasets/MultiSports/trainval/multisports_GT.pkl'
    all_gts = {}
    # 处理gt文件
    print('loading gts from ' + gt_dir)
    GT = pickle.load(open(gt_dir, 'rb'))
    videolist = GT['test_videos'][0]  # 测试集视频名称
    for video_name in videolist:
        tubes = GT['gttubes'][video_name]
        tubes_filted = {}
        for ilabel in tubes:
            if ilabel not in [15, 16, 17, 20, 64, 65]:
                ilabel_filted = ilabel - sum(ilabel > np.array([15, 16, 17, 20, 64, 65]))  # 修正换算到只有60个类别
                tubes_filted[ilabel_filted] = tubes[ilabel]
        all_gts[video_name] = tubes_filted

if __name__ == "__main__":
    test()
