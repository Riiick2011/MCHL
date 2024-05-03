import pathlib
import numpy as np
import os

import torch

from utils.utils import bbox_iou, nms_3d
from evaluator.link_utils import compute_score_one_class, ROADTube, ROAD_temporal_trim, Mass, dist_center, OJLATube, \
    ApproxMultiscanJPDAProbabilities, MCCLADet, MCCLATube, compare_with_gt, cascade_MCCLA
from copy import deepcopy
import cv2


@torch.no_grad()
def tube_link(all_dets, num_classes, videolist, link_method='viterbi', video_root=None):
    # 先把每个视频内关于每个类别的检测框梳理清楚
    # dets_all_videos是字典，索引是视频编号
    video_det_pkl = {i: [] for i in range(num_classes)}  # 按照类别放到pkl中 ,每个类别是一个列表
    # 对该视频该类别的帧级检测结果进行关联,这里开始根据关联算法不同而不同
    if link_method == 'viterbi':  # 维特比算法 类别之间独立进行
        for video_id, dets_video in all_dets.items():  # 对于每个视频的帧级检测结果而言
            video_name = videolist[video_id]
            print(video_name)
            # dets_video是字典，索引是类别，值是列表，列表内是多个数组，每个数组对应一帧
            # 对于维特比算法来说 dets_video是一个字典，索引是类别，第二个维度是6
            for cls_idx in sorted(dets_video.keys()):  # 对于该视频检测结果中存在的类别
                dets_cls_video = dets_video[cls_idx]  # 一个列表，该视频关于该类别的检测结果，列表内是多个数组，每个数组对应一帧
                dets_cls_video = np.concatenate(dets_cls_video, axis=0)  # array [n_box,6]
                tubes_cls_video = viterbi_link(dets_cls_video)
                # tubes_cls_video是元组构成的列表[(tube_score,tube_bboxes),...]
                # 每个元组对应一个管道，包含两项：tube_score(float)和tube_bboxes(array)
                # tube_bboxes数组的形状是(tube_length,6:(frame_id,bboxs,score))

                # 管道之间nms
                tubes_cls_video = tube_nms(tubes_cls_video)

                for tube_cls_video in tubes_cls_video:
                    video_det_pkl[cls_idx].append((video_name, tube_cls_video[0], tube_cls_video[1]))
                    # 每一项是一个元组(video_name, tube_score, tube_bboxes) 对应一个tube
    elif link_method == 'road':  # 类别之间独立进行
        for video_id, dets_video in all_dets.items():  # 对于每个视频的帧级检测结果而言
            video_name = videolist[video_id]
            print(video_name)
            # dets_video是字典，索引是类别，值是列表，列表内是多个数组，每个数组对应一帧
            # 对于road算法来说 dets_video是一个字典，索引是类别，第二个维度是6
            for cls_idx in sorted(dets_video.keys()):  # 对于该视频检测结果中存在的类别
                dets_cls_video = dets_video[cls_idx]  # 一个列表，该视频关于该类别的检测结果，列表内是多个数组，每个数组对应一帧
                dets_cls_video = np.concatenate(dets_cls_video, axis=0)  # array [n_box,6]
                tubes_cls_video = ROAD_link(dets_cls_video, simple_link=True, act_filter=False, temporal_trim=False)
                # tubes_cls_video是元组构成的列表[(tube_score,tube_bboxes),...]
                # 每个元组对应一个管道，包含两项：tube_score(float)和tube_bboxes(array)
                # tube_bboxes数组的形状是(tube_length,6:(frame_id,bboxs,score))

                # 管道之间nms
                tubes_cls_video = tube_nms(tubes_cls_video)

                for tube_cls_video in tubes_cls_video:
                    video_det_pkl[cls_idx].append((video_name, tube_cls_video[0], tube_cls_video[1]))
                    # 每一项是一个元组(video_name, tube_score, tube_bboxes) 对应一个tube
    elif link_method == 'ojla':  # 类别之间同步进行
        for video_id, dets_video in all_dets.items():  # 对于每个视频的帧级检测结果而言
            video_name = videolist[video_id]
            print(video_name)
            # dets_video是字典，索引是类别，值是列表，列表内是多个数组，每个数组对应一帧
            if len(dets_video) == 0:  # 如果该视频没有检测结果，则进行下一个视频
                continue
            dets_video = dets_video['multi_class']   # 一个列表，该视频关于的检测结果，列表内是多个数组，每个数组对应一帧，第二个维度是6+num_classes
            # 类别得分用置信度修正过了 后续不使用置信度
            # dets_cls_video = np.concatenate(dets_cls_video, axis=0)  # array [n_box,6]
            tubes_video = OJLA_link(dets_video, multilabel=True, num_classes=num_classes)  # 默认采用效果更好的multilabel模式
            # tubes_cls_video是元组构成的列表[(tube_score,tube_bboxes),...]
            # 每个元组对应一个管道，包含两项：tube_score(float)和tube_bboxes(array)
            # tube_bboxes数组的形状是(tube_length,6:(frame_id,bboxs,score))

            for cls_idx in tubes_video:
                if len(tubes_video[cls_idx]):  # 如果存在该类别的管道
                    # 同类别管道之间nms
                    tubes_cls_video = tube_nms(tubes_video[cls_idx])
                    # 要根据管道的类别存放
                    for tube_cls_video in tubes_cls_video:
                        video_det_pkl[cls_idx].append((video_name, tube_cls_video[0], tube_cls_video[1]))
                        # 每一项是一个元组(video_name, tube_score, tube_bboxes) 对应一个tube
    elif link_method == 'mccla':
        img_folder = None
        for video_id, dets_video in all_dets.items():  # 对于每个视频的帧级检测结果
            video_name = videolist[video_id]
            print(video_name)
            if video_root:  # 如果给了视频路径则说明det中不包含特征
                img_folder = os.path.join(video_root, video_name)  # 该视频的文件夹完整路径
            # dets_video是字典，索引是类别，值是列表，列表内是多个数组，每个数组对应一帧
            if len(dets_video) == 0:  # 如果该视频没有检测结果，则进行下一个视频
                continue
            dets_video = dets_video['multi_class']  # 一个列表，该视频关于的检测结果，列表内是多个数组，每个数组对应一帧，第二个维度是6+num_classes
            # 类别得分用置信度修正过了 后续不使用置信度
            # dets_cls_video = np.concatenate(dets_cls_video, axis=0)  # array [n_box,6]
            tubes_video = MCCLA_link(dets_video, multilabel=True, num_classes=num_classes, img_folder=img_folder)  # 默认采用效果更好的multilabel模式
            # tubes_cls_video是元组构成的列表[(tube_score,tube_bboxes),...]
            # 每个元组对应一个管道，包含两项：tube_score(float)和tube_bboxes(array)
            # tube_bboxes数组的形状是(tube_length,6:(frame_id,bboxs,score))

            for cls_idx in tubes_video:
                if len(tubes_video[cls_idx]):  # 如果存在该类别的管道
                    # 同类别管道之间nms
                    tubes_cls_video = tube_nms(tubes_video[cls_idx])
                    # 要根据管道的类别存放
                    for tube_cls_video in tubes_cls_video:
                        video_det_pkl[cls_idx].append((video_name, tube_cls_video[0], tube_cls_video[1]))
                        # 每一项是一个元组(video_name, tube_score, tube_bboxes) 对应一个tube
    return video_det_pkl


def tube_nms(tubes_cls_video):
    if len(tubes_cls_video) != 0:  # 进行管道之间的NMS
        keep = nms_3d(tubes_cls_video, 0.3)
        if np.array(keep).size:
            tubes_cls_video_keep = [tubes_cls_video[k] for k in keep]
            # max subarray with penalization -|Lc-L|/Lc
            tubes_cls_video = tubes_cls_video_keep
    return tubes_cls_video


# 对某视频某类别进行关联
def viterbi_link(dets):
    # dets是一个array形状为[n_box,6] 该视频该类别的所有帧级别检测   [frame_id,bboxes,score]
    tubes_cls_video = []  # 要返回的列表 其中每一项是一个元组(tube_score,tube_bboxes(array))，对应一个管道
    frame_ids = np.unique(dets[:, 0]).astype(int)  # 全部存在检测的帧id
    frame_start = min(frame_ids)
    frame_end = max(frame_ids)
    length = frame_end - frame_start + 1

    # 关联算法 维特比算法只适合非在线模式    管道长度固定、受漏检影响大、插帧方式不光滑
    # 首先填充中间帧   直接复制最近的帧
    detect = {}
    for frame_id in frame_ids:
        det_idx = np.where(dets[:, 0].astype(int) == frame_id)[0]  # 该帧检测框在数组中对应的行数
        frame_dets = dets[det_idx, 1:]  # 该帧的所有检测， 数组
        detect[frame_id] = frame_dets
    for frame_id in range(frame_start, frame_end):
        if frame_id not in frame_ids:
            frame_dist = abs(frame_id - frame_ids)
            neareast_frame_id = frame_ids[np.argmin(frame_dist)]  # 相同距离的话选坐标
            detect[frame_id] = detect[neareast_frame_id]

    res = []
    isempty_vertex = np.zeros([frame_start + length, ], dtype=bool)
    edge_scores = {i: compute_score_one_class(detect[i], detect[i + 1], w_iou=1.0, w_scores=1.0, w_scores_mul=0.5) for
                   i in range(frame_start, frame_end)}  # 每个相邻帧之间的边界得分[第i帧的检测框个数，第i+1帧的检测框个数]  i表示第i帧与第i+1帧

    while not np.any(isempty_vertex):  # 只要有一帧中无检测框 即结束
        # initialize
        scores = {d[0]: np.zeros([d[1].shape[0], ], dtype=np.float32) for d in detect.items()}
        index = {d[0]: np.nan * np.ones([d[1].shape[0], ], dtype=np.float32) for d in detect.items()}

        # viterbi  倒序，寻找最优路径，确定每一帧上的每个检测框最适合跟下一帧上的哪个检测框关联，score是按照关联一路累加的
        # from the second last frame back
        for i in range(frame_end - 1, frame_start - 1, -1):
            edge_score = edge_scores[i] + scores[i + 1]
            # find the maximum score for each bbox in the i-th frame and the corresponding index
            # 为第i帧上的每一个检测框找到最高的边界得分和对应的索引序号 即找到下一帧关联第几个检测框最好
            scores[i] = np.max(edge_score, axis=1)  # 是这一路关联的累计边界得分
            index[i] = np.argmax(edge_score, axis=1)  # 记录每个相邻帧之间的边界关联得分的大小序号

        # decode 解码出最佳路径
        idx = -np.ones([frame_start + length], dtype=np.int32)  # frame_start之后才是有效项
        idx[frame_start] = np.argmax(scores[frame_start])  # 根据一路倒推累加得到的score，找到最佳路径在第一帧上对应的检测框序号
        for i in range(frame_start, frame_end):
            idx[i + 1] = index[i][idx[i]]  # idx记录最佳路径在每一帧上对应的检测框序号

        # remove covered boxes and build output structures 去除用过的检测框
        this = np.empty((length, 6), dtype=np.float32)  # 随机数组   存储一个管道  管道长度都是固定的
        this[:, 0] = np.arange(frame_start, frame_end+1)  # 第0列代表帧序号
        k = 0
        for i in range(frame_start, frame_end+1):
            j = idx[i]  # 最佳路径在第一帧所选的检测框序号
            if i < frame_end:
                edge_scores[i] = np.delete(edge_scores[i], j, 0)  # 第i帧与第i+1帧之间的边界得分，关于第j个检测框的 删掉
            if i > frame_start:
                edge_scores[i - 1] = np.delete(edge_scores[i - 1], j, 1)
            this[k, 1:5] = detect[i][j, :4]  # bbox
            this[k, 5] = detect[i][j, 4]  # score
            k += 1
            detect[i] = np.delete(detect[i], j, 0)  # 第i帧的第j个检测框 删掉
            isempty_vertex[i] = (detect[i].size == 0)  # it is true when there is no detection in any frame 当任一帧中不含检测框时，记录该帧空了
        res.append(this)  # 保留该管道
        if len(res) == 3:
            break

    for tube in res:
        tube_score = np.mean(tube[:, -1])  # 管道得分是沿途所有帧上取平均
        tube_boxes = tube
        tubes_cls_video.append((tube_score, tube_boxes))

    # 返回一个列表，每一项是一个元组(tube_score, tube_bboxes) 对应一个tube
    return tubes_cls_video


def ROAD_link(dets, simple_link=True, act_filter=False, temporal_trim=True):
    """
    :param dets: 该视频该类别的所有帧级别检测
    :param simple_link: 原版算法-关联得分只看IoU，高于阈值直接关联   改进版-关联得分采用IoU和预测框得分加权，高于阈值直接关联
    :param act_filter: act算法的改进，过滤掉得分过低或者过短的管道，减少FP
    :param temporal_trim: 可以通过时间校准进一步修正管道的时间范围 可以在线使用也可以离线使用，但通常离线使用以提升得分
    :return:
    """

    # dets是一个array形状为[n_box,6] 该视频该类别的所有帧级别检测   [frame_id,bboxes,score]
    frame_ids = np.unique(dets[:, 0]).astype(int)  # 全部存在检测的帧id
    frame_start = min(frame_ids)
    frame_end = max(frame_ids)
    length = frame_end - frame_start + 1
    # 将该视频该类别的所有帧级别检测从数组表示转换为按照frame_id作为索引的字典表示
    detect = {}
    for frame_id in range(frame_start, frame_end+1):
        det_idx = np.where(dets[:, 0].astype(int) == frame_id)[0]  # 该帧检测框在数组中对应的行数
        frame_dets = dets[det_idx, 1:]  # 该帧的所有检测， 数组  如果没有检测则数组形状为(0,5)
        detect[frame_id] = frame_dets

    tube_list = []  # 该视频对应该动作类别的管道列表

    # 对于每一帧上的检测进行排序、关联
    for frame_id in range(frame_start, frame_end+1):
        tube_active_index = [tube.index for tube in tube_list if tube.active == 1]  # 活着的管道的索引列表
        tube_active_index_sorted = \
            sorted(tube_active_index, key=lambda i: tube_list[i].tube_score, reverse=True)  # 对活着的管道按照管道得分排序,降序

        # 找到txt中同一类别的检测框的索引，按照关于该动作类别的得分进行排序
        det_index_sorted = np.argsort(-detect[frame_id][:, -1]).tolist()  # 对该类别的检测按照得分排序,降序    改成计算得分矩阵

        # 对于得分最高的管道，如果检测框符合IoU阈值，则进行关联，从高到低
        for tube_index in tube_active_index_sorted:
            if simple_link:  # 原版
                linking_flag = False
                for det_index in det_index_sorted:
                    if bbox_iou(tube_list[tube_index].det_list[-1][:4],
                                detect[frame_id][det_index][:4], x1y1x2y2=True) >= 0.1:
                        # 作为一个超参数,如果过低，可能会导致错位关联；过高会漏关。要不然把conf和Iou折中一下
                        # print(iou(tube_list[tube_index].det_list[-1], txt['predict_micro_tube'][det_index][:4]))
                        tube_list[tube_index](frame_id, detect[frame_id][det_index])  # 管道关联该检测
                        det_index_sorted.remove(det_index)  # 并将该检测的索引删除
                        linking_flag = True
                        # print(tube_index, det_index)
                        break  # 跳出内循环，继续执行外循环
                if linking_flag:
                    continue  # 如果该管道在所有候选检测中成功进行了关联，则对下一个管道进行关联，否则对该管道记录为漏检
                tube_list[tube_index].miss_link(frame_id)  # 漏检则执行Tube对象的miss_link方法
            else:
                linking_flag = False
                if len(det_index_sorted):  # 只要还有候选检测，就可以关联
                    linking_score = np.array(
                        [0.6*bbox_iou(tube_list[tube_index].det_list[-1][:4], detect[frame_id][det_index][:4], x1y1x2y2=True)
                         + 0.4*detect[frame_id][det_index][4] for det_index in det_index_sorted])
                    if np.max(linking_score) >= 0.1:
                        tube_list[tube_index](frame_id,
                                              detect[frame_id][det_index_sorted[np.argmax(linking_score)]])  # 管道关联该检测
                        det_index_sorted.remove(det_index_sorted[np.argmax(linking_score)])
                        linking_flag = True
                        # 并将该检测的索引删除
                if linking_flag:
                    continue  # 如果该管道在所有候选检测中成功进行了关联，则对下一个管道进行关联，否则对该管道记录为漏检
                tube_list[tube_index].miss_link(frame_id)  # 漏检则执行Tube对象的miss_link方法

        # 对于没有进行关联的剩余检测，初始化管道
        for det_index in det_index_sorted:
            tube = ROADTube(frame_id, detect[frame_id][det_index], len(tube_list))  # det_index是检测索引，后面是管道唯一的管道序号
            tube_list.append(tube)

    # ACT的过滤  基本可以代替时间trim
    if act_filter:
        for tube in tube_list:
            if tube.tube_score < 0.005 or tube.frame_range < 15:
                tube_list.remove(tube)

    # temporal trim通过EM动态规划实现时间校准
    if temporal_trim:  # 通过维特比动态规划解能量函数的极大极小问题 每个新的检测框关联后就进行一次   时间标注校准
        tube_list_new = []
        for tube in tube_list:
            tube_list_new.extend(ROAD_temporal_trim(tube))
        tube_list = tube_list_new

    if len(tube_list) > 0:  # 如果该视频对于该动作类别检测出了动作管道，则进行内插外推
        for tube in tube_list:
            tube.interpolate()

    tubes_cls_video = [(tube.tube_score, tube.det_list_interpolated) for tube in tube_list]
    # 返回一个列表，每一项是一个元组(tube_score, tube_bboxes) 对应一个tube
    return tubes_cls_video


def OJLA_link(dets, multilabel=False, num_classes=24):
    """
    :param dets:
    :param multilabel:  管道的多类别模式， 可以在线使用也可以离线使用，但通常离线使用以提升得分
    :param num_classes:
    :return:
    """
    potts_matrix = np.ones(num_classes) - np.diag(np.ones(num_classes))  # 矩阵 24x24

    # dets是一个list,每一项对应一帧，是形状为[n_box,6+num_classes]的数组  该视频的所有帧级别检测   [frame_id,bboxes,score，cls_score]
    tubes_video = {i: [] for i in range(num_classes)}  # 按照类别索引 ,每个类别是一个列表
    tube_list = []  # 存放Tube类实例的列表

    for i in range(len(dets)):  # 逐帧进行
        frame_id = int(dets[i][0, 0])
        dets_frame = dets[i][:, 1:5]  # 2维数组
        scores_frame = dets[i][:, 6:]  # 2维数组  不含置信度
        det_indexs = np.argsort(-np.max(scores_frame, axis=1))[:3]  # 对该帧的检测框按照 最大类别得分进行降序排列 返回检测框的序号 只保留前三个
        dets_frame = dets_frame[det_indexs]  # 该帧经过筛选过后的检测框
        scores_frame = scores_frame[det_indexs]  # 该帧经过筛选过后的检测框

        # 计算每个活着的管道与每个检测之间的得分和损失
        tube_active_index = [tube.index for tube in tube_list if tube.active == 1]  # 活着的管道的索引列表
        tube_num = len(tube_active_index)
        det_num = len(dets_frame)

        mass_list = []  # 存放当前帧的管道关联情况 每一项是一个Mass类对象，对应一个管道的字典
        det_indexs_in_gate = []  # 用于存放与管道过近的检测框全局序号
        Mes_Tar = np.zeros((det_num, tube_num)).astype(np.bool)  # 确认矩阵， 记录管道和检测的可行关联情况
        for tube_index in range(tube_num):  # 局部索引
            mass = Mass()
            tube = tube_list[tube_active_index[tube_index]]
            last_det = tube.det_list[-1]
            last_score = tube.score_list[-1]
            tube_label_score = np.sum(tube.score_list[-7:], axis=0) if tube.det_num >= 7 \
                else np.sum(tube.score_list, axis=0)  # 还没加potts惩罚项  形状为(24,)
            if tube.det_num >= 7:  # 有效检测超过7次才需要减去potts惩罚项
                tube_label_score = tube_label_score - \
                                   np.array([potts_matrix[np.argmax(last_score), l] for l in range(num_classes)])
            gate = (last_det[2] - last_det[0]) / 2
            for det_index in range(det_num):  # 局部索引 对于该帧的每一个检测框  如果与该管道的最后一个有效检测中心距离处于门限之内 则计算标注得分
                if dist_center(last_det, dets_frame[det_index]) <= gate:
                    label_score = tube_label_score + scores_frame[det_index]
                    tube_cls = np.argmax(label_score)  # 如果该管道与该检测框进行关联，则管道的类别应该调整为该类别
                    overlap_score = bbox_iou(last_det, dets_frame[det_index])
                    cost = 1 / (label_score[tube_cls] + overlap_score * (tube.det_num + 1) / 2) if tube.det_num < 7 \
                        else 1 / (label_score[tube_cls] + overlap_score * 4)  # 该管道与该检测框关联的cost
                    mass.Cost = np.concatenate([mass.Cost, np.array(cost).reshape(1, 1)], axis=0)
                    mass.Meas_edge = np.concatenate([mass.Meas_edge, np.array([det_index, 1]).reshape(2, 1)], axis=1)
                    mass.tube_cls.append(tube_cls)
                    Mes_Tar[det_index, tube_index] = True  # 记录节点情况
                    det_indexs_in_gate.append(det_index)
            mass.Hypo = mass.Meas_edge[0].transpose().reshape(-1, 1)  # 一定包含序号-1代表漏检
            mass.Prob = np.exp(-mass.Cost)  # 包含了漏检的概率
            mass_list.append(mass)

        Mes_Tar2 = np.zeros((tube_num+det_num, tube_num+det_num)).astype(np.bool)
        Mes_Tar2[tube_num:, :tube_num] = Mes_Tar
        # 返回一个列表，每一项是一个数组对应一个mass管道，数组尺寸是（该管道可能的检测框数量（包含漏检），1）
        final_prob_matrix = ApproxMultiscanJPDAProbabilities(Mes_Tar2, mass_list)

        # 更新管道
        for j in range(tube_num):  # 对于每个管道，找到概率最高的检测框序号，检测框全局序号-1代表漏检
            link_prob = np.max(final_prob_matrix[j], axis=0)[0]  # float
            index = np.argmax(final_prob_matrix[j], axis=0)[0]  # int 检测框的本地索引  本地指的是该管道门限之内
            det_index = mass_list[j].Hypo[index][0]  # 全局索引
            tube_cls = mass_list[j].tube_cls[index]
            if det_index == -1:  # 漏检，增加漏检计数
                tube_list[tube_active_index[j]].miss_link(frame_id)
            else:
                tube_list[tube_active_index[j]](frame_id, dets_frame[det_index],
                                                scores_frame[det_index], link_prob, tube_cls)

        # 对于与所有管道的门控距离均超过阈值(那么肯定没有进行关联)并且最高类别得分高于0.1的检测框，则初始化管道
        det_indexs_out_of_gate = np.array([i for i in range(det_num) if i not in np.unique(det_indexs_in_gate)])
        if len(det_indexs_out_of_gate):  # 如果存在所有门控外的检测框 再检查最大类别得分
            det_indexs_out_of_gate = det_indexs_out_of_gate[np.max(scores_frame[det_indexs_out_of_gate], axis=1) > 0.1]
        for _ in det_indexs_out_of_gate:  # 全局索引
            tube = OJLATube(frame_id, dets_frame[_], scores_frame[_], len(tube_list))
            tube_list.append(tube)

    if len(tube_list) > 0:  # 如果该视频检测出了动作管道
        if multilabel:  # 判断是否复制和更改tube_cls   可以在线进行
            for cls_idx in range(num_classes):
                for tube in tube_list:
                    # if tube.frame_range >= 15:  # 自己加的
                    topkmean_score = np.mean(np.sort(np.array(tube.score_list)[:, cls_idx])[-40:])  # 前40个得分
                    if topkmean_score > 0.25:
                        tube_new = deepcopy(tube)
                        tube_new.tube_cls = cls_idx
                        tube_new.interpolate()
                        tubes_video[cls_idx].append((topkmean_score, tube_new.det_list_interpolated))
        else:
            for tube in tube_list:
                tube.interpolate()
                tubes_video[tube.tube_cls].append((tube.tube_score, tube.det_list_interpolated))

    # 返回一个字典，包含num_classes项，索引是cls序号，每一项是一个列表，列表的每一项是一个元组(tube_score, tube_bboxes) 对应该类别的一个tube
    return tubes_video


# 级联
def MCCLA_link(dets, multilabel=False, num_classes=24, img_folder=None):
    # dets是一个list,每一项对应一帧，是形状为[n_box,6+num_classes]的数组  该视频的所有帧级别检测
    # [frame_id,bboxes,score，cls_score]

    # 类别得分的计算方式
    use_score_dist = True

    # 用于与gt直观对比
    compare_gt = False
    if compare_gt:  # 用于对比真实管道
        compare_with_gt()

    tubes_video = {i: [] for i in range(num_classes)}  # 按照类别索引 ,每个类别是一个列表
    tube_list = []  # 存放Tube类实例的列表

    frame_ids = np.unique([det[0, 0] for det in dets]).astype(int)  # 全部存在检测的帧id
    frame_start = min(frame_ids)
    frame_end = max(frame_ids)
    for frame_id in range(frame_start, frame_end + 1):
        # 当前帧检测处理
        if frame_id in frame_ids:
            det_idx = np.where(frame_ids == frame_id)[0][0]  # 该帧检测框在数组中对应的行数
            frame_dets = dets[det_idx]  # 该帧的所有检测， 数组
        else:
            frame_dets = np.array([]).reshape(-1, 30)  # 如果该帧不存在检测，则用空数组代替
        dets_frame = frame_dets[:, 1:5]  # 2维数组
        confs_frame = frame_dets[:, 5]  # 2维数组
        scores_frame = frame_dets[:, 6:]  # 2维数组  不含置信度
        det_indexs = np.argsort(-np.max(scores_frame, axis=1))[:3]  # 对该帧的检测框按照 最大类别得分进行降序排列 返回检测框的序号 只保留前三个
        dets_frame = dets_frame[det_indexs]  # 该帧经过筛选过后的检测框
        scores_frame = scores_frame[det_indexs]  # 该帧经过筛选过后的检测框

        # 提取外观特征，构建带有外观特征的帧级检测实例
        if img_folder:
            ori_img = cv2.imread(img_folder + '/{:05d}.jpg'.format(frame_id)).transpose(1, 0, 2)  # HWC->WHC  BGR
            dets_frame = [MCCLADet(frame_id, dets_frame[i], confs_frame[i], scores_frame[i], ori_img=ori_img) for i in range(len(dets_frame))]
            for det in dets_frame:
                if not det.valid:
                    print(img_folder)
            dets_frame = [det for det in dets_frame if det.valid]
        else:  # 如果不给定图片文件夹，则说明det中包含了检测模型提取的中间特征
            dets_frame = [{}]
            dets_frame = \
                [MCCLADet(frame_id, dets_frame[i], confs_frame[i], scores_frame[i]) for i in range(len(dets_frame))]

        # 第一级 关联活着的管道    之前暂时漏检的管道还是可以与没漏检的管道竞争
        # 计算每个活着的管道与每个检测之间的得分和损失
        tube_active_list = [tube for tube in tube_list if tube.active == 1]  # 活着的管道的索引列表
        tube_suspect_list = [tube for tube in tube_list if tube.active == 2]  # 可疑的管道的索引列表
        dets_frame = cascade_MCCLA(tube_active_list, dets_frame, frame_id, use_score_dist, tube_active=True)

        # 第二级 关联可疑管道  调整可疑管道的cost 增加了iou权重
        dets_frame = cascade_MCCLA(tube_suspect_list, dets_frame, frame_id, use_score_dist, tube_active=False)

        # 第三级 对于与所有管道的门控距离均超过阈值(那么肯定没有进行关联)并且最高类别得分高于0.1的检测框，则初始化管道
        dets_frame = [det for det in dets_frame if np.max(det.score) > 0.1]
        for det in dets_frame:
            tube = MCCLATube(frame_id, det, len(tube_list))
            tube_list.append(tube)

    # 管道长度过滤、多标签管道复制
    if len(tube_list) > 0:  # 如果该视频检测出了动作管道
        tube_list = [tube for tube in tube_list if tube.frame_range >= 3]  # 用管道的最低长度过滤，根据数据集特性调整
        if multilabel:  # 判断是否复制和更改tube_cls   可以在线进行
            for cls_idx in range(num_classes):
                for tube in tube_list:
                    # 管道中所有帧里关于该类别的前40个得分取平均
                    topkmean_score = np.mean(np.sort(np.array(tube.score_list)[:, cls_idx])[-40:])
                    if topkmean_score > 0.25:
                        tube_new = deepcopy(tube)
                        tube_new.tube_cls = cls_idx
                        tube_new.interpolate()
                        tubes_video[cls_idx].append((topkmean_score, tube_new.det_list_interpolated))
        else:
            for tube in tube_list:
                tube.interpolate()
                tubes_video[tube.tube_cls].append((tube.tube_score, tube.det_list_interpolated))

    # 返回一个字典，包含num_classes项，索引是cls序号，每一项是一个列表，列表的每一项是一个元组(tube_score, tube_bboxes) 对应该类别的一个tube
    return tubes_video
