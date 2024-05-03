"""Pure Python NMS."""
import numpy as np
import torch
from .box_ops import calculate_iou


# 根据得分，对预测框进行nms   只对eval和test起作用
def nms(bboxes, scores, nms_thresh, nms_iou_type):
    """
    bboxes: （ndarray）[N, 4]
    scores: （ndarray）[N,]
    """
    keep = []
    order = scores.argsort()[::-1]  # 按分数大小降序排列的原始序号
    while order.size > 0:
        i = order[0]
        keep.append(i)  # 保留最高得分的序号
        # compute iou  计算最高得分预测框与所有剩余预测框的iou
        iou = calculate_iou(torch.tensor(bboxes[i]), torch.tensor(bboxes[order[1:]]),
                            iou_type=nms_iou_type).numpy().reshape(-1)
        # reserve all the boundingbox whose ovr less than thresh
        inds = np.where(iou <= nms_thresh)[0]  # iou重合程度小于阈值的 序号保留
        order = order[inds + 1]  # +1是因为 预测框不跟自己比较iou  每次循环均从剩余候选预测框中保留得分最高的预测框
    return keep


# 无视类别的多类别nms
def multiclass_nms_class_agnostic(scores, labels, bboxes, nms_thresh, nms_iou_type, topk=0):
    # 先排除 x1>=x2,y1>=y2的
    inds = np.where((bboxes[:, 2] - bboxes[:, 0] >= 1) * (bboxes[:, 3] - bboxes[:, 1] >= 1))[0]
    scores = scores[inds]
    labels = labels[inds]
    bboxes = bboxes[inds]

    # nms
    keep = nms(bboxes, scores, nms_thresh, nms_iou_type)

    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    # 再只保留前topk个  跨层级、跨类别
    if topk:
        top_ids = scores.argsort()[::-1][:topk]  # 从高到低 最高topk个的索引
        scores = scores[top_ids]
        labels = labels[top_ids]
        bboxes = bboxes[top_ids]
    return scores, labels, bboxes


# 关注类别的多类别nms，各个类别互不影响,独立进行
def multiclass_nms_class_aware(scores, labels, bboxes, nms_thresh, nms_iou_type, num_classes, topk=0):
    # 先排除 x1>=x2,y1>=y2的
    inds = np.where((bboxes[:, 2]-bboxes[:, 0] >= 1) * (bboxes[:, 3]-bboxes[:, 1] >= 1))[0]
    scores = scores[inds]
    labels = labels[inds]
    bboxes = bboxes[inds]

    # nms
    keep = np.zeros(len(bboxes), dtype=np.int)
    for i in range(num_classes):
        inds = np.where(labels == i)[0]   # np.where返回的是元组，元组尺寸等于labels的维度
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]  # 该类别下的检测框
        c_scores = scores[inds]  # 该类别下的检测框的N分
        c_keep = nms(c_bboxes, c_scores, nms_thresh, nms_iou_type)
        if topk:  # 每个类别保留topk个最高的
            top_ids = scores[inds[c_keep]].argsort()[::-1][:topk]  # 从高到低 最高3个的索引
            keep[inds[c_keep][top_ids]] = 1
        else:
            keep[inds[c_keep]] = 1

    keep = np.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return scores, labels, bboxes


# 多类别nms
def multiclass_nms(
        scores, labels, bboxes, nms_thresh, nms_iou_type, num_classes, topk, class_agnostic=False):
    if class_agnostic:
        return multiclass_nms_class_agnostic(
            scores, labels, bboxes, nms_thresh, nms_iou_type, topk)
    else:
        return multiclass_nms_class_aware(
            scores, labels, bboxes, nms_thresh, nms_iou_type, num_classes, topk)
