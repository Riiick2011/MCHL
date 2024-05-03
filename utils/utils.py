"""
本文件包含ucf24和jhmdb21计算video mAP用的评估标准
评估用的IoU是默认的无法改变的
"""
import numpy as np


# 两个检测框之间的IoU，默认两点式
def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(float(box1[0]-box1[2]/2.0), float(box2[0]-box2[2]/2.0))
        Mx = max(float(box1[0]+box1[2]/2.0), float(box2[0]+box2[2]/2.0))
        my = min(float(box1[1]-box1[3]/2.0), float(box2[1]-box2[3]/2.0))
        My = max(float(box1[1]+box1[3]/2.0), float(box2[1]+box2[3]/2.0))
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea


def area2d(b):
    """Compute the areas for a set of 2D boxes"""
    return (b[:, 2]-b[:, 0]+1)*(b[:, 3]-b[:, 1]+1)


def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""
    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2], b2[:, 2])
    ymax = np.minimum(b1[:, 3], b2[:, 3])

    width = np.maximum(0, xmax - xmin + 1)  # 增加一个像素点
    height = np.maximum(0, ymax - ymin + 1)

    return width * height


# 3D nms 管道之间的nms
def nms_3d(detections, overlap=0.5):
    # detections: [(tube1, score1), (tube2, score2)]
    if len(detections) == 0:
        return np.array([], dtype=np.int32)
    I = np.argsort([d[0] for d in detections])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0
    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([iou3dt(detections[ii][1], detections[i][1]) for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]
    return indices[:counter]


def iou3d(b1, b2):
    """Compute the IoU between two tubes with same temporal extent"""
    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(b1[:, 1:5], b2[:, 1:5])

    return np.mean(ov / (area2d(b1[:, 1:5]) + area2d(b2[:, 1:5]) - ov))


# 就是时间重叠部分的3D IoU额外乘以时间交并比
def iou3dt(b1, b2, spatialonly=False, temporalonly=False):
    """Compute the spatio-temporal IoU between two tubes"""
    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax <= tmin:
        return 0.0

    temporal_inter = tmax - tmin
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0])

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]): int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]): int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    if temporalonly:
        return temporal_inter / temporal_union
    return iou3d(tube1, tube2) * (1. if spatialonly else temporal_inter / temporal_union)


def pr_to_ap_voc(pr):
    """
    Compute VOC AP given precision and recall.
    """
    precision = pr[:, 0]
    recall = pr[:, 1]
    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Preprocess precision to be a non-decreasing array
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
    return average_precision
