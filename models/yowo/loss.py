import torch
import torch.nn as nn
import torch.nn.functional as F
from models.yowo.matcher import SimOTA
from utils.box_ops import calculate_iou
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized

from .matcher import bbox2dist, TaskAlignedAssigner


def wasserstein_loss(pred, target, eps=1e-7, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.
    Code is modified from https://github.com/Zzh-tju/CIoU.
    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """

    center1 = pred[:, :2] + pred[:, 2:] / 2
    center2 = target[:, :2] + target[:, 2:] / 2

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps #

    w1 = pred[:, 2] + eps
    h1 = pred[:, 3] + eps
    w2 = target[:, 2] + eps
    h2 = target[:, 3] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance
    return torch.exp(-torch.sqrt(wasserstein_2) / constant)


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).sum()
        return loss


class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False, nwd=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.nwd = nwd

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask,
                iou_type='ciou'):
        """
        :param pred_dist:  还没乘以stride  可以在计算损失的时候避免stride较大的预测框产生更大的损失值
        :param pred_bboxes:   还没乘以stride
        :param anchor_points:  还没乘以stride
        :param target_bboxes: 还没乘以stride
        :param target_scores:
        :param target_scores_sum:
        :param fg_mask:
        :param iou_type:
        :return:
        """
        # IoU loss
        # 用归一化后的正样本对齐得分(对齐gt：类别正确得分、重合程度得分)作为权重re-weight iou损失和dfl损失，评价的是这个正样本用来训练的优劣程度或者说可信程度
        weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)  # (N,1)

        iou = calculate_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], iou_type=iou_type)
        if self.nwd:  # 如果使用nwd，则用nwd修正iou
            nwd = wasserstein_loss(pred_bboxes[fg_mask], target_bboxes[fg_mask]).unsqueeze(-1)  # (N,1)
            iou_ratio = 0.8
            loss_box = (1 - iou_ratio) * (1.0 - nwd) + iou_ratio * (1.0 - iou)  # iou loss  (N,1)
        else:
            loss_box = 1.0 - iou  # (N,1)
        loss_iou = (loss_box * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]) * weight
            # [批次内的正样本数量, 1]
            loss_dfl = loss_dfl.sum() / target_scores_sum  # 类似于除以前景（正样本）数量，从而排除掉批次之间正样本数量不均匀的影响，只不过这里除以的是正样本的对齐得分之和
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        # Return sum of left and right DFL losses
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)
        # 这里交叉熵的含义是对于预测框关于锚点中心的距离预测(没乘以stride)，是处于0~regmax-1之间的，每个距离上都有一个概率，
        # 而目标是确定落在某个距离单位内的，左边界距离和右边界距离作为正确类别分别计算交叉熵，并且根据在该距离单位内的位置作为权重来修正


class NoConfCriterion:  # multi-hot功能未经过检验
    def __init__(self, m_cfg, img_size, num_classes, multi_hot=False):  # model must be de-paralleled
        self.num_classes = num_classes
        self.img_size = img_size

        self.reg_max = m_cfg['reg_max']
        self.use_dfl = self.reg_max > 1
        self.loss_iou_type = m_cfg['loss_iou_type']  # 训练时box_loss计算中用到的IoU类型，可选'iou'、'giou'等
        self.matcher_iou_type = m_cfg['matcher_iou_type']

        # loss
        self.VFL = m_cfg['VFL']
        self.multi_hot = multi_hot
        self.nwd = m_cfg['nwd']
        if self.VFL:
            self.cls_lossf = VarifocalLoss()
        else:
            self.cls_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.box_lossf = BboxLoss(self.reg_max, use_dfl=self.use_dfl, nwd=self.nwd)  # .to(device)
        self.loss_box_weight = m_cfg['NCCloss_box_weight']
        self.loss_cls_weight = m_cfg['NCCloss_cls_weight']
        self.loss_dfl_weight = m_cfg['NCCloss_dfl_weight']

        # 正样本分配器
        self.topk_candidate = m_cfg['topk_candidate']
        self.matcher = TaskAlignedAssigner(topk=self.topk_candidate,
                                           num_classes=self.num_classes,
                                           alpha=0.5,
                                           beta=6.0,
                                           matcher_iou_type=self.matcher_iou_type
                                           )

    def __call__(self, outputs, targets):
        """
            outputs: (Dict) -> {
                ("conf_pred": conf_pred,  # (Tensor) [B, M, 1])
                ("cls_pred": cls_pred,  # (Tensor) [B, M, Nc])
                ("score_pred": score_pred,  # (Tensor) [B, M, Nc])
                "bbox_pred": bbox_pred,  # (Tensor) [B, M, 4]  还没乘以stride
                ("dist_pred": dist_pred,  # (Tensor) [B, M, self.reg_max * 4])  还没乘以stride
                "anchor_point": anchor_point,  # (Tensor) [M, 2]
                "stride_tensor": stride_tensor}  # (Tensor) [M, 1]
            }
            targets: (List) [dict{'boxes': [...],
                                 'labels': [...],
                                 'orig_size': ...}, ...]   这里的真实标注框是两点式百分比形式
        """
        batch_size = outputs['bbox_pred'].shape[0]
        device = outputs['bbox_pred'].device
        anchor_point = outputs['anchor_point']
        stride_tensor = outputs['stride_tensor']
        score_pred = outputs['score_pred']  # 其实就是类别得分
        bbox_pred = outputs['bbox_pred']
        dist_pred = outputs['dist_pred']  # (Tensor) [B, M, self.reg_max * 4]

        target_bboxes = []
        target_labels = []
        target_scores = []
        fg_masks = []

        for batch_idx in range(batch_size):  # 对于每一个样本
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)

            # denormalize tgt_bbox  真实标注框恢复到两点式绝对坐标表示
            tgt_bboxes *= self.img_size

            # 返回每个预测框的目标框、归一化的对齐得分、作为正样本的预测框的mask
            target_label, target_bbox, target_score, fg_mask, _ = self.matcher(
                score_pred.detach()[batch_idx].sigmoid(),
                (bbox_pred.detach()[batch_idx] * stride_tensor).type(tgt_bboxes.dtype),
                anchor_point, stride_tensor, tgt_labels, tgt_bboxes)

            target_bbox /= stride_tensor
            target_bboxes.append(target_bbox)
            target_labels.append(target_label)
            target_scores.append(target_score)
            fg_masks.append(fg_mask)

        target_bboxes = torch.stack(target_bboxes, 0)  # [bs,M,4]
        target_labels = torch.stack(target_labels, 0)  # [bs,M,]
        target_scores = torch.stack(target_scores, 0)  # [bs,M,Nc]
        fg_masks = torch.stack(fg_masks, 0)  # [bs,M,]

        target_scores_sum = max(target_scores.sum(), torch.tensor(1.0).to(device))  # 前景(正样本)对齐得分的和，用来排除批次之间正样本(数量/对齐程度)分布不均的情况
        if is_dist_avail_and_initialized():  # 排除进程之间分布不均的情况
            torch.distributed.all_reduce(target_scores_sum)  # 默认是求和
        target_scores_sum = (target_scores_sum / get_world_size()).clamp(1.0)  # 跨进程平均，最低为1

        # cls loss
        if self.VFL:
            if self.multi_hot:
                target_labels = target_labels.float()
            else:
                target_labels = F.one_hot(target_labels.long(), self.num_classes)
            loss_cls = self.cls_lossf(score_pred, target_scores, target_labels) / target_scores_sum  # VFL way
        else:
            loss_cls = self.cls_lossf(score_pred, target_scores.to(score_pred.dtype)).sum() / target_scores_sum
        # BCE  对所有样本都计算分类。 正因为score_pred没有经过激活函数，因此对于负样本可以引导向着全类别输出0训练

        # bbox loss
        if fg_masks.sum():  # 都没有乘以stride
            loss_box, loss_dfl = self.box_lossf(dist_pred, bbox_pred, anchor_point, target_bboxes, target_scores,
                                                target_scores_sum, fg_masks, iou_type=self.loss_iou_type)  # 只对正样本计算回归损失
        else:
            loss_box = 0
            loss_dfl = 0

        losses = loss_box * self.loss_box_weight + \
                 loss_cls * self.loss_cls_weight + \
                 loss_dfl * self.loss_dfl_weight

        loss_dict = dict(
            loss_box=loss_box,
            loss_cls=loss_cls,
            loss_dfl=loss_dfl,
            losses=losses
        )

        return loss_dict


class ConfCriterion(object):
    def __init__(self, m_cfg, img_size, num_classes=80, multi_hot=False):
        self.num_classes = num_classes
        self.img_size = img_size
        self.multi_hot = multi_hot
        self.loss_iou_type = m_cfg['loss_iou_type']  # 训练时box_loss计算中用到的IoU类型，可选'iou'、'giou'等
        self.matcher_iou_type = m_cfg['matcher_iou_type']

        # loss
        self.conf_iou_aware = m_cfg['conf_iou_aware']  # 同时控制cls分支和conf分支
        self.cls_ori_iou_type = m_cfg['cls_ori_iou_type'] if not self.conf_iou_aware else False
        self.nwd = m_cfg['nwd']
        self.obj_lossf = nn.BCEWithLogitsLoss(reduction='none')
        self.cls_lossf = nn.BCEWithLogitsLoss(reduction='none')  # 自带了sigmoid计算，之后再计算BCE损失
        self.loss_conf_weight = m_cfg['CCloss_conf_weight']
        self.loss_cls_weight = m_cfg['CCloss_cls_weight']
        self.loss_box_weight = m_cfg['CCloss_box_weight']

        # matcher 正样本分配器
        self.center_sampling_radius = m_cfg['center_sampling_radius']
        self.topk_candidate = m_cfg['topk_candidate']
        self.matcher = SimOTA(
            num_classes=self.num_classes,
            center_sampling_radius=self.center_sampling_radius,
            topk_candidate=self.topk_candidate,
            matcher_iou_type=self.matcher_iou_type
            )

    def __call__(self, outputs, targets):        
        """
            outputs: (Dict) -> {
                ("conf_pred": conf_pred,  # (Tensor) [B, M, 1])
                ("cls_pred": cls_pred,  # (Tensor) [B, M, Nc])
                ("score_pred": score_pred,  # (Tensor) [B, M, Nc])
                "bbox_pred": bbox_pred,  # (Tensor) [B, M, 4]  还没乘以stride
                ("dist_pred": dist_pred,  # (Tensor) [B, M, self.reg_max * 4])
                "anchor_point": anchor_point,  # (Tensor) [M, 2]
                "stride_tensor": stride_tensor}  # (Tensor) [M, 1]
            }
            targets: (List) [dict{'boxes': [...], 
                                 'labels': [...], 
                                 'orig_size': ...}, ...]   这里的真实标注框是两点式百分比形式
        """
        batch_size = outputs['bbox_pred'].shape[0]
        device = outputs['bbox_pred'].device
        anchor_point = outputs['anchor_point']
        stride_tensor = outputs['stride_tensor']
        conf_pred = outputs['conf_pred']
        cls_pred = outputs['cls_pred']
        bbox_pred = outputs['bbox_pred'] * stride_tensor

        # label assignment 标注分配
        cls_targets = []
        box_targets = []
        conf_targets = []
        fg_masks = []

        for batch_idx in range(batch_size):  # 对于每一个样本
            tgt_labels = targets[batch_idx]["labels"].to(device)
            tgt_bboxes = targets[batch_idx]["boxes"].to(device)
            # denormalize tgt_bbox  真实标注框恢复到两点式绝对坐标表示
            tgt_bboxes *= self.img_size

            # check target
            if len(tgt_labels) == 0 or tgt_bboxes.max().item() == 0.:  # 如果该样本的真实标注数量为0或者真实标注框的坐标的最大值为0代表着没有真实标注框
                num_anchors = anchor_point.shape[0]  # 锚点框总数
                # There is no valid gt  各个分支的训练目标全部设置为0  0代表背景类
                cls_target = conf_pred.new_zeros((0, self.num_classes))
                box_target = conf_pred.new_zeros((0, 4))
                conf_target = conf_pred.new_zeros((num_anchors, 1))  # 存在目标的置信度目标为0
                fg_mask = conf_pred.new_zeros(num_anchors).bool()
            else:
                gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg_img = self.matcher(
                    anchor_point=anchor_point,
                    stride_tensor=stride_tensor,
                    conf_pred=conf_pred[batch_idx],
                    cls_pred=cls_pred[batch_idx],
                    bbox_pred=bbox_pred[batch_idx],
                    tgt_labels=tgt_labels,
                    tgt_bboxes=tgt_bboxes)
                # num_fg 该样本下分配出去的预测框总数
                # gt_matched_classes 分配出去的预测框对应的目标框类别 [num_fg,]
                # pred_ious_this_matching 分配出去的预测框与对应的目标框之间的IoU [num_fg,]
                # matched_gt_inds 分配出去的预测框对应的目标框序号 [num_fg,]
                # fg_mask 记录分配出去的预测框的mask [M,]

                conf_target = fg_mask.unsqueeze(-1)  # 置信度目标  是否存在动作实例的obj分支目标 [M, 1]
                box_target = tgt_bboxes[matched_gt_inds]  # reg回归目标 分配出的预测框对应的目标框 [num_fg,4]
                if self.multi_hot:
                    cls_target = gt_matched_classes.float()
                else:
                    cls_target = F.one_hot(gt_matched_classes.long(), self.num_classes)  # 分配出去的预测框对应的目标框类别
                if self.cls_ori_iou_type:  # 如果cls采用原始iou
                    cls_target = cls_target * pred_ious_this_matching.unsqueeze(-1)

            cls_targets.append(cls_target)
            box_targets.append(box_target)
            conf_targets.append(conf_target)
            fg_masks.append(fg_mask)

        cls_targets = torch.cat(cls_targets, 0)  # 批次内拼接 三个分支的目标，cls和box只对正样本有效
        box_targets = torch.cat(box_targets, 0)
        conf_targets = torch.cat(conf_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        num_foregrounds = fg_masks.sum()  # 批次内的前景样本或者叫正样本数量

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foregrounds)  # 默认是求和
        num_foregrounds = (num_foregrounds / get_world_size()).clamp(1.0)  # 跨进程平均，最低为1

        # box loss  默认采用1-GIoU作为回归损失
        matched_box_preds = bbox_pred.view(-1, 4)[fg_masks]
        iou = calculate_iou(matched_box_preds, box_targets, iou_type=self.loss_iou_type)  # （N,1）
        if self.nwd:  # 如果使用nwd，则用nwd修正iou
            nwd = wasserstein_loss(matched_box_preds, box_targets).unsqueeze(-1)  # (N,1)
            iou_ratio = 0.8
            loss_box = \
                ((1 - iou_ratio) * (1.0 - nwd) + iou_ratio * (1.0 - iou))  # iou loss
        else:
            loss_box = 1.0 - iou   # 前提是iou上限不超过1，对iou、giou、diou、ciou有效
        loss_box = loss_box.sum() / num_foregrounds

        # conf loss
        conf_targets = conf_targets.float()
        if self.conf_iou_aware:  # 如果loss关注iou，则用iou同时修正conf target和cls target，来体现该训练目标的可信度
            if self.nwd:
                iou = (iou.detach() * iou_ratio + nwd.detach() * (1 - iou_ratio))  # 如果使用nwd，则用nwd修正iou
            iou = ((iou.detach()+1)/2).clamp(0, 1)  # BCE损失计算输入限制在0~1之间,要根据iou类型的值域来钳位 giou diou ciou均为[-1,1]
            conf_targets[fg_masks] = iou * conf_targets[fg_masks]
            cls_targets = cls_targets * iou
        elif not self.cls_ori_iou_type:  # cls不使用原始iou-type修正的话，就使用一致的iou-type修正
            iou = ((iou.detach()+1)/2).clamp(0, 1)
            cls_targets = cls_targets * iou

        loss_conf = self.obj_lossf(conf_pred.view(-1, 1), conf_targets)  # 没看到使用focal loss啊？？  BCE损失目标conf可以乘以iou
        loss_conf = loss_conf.sum() / num_foregrounds  # 排除样本之间前景数量不均匀的影响
        
        # cls loss
        matched_cls_preds = cls_pred.view(-1, self.num_classes)[fg_masks]
        loss_cls = self.cls_lossf(matched_cls_preds, cls_targets)
        loss_cls = loss_cls.sum() / num_foregrounds
        
        # total loss
        losses = self.loss_conf_weight * loss_conf + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_box_weight * loss_box

        loss_dict = dict(
            loss_box=loss_box,
            loss_cls=loss_cls,
            loss_conf=loss_conf,
            losses=losses
        )

        return loss_dict


def build_criterion(m_cfg, img_size, num_classes, multi_hot=False, noconf=False):
    if noconf:
        criterion = NoConfCriterion(m_cfg, img_size, num_classes, multi_hot)
    else:
        criterion = ConfCriterion(m_cfg, img_size, num_classes, multi_hot)
    return criterion
    