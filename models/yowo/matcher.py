import torch.nn.functional as F
from utils.box_ops import *


class TaskAlignedAssigner(torch.nn.Module):

    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9, matcher_iou_type='ciou'):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.matcher_iou_type = matcher_iou_type  # matcher中的iou可以与loss中的iou独立

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anchor_point, stride_tensor, gt_labels, gt_bboxes):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pd_scores (Tensor): shape(M, Nc)
            pd_bboxes (Tensor): shape(M, 4)    with stride

            anchor_point (Tensor): shape(M, 2)
            stride_tensor (Tensor): shape(M, 1)
            gt_labels (Tensor): shape(N,)
            gt_bboxes (Tensor): shape(N, 4)

        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        anchor_point_with_stride = anchor_point * stride_tensor
        self.num_anchor = anchor_point_with_stride.shape[0]
        self.num_gt = len(gt_labels)  # N

        if self.num_gt == 0:  # 可以移到外面去
            device = gt_bboxes.device
            return (torch.full_like(pd_scores[..., 0], self.num_classes).to(device),
                    torch.zeros_like(pd_bboxes).to(device),
                    torch.zeros_like(pd_scores).to(device),
                    torch.zeros_like(pd_scores[..., 0], dtype=torch.bool).to(device),
                    torch.zeros_like(pd_scores[..., 0]).to(device))

        # mask_pos：描述每个gt对应的topk个正例预测框的mask
        # align_metric：每个gt与每个预测框的度量尺度（综合了正确类别得分、IoU） cIoU
        # overlaps：每个gt与每个预测框的重叠程度  默认cIoU
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_point_with_stride)

        # 对于一个预测框被分配给多个gt框的情况，把该预测框只分配给重合程度最高的gt框
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps)

        # assigned target
        # target_scores是one-hot的(M,Nc)，只在分配出去的预测框和对应的gt之间取1，其余取0
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # normalize 对于分配结果，在gt数量维度上进行归一化
        align_metric *= mask_pos  # N,M   一行中可能有多个非零项，一列中最多只有一个非零项
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # N,1   每一行一个非零项，对应对齐尺度(综合了类别得分和重合程度)最大的那个预测框
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # N,1 每一行一个非零项，对应重合程度最大的那个预测框
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)  # M,1
        target_scores = target_scores * norm_align_metric  # M,Nc

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anchor_point_with_stride):
        # get anchor_align metric, (N, M)  每个gt框与每个预测框之间的对齐度量(综合了对应类别得分和iou)，其中用了ciou
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)   # N,M
        # get in_gts mask, (N, M)  与每个gt框对应的候选锚点框
        mask_in_gts = self.select_candidates_in_gts(anchor_point_with_stride, gt_bboxes)
        # get topk_metric mask, (N, M) 根据对齐尺度，找到与每个gt框对应的topk个正例
        mask_topk = self.select_topk_candidates(align_metric * mask_in_gts)  # 为每一个gt挑选topk个正例anchor
        # merge all mask to a final mask, (N, M)
        mask_pos = mask_topk * mask_in_gts  # 标注每个gt对应的正例anchor的mask

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):
        bbox_scores = pd_scores.permute(1, 0).contiguous()[gt_labels, :]  # N,M

        overlaps = calculate_iou(gt_bboxes.unsqueeze(1), pd_bboxes.unsqueeze(0),
                                 iou_type=self.matcher_iou_type).squeeze(2).clamp(0)  # N,M
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps   # N,M

    def select_topk_candidates(self, metrics, largest=True):
        """
        Args:
            metrics: (N, M).
        """
        # (N, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=largest)

        # (N, topk, M) -> (N, M)
        is_in_topk = F.one_hot(topk_idxs, self.num_anchor).sum(-2)
        # filter invalid bboxes
        is_in_topk = torch.where(is_in_topk > 1, 0, is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Args:
            gt_labels: (N, 1)
            gt_bboxes: (N, 4)
            target_gt_idx: (M,)
            fg_mask: (M,)
        """
        target_bboxes = gt_bboxes[target_gt_idx]  # M,4
        target_labels = gt_labels[target_gt_idx]  # M,

        # assigned target scores
        target_labels.clamp(0)
        target_scores = F.one_hot(target_labels, self.num_classes)  # (M, Nc)
        fg_scores_mask = fg_mask[:, None].repeat(1, self.num_classes)  # (M, Nc)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)  # (M, Nc) 分配出去的预测框对应的行one-hot，其他行全是0

        return target_labels, target_bboxes, target_scores

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9):
        """select the positive anchor center in gt

        Args:
            xy_centers (Tensor): shape(M, 4)
            gt_bboxes (Tensor): shape(N, 4)
        Return:
            (Tensor): shape(N, M)
        """
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = \
            torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(self.num_gt, self.num_anchor, -1)
        # return (bbox_deltas.min(3)[0] > eps).to(gt_bboxes.dtype)
        return bbox_deltas.amin(2).gt_(eps)

    # 对于一个预测框被分配给多个gt框的情况，把该预测框只分配给重合程度最高的gt框
    def select_highest_overlaps(self, mask_pos, overlaps):  # 纳入类别方法
        """if an anchor box is assigned to multiple gts,
            the one with the highest iou will be selected.

        Args:
            mask_pos (Tensor): shape(N, M)
            overlaps (Tensor): shape(N, M)
        Return:
            target_gt_idx (Tensor): shape(M,)
            fg_mask (Tensor): shape(M,)
            mask_pos (Tensor): shape(N, M)
        """
        # (N, M) -> (M,)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(0) > 1).repeat([self.num_gt, 1])  # (N, M)  一个预测框对应多个gt框的那一列全是True，其余为False
            max_overlaps_idx = overlaps.argmax(0)  # (M,)  与每个预测框的重合程度最高的gt框序号
            is_max_overlaps = F.one_hot(max_overlaps_idx, self.num_gt)  # (M，N)
            is_max_overlaps = is_max_overlaps.permute(1, 0).contiguous().to(overlaps.dtype)  # (N,M) 每列有且仅有一个为1，其余为0
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (N,M)  多重分配的那个预测框 分配给为重合程度最高的那个gt
            fg_mask = mask_pos.sum(-2)  # 更新，指示哪个预测框被分配了出去
        # find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (M,) 每个预测框被分配给了哪个gt框  没分配出去的默认是0
        return target_gt_idx, fg_mask, mask_pos


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(lrwb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 1 - 0.01)
    # dist (lt, rb)限制距离的上下限


# SimOTA
class SimOTA(object):
    def __init__(self, num_classes, center_sampling_radius, topk_candidate, matcher_iou_type='iou'):
        self.num_classes = num_classes
        self.center_sampling_radius = center_sampling_radius   # 领域半径
        self.topk_candidate = topk_candidate  # 每个目标框最多得到topk_candidate个正样本
        self.matcher_iou_type = matcher_iou_type

    @torch.no_grad()
    def __call__(self,
                 anchor_point,  # [M, 2]
                 stride_tensor,     # [M, 1]
                 conf_pred,  # [M, 1]
                 cls_pred,  # [M, Nc]
                 bbox_pred,  # [M, 4]
                 tgt_labels,  # [N,]
                 tgt_bboxes):  # [N, 4]

        num_anchor = anchor_point.shape[0]  # 三个层级的anchor总数
        num_gt = len(tgt_labels)  # 该样本的目标框数量

        fg_mask, is_in_boxes_and_center = \
            self.get_in_boxes_info(
                tgt_bboxes,
                anchor_point,
                stride_tensor,
                num_anchor,
                num_gt
                )
        # 判断每个锚点框是否处于真实目标框之中，找到处于目标框附近的锚点框
        # fg_mask形状为[M,]表示该锚点框是否至少在一个目标框内或邻域内，Mp是候选锚点框数量
        # is_in_boxes_and_center形状为[N，Mp] 表示候选锚点框的中心是否既在该目标框的内部又在该目标框的邻域内  mask   [N，Mp]

        conf_pred_ = conf_pred[fg_mask]   # [Mp, 1]  候选预测框的置信度预测
        cls_pred_ = cls_pred[fg_mask]   # [Mp, Nc]
        bbox_pred_ = bbox_pred[fg_mask]   # [Mp, 4]
        num_in_boxes_anchor = bbox_pred_.shape[0]  # 候选预测框数量
        # [N, Mp]
        pair_wise_ious = calculate_iou(
            tgt_bboxes.unsqueeze(1), bbox_pred_.unsqueeze(0), iou_type=self.matcher_iou_type).squeeze(2)
        if self.matcher_iou_type in ['giou', 'diou', 'ciou']:  # 钳位
            pair_wise_ious = (pair_wise_ious + 1) / 2
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)  # IoU损失  作为回归代价

        if len(tgt_labels.shape) == 1:  # 如果目标框的类别不是one hot表示则转换为one hot表示
            gt_cls = F.one_hot(tgt_labels.long(), self.num_classes)
        elif len(tgt_labels.shape) == 2:
            gt_cls = tgt_labels  # [N, C]

        # [N, C] -> [N, Mp, C]
        gt_cls = gt_cls.float().unsqueeze(1).repeat(1, num_in_boxes_anchor, 1)

        with torch.cuda.amp.autocast(enabled=False):
            score_preds_ = torch.sqrt(
                cls_pred_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * conf_pred_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_())  # [N, Mp, C]  用置信度修正类别得分
            pair_wise_cls_loss = F.binary_cross_entropy(
                score_preds_, gt_cls, reduction="none").sum(-1)  # [N, Mp]  二元交叉熵作为代价
        del score_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )  # [N, Mp] 候选预测框与每个目标框的代价矩阵  第三项表示如果候选预测框对应的锚点框不在目标框的核心区域则施加巨大惩罚

        (
            num_fg,
            gt_matched_classes,         # [num_fg,]
            pred_ious_this_matching,    # [num_fg,]
            matched_gt_inds,            # [num_fg,]
        ) = self.dynamic_k_matching(
            cost,
            pair_wise_ious,
            tgt_labels,
            num_gt,
            fg_mask
            )
        # 动态分配  每个目标框所分配的正样本(预测框)数量不是固定的
        # num_fg 该样本下分配出去的预测框总数
        # gt_matched_classes 分配出去的预测框对应的目标框类别 [num_fg,]
        # pred_ious_this_matching 分配出去的预测框与对应的目标框之间的IoU [num_fg,]
        # matched_gt_inds 分配出去的预测框对应的目标框序号 [num_fg,]
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        # fg_mask 记录分配出去的预测框的mask [M,]
        return gt_matched_classes, fg_mask, pred_ious_this_matching, matched_gt_inds, num_fg

    # 判断锚点框是否处于目标框附近
    def get_in_boxes_info(
            self,
            gt_bboxes,    # [N, 4]
            anchor_point,      # [M, 2]
            stride_tensor,      # [M, 1]
            num_anchors,  # M
            num_gt,       # N
            ):
        # anchor center 锚点框的中心的绝对坐标
        anchor_point_with_stride = anchor_point * stride_tensor  # [M, 2]
        x_centers = anchor_point_with_stride[:, 0]  # [M,]
        y_centers = anchor_point_with_stride[:, 1]  # [M,]

        # [M,] -> [1, M] -> [N, M]  为了表示跟每个目标框的关系，重复N次
        x_centers = x_centers.unsqueeze(0).repeat(num_gt, 1)
        y_centers = y_centers.unsqueeze(0).repeat(num_gt, 1)

        # [N,] -> [N, 1] -> [N, M]  目标框的四个坐标重复M次
        gt_bboxes_l = gt_bboxes[:, 0].unsqueeze(1).repeat(1, num_anchors)  # x1
        gt_bboxes_t = gt_bboxes[:, 1].unsqueeze(1).repeat(1, num_anchors)  # y1
        gt_bboxes_r = gt_bboxes[:, 2].unsqueeze(1).repeat(1, num_anchors)  # x2
        gt_bboxes_b = gt_bboxes[:, 3].unsqueeze(1).repeat(1, num_anchors)  # y2

        b_l = x_centers - gt_bboxes_l  # 锚点框中心到目标框左上角的x差值 大于0才行
        b_r = gt_bboxes_r - x_centers
        b_t = y_centers - gt_bboxes_t
        b_b = gt_bboxes_b - y_centers
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0  # 差值全部大于0才说明这个锚点框的中心在这个目标框内部  mask [N,M,]
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0  # 该锚点框的中心在至少一个目标框的内部  mask  [M，]
        # in fixed center  固定的邻域
        center_radius = self.center_sampling_radius

        # [N, 2]  目标框的中心的绝对坐标
        gt_centers = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) * 0.5
        
        # [1, M]
        center_radius_ = center_radius * stride_tensor.reshape(1, -1)

        #  [N, M]
        gt_bboxes_l = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) - center_radius_  # x1  邻域的左上方x坐标
        gt_bboxes_t = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) - center_radius_  # y1
        gt_bboxes_r = gt_centers[:, 0].unsqueeze(1).repeat(1, num_anchors) + center_radius_  # x2
        gt_bboxes_b = gt_centers[:, 1].unsqueeze(1).repeat(1, num_anchors) + center_radius_  # y2

        c_l = x_centers - gt_bboxes_l  # 锚点框中心到目标框邻域左上角的x差值 大于0才行
        c_r = gt_bboxes_r - x_centers
        c_t = y_centers - gt_bboxes_t
        c_b = gt_bboxes_b - y_centers
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0  # 差值全部大于0才说明这个锚点框的中心在这个目标框邻域内  mask [N,M,]
        is_in_centers_all = is_in_centers.sum(dim=0) > 0  # 该锚点框的中心在至少一个目标框的邻域内  mask  [M，]

        # in boxes and in centers 该锚点框的中心至少在一个目标框的内部或者邻域内 mask   [M，]
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )  # 候选锚点框的中心是否既在该目标框的内部又在该目标框的邻域内  mask   [N，Mp]
        return is_in_boxes_anchor, is_in_boxes_and_center
    
    # 将候选预测框动态分配给目标框
    def dynamic_k_matching(
        self, 
        cost,            # 代价矩阵  [N,Mp]
        pair_wise_ious,  # iou矩阵  [N,Mp]
        gt_classes,      # [N,]
        num_gt,          # [N,]
        fg_mask          # [M,]  候选锚点框的mask，候选表示该锚点框是否至少在一个目标框内或邻域内
        ):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)  # [N,Mp]

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(self.topk_candidate, ious_in_boxes_matrix.size(1))  # 一个目标框能分到的正样本数量上限
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)  # 每个目标框的topk个IoU  [N,topk]
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)  # 判断每个目标框应该分配几(k)个正样本，最少1个  [N,]
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )  # 为该目标框找到损失最小的k个预测框，k对于不同目标框不一样
            matching_matrix[gt_idx][pos_idx] = 1  # 匹配矩阵，用来标记分配情况[N,Mp]

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)  # 检查是否有一个预测框被分配给多个目标框，将预测框分配给代价最小的那个目标框
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0  # 成功分配出去的预测框mask [Mp,]
        num_fg = fg_mask_inboxes.sum().item()  # 该样本下分配出去的预测框总数

        fg_mask[fg_mask.clone()] = fg_mask_inboxes  # fg_mask变成记录分配出去的预测框的mask [M,]

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)  # 分配出去的预测框对应的目标框序号 [num_fg,]
        gt_matched_classes = gt_classes[matched_gt_inds]  # 分配出去的预测框对应的目标框类别 [num_fg,]

        pred_ious_this_matching = \
            (matching_matrix * pair_wise_ious).sum(0)[fg_mask_inboxes]  # 这种分配后，分配出去的预测框与对应的目标框之间的IoU [num_fg,]

        # num_fg 该样本下分配出去的预测框总数
        # gt_matched_classes 分配出去的预测框对应的目标框类别 [num_fg,]
        # pred_ious_this_matching 分配出去的预测框与对应的目标框之间的IoU [num_fg,]
        # matched_gt_inds 分配出去的预测框对应的目标框序号 [num_fg,]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
        