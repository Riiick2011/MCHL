import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import build_backbone_2d
from ..backbone import build_backbone_3d
from models.yowo.encoder import build_channel_encoder
from models.yowo.fpn import build_fpn
from .head import build_head

from utils.nms import multiclass_nms
from utils.box_ops import calculate_iou

from .matcher import dist2bbox
from .cross_merge import cross_merge
from torchvision.transforms import Resize


# You Only Watch Once
class YOWO(nn.Module):
    def __init__(self, 
                 m_cfg,
                 device,
                 num_classes=20, 
                 conf_thresh=0.05,
                 nms_thresh=0.6,
                 nms_iou_type='iou',
                 totaltopk=40,
                 trainable=False,
                 multi_hot=False,
                 clstopk=0,
                 det_save_type='one_class',
                 bbox_with_feat=False):
        super(YOWO, self).__init__()
        self.m_cfg = m_cfg
        self.device = device
        self.stride = m_cfg['stride']  # stride是个列表，包含3项  表示骨架输出乃至最终输出的网格大小
        self.num_classes = num_classes
        self.trainable = trainable
        self.multi_hot = multi_hot

        # 推断时用
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.nms_iou_type = nms_iou_type
        self.clstopk = 0 if self.multi_hot else clstopk  # 只有在one-hot时才可以逐个类别取topk
        self.totaltopk = 0 if self.clstopk else totaltopk  # 如果给定了clstopk则totaltopk不起作用
        self.det_save_type = det_save_type
        self.bbox_with_feat = bbox_with_feat

        # ------------------ Network ---------------------
        # 2D backbone  构建骨架网络，如果有预训练模型并且处于训练模式则还载入预训练模型
        # bk_dim_2d是一个3项的列表，每一项表示一个层级的特征图通道数，[256,256,256]
        self.backbone_2d, bk_dim_2d,  = build_backbone_2d(
            self.m_cfg, pretrained=self.m_cfg['pretrained_2d'] and self.trainable)
            
        # 3D backbone  构建骨架网络，如果有预训练模型并且处于训练模式则还载入预训练模型
        # bk_dim_3d是一个3项的列表，每一项表示一个层级的特征图通道数,是[2048]或者[512,1024,2048]
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            self.m_cfg, pretrained=self.m_cfg['pretrained_3d'] and self.trainable)

        self.level_2d = len(bk_dim_2d)  # 特征图层级数量
        self.level_3d = len(bk_dim_3d)
        self.decoupled_early = self.m_cfg['decoupled_early']
        self.noconf = self.m_cfg['noconf']
        self.fpn = self.m_cfg['fpn']

        if self.decoupled_early:  # 早期解耦
            # 2D与3D特征融合并编码
            if self.level_3d == 1:  # 单层级3D
                # cls channel encoder 3个不同尺度的分类通道编码器
                self.cls_channel_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, bk_dim_2d[0]+bk_dim_3d[0], self.m_cfg['head_dim'])
                        for i in range(self.level_2d)])
                if self.fpn:
                    self.fpn = build_fpn(self.m_cfg['head_dim'])

                # reg channel & spatial encoder  3个不同尺度的回归通道编码器，仍然用的是通道注意力
                self.reg_channel_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, bk_dim_2d[0]+bk_dim_3d[0], self.m_cfg['head_dim'])
                        for i in range(self.level_2d)])
            else:  # 多层级3D  先两两组合，再通过通道注意力交换2D和3D的信息，再通过空间注意力来
                self.crossmerge = cross_merge(bk_dim_2d, bk_dim_3d)  # 交叉融合模块
                dim_merged = self.crossmerge.chout  # 256+2048
                # channel encoder 3*3个不同尺度的通道编码器，各自进行融合和注意力编码，编码结果拼接成列表送入头部
                self.cls_channel_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, dim_merged, self.m_cfg['head_dim'])
                     for _ in range(self.level_2d * self.level_3d)])
                self.reg_channel_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, dim_merged, self.m_cfg['head_dim'])
                     for _ in range(self.level_2d * self.level_3d)])

                # 9个特征图-》》3个特征图  压缩时间信息
                self.cls_temporal_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, self.m_cfg['head_dim'] * self.level_3d, self.m_cfg['head_dim'])
                     for _ in range(self.level_2d)])  # 融合3D特征
                self.reg_temporal_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, self.m_cfg['head_dim'] * self.level_3d, self.m_cfg['head_dim'])
                     for _ in range(self.level_2d)])  # 融合3D特征

            if self.noconf:
                self.reg_max = 1 if 'reg_max' not in self.m_cfg else self.m_cfg['reg_max']  # pred层的回归相关
                self.use_dfl = self.reg_max > 1  # 预测层是否采用dfl   只有noconf才可选是否使用dfl
                # 预测层，不输出conf_pred  内部不同层级是独立计算的
                self.heads = build_head(self.m_cfg, self.num_classes, decoupled_early=False,
                                        ch=[self.m_cfg['head_dim']] * self.level_2d)  # 不同尺度的特征图直接一起输入，在内部解耦
            else:
                # head 分别对应3个不同层级的头部  每个头部包含两个支路，并行输入，并行输出，内部独立的  均采用多层的3x3卷积  通道数不变
                self.heads = nn.ModuleList(
                    [build_head(self.m_cfg) for _ in range(self.level_2d)]
                )

                # pred 3个不同尺度的3种预测层
                self.conf_preds = nn.ModuleList(
                    [nn.Conv2d(self.m_cfg['head_dim'], 1, kernel_size=1)
                     for _ in range(self.level_2d)])
                self.cls_preds = nn.ModuleList(
                    [nn.Conv2d(self.m_cfg['head_dim'], self.num_classes, kernel_size=1)
                     for _ in range(self.level_2d)])
                self.reg_preds = nn.ModuleList(
                    [nn.Conv2d(self.m_cfg['head_dim'], 4, kernel_size=1)
                     for _ in range(self.level_2d)])

        else:  # 晚期解耦
            # 2D与3D特征融合并编码
            if self.level_3d == 1:  # 单层级3D
                # channel encoder 3个不同尺度的通道编码器，各自进行融合和注意力编码，编码结果拼接成列表送入头部
                self.channel_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, bk_dim_2d[0] + bk_dim_3d[0], self.m_cfg['head_dim'])
                     for i in range(self.level_2d)])
            else:   # 多层级3D  先两两组合，再通过通道注意力交换2D和3D的信息，再通过空间注意力来
                self.crossmerge = cross_merge(bk_dim_2d, bk_dim_3d)    # 交叉融合模块
                dim_merged = self.crossmerge.chout  # 256+2048
                # channel encoder 3*3个不同尺度的通道编码器，各自进行融合和注意力编码，编码结果拼接成列表送入头部
                self.channel_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, dim_merged, self.m_cfg['head_dim'])
                     for _ in range(self.level_2d * self.level_3d)])

                self.temporal_encoders = nn.ModuleList(
                    [build_channel_encoder(self.m_cfg, self.m_cfg['head_dim']*self.level_3d, self.m_cfg['head_dim'])
                     for _ in range(self.level_2d)])  # 融合3D特征
            if self.noconf:
                self.reg_max = 1 if 'reg_max' not in self.m_cfg else self.m_cfg['reg_max']  # pred层的回归相关
                self.use_dfl = self.reg_max > 1  # 预测层是否采用dfl   只有noconf才可选是否使用dfl
                # 预测层，不输出conf_pred  内部不同层级是独立计算的
                self.heads = build_head(self.m_cfg, self.num_classes, decoupled_early=False,
                                        ch=[self.m_cfg['head_dim']] * self.level_2d)  # 不同尺度的特征图直接一起输入，在内部解耦
            else:
                # head 分别对应3个不同层级的头部  每个头部包含两个支路，并行输入，并行输出，内部独立的  均采用多层的3x3卷积  通道数不变
                self.heads = nn.ModuleList(
                    [build_head(self.m_cfg) for _ in range(self.level_2d)]
                )

                # pred 3个不同尺度的3种预测层
                self.conf_preds = nn.ModuleList(
                    [nn.Conv2d(self.m_cfg['head_dim'], 1, kernel_size=1)
                     for _ in range(self.level_2d)])
                self.cls_preds = nn.ModuleList(
                    [nn.Conv2d(self.m_cfg['head_dim'], self.num_classes, kernel_size=1)
                     for _ in range(self.level_2d)])
                self.reg_preds = nn.ModuleList(
                    [nn.Conv2d(self.m_cfg['head_dim'], 4, kernel_size=1)
                     for _ in range(self.level_2d)])

        # init yowo
        self.init_yowo()

    def init_yowo(self):
        # Init yolo  初始化2维骨架yolo中的2D批次归一化
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

        if self.noconf:  # 无conf Detect
            self.heads.stride = torch.tensor(self.stride, dtype=torch.float32)
            self.heads.bias_init()
        else:
            # Init bias  单独初始化存在目标的置信度的偏置和分类得分的偏置
            # 回归部分的的偏置 采用默认的随机初始化
            init_prob = 0.01
            bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))  # -ln99
            # obj pred  # 存在目标的置信度
            for conf_pred in self.conf_preds:
                b = conf_pred.bias.view(1, -1)
                b.data.fill_(bias_value.item())
                conf_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            # cls pred  # 分类得分
            for cls_pred in self.cls_preds:
                b = cls_pred.bias.view(1, -1)
                b.data.fill_(bias_value.item())
                cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # 生成的锚点框是固定的，中心是网格中心，不是先验锚点框   独立一层的anchors
    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """
            Generate anchors from features.
            feats是一个列表，每一项是一个tensor对应一个层级的输出，形状为(B,C,H,W)
            strides是一个列表，每一项是一个int对应一个层级的stride
            返回沿着空间尺寸拼接好的anchor_points(M,2)和stride_tensor(M,1),M是所有层级的锚点框总数
        """
        anchor_points, stride_tensor = [], []
        assert feats is not None
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx)
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def bbox_decode(self, anchor_point, pred, xywh=False):
        """
        :param anchor_point: (Tensor)[M,2]
        :param pred:  (Tensor)[B,M,4 or reg_max*4]  检测框回归预测
        :param xywh:  (Bool) 输入的格式,True代表是相对锚点中心的xywh中心式百分比坐标，False代表是相对锚点中心的ltrb两点式百分比坐标
        :return: decoded_box (Tensor)[B,M,4] 返回两点式百分比坐标，还没有乘以stride
        """
        if xywh:  # 将相对锚点中心的xywh中心式百分比坐标预测结果转换为两点式百分比坐标
            # center of bbox  预测框中心的绝对坐标
            pred_ctr_xy = anchor_point + pred[..., :2]
            # size of bbox 预测框的宽高
            pred_box_wh = pred[..., 2:].exp()

            pred_x1y1 = pred_ctr_xy - 0.5 * pred_box_wh
            pred_x2y2 = pred_ctr_xy + 0.5 * pred_box_wh
            decoded_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)  # [B, M, 4] or [M, 4]  两点式百分比坐标
        else:  # 将相对锚点中心的ltrb距离两点式百分比坐标预测结果转换为两点式百分比坐标
            if self.use_dfl:
                b, m, c = pred.shape  # batch, anchors, channels
                pred = pred.view(b, m, 4, c // 4).softmax(3).matmul(
                    torch.arange(self.reg_max, dtype=torch.float, device=pred.device).type(pred.dtype))  # B,M,4  相对每个锚点的lrwb距离(还没乘以stride)
            decoded_box = dist2bbox(pred, anchor_point, xywh=False)  # xyxy两点式(还没乘以stride)
        return decoded_box

    # 给定经过NMS的bboxes坐标，和某一个层级的特征图，保存该特征图中被bboxes包含的特征
    def save_feature(self, batch_bboxes, featmap):
        """
        :param bboxes: list[array(N,4)]
        :param featmap:list[array(B,C,H,W)]
        :return:bboxes_feat
        """
        batch_size = len(batch_bboxes)
        if len(featmap) == 2:
            cls_featmaps=featmap[0]
        elif len(featmap) == 3:
            cls_featmaps=[featmap[_][0] for _ in range(len(featmap))]
        Hmax = max([cls_featmap.shape[2] for cls_featmap in cls_featmaps])
        torch_resize = Resize([Hmax, Hmax])
        cls_featmaps_ups = [F.interpolate(cls_featmaps[level], scale_factor=2 ** level) for level in range(self.level_2d)]
        cls_featmap = torch.stack(cls_featmaps_ups, dim=0).mean(dim=0)
        bboxes_feat = [[] for i in range(batch_size)]
        for i in range(batch_size):
            bboxes = batch_bboxes[i]
            for j in range(len(bboxes)):
                bbox = bboxes[j]
                x1, y1, x2, y2 = (bbox*Hmax).astype(int)
                bbox_feat = cls_featmap[i, :, x1:x2, y1:y2]
                bboxes_feat[i].append(torch_resize(bbox_feat))
        return np.array(bboxes_feat)

    # one_hot模式的后处理  只对eval和test起作用
    def post_process_one_hot(self, bbox_pred, cls_pred, conf_pred=None, noconf=False, person_proposal=None):
        """
        Input: 一个样本的预测输出
            conf_pred: (Tensor) [M, 1]
            cls_pred(score_pred): (Tensor) [M, Nc]
            bbox_pred: (Tensor) [M, 4]  中心式相对坐标  两点式绝对坐标
            person_proposal: (Tensor) [N, 4] or None
        """
        if person_proposal is not None:
            person_iou = calculate_iou(bbox_pred.unsqueeze(1), person_proposal.unsqueeze(0)).squeeze(-1)
            person_mask = person_iou >= 0.5  # 只保留与人员检测器给出的人员提议检测框重合程度超过0.8的预测框
            person_mask = torch.sum(person_mask, axis=1)
            person_mask = torch.where(person_mask != 0)
            bbox_pred = bbox_pred[person_mask]
            cls_pred = cls_pred[person_mask]
            if not noconf:
                conf_pred = conf_pred[person_mask]

        if self.det_save_type == 'multi_class':  # 如果每个检测保存所有类别得分
            if noconf:
                raise Exception('multi_class保存模式不支持noconf模式')
            else:
                # Keep top k top scoring indices only.  把所有预测框按照目标置信度排序，只保留最高的totaltopk
                num_topk = min(self.totaltopk, bbox_pred.shape[0])
                conf_pred = conf_pred.sigmoid()
                cls_pred = cls_pred.sigmoid()
                cls_pred = torch.sqrt(conf_pred * cls_pred)
                predicted_prob, topk_idxs = conf_pred.sort(axis=0, descending=True)
                topk_confs = predicted_prob[:num_topk]  # 最高的topk个得分
                topk_idxs = topk_idxs[:num_topk]  # 最高的topk个得分的原序号

                # filter out the proposals with low confidence score 保留修正分类得分大于置信度阈值的预测框以及对应原始序号，这样一个预测框可以对应多个类别
                keep_idxs = topk_confs > self.conf_thresh
                conf_pred = topk_confs[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]
                bboxes = bbox_pred[topk_idxs]  # 保留的预测框  两点式绝对坐标表示
                scores = cls_pred[topk_idxs]

                # to cpu
                conf_pred = conf_pred.cpu().numpy()
                scores = scores.cpu().numpy()
                bboxes = bboxes.cpu().numpy()

                # nms 不关注类别的多类别nms，各个类别互相影响
                conf_pred, scores, bboxes = multiclass_nms(
                    conf_pred, scores, bboxes, self.nms_thresh, self.nms_iou_type, num_classes=self.num_classes,
                    topk=self.totaltopk, class_agnostic=True)

            return conf_pred, scores, bboxes  # 输出该样本经过置信度筛选和nms筛选过后的预测框输出(不分层级)，其中预测框坐标是两点式绝对坐标
        else:
            if noconf:
                scores = cls_pred.sigmoid().flatten()  # [M * Nc,]  先排列同一个预测框的C个类别，再排列不同预测框
            else:
                # (H x W x C,)用置信度得分来修正每一个类别的得分，先排列同一个预测框的C个类别，再排列不同预测框
                scores = (torch.sqrt(conf_pred.sigmoid() * cls_pred.sigmoid())).flatten()

            # torch.sort is actually faster than .topk (at least on GPUs)
            predicted_prob, topk_idxs = scores.sort(descending=True)
            if self.totaltopk:  # 容易产生高分错误类别对低分正确类别的压制 虽然极大的降低了FP提升了FmAP但是会减少TP  TP对于ROAD管道关联来说比FP更重要，降低了VmAP
                # Keep top k top scoring indices only.  把所有预测框关于所有类别的修正分类得分排序，只保留最高的topk
                num_topk = self.totaltopk
                topk_scores = predicted_prob[:num_topk]  # 最高的topk个得分
                topk_idxs = topk_idxs[:num_topk]  # 最高的topk个得分的原序号

                # filter out the proposals with low confidence score 保留修正分类得分大于置信度阈值的预测框以及对应原始序号，这样一个预测框可以对应多个类别
                keep_idxs = topk_scores > self.conf_thresh
                scores = topk_scores[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]
            else:
                num_topk = self.clstopk
                # filter out the proposals with low confidence score 保留修正分类得分大于置信度阈值的预测框以及对应原始序号，这样一个预测框可以对应多个类别
                keep_idxs = predicted_prob > self.conf_thresh
                scores = predicted_prob[keep_idxs]
                topk_idxs = topk_idxs[keep_idxs]

            bbox_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')  # 保留的预测框所对应的锚点框序号
            labels = topk_idxs % self.num_classes  # 保留的预测框所对应的类别(在这里一个预测框和不同的类别算作多个预测)

            bboxes = bbox_pred[bbox_idxs]  # 保留的预测框  两点式绝对坐标表示

            # to cpu
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            bboxes = bboxes.cpu().numpy()

            # nms 关注类别的多类别nms，各个类别互不影响
            scores, labels, bboxes = multiclass_nms(
                scores, labels, bboxes, self.nms_thresh, self.nms_iou_type, num_classes=self.num_classes,
                topk=num_topk, class_agnostic=False)  # 对于totaltopk来说不起作用，对clstopk来说每个类别只保留clstopk个

            return scores, labels, bboxes  # 输出该样本经过置信度筛选和nms筛选过后的预测框输出(不分层级)，其中预测框坐标是两点式绝对坐标

    # multi_hot模式的后处理  只对eval和test起作用
    def post_process_multi_hot(self, bbox_pred, cls_pred, conf_pred=None, noconf=False, person_proposal=None):
        """
        Input: 一个样本的预测输出
            conf_pred: (Tensor) [M, 1]
            cls_pred: (Tensor) [M, Nc]
            bbox_pred: (Tensor) [M, 4]  中心式相对坐标  两点式绝对坐标
            person_proposal: (Tensor) [N, 4]
        """
        if person_proposal is not None:
            person_iou = calculate_iou(bbox_pred.unsqueeze(1), person_proposal.unsqueeze(0)).squeeze(-1)
            person_mask = person_iou >= 0.8
            person_mask = torch.sum(person_mask, axis=1)
            person_mask = torch.where(person_mask != 0)
            bbox_pred = bbox_pred[person_mask]
            cls_pred = cls_pred[person_mask]
            if not noconf:
                conf_pred = conf_pred[person_mask]

        if noconf:
            # cls_pred
            cls_pred = torch.sigmoid(cls_pred)  # [M, Nc]

            # topk
            _, topk_inds = cls_pred.amax(1).sort(descending=True)  # 按照预测框的最大得分类别排序
            topk_inds = topk_inds[:self.totaltopk]
            topk_cls_pred = cls_pred[topk_inds]  # [k, Nc]
            topk_bbox_pred = bbox_pred[topk_inds]  # [k, 4]

            # threshold
            keep = topk_cls_pred.amax(1).gt(self.conf_thresh)
            cls_pred = topk_cls_pred[keep]
            bbox_pred = topk_bbox_pred[keep]

            # to cpu
            scores = cls_pred.amax(1).cpu().numpy()  # 预测框的最大得分类别作为得分
            labels = cls_pred.cpu().numpy()
            bboxes = bbox_pred.cpu().numpy()
        else:
            # conf pred
            conf_pred = torch.sigmoid(conf_pred.squeeze(-1))   # [M, ]

            # cls_pred
            cls_pred = torch.sigmoid(cls_pred)                 # [M, Nc]

            # topk
            topk_conf_pred, topk_inds = torch.topk(conf_pred, self.totaltopk)
            topk_cls_pred = cls_pred[topk_inds]
            topk_bbox_pred = bbox_pred[topk_inds]

            # threshold
            keep = topk_conf_pred.gt(self.conf_thresh)
            conf_pred = topk_conf_pred[keep]
            cls_pred = topk_cls_pred[keep]
            bbox_pred = topk_bbox_pred[keep]

            # to cpu
            scores = conf_pred.cpu().numpy()
            labels = cls_pred.cpu().numpy()
            bboxes = bbox_pred.cpu().numpy()

        # 无视类别的多类别nms
        scores, labels, bboxes = multiclass_nms(
            scores, labels, bboxes, self.nms_thresh, self.nms_iou_type, num_classes=self.num_classes,
            topk=self.totaltopk, class_agnostic=True)

        # [M, 4 + 1 + Nc]
        out_boxes = np.concatenate([bboxes, scores[..., None], labels], axis=-1)

        return out_boxes

    # 表明当前计算不需要反向传播，使用之后，强制后边的内容不进行计算图的构建  只对eval和test起作用 不用于训练
    @torch.no_grad()
    def inference(self, video_clips, batch_target=None):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        return:
        """
        batch_size, _, _, img_h, img_w = video_clips.shape
        # key frame
        key_frame = video_clips[:, :, -1, :, :]
        # 3D backbone
        feats_3d = self.backbone_3d(video_clips)  # 列表表示

        if self.decoupled_early:  # Free-YOLO-LARGE or YOLOv8L
            # 2D backbone
            feats_2d = self.backbone_2d(key_frame)
            if len(feats_2d) == 2:  # Free YOLO
                cls_feats, reg_feats = self.backbone_2d(key_frame)
            else:  # YOLOv8
                cls_feats = reg_feats = self.backbone_2d(key_frame)

            if self.level_3d == 1:
                feat_3d_ups = [F.interpolate(feats_3d[0], scale_factor=2 ** (2 - level)) for level in
                               range(self.level_2d)]
                cls_feats = [self.cls_channel_encoders[level](cls_feats[level], feat_3d_ups[level])
                             for level in range(self.level_2d)]
                reg_feats = [self.reg_channel_encoders[level](reg_feats[level], feat_3d_ups[level])
                             for level in range(self.level_2d)]
            else:
                cls_feats = self.crossmerge(cls_feats, feats_3d)  # 列表，9个列表，每个都是2D，3D
                reg_feats = self.crossmerge(reg_feats, feats_3d)  # 列表，9个列表，每个都是2D，3D
                cls_feats = [self.cls_channel_encoders[level](cls_feats[level][0], cls_feats[level][1]) for level in
                             range(self.level_2d * self.level_3d)]
                reg_feats = [self.reg_channel_encoders[level](reg_feats[level][0], reg_feats[level][1]) for level in
                             range(self.level_2d * self.level_3d)]
                cls_feats = [
                    [cls_feats[idx_2d],
                     torch.cat([cls_feats[idx_2d + self.level_2d], cls_feats[idx_2d + self.level_2d * 2]], dim=1)]
                    for idx_2d in range(self.level_2d)]  # 同一个2D的不同tensor组合起来
                reg_feats = [
                    [reg_feats[idx_2d],
                     torch.cat([reg_feats[idx_2d + self.level_2d], reg_feats[idx_2d + self.level_2d * 2]], dim=1)]
                    for idx_2d in range(self.level_2d)]  # 同一个2D的不同tensor组合起来

                cls_feats = [self.cls_temporal_encoders[level](cls_feats[level][0], cls_feats[level][1])
                             for level in range(self.level_2d)]
                reg_feats = [self.reg_temporal_encoders[level](reg_feats[level][0], reg_feats[level][1])
                             for level in range(self.level_2d)]

            if self.noconf:
                # pred-decoupled head
                feats = [cls_feats, reg_feats]  # 嵌套列表，先解耦后等级
            else:
                feats = [self.heads[level](cls_feats[level], reg_feats[level]) for level in
                         range(self.level_2d)]  # 嵌套列表，先等级后解耦

        else:  # YOLOv8l
            # 2D backbone
            feats_2d = self.backbone_2d(key_frame)
            if self.level_3d == 1:
                feat_3d_ups = [F.interpolate(feats_3d[0], scale_factor=2 ** (2 - level)) for level in
                               range(len(feats_2d))]
                feats = [self.channel_encoders[level](feats_2d[level], feat_3d_ups[level]) for level in
                         range(len(feats_2d))]
            else:
                feats = self.crossmerge(feats_2d, feats_3d)  # 列表，9个列表，每个都是2D，3D
                feats = [self.channel_encoders[level](feats[level][0], feats[level][1]) for level in
                         range(self.level_2d * self.level_3d)]
                feats = [[feats[idx_2d],
                          torch.cat([feats[idx_2d + self.level_3d], feats[idx_2d + self.level_3d * 2]], dim=1)]
                         for idx_2d in range(self.level_2d)]  # 同一个2D的不同tensor组合起来
                feats = [self.temporal_encoders[level](feats[level][0], feats[level][1])
                         for level in range(self.level_2d)]

            if not self.noconf:
                feats = [self.heads[level](feats[level], feats[level]) for level in range(self.level_2d)]  # 嵌套列表,先等级后解耦

        if self.noconf:
            # 列表，含有层级数量项，每一项是一个tensor(由reg和cls在通道维度上拼接而成)
            preds = self.heads(feats, decoupled_in=self.decoupled_early)  # B,C,H,W
            dist_pred, score_pred = \
                torch.cat([xi.view(batch_size, self.reg_max * 4 + self.num_classes, -1) for xi in preds], 2
                          ).permute(0, 2, 1).contiguous().split(
                    (self.reg_max * 4, self.num_classes), 2)  # 跨层级，在锚点框个数上拼接起来，然后再把回归和分类结果分开  B,M,C

            # decode
            anchor_point, stride_tensor = self.make_anchors(preds, self.stride)  # 所有层级拼接在一起   层级拼在一起的anchor
            bbox_pred = self.bbox_decode(anchor_point, dist_pred)  # B,M,4  xyxy两点式坐标(还没乘以stride)
        else:
            # pred
            conf_pred = [self.conf_preds[level](feats[level][1]) for level in range(self.level_2d)]  # B,C,H,W
            anchor_point, stride_tensor = self.make_anchors(conf_pred, self.stride)  # 所有层级拼接在一起   层级拼在一起的anchor
            # B,M,C
            conf_pred = torch.cat([_.flatten(2, 3) for _ in conf_pred], dim=-1
                                  ).permute(0, 2, 1).contiguous()
            cls_pred = torch.cat([self.cls_preds[level](feats[level][0]).flatten(2, 3)
                                  for level in range(self.level_2d)], dim=-1).permute(0, 2, 1).contiguous()
            reg_pred = torch.cat([self.reg_preds[level](feats[level][1]).flatten(2, 3)
                                  for level in range(self.level_2d)], dim=-1).permute(0, 2, 1).contiguous()

            # decode
            bbox_pred = self.bbox_decode(anchor_point, reg_pred, xywh=True)  # B,M,4  xyxy两点式坐标(还没乘以stride)

        # 后处理
        bbox_pred = bbox_pred * stride_tensor
        if self.multi_hot:
            batch_bboxes = []
            for batch_idx in range(batch_size):

                # 单独的人员检测器输出的人员提议
                if batch_target is not None:   # 只有multisports中有
                    person_proposal = batch_target[batch_idx]['person_proposal']
                    if person_proposal is not None:
                        person_proposal = person_proposal[:, :4] * img_w
                        person_proposal = torch.tensor(person_proposal).to(bbox_pred.device)
                else:
                    person_proposal = None

                # post-process
                if self.noconf:
                    out_boxes = self.post_process_multi_hot(bbox_pred[batch_idx], score_pred[batch_idx],
                                                            noconf=True, person_proposal=person_proposal)
                else:
                    out_boxes = self.post_process_multi_hot(bbox_pred[batch_idx], cls_pred[batch_idx],
                                                            conf_pred[batch_idx], noconf=False,
                                                            person_proposal=person_proposal)

                # normalize bbox 归一化输出
                out_boxes[..., :4] /= max(img_h, img_w)
                out_boxes[..., :4] = out_boxes[..., :4].clip(0., 1.)

                batch_bboxes.append(out_boxes)
            if self.bbox_with_feat:
                batch_bboxes_feat = self.save_feature(batch_bboxes, feats)
                return batch_bboxes, batch_bboxes_feat
            else:
                return batch_bboxes
        else:
            batch_scores = []  # 共有batch size项，每一项是一个tensor对应一个样本的输出
            batch_labels = []
            batch_bboxes = []  # 是两点式百分比坐标
            for batch_idx in range(batch_size):  # 批次内的第batch_idx个样本，逐个样本进行后处理

                # 单独的人员检测器输出的人员提议
                if batch_target is not None:  # 只有multisports中有
                    person_proposal = batch_target[batch_idx]['person_proposal']
                    if person_proposal is not None:
                        person_proposal = person_proposal[:, :4] * img_w
                        person_proposal = torch.tensor(person_proposal).to(bbox_pred.device)
                else:
                    person_proposal = None

                # [B, M, C] -> [M, C]
                # post-process  对该样本的多层级输出进行后处理
                # 输入的坐标是两点式绝对坐标
                # 输出该样本经过置信度筛选和nms筛选过后的预测框输出(不分层级)，其中预测框坐标是两点式绝对坐标
                if self.noconf:
                    scores, labels, bboxes = self.post_process_one_hot(bbox_pred[batch_idx], score_pred[batch_idx],
                                                                       noconf=True, person_proposal=person_proposal)
                else:
                    scores, labels, bboxes = self.post_process_one_hot(bbox_pred[batch_idx], cls_pred[batch_idx],
                                                                       conf_pred[batch_idx], noconf=False,
                                                                       person_proposal=person_proposal)
                # normalize bbox  再将坐标归一化并且钳位
                bboxes /= max(img_h, img_w)
                bboxes = bboxes.clip(0., 1.)

                batch_scores.append(scores)
                batch_labels.append(labels)
                batch_bboxes.append(bboxes)
            if self.bbox_with_feat:
                batch_bboxes_feat = self.save_feature(batch_bboxes, feats)
                return batch_scores, batch_labels, batch_bboxes, batch_bboxes_feat
            else:
                return batch_scores, batch_labels, batch_bboxes  # 列表，含有batch_size项，每一项是一个tensor对应一个样本，其中预测框坐标是两点式百分比坐标

    # 该方法用于训练
    def forward(self, video_clips, batch_target=None):
        """
        Input:
            video_clips: (Tensor) -> [B, 3, T, H, W].
        return:
            outputs: (Dict) -> {
                ("conf_pred": conf_pred,  # (Tensor) [B, M, 1])
                ("cls_pred": cls_pred,  # (Tensor) [B, M, Nc])
                ("score_pred": score_pred,  # (Tensor) [B, M, Nc])
                "bbox_pred": bbox_pred,  # (Tensor) [B, M, 4]  还没乘以stride
                "anchor_point": anchor_point,  # (Tensor) [M, 2]
                "stride_tensor": stride_tensor}  # (Tensor) [M, 1]
            }
        """                        
        if not self.trainable:  # not
            return self.inference(video_clips, batch_target)
        else:
            batch_size, _, _, img_h, img_w = video_clips.shape
            # key frame
            key_frame = video_clips[:, :, -1, :, :]
            # 3D backbone
            feats_3d = self.backbone_3d(video_clips)  # 列表表示

            if self.decoupled_early:  # Free-YOLO-LARGE or YOLOv8L
                # 2D backbone
                feats_2d = self.backbone_2d(key_frame)
                if len(feats_2d) == 2:  # Free YOLO
                    cls_feats, reg_feats = self.backbone_2d(key_frame)
                else:  # YOLOv8
                    cls_feats = reg_feats = self.backbone_2d(key_frame)

                if self.level_3d == 1:
                    feat_3d_ups = [F.interpolate(feats_3d[0], scale_factor=2 ** (2 - level)) for level in range(self.level_2d)]
                    cls_feats = [self.cls_channel_encoders[level](cls_feats[level], feat_3d_ups[level])
                                 for level in range(self.level_2d)]
                    reg_feats = [self.reg_channel_encoders[level](reg_feats[level], feat_3d_ups[level])
                                 for level in range(self.level_2d)]
                else:
                    cls_feats = self.crossmerge(cls_feats, feats_3d)  # 列表，9个列表，每个都是2D，3D
                    reg_feats = self.crossmerge(reg_feats, feats_3d)  # 列表，9个列表，每个都是2D，3D
                    cls_feats = [self.cls_channel_encoders[level](cls_feats[level][0], cls_feats[level][1]) for level in
                             range(self.level_2d * self.level_3d)]
                    reg_feats = [self.reg_channel_encoders[level](reg_feats[level][0], reg_feats[level][1]) for level in
                             range(self.level_2d * self.level_3d)]
                    cls_feats = [
                        [cls_feats[idx_2d],
                         torch.cat([cls_feats[idx_2d + self.level_2d], cls_feats[idx_2d + self.level_2d * 2]], dim=1)]
                        for idx_2d in range(self.level_2d)]  # 同一个2D的不同tensor组合起来
                    reg_feats = [
                        [reg_feats[idx_2d],
                         torch.cat([reg_feats[idx_2d + self.level_2d], reg_feats[idx_2d + self.level_2d * 2]], dim=1)]
                        for idx_2d in range(self.level_2d)]  # 同一个2D的不同tensor组合起来

                    cls_feats = [self.cls_temporal_encoders[level](cls_feats[level][0], cls_feats[level][1])
                             for level in range(self.level_2d)]
                    reg_feats = [self.reg_temporal_encoders[level](reg_feats[level][0], reg_feats[level][1])
                                 for level in range(self.level_2d)]

                if self.noconf:
                    # pred-decoupled head
                    feats = [cls_feats, reg_feats]  # 嵌套列表，先解耦后等级
                else:
                    feats = [self.heads[level](cls_feats[level], reg_feats[level]) for level in range(self.level_2d)] # 嵌套列表，先等级后解耦

            else:  # YOLOv8l
                # 2D backbone
                feats_2d = self.backbone_2d(key_frame)
                if self.level_3d == 1:
                    feat_3d_ups = [F.interpolate(feats_3d[0], scale_factor=2 ** (2 - level)) for level in range(len(feats_2d))]
                    feats = [self.channel_encoders[level](feats_2d[level], feat_3d_ups[level]) for level in range(len(feats_2d))]
                else:
                    feats = self.crossmerge(feats_2d, feats_3d)  # 列表，9个列表，每个都是2D，3D
                    feats = [self.channel_encoders[level](feats[level][0], feats[level][1]) for level in
                             range(self.level_2d * self.level_3d)]
                    feats = [[feats[idx_2d],
                              torch.cat([feats[idx_2d + self.level_3d], feats[idx_2d + self.level_3d * 2]], dim=1)]
                             for idx_2d in range(self.level_2d)]  # 同一个2D的不同tensor组合起来
                    feats = [self.temporal_encoders[level](feats[level][0], feats[level][1])
                             for level in range(self.level_2d)]

                if not self.noconf:
                    feats = [self.heads[level](feats[level], feats[level]) for level in range(self.level_2d)]  # 嵌套列表,先等级后解耦

            if self.noconf:
                # 列表，含有层级数量项，每一项是一个tensor(由reg和cls在通道维度上拼接而成)
                preds = self.heads(feats, decoupled_in=self.decoupled_early)  # B,C,H,W
                dist_pred, score_pred = \
                    torch.cat([xi.view(batch_size, self.reg_max * 4 + self.num_classes, -1) for xi in preds], 2
                              ).permute(0, 2, 1).contiguous().split(
                        (self.reg_max * 4, self.num_classes), 2)  # 跨层级，在锚点框个数上拼接起来，然后再把回归和分类结果分开  B,M,C

                # decode
                anchor_point, stride_tensor = self.make_anchors(preds, self.stride)  # 所有层级拼接在一起   层级拼在一起的anchor
                bbox_pred = self.bbox_decode(anchor_point, dist_pred)  # B,M,4  xyxy两点式坐标(还没乘以stride)

                # output dict
                outputs = {"score_pred": score_pred,  # (Tensor) [B, M, Nc]
                           "bbox_pred": bbox_pred,  # (Tensor) [B, M, 4]  还没乘以stride
                           "dist_pred": dist_pred,
                           # (Tensor) [B, M, self.reg_max * 4]dist_pred是分布式表示的ltrb两点式(相对锚点、还没乘以stride)
                           "anchor_point": anchor_point,  # (Tensor) [M, 2]
                           "stride_tensor": stride_tensor}  # (Tensor) [M, 1]
                return outputs
            else:
                # pred
                conf_pred = [self.conf_preds[level](feats[level][1]) for level in range(self.level_2d)]  # B,C,H,W
                anchor_point, stride_tensor = self.make_anchors(conf_pred, self.stride)  # 所有层级拼接在一起   层级拼在一起的anchor
                # B,M,C
                conf_pred = torch.cat([_.flatten(2, 3) for _ in conf_pred], dim=-1
                                      ).permute(0, 2, 1).contiguous()
                cls_pred = torch.cat([self.cls_preds[level](feats[level][0]).flatten(2, 3)
                                      for level in range(self.level_2d)], dim=-1).permute(0, 2, 1).contiguous()
                reg_pred = torch.cat([self.reg_preds[level](feats[level][1]).flatten(2, 3)
                                      for level in range(self.level_2d)], dim=-1).permute(0, 2, 1).contiguous()

                # decode
                bbox_pred = self.bbox_decode(anchor_point, reg_pred, xywh=True)  # B,M,4  xyxy两点式坐标(还没乘以stride)

                # output dict
                outputs = {"conf_pred": conf_pred,  # (Tensor) [B, M, 1]
                           "cls_pred": cls_pred,  # (Tensor) [B, M, Nc]
                           "bbox_pred": bbox_pred,  # (Tensor) [B, M, 4]  还没乘以stride
                           "anchor_point": anchor_point,  # (Tensor) [M, 2]
                           "stride_tensor": stride_tensor}  # (Tensor) [M, 1]
                return outputs
