"""
这里定义了多种2D IoU，用于训练中的损失计算 和 评估中的nms计算
"""

import numpy as np
import torch
import math


"""
# yolov8    weight是用来
if type(iou) is tuple:
    if len(iou) == 2:
        loss_iou = ((1.0 - iou[0]) * iou[1].detach() * weight).sum() / target_scores_sum   # 加了focal loss作为梯度增益的
    else:
        loss_iou = (iou[0] * iou[1] * weight).sum() / target_scores_sum  # WIoU
else:
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum   # 其他IoU

"""


class WIoU_Scale:
    """
    monotonous: {
            None: origin v1
            True: monotonic FM v2
            False: non-monotonic FM v3
        }
    momentum: The momentum of running mean
    """
    iou_mean = 1.
    monotonous = False
    _momentum = 1 - 0.5 ** (1 / 7000)
    _is_train = True

    def __init__(self, iou):  # 每次新建实例时等于传入最新的iou，并且更新iou均值
        self.iou = iou
        self._update(self)

    @classmethod
    def _update(cls, self):  # 更新iou均值
        if cls._is_train:
            cls.iou_mean = (1 - cls._momentum) * cls.iou_mean + \
                                         cls._momentum * self.iou.detach().mean().item()

    @classmethod
    def _scaled_loss(cls, self, gamma=1.9, delta=3):
        if isinstance(self.monotonous, bool):
            if self.monotonous:
                return (self.iou.detach() / self.iou_mean).sqrt()
            else:
                beta = self.iou.detach() / self.iou_mean
                alpha = delta * torch.pow(gamma, beta - delta)
                return beta / alpha
        return 1


def calculate_iou(box1, box2, xywh=False, iou_type='iou',
        Focal=0, alpha=1, gamma=0.5, scale=False, eps=1e-7):
    """
        Compute iou
        Args:
            box1 (tensor): pred values   shape(n,4)
            box2 (tensor): target values  shape(n,4)
            xywh (bool): True or False
            iou_type (str): 'iou', giou', 'diou', 'ciou', 'eiou', 'siou', 'riou', 'wiou, 'riou'
            Focal (int): 0 关闭Focal， 1 Focal困难样本(1-IoU).pow， 2 Focal高质量样本IoU.pow  开启则额外返回一项梯度增益
                         如果开启最好像FocalEIoU一样选2，避免困难样本带来的负面梯度
            alpha (int): AlphaIoU的参数 1则不开启，开启的话一般为3   梯度加速
            gamma (float): FocalLoss的参数  梯度增益计算中的指数
            scale (bool): True or False WIoU的参数 三种模式 默认为v3
            eps (float): 避免为0的极小数

        Returns:
            iou (1 tensor or tuple of 2 tensor): tensor shape(n,1)   当开启Focal时候，返回一个2元元组
    """

    # Returns Intersection over Union (IoU) of box1 to box2
    # box1(n,4）             box2(n,4)                => (n,1)
    # box1(1,4）or box1(4,)  box2(n,4)                => (n,1)
    # box1(n,4）             box2(1,4) or box2(4,)    => (n,1)
    # box1(n,1,4）           box2(1,m,4)              => (n,m,1)
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter

    # IoU
    iou = inter / (union + eps)
    if iou_type in ['giou', 'diou', 'ciou', 'eiou', 'siou', 'riou', 'wiou']:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width   最小外接矩形的宽度
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if iou_type == 'giou':  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch  # convex area 最小外接矩形的面积
            if Focal != 0:
                gradient_gain = (1-iou).pow(gamma) if Focal == 1 else iou.pow(gamma)
                return iou.pow(alpha) - ((c_area - union) / (c_area + eps)).pow(alpha), gradient_gain  # Focal_GIoU
            else:
                return iou.pow(alpha) - ((c_area - union) / (c_area + eps)).pow(alpha)

        elif iou_type in ['diou', 'ciou', 'eiou', 'wiou']:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2  # convex diagonal squared  最小外接矩形的对角线的平方
            # center dist ** 2  两个边界框中心的距离平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if iou_type == 'diou':
                if Focal != 0:
                    gradient_gain = (1 - iou).pow(gamma) if Focal == 1 else iou.pow(gamma)
                    return iou.pow(alpha) - (rho2 / (c2 + eps)).pow(alpha), gradient_gain  # Focal_DIoU
                else:
                    return iou.pow(alpha) - (rho2 / (c2 + eps)).pow(alpha)  # DIoU
            elif iou_type == 'ciou':  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)  # 表示宽高比的相似程度
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))  # 用来平衡形状损失所占比例的系数
                if Focal != 0:
                    gradient_gain = (1 - iou).pow(gamma) if Focal == 1 else iou.pow(gamma)
                    return iou.pow(alpha) - (rho2 / (c2 + eps)).pow(alpha) - (v * alpha_ciou).pow(alpha), gradient_gain  # Focal_CIoU
                return iou.pow(alpha) - (rho2 / (c2 + eps)).pow(alpha) - (v * alpha_ciou).pow(alpha)  # CIoU
            elif iou_type == 'eiou':
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2  # 宽度差的平方
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2  # 高度差的平方
                cw2 = cw ** 2   # 最小外接矩形的宽度
                ch2 = ch ** 2   # 最小外接矩形的高度
                if Focal != 0:   # 别的Focal都是强化低质量样本的损失，而EIoU刚好相反希望强化高质量样本的损失
                    gradient_gain = (1 - iou).pow(gamma) if Focal == 1 else iou.pow(gamma)
                    return iou.pow(alpha) - (rho2 / (c2 + eps)).pow(alpha) \
                        - (rho_w2 / (cw2 + eps)).pow(alpha) - (rho_h2 / (ch2 + eps)).pow(alpha), gradient_gain  # Focal_EIoU
                else:
                    return iou.pow(alpha) - (rho2 / (c2 + eps)).pow(alpha) \
                        - (rho_w2 / (cw2 + eps)).pow(alpha) - (rho_h2 / (ch2 + eps)).pow(alpha)  # EIoU
            elif iou_type == 'wiou':  # WIoU https://arxiv.org/abs/2301.10051  也是避免低质量样本带来的有害梯度增益
                if Focal:  # 自带了类似Focal的梯度增益设计
                    raise RuntimeError("WIoU do not support Focal.")
                elif scale:
                    return getattr(WIoU_Scale, '_scaled_loss')(WIoU_Scale(1 - iou)), (1 - iou) * torch.exp(
                        (rho2 / c2)), iou
                else:
                    return iou, torch.exp((rho2 / c2))  # WIoU v1
        elif iou_type in ['siou', 'riou']:  # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = torch.abs((b2_x1 + b2_x2 - b1_x1 - b1_x2) / 2)
            s_ch = torch.abs((b2_y1 + b2_y2 - b1_y1 - b1_y2) / 2)
            if iou_type == 'riou':
                rho_w = torch.abs((b2_x2 - b2_x1) - (b1_x2 - b1_x1))   # 宽度差
                rho_h = torch.abs((b2_y2 - b2_y1) - (b1_y2 - b1_y1))   # 高度差
                if Focal != 0:
                    gradient_gain = (1 - iou).pow(gamma) if Focal == 1 else iou.pow(gamma)
                    return iou.pow(alpha) - ((s_cw / cw + s_ch / ch) / 2).pow(alpha) - (rho_w / (cw + eps)).pow(
                        alpha) - (rho_h / (ch + eps)).pow(alpha), gradient_gain  # Focal_RIoU
                else:
                    return iou - (s_cw / (cw + eps) + s_ch / (ch + eps)) / 2 - rho_w / (cw + eps) - rho_h / (ch + eps)  # RIoU
            else:
                # 为了对比运算速度 可以先arcsin求角度再反过来求sin
                sin_alpha = torch.arctan(s_ch / (s_cw + eps)) if s_cw > s_ch \
                    else torch.arctan(s_cw / (s_ch + eps))  # 保证取0 ~ pi/4之间
                angle_cost = torch.cos((math.pi / 2 - sin_alpha * 2))
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma_siou = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma_siou * rho_x) - torch.exp(gamma_siou * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = (1 - torch.exp(-1 * omiga_w)).pow(4) + (1 - torch.exp(-1 * omiga_h)).pow(4)
                if Focal != 0:
                    gradient_gain = (1 - iou).pow(gamma) if Focal == 1 else iou.pow(gamma)
                    return iou.pow(alpha) - (
                                distance_cost.pow(alpha) + shape_cost.pow(alpha)) / 2, gradient_gain  # Focal_SIoU
                else:
                    return iou.pow(alpha) - (distance_cost.pow(alpha) + shape_cost.pow(alpha)) / 2  # SIoU
    else:
        if Focal != 0:
            gradient_gain = (1 - iou).pow(gamma) if Focal == 1 else iou.pow(gamma)
            return iou.pow(alpha), gradient_gain  # Focal_IoU
        else:
            return iou.pow(alpha)  # IoU


def rescale_bboxes(bboxes, orig_size):  # 将两点式归一化小数表示的预测框变换为原始图片尺寸下的两点式绝对坐标表示
    orig_w, orig_h = orig_size[0], orig_size[1]
    bboxes[..., [0, 2]] = np.clip(
        bboxes[..., [0, 2]] * orig_w, a_min=0., a_max=orig_w
        )
    bboxes[..., [1, 3]] = np.clip(
        bboxes[..., [1, 3]] * orig_h, a_min=0., a_max=orig_h
        )
    
    return bboxes


if __name__ == '__main__':
    box1 = torch.tensor([151, 170, 182, 197]).unsqueeze(0)
    box2 = torch.tensor([162, 171, 193, 197]).unsqueeze(0)
    box3 = torch.tensor([200, 180, 220, 200]).unsqueeze(0)
    print("=" * 20)
    print('box1:', box1.data)
    print('box2:', box2.data)
    print('box3:', box3.data)
    print("=" * 20)
    print(" IoU有交集：", calculate_iou(box1, box2))
    print("GIoU有交集：", calculate_iou(box1, box2, iou_type='giou'))
    print("DIoU有交集：", calculate_iou(box1, box2, iou_type='diou'))
    print("CIoU有交集：", calculate_iou(box1, box2, iou_type='ciou'))
    print("EIoU有交集：", calculate_iou(box1, box2, iou_type='eiou'))
    print("SIoU有交集：", calculate_iou(box1, box2, iou_type='siou'))
    print("RIoU有交集：", calculate_iou(box1, box2, iou_type='riou'))
    print("WIoU有交集：", calculate_iou(box1, box2, iou_type='wiou', scale=True))
    print("=" * 20)
    print(" IoU无交集：", calculate_iou(box1, box3))
    print("GIoU无交集：", calculate_iou(box1, box3, iou_type='giou'))
    print("DIoU无交集：", calculate_iou(box1, box3, iou_type='diou'))
    print("CIoU无交集：", calculate_iou(box1, box3, iou_type='ciou'))
    print("EIoU无交集：", calculate_iou(box1, box3, iou_type='eiou'))
    print("SIoU无交集：", calculate_iou(box1, box3, iou_type='siou'))
    print("RIoU无交集：", calculate_iou(box1, box3, iou_type='riou'))
    print("WIoU无交集：", calculate_iou(box1, box3, iou_type='wiou', scale=True))

