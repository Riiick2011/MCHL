import numpy as np
import time
import scipy
import random
import numpy as np
import time
import scipy.sparse.csgraph
import random
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
from utils.utils import bbox_iou
import cv2
from evaluator import kalman_filter, nn_matching


# viterbi算法中相邻帧之间的关联得分基于iou、类别得分之和、类别得分之积   输入是数组
def compute_score_one_class(bbox1, bbox2, w_iou=1.0, w_scores=1.0, w_scores_mul=0.5):
    # bbx: <x1> <y1> <x2> <y2> <class score>
    n_bbox1 = bbox1.shape[0]
    n_bbox2 = bbox2.shape[0]
    # for saving all possible scores between each two bbxes in successive frames
    scores = np.zeros([n_bbox1, n_bbox2], dtype=np.float32)
    for i in range(n_bbox1):
        box1 = bbox1[i, :4]
        for j in range(n_bbox2):
            box2 = bbox2[j, :4]
            bbox_iou_frames = bbox_iou(box1, box2, x1y1x2y2=True)
            sum_score_frames = bbox1[i, 4] + bbox2[j, 4]
            mul_score_frames = bbox1[i, 4] * bbox2[j, 4]
            scores[i, j] = w_iou * bbox_iou_frames + w_scores * sum_score_frames + w_scores_mul * mul_score_frames
    return scores


# ROAD算法的Tube类别
class ROADTube(object):  # 通过读取txt文件转换为的检测管道对象
    def __init__(self, frame_id, det, tube_index, tube_cls=-1):
        # det是对应一个txt的字典，det_index是检测索引，tube_index是管道唯一(同一视频同一类别下)的索引号
        self.start_frame = frame_id  # 管道起始帧的序号
        self.end_frame = self.start_frame  # 管道结束帧的序号
        self.frame_range = self.end_frame - self.start_frame + 1  # 管道跨度
        self.det_num = 1  # 管道中包含的检测数(帧级检测的一个检测框是一个检测)
        self.det_list = [det]  # 用列表表示,每一项是一个形状为(5,)数组  bboxs,score
        self.det_list_interpolated = []  # 存放内插外推后的检测框，是一个数组
        self.frame_ids = [frame_id]  # 帧序号，一个列表
        self.miss_link_times = 0  # 连续漏检的次数，超过多次就判定管道结束
        self.tube_score_sum = det[-1]
        self.tube_score = self.tube_score_sum / self.det_num
        self.active = True
        self.index = tube_index  # 该管道在管道列表中的序号
        self.edge_scores = [det[-1]]
        self.tube_cls = tube_cls

    def __call__(self, frame_id, det, tube_cls=-1):  # 关联新的检测, txt为字典，det_index为字典上该检测的索引
        self.det_list.append(det)
        self.frame_ids.append(frame_id)  # 帧序号

        self.end_frame = frame_id  # 管道结束帧的序号
        self.frame_range = self.end_frame - self.start_frame + 1  # 管道跨度
        self.det_num += 1
        self.tube_score_sum = self.tube_score_sum + det[-1]
        self.tube_score = self.tube_score_sum / self.det_num  # 更新管道得分
        self.miss_link_times = 0
        self.edge_scores.append(det[-1])
        self.tube_cls = tube_cls

    def miss_link(self, frame_id):  # 输入当前帧序号(列表)
        self.miss_link_times += 1
        if self.miss_link_times == 5:  # 连续漏检5次就判定管道结束,应该根据采样率不同而进行调节
            self.active = False

    def interpolate(self):  # 当一个视频关于某一类别的管道关联结束时，进行内插外推
        self.det_list_interpolated.append(self.det_list[0])
        # 同一视频内不同txt之间的帧间隔是不同的，需要根据frame_id的差值计算
        gap_list = np.array(self.frame_ids[1:]) - np.array(self.frame_ids[:-1])
        for i, gap in enumerate(gap_list):  # 循环原总帧数-1次
            for j in range(gap):
                det_box_interpolated = self.det_list[i] + (j + 1) / gap * (self.det_list[i + 1] - self.det_list[i])
                self.det_list_interpolated.append(det_box_interpolated)
        self.det_list_interpolated = np.array(self.det_list_interpolated)
        if self.frame_range != len(self.det_list_interpolated):  # 如果内插失败则打印出错误
            print('interpolate failure')
        self.det_list_interpolated = np.concatenate(
            [np.array(np.arange(self.start_frame, self.end_frame + 1)).reshape(-1, 1),
             self.det_list_interpolated], axis=-1)


# 对untrimmed数据集比较有效，既是一种分割机制也是一种过滤机制，但是只能离线使用
def ROAD_temporal_trim(tube):
    alpha = 3
    scores = np.zeros((2, tube.det_num + 1))  # 动态规划前向过程的得分矩阵 行数为2代表 背景类和该动作类
    scores[:, 0] = 0
    scores[:, 1:] = np.array([1 - np.array(tube.edge_scores), tube.edge_scores])
    v = np.array([0, 1])
    index = np.zeros((2, tube.det_num)).astype(np.int)  # 存放前向过程中每一帧的最大索引，反向过程中，通过此找到最佳路径
    for j in range(1, tube.det_num + 1):  # 在每一次检测中 判断取背景类得分高 还是取该动作类得分高
        for i in range(2):  # 对于每个类别 找寻最优路径  变号有惩罚
            dmax = np.max(scores[:, j - 1] - alpha * (v != i))
            tb = np.argmax(scores[:, j - 1] - alpha * (v != i))
            scores[i, j] = scores[i, j] + dmax  # 记录第j-1帧的第i个类别与第j-2帧不同类别关联的最高得分
            index[i, j - 1] = tb  # 记录第j-1帧第i个类别与第j-2帧哪个类别关联的得分最高    第-1帧代表初始值

    # 从最后一帧开始倒推
    scores = scores[:, 1:]  # 变为跟记录序号选择的矩阵phi形状一致的矩阵，列数为检测数目
    # phi和D记录了不同类别的最佳路径

    # 反向过程
    det_id_list = [tube.det_num - 1]
    best_cls_list = [np.argmax(scores[:, tube.det_num - 1])]  # 一个列表，记录每一帧上的最佳类别，初始只有最后一帧检测上的最佳类别
    best_cls = best_cls_list[0]  # 最后一帧检测上的最佳类别

    for j in range(tube.det_num - 1, 0, -1):  # 反向过程，一直到第一帧
        best_cls = index[best_cls, j]
        best_cls_list.insert(0, best_cls)  # 将该帧的最佳类别从前面放到列表中
        det_id_list.insert(0, j - 1)  # 共计det_num项，从0到det_num-1

    fore_idx_list = np.where(np.array(best_cls_list) == 1)[0]  # 整个最优路径中取类别c的帧序号
    tube_trimmed_list = []
    if len(fore_idx_list) > 1:
        gap = np.where((np.concatenate([fore_idx_list, fore_idx_list[-1].reshape(1,)+1]) -
                        np.concatenate([fore_idx_list[0].reshape(1,)-2, fore_idx_list])) > 1)[0]  # 相邻大于1就断开了
        if len(gap) > 1:  # 有多段
            ts = gap
            te = np.concatenate([np.array(gap[1:]-1), np.array(len(fore_idx_list)-1).reshape(1,)])
            ts = np.array(det_id_list)[fore_idx_list[ts]].tolist()
            te = np.array(det_id_list)[fore_idx_list[te]].tolist()
        else:  # 只有1段  可能是原长
            ts = np.array(det_id_list)[fore_idx_list[gap]].tolist()
            te = np.array(det_id_list)[fore_idx_list[[-1]]].tolist()
        # 根据ts和te两个列表，将tube分割成多个tube
        for s, e in zip(ts, te):
            tube_trimmed = deepcopy(tube)
            tube_trimmed.frame_ids = tube_trimmed.frame_ids[s:e+1]
            tube_trimmed.det_num = e-s+1
            tube_trimmed.start_frame = tube_trimmed.frame_ids[0]
            tube_trimmed.end_frame = tube_trimmed.frame_ids[-1]
            tube_trimmed.frame_range = tube_trimmed.end_frame - tube_trimmed.start_frame + 1
            tube_trimmed.det_list = tube_trimmed.det_list[s:e+1]
            tube_trimmed.tube_score = np.sum(np.array(tube_trimmed.det_list)[:, -1])/tube_trimmed.det_num
            tube_trimmed_list.append(tube_trimmed)
    return tube_trimmed_list


class Mass(object):
    def __init__(self):
        self.Meas_edge = np.array([[-1], [1]])  # 序号取-1代表漏检 取其他代表检测框的序号
        self.Hypo = None
        self.Prob = None
        self.Cost = np.array([[10]])
        self.tube_cls = [-1]


def dist_center(box1, box2):
    x1 = (box1[0] + box2[2]) / 2
    y1 = (box1[1] + box2[3]) / 2
    x2 = (box2[0] + box2[2]) / 2
    y2 = (box2[1] + box2[3]) / 2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


class OJLATube(object):
    def __init__(self, frame_id, det, score, tube_index):
        """
        :param frame_id: int
        :param det: array(4,)
        :param score: array(num_classes,)
        :param tube_index: int
        """
        self.start_frame = frame_id  # 管道起始帧的序号
        self.end_frame = self.start_frame  # 管道结束帧的序号
        self.frame_range = self.end_frame - self.start_frame + 1  # 管道跨度
        self.det_num = 1  # 管道中包含的检测数(帧级检测的一个检测框是一个检测)
        self.det_list = [det]  # 用列表表示,每一项是一个形状为(4,)数组，对应一个检测框的维度，不包含背景检测框
        self.score_list = [score]  # 用列表表示，每一项对应一个检测框的全类别得分，不包含背景检测框
        self.det_list_interpolated = []  # 存放内插外推后的检测框，是一个数组
        self.frame_ids = [frame_id]  # 帧序号，一个列表
        self.miss_link_times = 0  # 连续漏检的次数，超过多次就判定管道结束
        self.tube_score_sum = np.max(score)  # 用第一个检测框的最大类别得分作为管道初始得分
        self.tube_score = self.tube_score_sum / self.det_num
        self.active = True
        self.index = tube_index  # 该管道在管道列表中的序号
        self.tube_cls = np.argmax(score)  # 用检测框的最大得分类别作为管道的初始类别

    def __call__(self, frame_id, det, score, link_prob, tube_cls):  # 关联新的检测
        self.det_list.append(det)
        self.score_list.append(score)
        self.frame_ids.append(frame_id)  # 帧序号
        self.end_frame = frame_id  # 管道结束帧的序号
        self.frame_range = self.end_frame - self.start_frame + 1  # 管道跨度
        self.det_num += 1
        self.tube_score_sum = self.tube_score_sum + link_prob  # 增加修正过的关联得分
        self.tube_score = self.tube_score_sum / self.det_num  # 更新管道得分
        self.tube_cls = tube_cls  # 更新管道类别
        self.miss_link_times = 0

    def miss_link(self, frame_id):  # 输入当前帧序号(列表)
        self.miss_link_times += 1
        if self.miss_link_times == 5:  # 连续漏检5次就判定管道结束,应该根据采样率不同而进行调节
            self.active = False

    def interpolate(self):  # 当一个视频关于某一类别的管道关联结束时，进行内插外推
        for i in range(len(self.det_list)):  # det(4,) ->det(5,)
            self.det_list[i] = np.concatenate([self.det_list[i], [self.score_list[i][self.tube_cls]]])
        self.det_list_interpolated.append(self.det_list[0])
        # 同一视频内不同txt之间的帧间隔是不同的，需要根据frame_id的差值计算
        gap_list = np.array(self.frame_ids[1:]) - np.array(self.frame_ids[:-1])
        for i, gap in enumerate(gap_list):  # 循环原总帧数-1次
            for j in range(gap):
                det_box_interpolated = self.det_list[i] + (j + 1) / gap * (self.det_list[i + 1] - self.det_list[i])
                self.det_list_interpolated.append(det_box_interpolated)
        self.det_list_interpolated = np.array(self.det_list_interpolated)
        if self.frame_range != len(self.det_list_interpolated):  # 如果内插失败则打印出错误
            print('interpolate failure')
        self.det_list_interpolated = np.concatenate(
            [np.array(np.arange(self.start_frame, self.end_frame + 1)).reshape(-1, 1),
        self.det_list_interpolated], axis=-1)  # det(5,) ->det(6,)


def JPDA_Probabilty_Calculator(M):  # 用条件概率的方式分析所有可行假设，并更新每个管道取每个检测框的概率
    # M是一个该连通图的列表，其中每一项是一个Mass类的实例对应一个管道在当前时刻的关联可能性
    N_T = len(M)  # 该连通图中的管道数量
    F_Pr = []  # 存放最终概率，包含管道数目项，每一项是一个数组，数组形状为（该管道可能关联的检测框数目（包含漏检），1）
    PT = np.zeros((1, N_T), dtype=float)
    msp = M[0].Meas_edge[1, -1]  # 扫描的次数  在这里一个时刻只扫描一次
    Hypo_indx = np.asarray([len(item.Prob) - 1 for item in M])  # 数组，形状为(管道数量，)  每个管道的假设数量 从0开始计数
    for i in range(N_T):  # 对于每个管道
        ind0 = [item for item in range(len(M[i].Meas_edge[1])) if M[i].Meas_edge[1][item] == 1]  # 可能关联的检测框 本地序号 包括漏检
        F_Pr.append(np.zeros((len(ind0), 1), dtype=float))  # 用来存放每个管道关于可能的检测框之间的关联概率 包含漏检  每一项是一个数组对应一个管道

    if N_T == 1:
        P_T = np.prod(M[0].Prob, axis=1)
        for kk in range(len(ind0)):  # 检测框序号的本地索引
            temp = [P_T[item] for item in range(len(M[0].Hypo[:, 0])) if
                    M[0].Hypo[item, 0] == M[-1].Meas_edge[0, ind0[kk]]]
            F_Pr[-1][ind0[kk], 0] = sum(temp)
    else:
        a = np.zeros((1, N_T), dtype=int)    # 为了穷举 a表示假设情况
        a[0, -1] = -1
        temp = np.zeros((1, N_T), dtype=int)  # 每次穷举，都是最后一个管道的假设序号+1
        temp[0, -1] = 1
        t = 0

        while max(np.abs(a[0] - Hypo_indx)) > 1e-3 and (t < 1000):
            a[0] += temp[0]
            hypothesis = np.zeros((msp, N_T))
            for j in range(N_T - 1, -1, -1):  # 循环一遍 进位设计 每个管道的假设序号不得超过该管道具有的假设总数
                if a[0][j] > Hypo_indx[j]:
                    a[0][j] = 0
                    a[0][j - 1] += 1
                PT[0, j] = np.prod(M[j].Prob[a[0][j], :])  # 用于记录该假设下，各个管道取对应检测的概率
                hypothesis[:, j] = M[j].Hypo[a[0][j], :].T  # 该种假设下，第j个管道所取的检测全局序号  检测框全局序号-1代表漏检

            chkk = 0  # 用于判断该假设是否合理， 如果合理，则每次扫描都应该合理
            # 合理：在一次扫描中 一个有效检测框不能同时分配给多个管道，一个管道不能获得超过一个检测框（包括漏检）
            for jj in range(msp):  # 每次扫描
                zhpo = [item for item in range(len(hypothesis[jj])) if hypothesis[jj, item] == -1]  # 找到本次假设中漏检的管道的本地序号
                if ((zhpo == []) and (len(np.unique(hypothesis[jj])) == N_T)) or (
                    len(np.unique(hypothesis[jj])) == N_T - len(zhpo) + 1):
                    # 当所有管道均没有漏检时，各自应分配到一个互不重复的检测框时
                    # 当存在漏检时，不漏检的哪些管道各自应分配到一个互不重复的检测框时
                    chkk += 1
                else:
                    break
            if chkk == msp:  # 如果本次假设合理
                for i in range(N_T):
                    # 找到本假设下，该管道对应的检测的本地序号
                    indd = [item for item in range(len(M[i].Meas_edge[1]))
                            if M[i].Meas_edge[1, item] == 1 and M[i].Meas_edge[0, item] == M[i].Hypo[a[0][i], 0]]
                    for itemTemp in indd:
                        F_Pr[i][itemTemp][0] += np.prod(PT)  # 每存在一个假设中该管道取该检测框，就增加一次该管道取该检测框的概率
            t += 1

    for item in range(len(F_Pr)):
        F_Pr[item] = F_Pr[item] / sum(F_Pr[item])  # 将该管道能取得的所有检测框的概率归一化
    return F_Pr   # 存放最终概率，包含管道数目项，每一项是一个数组，数组形状为（该管道可能关联的检测框数目（包含漏检），1）


def ApproxMultiscanJPDAProbabilities(M, obj_info):
    U = len(obj_info)  # 管道数量
    Final_probabilty = [None for _ in range(U)]

    # n_components是连通图数量，labels是个数组表示每个节点属于的连通图序号
    n_components, labels = scipy.sparse.csgraph.connected_components(M, False, 'strong')  # 构建无向图  'strong'无所谓
    C2 = labels[:U]  # 每个管道节点属于哪一个连通图

    for i in np.unique(C2):  # 对于每张连通图计算互联概率    一个节点只可能属于一个连通图  连通图之间的计算互相独立
        ix = np.where(C2 == i)[0].tolist()  # 数组，用于指示属于该连通图的管道的序号
        tempInput = [obj_info[_] for _ in ix]
        temp = JPDA_Probabilty_Calculator(tempInput)
        for j in ix:
            Final_probabilty[j] = temp[ix.index(j)]
    return Final_probabilty


# MCHL算法的帧级检测类
class MCHLDet(object):
    def __init__(self, frame_id, det, conf, score, ori_img=None):
        if ori_img is not None:  # 如果使用原始图片，则det是一个1维数组
            self.bbox = det  # x1y1x2y2
            self.conf = conf
            self.score = score
            self.frame_id = frame_id
            x1, y1, x2, y2 = self.bbox.astype(int)
            self.xyah = self.to_xyah()
            self.feat = cv2.resize(ori_img[x1:x2+1, y1:y2+1, :], (14, 14)).reshape(1, -1)  # （1，588）
            self.valid = True
            if np.sum(self.feat) == 0:  # 全是0的特征图无法进行对角化
                self.valid = False
        else:  # 如果不传入原始图片，则说明det内包含了检测模型中提取的特征
            self.bbox = det  # x1y1x2y2
            self.score = score
            self.frame_id = frame_id
            x1, y1, x2, y2 = self.bbox.astype(int)
            self.xyah = self.to_xyah()
            self.valid = True

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.bbox.copy()
        xc = (self.bbox[0] + self.bbox[2]) / 2
        yc = (self.bbox[1] + self.bbox[3]) / 2
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        ret = [xc, yc, width/height, height]
        return ret


class MCHLTube(object):
    def __init__(self, frame_id, det, tube_index):
        """
        :param frame_id: int
        :param det: array(4,)
        :param score: array(num_classes,)
        :param tube_index: int
        """
        self.start_frame = frame_id  # 管道起始帧的序号
        self.end_frame = self.start_frame  # 管道结束帧的序号
        self.frame_range = self.end_frame - self.start_frame + 1  # 管道跨度
        self.det_num = 1  # 管道中包含的检测数(帧级检测的一个检测框是一个检测)
        self.det_list = [det.bbox]  # 用列表表示,每一项是一个形状为(4,)数组，对应一个检测框的维度，不包含背景检测框
        self.conf_list = [det.conf]
        self.score_list = [det.score]  # 用列表表示，每一项对应一个检测框的全类别得分，不包含背景检测框
        self.det_list_interpolated = []  # 存放内插外推后的检测框，是一个数组
        self.frame_ids = [frame_id]  # 帧序号，一个列表
        self.miss_link_times = 0  # 连续漏检的次数，超过多次就判定管道结束
        self.tube_score_sum = np.max(det.score)  # 用第一个检测框的最大类别得分作为管道初始得分
        self.tube_score = self.tube_score_sum / self.det_num
        self.index = tube_index  # 该管道在管道列表中的序号
        self.tube_cls = np.argmax(det.score)  # 用检测框的最大得分类别作为管道的初始类别

        # 外观特征采用EMA更新
        self.ema_alpha = 0.9
        self.feat = det.feat / np.linalg.norm(det.feat)

        # 卡尔曼预测
        self.kf = kalman_filter.KalmanFilter()
        self.mean, self.covariance = self.kf.initiate(det.xyah)

        self.active = 2  # 创建时处于待定状态

    def __call__(self, frame_id, det, link_prob, tube_cls):  # 关联新的检测
        self.det_list.append(det.bbox)
        self.conf_list.append(det.conf)
        self.score_list.append(det.score)
        self.frame_ids.append(frame_id)  # 帧序号
        self.end_frame = frame_id  # 管道结束帧的序号
        self.frame_range = self.end_frame - self.start_frame + 1  # 管道跨度
        self.det_num += 1
        self.tube_score_sum = self.tube_score_sum + link_prob  # 增加修正过的关联得分
        self.tube_score = self.tube_score_sum / self.det_num  # 更新管道得分
        self.tube_cls = tube_cls  # 更新管道类别
        self.miss_link_times = 0
        # 更新外观特征
        self.feat = (self.feat * self.ema_alpha + det.feat * (1-self.ema_alpha))/np.linalg.norm(self.feat)
        # 用空间信息的测量值更新卡尔曼滤波器
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, det.xyah, det.conf)

        if self.det_num >= 3:  # 当管道长度超过2时，只要进行一次额外关联，就处于活跃状态或者重新回到激活状态
            self.active = 1

    def miss_link(self):  # 输入当前帧序号(列表)  # 增加一个临界判定超过3帧
        self.miss_link_times += 1
        self.active = 2  # 出现漏检就回到待定状态
        if self.miss_link_times == 5:  # 连续漏检5次就判定管道结束,应该根据采样率不同而进行调节
            self.active = 0

    def interpolate(self):  # 当一个视频关于某一类别的管道关联结束时，进行内插外推
        for i in range(len(self.det_list)):  # det(4,) ->det(5,)
            self.det_list[i] = np.concatenate([self.det_list[i], [self.score_list[i][self.tube_cls]]])
        self.det_list_interpolated.append(self.det_list[0])
        # 同一视频内不同txt之间的帧间隔是不同的，需要根据frame_id的差值计算
        gap_list = np.array(self.frame_ids[1:]) - np.array(self.frame_ids[:-1])
        for i, gap in enumerate(gap_list):  # 循环原总帧数-1次
            for j in range(gap):
                det_box_interpolated = self.det_list[i] + (j + 1) / gap * (self.det_list[i + 1] - self.det_list[i])
                self.det_list_interpolated.append(det_box_interpolated)
        self.det_list_interpolated = np.array(self.det_list_interpolated)
        if self.frame_range != len(self.det_list_interpolated):  # 如果内插失败则打印出错误
            print('interpolate failure')
        self.det_list_interpolated = np.concatenate(
            [np.array(np.arange(self.start_frame, self.end_frame + 1)).reshape(-1, 1),
             self.det_list_interpolated], axis=-1)  # det(5,) ->det(6,)


# 用于MCHL中展示gt
def compare_with_gt():
    videolist = []
    with open('/media/su/d/datasets/UCF24-YOWO/testlist_video.txt', 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.rstrip()
            videolist.append(line)
    videolist.sort()
    from scipy.io import loadmat
    gt_data = loadmat('/media/su/d/datasets/UCF24-YOWO/splitfiles/finalAnnots.mat')['annot']
    n_videos = gt_data.shape[1]
    print('loading gt tubes ...')  # 读取真实管道标注
    tube_length_min = []
    for i in range(n_videos):
        video_name = gt_data[0][i][1][0]
        if video_name == os.path.join(img_folder.split('/')[-2], img_folder.split('/')[-1]):
            n_tubes = len(gt_data[0][i][2][0])
            v_annotation = {}
            all_gt_boxes = []
            for j in range(n_tubes):  # 利用了同一个视频中只有一种类别的管道的先验知识
                gt_one_tube = []
                tube_start_frame = gt_data[0][i][2][0][j][1][0][0]
                tube_end_frame = gt_data[0][i][2][0][j][0][0][0]
                tube_class = gt_data[0][i][2][0][j][2][0][0] - 1  # 类别改为从0开始计数
                tube_data = gt_data[0][i][2][0][j][3]
                tube_length = tube_end_frame - tube_start_frame + 1
                if tube_length == 4:
                    print(video_name)
                for k in range(tube_length):  # 每一个管道
                    # gt_boxes是一个列表，包含5项，对应帧序号、一帧上的一个检测框的点模式坐标
                    gt_boxes = [int(tube_start_frame + k),
                                float(tube_data[k][0]),
                                float(tube_data[k][1]),
                                float(tube_data[k][0]) + float(tube_data[k][2]),
                                float(tube_data[k][1]) + float(tube_data[k][3])]
                    gt_one_tube.append(gt_boxes)
                all_gt_boxes.append(np.array(gt_one_tube))  # 包含管道个数项，每一项是一个数组


def cascade_MCHL(tube_list, dets_frame, frame_id, use_score_dist=True, tube_active=True):
    tube_num = len(tube_list)
    det_num = len(dets_frame)
    mass_list = []  # 存放当前帧的管道关联情况 每一项是一个Mass类对象，对应一个管道的字典
    det_in_gate_indexes = []  # 用于存放与管道过近的检测框序号
    det_linked_indexes = []  # 用于记录关联过的检测框序号
    Mes_Tar = np.zeros((det_num, tube_num)).astype(np.bool)  # 确认矩阵， 记录管道和检测的可行关联情况
    for tube_index in range(tube_num):
        mass = Mass()
        tube = tube_list[tube_index]
        last_det = tube.det_list[-1]
        # last_det = np.mean(tube.det_list[-7:], axis=0)
        gate = (last_det[2] - last_det[0]) / 2  # 因为人类是竖长的 宽度要远远小于高度
        for det_index in range(det_num):
            det = dets_frame[det_index]
            motion_cost = tube.kf.gating_distance(tube.mean, tube.covariance, det.xyah)  # 马氏距离的平方
            center_dist = dist_center(last_det, det.bbox)
            if center_dist <= gate or motion_cost <= 10:  # 检测框邻域映射
                tube_cls_score = np.sum(tube.score_list[-3:], axis=0) + det.score
                tube_cls = np.argmax(tube_cls_score)  # 如果该管道与该检测框进行关联，则管道的类别应该调整为该类别
                overlap_score = bbox_iou(tube.det_list[-1], det.bbox)
                appear_cost = nn_matching._nn_cosine_distance(tube.feat, det.feat)[0]  # 余弦距离
                appear_score = 1 - 1 / (1 + np.exp(-appear_cost))
                conf_score = np.mean(tube.conf_list)
                if use_score_dist:  # 如果使用类别得分向量的欧氏距离计算关联成本中的类别得分部分
                    score_dist = np.clip(np.linalg.norm(np.mean(tube.score_list[-3:], axis=0) - det.score), 0, 1)  # 1-
                    label_score = 1 - score_dist
                    label_score_weight = 0.6 if not tube_active else 1
                    label_appear_weight = 0.125
                    cost = 1 / np.clip((label_score_weight * label_score + overlap_score +
                                        label_appear_weight * appear_score) * conf_score *
                                       min([(tube.det_num + 1) / 2, 2]), 1e-7, 10)  # 该管道与该检测框关联的cost
                else:  # 原版exp0
                    label_score = tube_cls_score
                    cost = 1 / (label_score[tube_cls] + overlap_score * min([(tube.det_num + 1) / 2, 2]))
                mass.Cost = np.concatenate([mass.Cost, np.array(cost).reshape(1, 1)], axis=0)
                mass.Meas_edge = np.concatenate([mass.Meas_edge, np.array([det_index, 1]).reshape(2, 1)], axis=1)
                mass.tube_cls.append(tube_cls)
                Mes_Tar[det_index, tube_index] = True  # 记录节点情况
                if center_dist <= gate / 2:
                    det_in_gate_indexes.append(det_index)
        mass.Hypo = mass.Meas_edge[0].transpose().reshape(-1, 1)  # 一定包含序号-1代表漏检
        mass.Prob = np.exp(-mass.Cost)  # 包含了漏检的概率
        mass_list.append(mass)
    Mes_Tar2 = np.zeros((tube_num + det_num, tube_num + det_num)).astype(np.bool)
    Mes_Tar2[tube_num:, :tube_num] = Mes_Tar
    # 返回一个列表，每一项是一个数组对应一个mass管道，数组尺寸是（该管道可能的检测框数量（包含漏检），1）
    final_prob_matrix = ApproxMultiscanJPDAProbabilities(Mes_Tar2, mass_list)

    # 更新管道
    for tube_index in range(tube_num):  # 局部索引  对于每个管道，找到概率最高的检测框序号，检测框全局序号-1代表漏检
        link_prob = np.max(final_prob_matrix[tube_index], axis=0)[0]  # float
        index = np.argmax(final_prob_matrix[tube_index], axis=0)[0]  # int 检测框的本地索引  本地指的是该管道门限之内
        det_index = mass_list[tube_index].Hypo[index][0]  # 全局索引
        tube_cls = mass_list[tube_index].tube_cls[index]
        if det_index == -1:  # 漏检，增加漏检计数
            tube_list[tube_index].miss_link()
        else:
            det_linked_indexes.append(det_index)
            tube_list[tube_index](frame_id, dets_frame[det_index], link_prob, tube_cls)

    det_excluded_indexes = det_in_gate_indexes + det_linked_indexes
    det_left_indexes = np.array([i for i in range(det_num) if i not in np.unique(det_excluded_indexes)])
    dets_frame = [dets_frame[_] for _ in det_left_indexes]
    return dets_frame
