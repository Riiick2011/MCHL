import pickle
import numpy as np
import cv2
import os
from scipy.io import loadmat


# 本函数将检测框和对应的文本标注加到原始图片上并返回
def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)

    if label is not None:  # 如果有文本标注则在检测框附近再绘一个小框并在其中写上文本标注
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1 + 5), int(y1 + 15)), 0, text_scale, cls_color, 1, lineType=cv2.LINE_AA)
    return img


class_names = [
    'Basketball',     'BasketballDunk',    'Biking',            'CliffDiving',
    'CricketBowling', 'Diving',            'Fencing',           'FloorGymnastics',
    'GolfSwing',      'HorseRiding',       'IceDancing',        'LongJump',
    'PoleVault',      'RopeClimbing',      'SalsaSpin',         'SkateBoarding',
    'Skiing',         'Skijet',            'SoccerJuggling',    'Surfing',
    'TennisSwing',    'TrampolineJumping', 'VolleyballSpiking', 'WalkingWithDog']

video_det_dir = '/home/su/YOWOv3/results/ucf24/yowo_v3_large/exp24_untrim_untrim/'
linking_algorithms = ['mccla', 'viterbi', 'road', 'ojla_multilabel']
with open('/media/su/d/datasets/UCF24-YOWO/testlist_video.txt', 'r') as file:
    video_name_list = file.readlines()

# 读取真实标注
all_gts = {}
gt_data = loadmat('/media/su/d/datasets/UCF24-YOWO/splitfiles/finalAnnots.mat')['annot']  # ucf的管道标注存放在mat文件中
n_videos = gt_data.shape[1]
print('loading gt tubes ...')  # 读取真实管道标注
for i in range(n_videos):
    video_name = gt_data[0][i][1][0]
    if video_name+'\n' in video_name_list:
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
        all_gts[video_name] = v_annotation


for video_name in video_name_list:
    video_name = video_name.rstrip()
    # video_name = 'SalsaSpin/v_SalsaSpin_g01_c01'  #'Diving/v_Diving_g06_c06' #'SalsaSpin/v_SalsaSpin_g01_c02'  # 根据视频更换
    class_id = class_names.index(video_name.split('/')[0])

    # 只考虑这4个类别
    if class_id not in [4, 5, 14, 22]:
        continue

    # 真实管道绘制
    video_save_dir = os.path.join(video_det_dir, 'GT', video_name)  # 视频绘制后的保存路径
    os.makedirs(video_save_dir, exist_ok=True)
    video_root = '/media/su/d/datasets/UCF24-YOWO/rgb-images'
    video_dir = os.path.join(video_root, video_name)
    tubes = all_gts[video_name][class_id]
    tubes = sorted(tubes, key=len, reverse=True)[:6]
    tubes = sorted(tubes, key=lambda x: x[0][0])
    cls_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
    for tube_id, tube in enumerate(tubes):
        cls_color = cls_colors[tube_id % len(cls_colors)]
        for det in tube:
            img_id = det[0].astype(int)
            bbox = det[1:].astype(int)
            label = 'gttube' + str(tube_id + 1)
            img_dir = os.path.join(video_dir, '{:05d}.jpg'.format(img_id))
            img_save_dir = os.path.join(video_save_dir, '{:05d}.jpg'.format(img_id))
            img = cv2.imread(img_dir)
            if img is None:
                continue
            img_plotted = plot_bbox_labels(img, bbox, label=label, cls_color=cls_color)
            print(video_name, img_id, 'GT')
            cv2.imwrite(img_save_dir, img_plotted)
            # cv2.imshow('img_plotted', img_plotted)
            # cv2.waitKey()
        video_dir = video_save_dir

    # 关联算法管道绘制
    for linking_algorithm in linking_algorithms:
        video_det_pth = video_det_dir+'video_det_2_'+linking_algorithm+'.pkl'  # 关联结果的路径
        video_save_dir = os.path.join(video_det_dir, linking_algorithm, video_name)  # 视频绘制后的保存路径
        os.makedirs(video_save_dir, exist_ok=True)
        video_root = '/media/su/d/datasets/UCF24-YOWO/rgb-images'
        video_dir = os.path.join(video_root, video_name)
        video_det = pickle.load(open(video_det_pth, 'rb'))[class_id]
        tubes = [tube_tuple[2] for tube_tuple in video_det if video_name == tube_tuple[0]]
        tubes = sorted(tubes, key=len, reverse=True)[:6]
        tubes = sorted(tubes, key=lambda x: x[0][0])
        cls_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
        for tube_id, tube in enumerate(tubes):
            cls_color = cls_colors[tube_id]
            for det in tube:
                img_id = det[0].astype(int)
                bbox = det[1:-1].astype(int)
                score = np.around(det[-1], decimals=2)
                label = 'tube'+str(tube_id+1)
                img_dir = os.path.join(video_dir, '{:05d}.jpg'.format(img_id))
                img_save_dir = os.path.join(video_save_dir, '{:05d}.jpg'.format(img_id))
                img = cv2.imread(img_dir)
                if img is None:
                    continue
                img_plotted = plot_bbox_labels(img, bbox, label=label, cls_color=cls_color)
                print(video_name, img_id, linking_algorithm)
                cv2.imwrite(img_save_dir, img_plotted)
                # cv2.imshow('img_plotted', img_plotted)
                # cv2.waitKey()
            video_dir = video_save_dir





