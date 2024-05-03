"""
本文件包含可视化所需函数
"""
import cv2
import numpy as np


# 本函数将真实检测框标注加到原始图片上，生成包含真实检测框的图片并返回
def vis_targets(video_clips, targets):
    """
        video_clips: (Tensor) -> [B, C, T, H, W]
        targets: List[Dict] -> [{'boxes': (Tensor) [N, 4],
                                 'labels': (Tensor) [N,]}, 
                                 ...],
    """
    batch_size = len(video_clips)

    for batch_index in range(batch_size):
        video_clip = video_clips[batch_index]
        target = targets[batch_index]

        key_frame = video_clip[:, :, -1, :, :]
        tgt_bboxes = target['boxes']
        tgt_labels = target['labels']

        key_frame = convert_tensor_to_cv2img(key_frame)
        width, height = key_frame.shape[:-1]

        for box, label in zip(tgt_bboxes, tgt_labels):
            x1, y1, x2, y2 = box
            label = int(label)

            x1 *= width
            y1 *= height
            x2 *= width
            y2 *= height

            # draw bbox
            cv2.rectangle(key_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.imshow('groundtruth', key_frame)
        cv2.waitKey(0)


# 本函数将图像tensor变换回图像格式(H,W,RGB)
def convert_tensor_to_cv2img(img_tensor):
    """ convert torch.Tensor to cv2 image """
    # to numpy 转换为（H,W,C）
    img_tensor = img_tensor.permute(1, 2, 0).cpu().numpy()
    # to cv2 img Mat
    cv2_img = img_tensor.astype(np.uint8)
    # to BGR
    cv2_img = cv2_img.copy()[..., (2, 1, 0)]

    return cv2_img


# 本函数将一个检测框和对应的文本标注加到原始图片上并返回
def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:  # 如果有文本标注则在检测框附近再绘一个小框并在其中写上文本标注
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


# 本函数将检测结果加到原始图片上，生成包含检测框的图片并返回
def vis_detection(frame, scores, labels, bboxes, vis_thresh, class_names, class_colors):
    ts = 0.4  # 图片上的文字标注的字号大小
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:  # 得分高于可视化阈值的检测框才进行显示
            label = int(labels[i])
            cls_color = class_colors[label]
                
            if len(class_names) > 1:  # 判断是否有文本标注
                mess = '%s: %.2f' % (class_names[label], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
                # visualize bbox 逐个检测框往原图上绘制
            frame = plot_bbox_labels(frame, bbox, mess, cls_color, text_scale=ts)

    return frame
        