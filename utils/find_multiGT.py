'''
用于查找哪些图片中具有多个真实检测框
只有UCF才会出现一张图片中有多个真实检测框的情况
UCF的每个视频中都只有一种类别，可能有多个actor/tube
JHMDB不会出现一张图片中有多个真实检测框的情况
JHMDB的每个视频中都只有一种类别，一个actor/tube
'''
import os
import numpy as np

video_testlist = []
with open('/media/su/d/datasets/UCF24-YOWO/testlist.txt', 'r') as file:  # 存放测试集图片的文件名
    lines = file.readlines()
    for line in lines:
        line = line.rstrip()
        video_testlist.append(line)

for line in video_testlist:
    f = open(os.path.join('/media/su/d/datasets/UCF24-YOWO/labels', line), 'r')
    labels = f.readlines()
    if len(labels) > 1:
        print(line)
