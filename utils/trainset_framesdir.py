"""
用于保存untrimmed的ucf24数据集和multisports数据集中，训练集的所有图片 包括unlabelled
"""
import os
import pickle


def save_trainset_all_framesdir(dataset):
    if dataset == 'ucf24':
        video_trainlist = []
        with open('/media/su/d/datasets/UCF24-YOWO/splitfiles/testlist01.txt', 'r') as file:  # 存放训练集的视频文件名
            lines = file.readlines()
            for line in lines:
                line = line.rstrip()
                video_trainlist.append(line)

        with open('/media/su/d/datasets/UCF24-YOWO/testlist_untrimmed.txt', 'w') as file:  # 存放训练集的视频文件名
            for line in video_trainlist:
                f = sorted(os.listdir(os.path.join('/media/su/d/datasets/UCF24-YOWO/rgb-images', line)))
                img_dirs = [os.path.join(line, img_id) + '\n' for img_id in f]
                file.writelines(img_dirs)
                file.flush()
    elif dataset == 'jhmdb21':
        video_trainlist = []
        with open('/media/su/d/datasets/JHMDB-YOWO/testlist_video.txt', 'r') as file:  # 存放训练集的视频文件名
            lines = file.readlines()
            for line in lines:
                line = line.rstrip()
                video_trainlist.append(line)

        with open('/media/su/d/datasets/JHMDB-YOWO/testlist_untrimmed.txt', 'w') as file:  # 存放训练集的视频文件名
            for line in video_trainlist:
                f = sorted(os.listdir(os.path.join('/media/su/d/datasets/JHMDB-YOWO/rgb-images', line)))
                img_dirs = [os.path.join(line, img_id) + '\n' for img_id in f]
                file.writelines(img_dirs)
                file.flush()
    elif dataset == 'multisports':
        GT_path = os.path.join('/media/su/d/datasets/MultiSports/trainval/multisports_GT.pkl')
        GT = pickle.load(open(GT_path, 'rb'))
        video_trainlist = GT['train_videos'][0]  # 训练集视频名称

        with open('/media/su/d/datasets/MultiSports/trainval/trainlist_untrimmed.txt', 'w') as file:  # 存放训练集的视频文件名
            for line in video_trainlist:
                f = sorted(os.listdir(os.path.join('/media/su/d/datasets/MultiSports/trainval/rawframes', line)))
                img_dirs = [os.path.join(line, img_id) + '\n' for img_id in f]
                file.writelines(img_dirs)
                file.flush()


if __name__ == '__main__':
    dataset = 'jhmdb21'
    save_trainset_all_framesdir(dataset)

