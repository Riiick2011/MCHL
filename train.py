import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import time
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.misc import CollateFunc, build_dataset, build_dataloader
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from config import build_dataset_config, build_model_config
from models import build_model

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

GLOBAL_SEED = 42


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    # CUDA
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')

    # Dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24,jhmdb21, ava_v2.2')
    parser.add_argument('--data_root', default='/media/su/d/datasets/UCF24-YOWO/',
                        help='data root')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers used in dataloading')

    # Batchsize
    parser.add_argument('-bs', '--batch_size', default=8, type=int,
                        help='total batch size.')
    parser.add_argument('-tbs', '--test_batch_size', default=8, type=int,
                        help='total test batch size')
    parser.add_argument('-accu', '--accumulate', default=16, type=int,
                        help='gradient accumulate.')  # 当增大单卡bs时候，应该减少accu  默认是 单卡bs8xaccu16=bs128

    # Model
    parser.add_argument('-v', '--version', default='yowo_v2_large', type=str,
                        help='build YOWOv2')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='keep training')  # 恢复训练机制，模型恢复完整路径
    parser.add_argument('-K', '--len_clip', default=16, type=int,
                        help='video clip length.')
    parser.add_argument('--freeze_backbone_2d', action="store_true", default=False,
                        help="freeze 2D backbone.")
    parser.add_argument('--freeze_backbone_3d', action="store_true", default=False,
                        help="freeze 3d backbone.")
    parser.add_argument('--weight_folder', default='./weights/', type=str,
                        help='path to load and save weight')  # 保存训练结果模型权重的文件夹  与预训练模型的路径无关，预训练模型权重在各自的py文件中指定

    # Epoch
    parser.add_argument('--max_epoch', default=10, type=int,
                        help='max epoch.')
    parser.add_argument('--lr_epoch', nargs='+', default=[2, 3, 4], type=int,
                        help='lr epoch to decay')
    parser.add_argument('-lr', '--base_lr', default=0.0001, type=float,
                        help='base lr.')  # 基础学习率
    parser.add_argument('-ldr', '--lr_decay_ratio', default=0.5, type=float,
                        help='base lr.')

    # Evaluation
    parser.add_argument('--eval', action='store_true', default=False,
                        help='do evaluation during training.')  # 是否在训练过程中进行评估，默认为否
    parser.add_argument('--eval_epoch', default=1, type=int,
                        help='after eval epoch, the model is evaluated on val dataset.')  # 默认间隔1个周期进行一次评估
    parser.add_argument('-ct', '--conf_thresh', default=0.005, type=float,
                        help='confidence threshold. We suggest 0.005 for UCF24 and 0.1 for AVA.')  # 只对评估时有用
    parser.add_argument('--totaltopk', default=40, type=int,
                        help='topk prediction candidates.')  # 评估时在nms之前，是为了配合低conf阈值提升困难样本检测效果的同时不显著增加简单样本保存的检测数
    parser.add_argument('--infer_dir', default='./results/',
                        type=str, help='save inference results.')  # 评估的推断结果保存地址
    parser.add_argument('--map_path', default='./evaluator/eval_results/',
                        type=str, help='path to save mAP results')  # frame mAP结果的保存路径
    parser.add_argument('--link_method', default='viterbi',
                        type=str, help='link method in evaluating video mAP')  # 计算video mAP时采用的关联算法
    parser.add_argument('--det_save_type', default='one_class',
                        type=str, help='')  # 默认one_class表示一个检测框只对应一个类别得分，multi_class表示一个检测框对应所有类别得分

    # DDP train  分布式训练
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')  # 用于设置分布式训练的url
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')  # 分布式进程数
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')  # 是否使用批次归一化分布式同步
    parser.add_argument('--local_rank', type=int, default=0, help='local rank')  # local rank

    # Experiment
    parser.add_argument('--multi_sampling_rate', default=1, type=int,
                        help='number of sampling rates')  # 是否开启多采样率模块，几种采样率，
    # 1代表不开启，2代表2种（1、2），4代表4种（1、2、4、8）
    parser.add_argument('--clstopk', default=0, type=int,
                        help='topk per cls after nms.')  # 在nms之后每个类别最终最多保留的class-aware检测个数  默认0或者3，
    # 0表示全部保留，3表示只保留最高的3个
    parser.add_argument('--untrimmed', action='store_true', default=False,
                        help='use untrimmed frames to train')  # 是否使用untrimmed frames进行训练以获得鉴定动作起始点的能力
    # 训练出的模型只用于评估untrimmed数据集上的video mAP
    parser.add_argument('--bbox_with_feat', action='store_true', default=False,
                        help='bbox_with_feat for MCCLA link')  # 是否保存bbox_feat用于MCCLA关联算法计算video mAP

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist 分布式
    print('World size: {}'.format(distributed_utils.get_world_size()))  # 分布式进程数量，还没启用分布式则为1
    if args.distributed:
        distributed_utils.init_distributed_mode(args)  # 初始化分布式模式
        print("git:\n  {}\n".format(distributed_utils.get_sha()))  # 加密

    # path to save model  保存模型的路径
    path_to_save = os.path.join(args.weight_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True  # 开启后cudnn会自动寻找最快的卷积实现算法  如果网络模型结构不停变化，反而不应开启
        device = torch.device("cuda")  # 可见GPU中的软编号,默认选第一块
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # dataset and evaluator  构建训练集实例、frame mAP评估器实例(包含测试集实例，如果要求进行评估)并返回
    dataset, evaluator, num_classes = build_dataset(d_cfg, args, is_train=True)

    # dataloader
    batch_size = args.batch_size // distributed_utils.get_world_size()  # 单卡的batch size是batch size除以分布式进程(使用的显卡数量)个数
    dataloader = build_dataloader(args, dataset, batch_size, CollateFunc())

    # build model
    model, criterion = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device,
        num_classes=num_classes, 
        trainable=True,
        resume=args.resume)
    model = model.to(device).train()  # 移动到指定设备上并切换为训练模式，将batch normalization层和dropout层调整到训练模式

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    """
    # Compute FLOPs and Params  会导致进程之间不平衡
    if distributed_utils.is_main_process():
        model_copy = deepcopy(model_without_ddp)
        FLOPs_and_Params(
            model=model_copy,
            img_size=d_cfg['test_size'],
            len_clip=args.len_clip,
            device=device)
        del model_copy
    """

    # optimizer 优化器     返回优化器和周期  可以恢复训练
    # 通常base_lr默认作为初始学习率
    # last_epoch表示已经结束训练并且保存的epoch   -1表示重新开始训练
    optimizer, last_epoch = build_optimizer(
        d_cfg,
        model=model_without_ddp,
        base_lr=args.base_lr,
        resume=args.resume)

    # lr scheduler
    # optimizer  : 要更改学习率的优化器；
    # milestones : 递增的list，存放要更新lr的epoch；
    # gamma      : 更新lr的乘法因子；
    # last_epoch : 最后一个epoch的index，如果是训练了很多个epoch后中断了，继续训练，这个值就等于加载的模型的epoch。默认为-1表示从头开始训练，即从epoch=1开始。
    # 创建时，或者step时，lr_scheduler中的last_epoch就会+1，当等于milestones时，就对optimizer中的lr乘一次lr_decay
    # lr_scheduler既不知道降低了几次也不知道base_lr是多少
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=args.lr_epoch,
        gamma=args.lr_decay_ratio,
        last_epoch=last_epoch)
    start_epoch = last_epoch + 1

    # warmup scheduler 返回一个预热器，预热wp_iter次迭代
    # base_lr在这里只作为一个运算过程中的基本单位，不影响当前学习率，最后lr=init_lr * warmup_factor
    warmup_scheduler = build_warmup(d_cfg, base_lr=args.base_lr)

    # training configuration
    epoch_size = len(dataloader)  # 一个epoch内的迭代数
    warmup = True  # 处于warmup阶段的标志位

    # start to train
    t0 = time.time()
    for epoch in range(start_epoch, args.max_epoch):  # 从0开始
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (batch_img_name, batch_video_clips, batch_target) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size  # 总的迭代数

            # warmup
            if ni < d_cfg['wp_iter'] and warmup:  # 处于warmup阶段，不断预热，提高学习率
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == d_cfg['wp_iter'] and warmup:  # 结束warmup阶段，学习率设置为base_lr
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, lr=args.base_lr, base_lr=args.base_lr)

            # to device
            batch_video_clips = batch_video_clips.to(device)

            # inference
            batch_output = model(batch_video_clips)
            
            # loss
            loss_dict = criterion(batch_output, batch_target)
            losses = loss_dict['losses']

            # reduce 输入多进程分布式的损失字典，返回平均后的损失字典   默认采用平均模式
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward and Optimize
            losses = losses / args.accumulate  # 累加梯度 近似与更大的batch等价
            losses.backward()
            if ni % args.accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                    
            # Display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr'] for param_group in optimizer.param_groups]
                print_log(cur_lr, epoch, args.max_epoch - 1, iter_i, epoch_size,
                          loss_dict_reduced, t1-t0, args.accumulate)  # 打印loss_dict_reduced是因为reduced后的是多进程平均后的更稳定
                t0 = time.time()

        # evaluation
        if epoch % args.eval_epoch == 0 or (epoch + 1) == args.max_epoch:
            eval_one_epoch(args, model_without_ddp, optimizer, evaluator, epoch, path_to_save)

        lr_scheduler.step()  # lr_scheduler的last_epoch+1并与milestone比较，如果与任何一个相等，则降低一次optimizer中的学习率


def eval_one_epoch(args, model_eval, optimizer, evaluator, epoch, path_to_save):
    # check evaluator
    if distributed_utils.is_main_process():
        if evaluator is None:
            print('No evaluator ... save model and go on training.')
            
        else:
            print('eval ...')
            # set eval mode
            model_eval.trainable = False
            model_eval.eval()

            # evaluate
            evaluator.evaluate_frame_map(model_eval, epoch)
                
            # set train mode.
            model_eval.trainable = True
            model_eval.train()

        # save model
        print('Saving state, epoch:', epoch)
        weight_name = '{}_epoch_{}.pth'.format(args.version, epoch)
        checkpoint_path = os.path.join(path_to_save, weight_name)
        torch.save({'model': model_eval.state_dict(),
                    'optimizer': optimizer.state_dict(),  # 保存时学习率还没有更新
                    'epoch': epoch,
                    'args': args}, checkpoint_path)  # epoch代表已经结束该epoch的训练，epoch从0开始

    if args.distributed:
        # wait for all processes to synchronize  同步屏障，用来同步并行计算
        dist.barrier()


def print_log(lr, epoch, max_epoch, iter_i, epoch_size, loss_dict, time, accumulate):
    # basic infor
    log = '[Epoch: {}/{}]'.format(epoch, max_epoch)
    log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
    log += '[lr: {:.6f}]'.format(lr[0])
    # loss infor
    for k in loss_dict.keys():
        if k == 'losses':
            log += '[{}: {:.2f}]'.format(k, loss_dict[k] * accumulate)
        else:
            log += '[{}: {:.2f}]'.format(k, loss_dict[k])

    # other infor
    log += '[time: {:.2f}]'.format(time)

    # print log infor
    print(log, flush=True)


if __name__ == '__main__':
    train()
