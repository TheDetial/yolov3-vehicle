# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.utils import *
from utils.log import LossHistory
from datasets.dataloader import yolo_dataset_collate
from datasets import __datasets__
from backbones import __backbones__
from models import __models__

# @function:    yolov4训练脚本
# @author:      TheDetial
# @date:        2022/06
# @last_edit:   2022/06

#  ---  start  ---
parser = argparse.ArgumentParser(description='RetinaFace face/landmarks detection!')
parser.add_argument('--backbone', default='fd', help='select a backbone structure', choices=__backbones__.keys())  # "cspdarknet53"
parser.add_argument('--model', default='fd', help='select a model structure', choices=__models__.keys())  # "yolov4"
parser.add_argument('--dataloader', required=True, help='dataset loader name', choices=__datasets__.keys())  # "voc detection dataload"
parser.add_argument('--trainlist', required=True, help='training list')  # train list.txt  --训练集
parser.add_argument('--vallist', required=True, help='val list')  # val list.txt  --验证集
parser.add_argument('--batch_size', type=int, default=64, metavar='batch_size', help='Size of batch)')  # 默认给解冻训练时的batch_size
parser.add_argument('--freeze_lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--unfreeze_lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--trainloss', default='fd', help='select a model structure', choices=__models__.keys())  # "loss function"
parser.add_argument('--bbpretrain', action='store_true', help='continue training the model from pretrained weights')  # backbone是否使用预训练模型
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epoch to train')
parser.add_argument('--freeze_epoches', type=int, required=True, help='number of epochs to train')
parser.add_argument('--max_epoches', type=int, required=True, help='number of epochs to train')
parser.add_argument('--freeze_train', action='store_true', help='split net training for freeze and unfreeze stage')  # 是否使用冻结和解冻训练
# add
parser.add_argument('--input_height', type=int, default=416, help='input height')  # 网络预设输入尺寸
parser.add_argument('--input_width', type=int, default=416, help='input width')  # 网络预设输入尺寸
parser.add_argument('--anchors_path', required=True, help='anchor path')  # 预设的anchor_size.txt
parser.add_argument('--classes_path', required=True, help='classes path')  # 预设的classes种类.txt
parser.add_argument('--mosaic', action='store_true', help='continue training the model')  # 是否马赛克增强
parser.add_argument('--Cosine_lr', action='store_true', help='continue training the model')  # 是否余弦退火学习率
parser.add_argument('--label_smoothing', type=float, default=0.01, help='label smoothing rate')  # 标签平滑项
# if resume
parser.add_argument('--resume', action='store_true', help='continue training the model')  # 是否继续训练
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')  # resume 整体网络模型

# start
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True
Cuda = True

imgload = __datasets__[args.dataloader]
criterionLoss = __models__[args.trainloss]

# epoch
# train+val+save
def train_one_epoch(net_train, net, optimizer, criterion, epoch, epoch_step, epoch_step_val, Epoches, gen_train, gen_val, loss_history):
    loss = 0
    val_loss = 0
    # 1、开始训练
    net_train.train()
    print("Train")
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoches}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_train):  # gen是通过dataLoader载入的数据img和box真实标注已经全部做了归一化处理
            if iteration >= epoch_step:
                break
            # ----------------------------------------------
            # targets 真实框的标签情况 [batch_size, num_gt, 5]
            # targets中存储了归一化后的真实box信息
            # ----------------------------------------------
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]

            optimizer.zero_grad()
            outputs = net_train(images)  # 模型输出
            loss_value_all = 0
            num_pos_all = 0
            # 有效特征图循环计算： 13*13-->26*26-->52*52尺寸依次进行
            for th in range(len(outputs)):
                loss_item, num_pos = criterion(th, outputs[th], targets)
                loss_value_all += loss_item
                num_pos_all += num_pos
            loss_value = loss_value_all / num_pos_all

            loss_value.backward()
            optimizer.step()

            loss += loss_value.item()
            pbar.set_postfix(**{'train loss': loss / (iteration + 1), 'lr': get_lr(optimizer)})
            pbar.update(1)
    # print("finish train")

    # 2、开始验证
    print("Validation")
    net_train.eval()
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoches}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                targets = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in targets]

                optimizer.zero_grad()
                outputs = net_train(images)
                loss_value_all = 0
                num_pos_all = 0
                for th in range(len(outputs)):
                    loss_item, num_pos = criterion(th, outputs[th], targets)
                    loss_value_all += loss_item
                    num_pos_all += num_pos
                loss_value = loss_value_all / num_pos_all
            val_loss += loss_value.item()
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
    # print('finish Validation')
    # 3、save
    loss_history.append_loss(loss / epoch_step, val_loss / epoch_step_val)
    print('Epoch:' + str(epoch + 1) + '/' + str(epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    torch.save(net.state_dict(), args.logdir + 'Epoch%d-train_loss%.3f-val_loss%.3f.pth' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))
    print(" : ----- Epoch%d.pth saved ----- : " % (epoch + 1))

# train all
def train(train_lines, val_lines, net_train, net, learning_rate, input_shape, num_classes, start_epoch, end_epoches, batch_size, criterion, loss_history):

    epoch_step = len(train_lines) // batch_size
    epoch_step_val = len(val_lines) // (batch_size//2)

    optimizer = optim.Adam(net_train.parameters(), learning_rate, weight_decay=5e-4)
    # 学习率更新方式
    if args.Cosine_lr:
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)
    # 载入训练和验证集
    train_dataset = imgload(train_lines, input_shape, num_classes, mosaic=args.mosaic, train=True)
    val_dataset = imgload(val_lines, input_shape, num_classes, mosaic=False, train=False)
    gen_train = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                           pin_memory=True, drop_last=False, collate_fn=yolo_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size//2, num_workers=4,
                           pin_memory=True, drop_last=False, collate_fn=yolo_dataset_collate)
    # 
    for epoch in range(start_epoch, end_epoches):
        train_one_epoch(net_train, net, optimizer, criterion, epoch, epoch_step, epoch_step_val, end_epoches, gen_train, gen_val, loss_history)
        lr_scheduler.step()


# 主函数
def main():

    # 0、预设部分参数
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # 预设9组anchors尺寸
    input_shape = [args.input_height, args.input_width]
    class_names, num_classes = get_classes(args.classes_path)
    anchors, num_anchors = get_anchors(args.anchors_path)
    model = __models__[args.model](anchors_mask, num_classes, pretrain=args.bbpretrain)

    # 1、若不使用预训练模型进行权重初始化  --从零开始训练
    if not args.bbpretrain:
        weights_init(model)

    # 2、训练模式选择  --resume 继续训练
    if args.resume and not args.bbpretrain:
        # assert os.path.isfile(args.resume)
        # checkpoint = torch.load(args.resume)
        # print('pretrain...')
        print("loading the lastest model in logdir: {}".format(args.checkpoint_path))
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint)
        print('resume from a model ... ... ')

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)  # 多卡
        model_train = model_train.cuda()
    print("Number of model parameters: {}".format(sum([p.data.nelement() for p in model.parameters()])))

    criterion = criterionLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, args.label_smoothing)
    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    loss_history = LossHistory(args.logdir)

    with open(args.trainlist) as f:
        train_lines = f.readlines()
    with open(args.vallist) as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    print("train data nums: ", num_train)
    print("val data nums: ", num_val)

    if args.freeze_train:  # 5.1 在先冻结-后解冻的训练模式下
        # （1）先冻结训练
        for param in model.backbone.parameters():  # 此处看yolov4中网络定义backbone=darknet53()
            param.requires_grad = False
        train(train_lines, val_lines, model_train, model, args.freeze_lr, input_shape, num_classes, args.start_epoch, args.freeze_epoches, args.batch_size * 2, criterion, loss_history)  # 冻结训练时，batch_size大一点
        # （2）后解冻训练
        for param in model.backbone.parameters():
            param.requires_grad = True
        train(train_lines, val_lines, model_train, model, args.unfreeze_lr, input_shape, num_classes, args.freeze_epoches, args.max_epoches, args.batch_size, criterion, loss_history)  # 解冻训练时，batch_size小一点
    else:  # 5.2 在直接更新所有参数的训练模式下(默认状态：参数全部更新)
        train(train_lines, val_lines, model_train, model, args.unfreeze_lr, input_shape, num_classes, args.start_epoch, args.max_epoches, args.batch_size, criterion, loss_history)  # 训练更新所有参数==解冻训练，batch_size小一点

if __name__ == '__main__':
    main()
