# -*- coding: utf-8 -*-
from collections import OrderedDict
import torch
import torch.nn as nn
from models.modules import *
from backbones import darknet53

# @function:    yolov4整体结构定义
# @author:      TheDetial
# @date:        2022/06
# @last_edit:   2022/06

# yolov4架构定义
class yoloV4Model(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrain=False):
        super(yoloV4Model, self).__init__()

        self.backbone = darknet53(pretrain)

        self.conv1 = make_three_conv([512, 1024], 1024)
        self.SPP = SpatialPyramidPooling()
        self.conv2 = make_three_conv([512, 1024], 2048)

        self.upsample1 = Upsample(512, 256)
        self.conv_for_P4 = conv2d(512, 256, 1)
        self.make_five_conv1 = make_five_conv([256, 512], 512)

        self.upsample2 = Upsample(256, 128)
        self.conv_for_P3 = conv2d(256, 128, 1)
        self.make_five_conv2 = make_five_conv([128, 256], 256)

        # 检测头
        # 有效特征层1
        self.yolo_head3 = yolo_head([256, len(anchors_mask[0])*(5+num_classes)], 128)
        self.down_sample1 = conv2d(128, 256, 3, stride=2)
        self.make_five_conv3 = make_five_conv([256, 512], 512)
        # 有效特征层2
        self.yolo_head2 = yolo_head([512, len(anchors_mask[1])*(5+num_classes)], 256)
        self.down_sample2 = conv2d(256, 512, 3, stride=2)
        self.make_five_conv4 = make_five_conv([512, 1024], 1024)

        # 有效特征层
        self.yolo_head1 = yolo_head([1024, len(anchors_mask[2])*(5+num_classes)], 512)


    def forward(self, x):

        x2, x1, x0 = self.backbone(x)

        P5 = self.conv1(x0)
        P5 = self.SPP(P5)
        P5 = self.conv2(P5)

        P5_upsample = self.upsample1(P5)
        P4 = self.conv_for_P4(x1)
        P4 = torch.cat([P4, P5_upsample], axis=1)
        P4 = self.make_five_conv1(P4)

        P4_upsample = self.upsample2(P4)
        P3 = self.conv_for_P3(x2)
        P3 = torch.cat([P3, P4_upsample], axis=1)
        P3 = self.make_five_conv2(P3)

        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], axis=1)
        P4 = self.make_five_conv3(P4)

        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], axis=1)
        P5 = self.make_five_conv4(P5)
        # 预设[416,416]网络输入
        # 大特征层：[bs, 75, 52, 52]
        out2 = self.yolo_head3(P3)
        # 中特征层：[bs, 75, 26, 26]
        out1 = self.yolo_head2(P4)
        # 小特征层：[bs, 75, 13, 13]
        out0 = self.yolo_head1(P5)

        return out0, out1, out2
