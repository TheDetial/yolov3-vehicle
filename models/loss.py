# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
import numpy as np

# yolov4损失函数定义
class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes, input_shape, cuda, anchors_mask, label_smoothing):
        super(YOLOLoss, self).__init__()

        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_shape = input_shape
        self.cuda = cuda
        self.anchors_mask = anchors_mask
        self.label_smoothing = label_smoothing

        self.ignore_threshold = 0.5

    # ciou计算函数定义，非ciou损失定义  --use
    def box_ciou(self, b1, b2):
        """
        输入为：
        ----------
        b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
        b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

        返回为：
        -------
        ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
        """
        # ----------------------------------------------------#
        #   求出预测框左上角右下角
        # ----------------------------------------------------#
        b1_xy = b1[..., :2]
        b1_wh = b1[..., 2:4]
        b1_wh_half = b1_wh / 2.
        b1_mins = b1_xy - b1_wh_half
        b1_maxes = b1_xy + b1_wh_half
        # ----------------------------------------------------#
        #   求出真实框左上角右下角
        # ----------------------------------------------------#
        b2_xy = b2[..., :2]
        b2_wh = b2[..., 2:4]
        b2_wh_half = b2_wh / 2.
        b2_mins = b2_xy - b2_wh_half
        b2_maxes = b2_xy + b2_wh_half

        # ----------------------------------------------------#
        #   求真实框和预测框所有的iou
        # ----------------------------------------------------#
        intersect_mins = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area = b2_wh[..., 0] * b2_wh[..., 1]
        union_area = b1_area + b2_area - intersect_area
        iou = intersect_area / torch.clamp(union_area, min=1e-6)

        # ----------------------------------------------------#
        #   计算中心的差距
        # ----------------------------------------------------#
        center_distance = torch.sum(torch.pow((b1_xy - b2_xy), 2), axis=-1)

        # ----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        # ----------------------------------------------------#
        enclose_mins = torch.min(b1_mins, b2_mins)
        enclose_maxes = torch.max(b1_maxes, b2_maxes)
        enclose_wh = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        # ----------------------------------------------------#
        #   计算对角线距离
        # ----------------------------------------------------#
        enclose_diagonal = torch.sum(torch.pow(enclose_wh, 2), axis=-1)
        ciou = iou - 1.0 * (center_distance) / torch.clamp(enclose_diagonal, min=1e-6)

        v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(
            b1_wh[..., 0] / torch.clamp(b1_wh[..., 1], min=1e-6)) - torch.atan(
            b2_wh[..., 0] / torch.clamp(b2_wh[..., 1], min=1e-6))), 2)
        alpha = v / torch.clamp((1.0 - iou + v), min=1e-6)
        ciou = ciou - alpha * v
        return ciou

    # --use
    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    # 交叉熵损失：置信度conf、物体种类class  --use
    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0-epsilon)
        output = -target * torch.log(pred)-(1.0-target)*torch.log(1.0-pred)  # sigmoid cross-entropy() loss

        return output

    # 标签平滑函数定义  --use
    def smooth_labels(self, y_true, label_smoothing, num_classes):
        return y_true * (1.0-label_smoothing) + label_smoothing / num_classes

    # 计算IOU  --use
    def calculate_iou(self, _box_a, _box_b):
        #-----------------------------------------------------------#
        #   计算真实框的左上角和右下角
        #-----------------------------------------------------------#
        b1_x1, b1_x2 = _box_a[:, 0] - _box_a[:, 2] / 2, _box_a[:, 0] + _box_a[:, 2] / 2
        b1_y1, b1_y2 = _box_a[:, 1] - _box_a[:, 3] / 2, _box_a[:, 1] + _box_a[:, 3] / 2
        #-----------------------------------------------------------#
        #   计算先验框获得的预测框的左上角和右下角
        #-----------------------------------------------------------#
        b2_x1, b2_x2 = _box_b[:, 0] - _box_b[:, 2] / 2, _box_b[:, 0] + _box_b[:, 2] / 2
        b2_y1, b2_y2 = _box_b[:, 1] - _box_b[:, 3] / 2, _box_b[:, 1] + _box_b[:, 3] / 2

        #-----------------------------------------------------------#
        #   将真实框和预测框都转化成左上角右下角的形式
        #-----------------------------------------------------------#
        box_a = torch.zeros_like(_box_a)
        box_b = torch.zeros_like(_box_b)
        box_a[:, 0], box_a[:, 1], box_a[:, 2], box_a[:, 3] = b1_x1, b1_y1, b1_x2, b1_y2
        box_b[:, 0], box_b[:, 1], box_b[:, 2], box_b[:, 3] = b2_x1, b2_y1, b2_x2, b2_y2

        #-----------------------------------------------------------#
        #   A为真实框的数量，B为先验框的数量
        #-----------------------------------------------------------#
        A = box_a.size(0)
        B = box_b.size(0)
        #-----------------------------------------------------------#
        #   计算交的面积
        #-----------------------------------------------------------#
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        #-----------------------------------------------------------#
        #   计算预测框和真实框各自的面积
        #-----------------------------------------------------------#
        area_a = ((box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2]-box_b[:, 0]) * (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        #-----------------------------------------------------------#
        #   求IOU
        #-----------------------------------------------------------#
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    # --use
    def get_target(self, th, targets, anchors, in_h, in_w):
        bs = len(targets)
        # 用于选取哪些先验框不包含物体
        noobj_mask = torch.ones(bs, len(self.anchors_mask[th]), in_h, in_w, requires_grad=False)  # 全1初始化
        # 让网络更加关注小目标
        box_loss_scale = torch.zeros(bs, len(self.anchors_mask[th]), in_h, in_w, requires_grad=False)  # 全0初始化
        # 真实框映射到y_true和网络预测输出结果shape对应[batch_size, 3, 13, 13, 5+num_classes]
        y_true = torch.zeros(bs, len(self.anchors_mask[th]), in_h, in_w, self.bbox_attrs, requires_grad=False)

        # 循环batch_size(本质为循环一张一张的图像)
        for b in range(bs):
            if len(targets[b]) == 0:  # 如果该张图像的真实框个数为0，则直接跳过不进行处理，直接进行下一张图像处理即可
                continue
            batch_target = torch.zeros_like(targets[b])  # 全0初始化
            # -------------------------------------------------------
            #   计算出正样本在特征层上的中心点
            # 将真实框从归一化的状态调整至有效特征图尺寸大小(targets中的box 真实标签 为经过归一化后处理，以网络输入尺寸416做归一化)
            # GT box ---> dataloader时以416做归一化 --->此处再将归一化的坐标放回到有效特征图尺寸大小
            # batch_target：归一化的坐标全部拉回到13*13尺寸, class不变
            # ----------------------------------------------
            batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w  # x,w
            batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h  # y,h
            batch_target[:, 4] = targets[b][:, 4]  # 取类别标签真实class label，不变直接取即可
            batch_target = batch_target.cpu()
            # 将真实框转换一个形式：方便iou计算
            gt_box = torch.FloatTensor(torch.cat((torch.zeros((batch_target.size(0), 2)), batch_target[:, 2:4]), 1))
            # 将先验框转换一个形式
            # 此处的先验框anchors为scaled_anchors=预设的anchor(为416尺寸)转换到有效特征图尺寸下
            anchor_shapes = torch.FloatTensor(torch.cat((torch.zeros((len(anchors), 2)), torch.FloatTensor(anchors)), 1))
            # -------------------------------------------------------#
            #   计算交并比： self.calculate_iou()
            #   self.calculate_iou(gt_box, anchor_shapes) = [num_true_box, 9] ：计算每一个真实框和9个先验框的重合情况
            #   以num_true_box=3为例,
            #   (1)、self.calculate_iou()计算后的结果为：
            #   [
            #     [iou0, iou1, iou2, iou3, iou4, iou5, iou6, iou7, iou8],
            #     [iou0, iou1, iou2, iou3, iou4, iou5, iou6, iou7, iou8],
            #     [iou0, iou1, iou2, iou3, iou4, iou5, iou6, iou7, iou8],
            #                    ]
            #   (2)、取torch.argmax()后结果best_ns为：
            #        假设上述三个真实框分别对应的最大iou分别为：iou6,iou2,iou5
            #    [
            #      [iou6, 6],
            #      [iou2, 2],
            #      [iou5, 5],
            #                   ]
            #
            # -------------------------------------------------------#
            # 计算每一个真实框和9个先验框的重合情况，并返回最大IOU和每个真实框最重合的anchors的序号
            #   best_ns中包括:
            #   [每个真实框最大的重合度max_iou, 每一个真实框最重合的先验框的序号]
            best_ns = torch.argmax(self.calculate_iou(gt_box, anchor_shapes), dim=-1)
            # t:每个真实框计算后得到的最大重合度max_iou,  best_n:每一个真实框最重合的先验框的序号
            for t, best_n in enumerate(best_ns):
                # 若该最大IOU的anchor标号不在该有效特征图对应的预设anchors中，则直接跳过  13*13尺寸对应的是在[6,7,8]序号的预设anchors中的一个
                if best_n not in self.anchors_mask[th]:
                    continue
                # (1)、判断这个先验框是当前特征点的哪一个先验框
                k = self.anchors_mask[th].index(best_n)  # 需要知道到底是哪个预设的anchor,k是符合条件的预设anchor的序号
                # (2)、获得真实框属于哪个网格点
                #       (x, y)向下取整
                i = torch.floor(batch_target[t, 0]).long()  # x
                j = torch.floor(batch_target[t, 1]).long()  # y
                # (3)、取物体所属类别label号，整数
                c = batch_target[t, 4].long()  # 物体类别label号
                # (4)、noobj_mask代表无目标的特征点  全1初始化
                # noobj_mask：shape: (b,3,13,13,)
                noobj_mask[b, k, j, i] = 0
                #   (5)、对y_true进行赋值
                #   tx、ty代表中心调整参数的真实值
                #   y_true shape: shape: (b,3,13,13,5+20)
                y_true[b, k, j, i, 0] = batch_target[t, 0]
                y_true[b, k, j, i, 1] = batch_target[t, 1]
                y_true[b, k, j, i, 2] = batch_target[t, 2]
                y_true[b, k, j, i, 3] = batch_target[t, 3]
                y_true[b, k, j, i, 4] = 1  # 是否包含物体种类，包含为1
                y_true[b, k, j, i, c+5] = 1  # 在其种类label的位置index置1
                # (6)、用于获取xywh比例
                # 大目标loss权重小， 小目标loss权重大
                # 将相对于有效特征层的真实宽高进行相乘，再乘以有效特征层的宽高 -->相当于做了归一化到(0, 1)之间
                box_loss_scale[b, k, j, i] = batch_target[t, 2] * batch_target[t, 3]/in_w/in_h
        return y_true, noobj_mask, box_loss_scale

    # --use
    def get_ignore(self, l, x, y, h, w, targets, scaled_anchors, in_h, in_w, noobj_mask):
        # -----------------------------------------------------#
        #   计算一共有多少张图片
        # -----------------------------------------------------#
        bs = len(targets)
        # ----------------------------------------------
        # 1、 对网络输出的预测结果进行解码
        # ----------------------------------------------
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        # -----------------------------------------------------#
        #   生成网格，先验框中心，网格左上角
        # -----------------------------------------------------#
        # -------------------例子：---------------------------
        # (1)、torch.linspace(0, in_w - 1, in_w)生成从0到12的13个数组成的等差数列 输出tensor([0,1,2,3,4,5,6,7,8,9,10,11,12])；
        # (2)、torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1):将(1)的结果复制in_h次；
        # (3)、grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
        #             int(bs * len(self.anchors_mask[l])), 1, 1)：将(2)的结果复制int(bs * len(self.anchors_mask[l]))次；
        # ----------------------------------------------
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_h, 1).repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_w, 1).t().repeat(
            int(bs * len(self.anchors_mask[l])), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        scaled_anchors_l = np.array(scaled_anchors)[self.anchors_mask[l]]  # 只取该l标号有效特征层的scaled_anchors
        anchor_w = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors_l).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)
        # -------------------------------------------------------#
        #  2、 计算调整后的先验框中心与宽高
        # -------------------------------------------------------#
        pred_boxes_x = torch.unsqueeze(x + grid_x, -1)  # x+grid_x: 预测的偏移量+中心坐标
        pred_boxes_y = torch.unsqueeze(y + grid_y, -1)  # y+grid_y：预测的偏移量+中心坐标
        pred_boxes_w = torch.unsqueeze(torch.exp(w) * anchor_w, -1)  # anchor_w * e^w
        pred_boxes_h = torch.unsqueeze(torch.exp(h) * anchor_h, -1)  # anchor_h * e^h
        pred_boxes = torch.cat([pred_boxes_x, pred_boxes_y, pred_boxes_w, pred_boxes_h],
                               dim=-1)  # 对scaled_anchors使用网络预测出的参数进行调整之后的boxes
        for b in range(bs):
            # -------------------------------------------------------#
            #   将预测结果转换一个形式
            #   pred_boxes_for_ignore      num_anchors, 4
            # -------------------------------------------------------#
            pred_boxes_for_ignore = pred_boxes[b].view(-1, 4)
            # -------------------------------------------------------#
            #   计算真实框，并把真实框转换成相对于特征层的大小
            #   gt_box      num_true_box, 4
            # -------------------------------------------------------#
            if len(targets[b]) > 0:
                batch_target = torch.zeros_like(targets[b])
                # -------------------------------------------------------#
                #   计算出正样本在特征层上的中心点
                # 和get_target中一样，拉回到13尺寸
                # batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w  # x,w
                # batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h  # y,h
                # batch_target[:, 4] = targets[b][:, 4]  # 取类别标签真实class label，不变直接取即可

                # -------------------------------------------------------#
                batch_target[:, [0, 2]] = targets[b][:, [0, 2]] * in_w
                batch_target[:, [1, 3]] = targets[b][:, [1, 3]] * in_h
                batch_target = batch_target[:, :4]
                # -------------------------------------------------------#
                #   计算交并比
                #   anch_ious       num_true_box, num_anchors
                # -------------------------------------------------------#
                anch_ious = self.calculate_iou(batch_target, pred_boxes_for_ignore)  # gt_box和pred_boxes计算IOU
                # -------------------------------------------------------#
                #   每个先验框对应真实框的最大重合度
                #   anch_ious_max   num_anchors
                # -------------------------------------------------------#
                anch_ious_max, _ = torch.max(anch_ious, dim=0)
                anch_ious_max = anch_ious_max.view(pred_boxes[b].size()[:3])
                noobj_mask[b][anch_ious_max > self.ignore_threshold] = 0
        return noobj_mask, pred_boxes

    def forward(self, th, input, targets=None):
        '''
        input:
                th：有效特征图的编号
                input：网络的直接输出
                targets：GT标签
        return:
                loss：三个损失之和
                num_pos：正样本个数
        '''

        # 1、获取网络预测输出的batch_size和有效特征层宽高
        bs = input.size(0)
        in_h = input.size(2)
        in_w = input.size(3)

        # 2、计算当前有效特征层降采样步长
        stride_h = self.input_shape[0] / in_h
        stride_w = self.input_shape[1] / in_w

        # 3、获取scaled_anchors尺寸大小
        scaled_anchors = [(a_w / stride_w, a_h / stride_h) for a_w, a_h in self.anchors]  # anchors.txt中预设的anchor尺寸本身是在416尺寸下的，此处将其调整至有效特征图尺寸

        # 4、网络输出shape调整：# bs, 3 * (5 + num_classes), 13, 13 = > bs, 3, 5 + num_classes, 13, 13 = > batch_size, 3, 13, 13, 5 + num_classes
        prediction = input.view(bs, len(self.anchors_mask[th]), self.bbox_attrs, in_h, in_w).permute(0, 1, 3, 4, 2).contiguous()

        # 5、获取网络输出的具体内容--->转为相应的调整参数
        # (1)框的中心坐标
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # (2)框的宽高
        w = prediction[..., 2]
        h = prediction[..., 3]
        # (3)是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # (4)物体种类
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 6、获取网络应该有的预测结果
        y_true, noobj_mask, box_loss_scale = self.get_target(th, targets, scaled_anchors, in_h, in_w)

        # 7、
        noobj_mask, pred_boxes = self.get_ignore(th, x, y, w, h, targets, scaled_anchors, in_h, in_w, noobj_mask)

        if self.cuda:
            y_true = y_true.cuda()
            noobj_mask = noobj_mask.cuda()
            box_loss_scale = box_loss_scale.cuda()
        box_loss_scale = 2-box_loss_scale
        # 8、损失函数：bbox框  --CIOU
        ciou = (1-self.box_ciou(pred_boxes[y_true[..., 4]==1], y_true[..., :4][y_true[..., 4]==1])) * box_loss_scale[y_true[..., 4]==1]
        loss_loc = torch.sum(ciou)
        # 9、损失函数：conf置信度  --是否有物体
        loss_conf = torch.sum(self.BCELoss(conf, y_true[..., 4])*y_true[..., 4]) + torch.sum(self.BCELoss(conf, y_true[..., 4])*noobj_mask)
        # 10、损失函数：class类别  --多标签分类
        loss_cls = torch.sum(self.BCELoss(pred_cls[y_true[..., 4]==1], self.smooth_labels(y_true[..., 5:][y_true[..., 4]==1], self.label_smoothing, self.num_classes)))
        # 总损失函数
        loss = loss_loc + loss_conf + loss_cls
        num_pos = torch.sum(y_true[..., 4])
        num_pos = torch.max(num_pos, torch.ones_like(num_pos))

        return loss, num_pos
