# -*- coding: utf-8 -*-
import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import colorsys
from PIL import ImageDraw, ImageFont

from utils.utils import *
from utils.decoding import DecodeBox
from backbones import __backbones__
from models import __models__

# 测试模式1：	只对图像进行检测可视化并存到本地，不计算评价指标；
# 测试模式2：	只计算评价指标不进行检测可视化；
#
# @function:    测试脚本：对测试模式1的实现;
# @author:      TheDetial
# @date:        2022/05
# @last_edit:   2022/06

#  ---  start  ---
parser = argparse.ArgumentParser(description='voc yolov4 detection!')
# parser.add_argument('--backbone', default='fd', help='select a backbone structure', choices=__backbones__.keys())  # "resnet/mobilenet"
parser.add_argument('--model', default='fd', help='select a model structure', choices=__models__.keys())  # "retinaface"
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')  # 载入训练好的模型 (路径+模型名字)
parser.add_argument('--input_height', type=int, default=640, help='cnn input height')
parser.add_argument('--input_width', type=int, default=360, help='cnn input width')
parser.add_argument('--conf_thres', type=float, default=0.5, help='nms-confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.5, help='nms-iou threshold')
parser.add_argument('--letterbox_image', action='store_true', help='limitation image size')  # 是否开启预处理：图像的不失真resize
# parser.add_argument('--mode', default='dir_predict', help='select a test mode')  # 测试模式设置
parser.add_argument('--img_dir', type=str, required=True, help='directory of test images')  # 测试图像所在的文件夹目录
parser.add_argument('--save_dir', type=str, required=True, help='save directory of test images results')  # 测试结果保存文件夹目录
# add
parser.add_argument('--anchors_path', required=True, help='anchor path')  # 预设的anchor_size.txt
parser.add_argument('--classes_path', required=True, help='classes path')  # 预设的classes种类.txt
args = parser.parse_args()

# yolo detection 类定义
class yolo_detection(object):

    # 初始化
    def __init__(self, **kwargs):

        # 0、预设部分参数
        self.anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]  # 预设9组anchors尺寸
        self.input_shape = [args.input_height, args.input_width]
        self.class_names, self.num_classes = get_classes(args.classes_path)
        self.anchors, self.num_anchors = get_anchors(args.anchors_path)
        self.pretrain = False

        # 1、 模型设置，默认使用cuda
        self.model = __models__[args.model](self.anchors_mask, self.num_classes, pretrain=self.pretrain)
        #  前
        checkpoint = torch.load(args.loadckpt)
        self.model.load_state_dict(checkpoint)
        print("loading the model in logdir: {}".format(args.loadckpt))
        # 后
        self.model = nn.DataParallel(self.model)
        self.model.cuda()
        self.model.eval()
        self.letterbox_image = args.letterbox_image
        # 阈值
        self.conf_thres = args.conf_thres
        self.nms_thres = args.nms_thres
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        # 画框设置不同的颜色
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # # 图片检测+解码+nms+可视化绘制+存图：返回可视化绘制后的图像并保存
    def detect_img(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda()
            outputs = self.model(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, image_shape, self.letterbox_image, conf_thres=self.conf_thres, nms_thres=self.nms_thres)
            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4]*results[0][:, 5]  # score=conf*class_pre
            top_boxes = results[0][:, :4]  # 取box坐标
        font = ImageFont.truetype(font='list/simhei.ttf', size=np.floor(3e-2*image.size[1]+0.5).astype('int32'))
        thickness = int(max((image.size[0]+image.size[1]) // np.mean(self.input_shape), 1))

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]
            top, left, bottom, right = box
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

# 对一个文件夹中的图片进行检测，并进行可视化绘制后保存到本地
def main():
    yolov4 = yolo_detection()
    # 对文件夹内的图像进行检测并可视化
    img_names = os.listdir(args.img_dir)
    for img_name in tqdm(img_names):
        if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
            image_path = os.path.join(args.img_dir, img_name)
            image = Image.open(image_path)
            img_res = yolov4.detect_img(image)
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            img_res.save(os.path.join(args.save_dir, img_name))

            # image = cv2.imread(image_path)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # img_res = retinaface.detect_img(image)  # 调用检测函数
            # img_res = cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR)
            # if not os.path.exists(args.save_dir):
            #     os.makedirs(args.save_dir)
            # cv2.imwrite(os.path.join(args.save_dir, img_name), img_res)

if __name__ == '__main__':
    main()
