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
import xml.etree.ElementTree as ET

from utils.utils import *
from utils.utils_map import *
from utils.decoding import DecodeBox
from backbones import __backbones__
from models import __models__

# 测试模式1：	只对图像进行检测可视化并存到本地，不计算评价指标；
# 测试模式2：	指标计算：只有voc2007的test测试集释放了真值标签，可用于评估map好坏
#
# @function:    测试脚本：对测试模式2的实现;
# @author:      TheDetial
# @date:        2022/05
# @last_edit:   2022/06

#  ---  start  ---
parser = argparse.ArgumentParser(description='voc yolov4 detection!')
# parser.add_argument('--backbone', default='fd', help='select a backbone structure', choices=__backbones__.keys())  # "resnet/mobilenet"
parser.add_argument('--model', default='fd', help='select a model structure', choices=__models__.keys())  # "yolov4"
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')  # 载入训练好的模型 (路径+模型名字)
parser.add_argument('--input_height', type=int, default=640, help='cnn input height')
parser.add_argument('--input_width', type=int, default=360, help='cnn input width')
parser.add_argument('--conf_thres', type=float, default=0.5, help='nms-confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.5, help='nms-iou threshold')
parser.add_argument('--letterbox_image', action='store_true', help='limitation image size')  # 是否开启预处理：图像的不失真resize
# parser.add_argument('--mode', default='dir_predict', help='select a test mode')  # 测试模式设置
parser.add_argument('--img_dir', type=str, required=True, help='directory of test images')  # 测试图像所在的文件夹目录
parser.add_argument('--save_dir', type=str, required=True, help='save directory of test images results')  # 测试结果保存文件夹目录
parser.add_argument('--anchors_path', required=True, help='anchor path')  # 预设的anchor_size.txt
parser.add_argument('--classes_path', required=True, help='classes path')  # 预设的classes种类.txt
# add
parser.add_argument('--map_vis', action='store_true', help='visualizaion')  # 是否开启voc_map的计算可视化
parser.add_argument('--MINOVERLAP', type=float, default=0.5, help='nms-confidence threshold')  # 获取指定的mAP0.X
parser.add_argument('--map_mode', type=int, default=0, help='select map mode for test')
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

    # 图片检测+解码+nms+可视化绘制+存图：返回可视化绘制后的图像并保存
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

    # get_map.py脚本调用,对测试集所有图片预测结果，并写入txt文档
    # 网络预测：包括载入原始数据做resize+送入网络+网络预测结果解码+nms+点坐标对回到图像原始尺寸
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")  # 打开每张图片对应的检测结果detection下的 xxx.txt
        image_shape = np.array(np.shape(image)[0:2])
        # 代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        image = cvtColor(image)
        #  给图像增加灰条，实现不失真的resize
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 添加batch_size维度
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.cuda()
            #  将图像输入网络当中进行预测！
            outputs = self.model(images)
            outputs = self.bbox_util.decode_box(outputs)  # 解码
            # 将预测框进行堆叠，然后进行非极大抑制  # nms后已经对回到原始图像尺寸
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.conf_thres,
                                                         nms_thres=self.nms_thres)

            if results[0] is None:
                return
            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]  # confidence*class_pre
            top_boxes = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])  # score=confidence*class_pre

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))
        f.close()
        return


# 计算带gt测试数据的map结果
def main():
    yolov4 = yolo_detection()
    class_names, num_classes = get_classes(args.classes_path)
    image_ids = open(os.path.join(args.img_dir, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, 'ground-truth')):
        os.makedirs(os.path.join(args.save_dir, 'ground-truth'))
    if not os.path.exists(os.path.join(args.save_dir, 'detection-results')):
        os.makedirs(os.path.join(args.save_dir, 'detection-results'))
    if not os.path.exists(os.path.join(args.save_dir, 'images-optional')):
        os.makedirs(os.path.join(args.save_dir, 'images-optional'))

    # 1、加载YOLO类，对测试集所有图片预测结果，并写入txt文档
    if args.map_mode == 0 or args.map_mode == 1:
        print("Load model.")
        # yolo = YOLO(confidence=0.001, nms_iou=0.5)  # 调用YOLO class
        print("Load model done.")
        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(args.img_dir, "VOC2007/JPEGImages/"+image_id+".jpg")
            image = Image.open(image_path)
            # 是否开启可视化存储
            if args.map_vis:
                image.save(os.path.join(args.save_dir, "images-optional/" + image_id + ".jpg"))  # 所有的测试图片复制存储在images-optional文件夹内
            # yolo.get_map_txt(image_id, image, class_names, map_out_path)  # 载入图像，获取网络输出，写入到detection-results/中xxx.txt中
            yolov4.get_map_txt(image_id, image, class_names, args.save_dir)  # 载入图像，获取网络输出，写入到detection-results/中xxx.txt中
        print("Get predict result done.")

    # 2、读取gt数据，将每张图像的类别+box坐标写入xxx.txt
    if args.map_mode == 0 or args.map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(args.save_dir, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(args.img_dir, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    # 3、进行预测结果和GT的map计算
    if args.map_mode == 0 or args.map_mode == 3:
        print("Get map.")
        get_map(args.MINOVERLAP, True, path=args.save_dir)
        print("Get map done.")

    if args.map_mode == 4:
        print("Get map.")
        get_coco_map(class_names=class_names, path=args.save_dir)
        print("Get map done.")

if __name__ == '__main__':
    main()
