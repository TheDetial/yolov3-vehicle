# -*- coding: utf-8 -*-
from models.yolov4_CSPdark53 import yoloV4Model
from models.loss import YOLOLoss

__models__ = {
    "yolov4": yoloV4Model,
    "yololoss": YOLOLoss,
}
