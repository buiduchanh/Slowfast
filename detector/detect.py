#!python3
# -*- coding: utf-8 -*-
import os
import colorsys
import cv2
import numpy as np
from ctypes import *

class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int),
                ("uc", POINTER(c_float)),
                ("points", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]


class DARKNET(object):
    def __init__(self, darknetlib_path, config_path, meta_path, weight_path):
        self.lib = CDLL(darknetlib_path, RTLD_GLOBAL)
        self.lib.network_width.argtypes = [c_void_p]
        self.lib.network_width.restype = c_int
        self.lib.network_height.argtypes = [c_void_p]
        self.lib.network_height.restype = c_int


        make_image = self.lib.make_image
        make_image.argtypes = [c_int, c_int, c_int]
        make_image.restype = IMAGE

        load_net_custom = self.lib.load_network_custom
        load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
        load_net_custom.restype = c_void_p

        load_meta = self.lib.get_metadata
        load_meta.argtypes = [c_char_p]
        load_meta.restype = METADATA

        self.copy_image_from_bytes = self.lib.copy_image_from_bytes
        self.copy_image_from_bytes.argtypes = [IMAGE, c_char_p]

        self.get_network_boxes = self.lib.get_network_boxes
        self.get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int), c_int]
        self.get_network_boxes.restype = POINTER(DETECTION)

        self.free_detections = self.lib.free_detections
        self.free_detections.argtypes = [POINTER(DETECTION), c_int]

        self.do_nms_sort = self.lib.do_nms_sort
        self.do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

        self.predict_image = self.lib.network_predict_image
        self.predict_image.argtypes = [c_void_p, IMAGE]
        self.predict_image.restype = POINTER(c_float)

        self.net_main = load_net_custom(config_path.encode('ascii'), 
            weight_path.encode('ascii'), 0, 1)  # batch size = 1

        self.meta_main = load_meta(meta_path.encode('ascii'))

        # Create an image we reuse for each detect
        self.darknet_image = make_image(self.network_width(self.net_main), self.network_height(self.net_main), 3)

    def network_width(self, net):
        return self.lib.network_width(net)

    def network_height(self, net):
        return self.lib.network_height(net)


class YOLO():
    # Here will be the instance stored.
    __instance = None

    @staticmethod
    def get_instance(darknetlib_path, config_path, meta_path, classes_path, weight_path):
        ''' Static access method. '''
        if YOLO.__instance == None:
            YOLO(darknetlib_path, config_path, meta_path, classes_path, weight_path)
        return YOLO.__instance

    def __init__(self, darknetlib_path, config_path, meta_path, classes_path, weight_path):
        if YOLO.__instance != None:
            raise Exception('This class is a singleton!')
        else:
            YOLO.__instance = self

        self.score = 0.3
        self.iou = 0.35

        self.class_names = self._get_class(classes_path)
        self.darknet = DARKNET(darknetlib_path, config_path, meta_path, weight_path)

        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    def _get_class(self, classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def convert_bbox(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def array_to_image(self, arr):
        # need to return old values to avoid python freeing memory
        arr = arr.transpose(2,0,1)
        c = arr.shape[0]
        h = arr.shape[1]
        w = arr.shape[2]
        arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
        data = arr.ctypes.data_as(POINTER(c_float))
        im = IMAGE(w,h,c,data)
        return im, arr

    def detect(self, net, meta, im, scale, thresh=.2, nms=.35, hier_thresh=.5):
        num = c_int(0)
        pnum = pointer(num)

        self.darknet.predict_image(net, im)

        letter_box = 0
        dets = self.darknet.get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum, letter_box)

        num = pnum[0]

        if nms:
            self.darknet.do_nms_sort(dets, num, meta.classes, nms)

        res = []
        for j in range(num):
            for i in range(meta.classes):
                if dets[j].prob[i] > 0:
                    b = dets[j].bbox
                    xmin, ymin, xmax, ymax = self.convert_bbox(float(b.x), float(b.y), float(b.w), float(b.h))
                    class_name = self.class_names[i]
                    res.append(([int(xmin * scale[1]), int(ymin * scale[0]), int(xmax * scale[1]), int(ymax * scale[0])], 
                                i, class_name, dets[j].prob[i]))

        res = sorted(res, key=lambda x: -x[1])
        self.darknet.free_detections(dets, num)

        return res
    
    def fix_overlap(self, boxes):
        if len(boxes) == 0:
            return boxes

        new_boxes = [boxes[0]]
        for i in range(1, len(boxes)):
            is_new = True
            for j in range(len(new_boxes)):
                if self.bbox_iou(boxes[i][:-1], new_boxes[j][:-1]) > 0.5:
                    is_new = False
            if is_new:
                new_boxes.append(boxes[i])

        return new_boxes

    def interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                if x4 < x2:
                    return x4 - x1
                else:
                    return x2 - x1
        else:
            if x2 < x3:
                return 0
            else:
                if x4 < x2:
                    return x4 - x3
                else:
                    return x2 - x3

    def bbox_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        w1 = x1_max - x1_min
        h1 = y1_max - y1_min

        w2 = x2_max - x2_min
        h2 = y2_max - y2_min

        intersect_w = self.interval_overlap([x1_min, x1_max], [x2_min, x2_max])
        intersect_h = self.interval_overlap([y1_min, y1_max], [y2_min, y2_max])

        intersect = intersect_w * intersect_h

        union = w1 * h1 + w2 * h2 - abs(intersect)

        return float(intersect) / union
        
    def detect_image(self, image):
        height, width = image.shape[:2]
        scale = ( height / self.darknet.network_height(self.darknet.net_main), 
                width / self.darknet.network_width(self.darknet.net_main))

        image_rgb = image[...,::-1]
        image_resized = cv2.resize(image_rgb, 
                                   (self.darknet.network_width(self.darknet.net_main), 
                                   self.darknet.network_height(self.darknet.net_main)),
                                   interpolation=cv2.INTER_LINEAR)
        image_rgb, _ = self.array_to_image(image_resized)

        detections = self.detect(self.darknet.net_main, self.darknet.meta_main, image_rgb, scale, self.score, self.iou)


        detected_boxes = []

        h, w, _ = image.shape
        for i in range(len(detections)):

            c = detections[i][1]
            if c == 0:
                left, top, right, bottom = detections[i][0]
                detected_boxes.append([left, top, right, bottom, c])

        detected_boxes = self.fix_overlap(detected_boxes)

        return np.array(detected_boxes)[:,:-1]
