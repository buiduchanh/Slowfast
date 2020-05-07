
import os, sys
from time import time

import numpy as np
import pandas as pd
import cv2
import torch

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.utils import logging
from slowfast.utils import misc
from slowfast.datasets import cv2_transform
from slowfast.models import model_builder
from slowfast.datasets.cv2_transform import scale
import math

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_config

from detector.detect import YOLO
import argparse

import queue
from threading import Thread
import time

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class SlowFast():
    # Here will be the instance stored.
    __instance = None

    @staticmethod
    def get_instance(cfg, backbone):
        """ Static access method. """
        if SlowFast.__instance == None:
            print('Loading Safety Processing ... ')
            SlowFast(cfg, backbone)
        return SlowFast.__instance

    def __init__(self, cfg, backbone):
        if SlowFast.__instance != None:
            raise Exception('This class is a singleton!')
        else:
            SlowFast.__instance = self
        
        self.cfg = cfg
        self.backbone = backbone
        print(self.cfg.MODEL.ARCH)
        self.model = model_builder.build_model(self.cfg)
        self.model.eval()
        misc.log_model_info(self.model)
        
        self.checkpoint = cfg.TRAIN.CHECKPOINT_FILE_PATH

        cu.load_checkpoint(
                            self.checkpoint,
                            self.model,
                            self.cfg.NUM_GPUS > 1,
                            None,
                            inflation=False,
                            convert_from_caffe2= "caffe2" in [self.cfg.TEST.CHECKPOINT_TYPE, self.cfg.TRAIN.CHECKPOINT_TYPE],
                        )

        if self.backbone == 'yolo':
            self.object_predictor =  YOLO.get_instance(self.cfg.DEMO.YOLO_LIB, self.cfg.DEMO.YOLO_CFG, self.cfg.DEMO.YOLO_META, self.cfg.DEMO.YOLO_CLASS, self.cfg.DEMO.YOLO_WEIGHTS)
        else:
            dtron2_cfg_file = self.cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_CFG
            dtron2_cfg = get_cfg()
            dtron2_cfg.merge_from_file(model_zoo.get_config_file(dtron2_cfg_file))
            dtron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
            dtron2_cfg.MODEL.WEIGHTS = self.cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_WEIGHTS
            self.object_predictor = DefaultPredictor(dtron2_cfg)

        with open(self.cfg.DEMO.LABEL_FILE_PATH) as f:
            self.labels = f.read().split('\n')[:-1]
        self.palette = np.random.randint(64, 128, (len(self.labels), 3)).tolist()

        self.images_queue = queue.Queue()
        self.write_queue = queue.Queue()

        self.cap  = cv2.VideoCapture(self.cfg.DEMO.DATA_SOURCE)
        self.seq_len = self.cfg.DATA.NUM_FRAMES*self.cfg.DATA.SAMPLING_RATE



    def detector(self, image, display_height, display_width):

        if self.backbone == 'yolo':
            boxes = self.object_predictor.detect_image(image)
            boxes = torch.as_tensor(boxes).float().cuda()
            return boxes
        else:
            outputs = self.object_predictor(image)
            fields = outputs["instances"]._fields
            pred_classes = fields["pred_classes"]
            selection_mask = pred_classes == 0
            # acquire person boxes
            pred_classes = pred_classes[selection_mask]
            pred_boxes = fields["pred_boxes"].tensor[selection_mask]
            scores = fields["scores"][selection_mask]
            boxes = cv2_transform.scale_boxes(self.cfg.DATA.TEST_CROP_SIZE,
                                                pred_boxes,
                                                display_height,
                                                display_width)
            return boxes

    def reconnect_cam(self):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.cfg.DEMO.DATA_SOURCE)
        self.clear_queue_data()

    def clear_queue_data(self):
        self.images_queue.queue.clear()

    def read_frames(self):

        frames, org_frames, success, count_err = [] , [], True, 0

        while True:
            success, frame = self.cap.read()
            if success:
                frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_processed = scale(self.cfg.DATA.TEST_CROP_SIZE, frame_processed)
                frames.append(frame_processed)
                org_frames.append(frame)
            else:
                count_err += 1
                if count_err % 5 == 0:
                    self.reconnect_cam()
                continue

            if len(frames) % self.seq_len == 0:
                if self.images_queue.qsize() >= 2:
                    self.images_queue.get()

                self.images_queue.put([frames, org_frames])

                frames = []
                org_frames = []

    def process(self):

        while True:
            if self.images_queue.qsize() <= 0:
                time.sleep(0.5)

            frames, org_frames = self.images_queue.get()
            midframe = org_frames[self.seq_len//2 - 2]
            display_height, display_width = midframe.shape[:2]

            boxes = self.detector(midframe, display_height, display_width)
            boxes = torch.cat([torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes], axis=1)

            inputs = torch.from_numpy(np.array(frames)).float()
            inputs = inputs / 255.0
            # Perform color normalization.
            inputs = inputs - torch.tensor(self.cfg.DATA.MEAN)
            inputs = inputs / torch.tensor(self.cfg.DATA.STD)
            # T H W C -> C T H W.
            inputs = inputs.permute(3, 0, 1, 2)

            # 1 C T H W.
            inputs = inputs.unsqueeze(0)

            # Sample frames for the fast pathway.
            index = torch.linspace(0, inputs.shape[2] - 1, self.cfg.DATA.NUM_FRAMES).long()
            fast_pathway = torch.index_select(inputs, 2, index)
            
            # Sample frames for the slow pathway.
            index = torch.linspace(0, fast_pathway.shape[2] - 1, 
                                    fast_pathway.shape[2]//self.cfg.SLOWFAST.ALPHA).long()
            slow_pathway = torch.index_select(fast_pathway, 2, index)
            inputs = [slow_pathway, fast_pathway]

            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # use a dummy variable to disable all computations below.
            if not len(boxes):
                preds = torch.tensor([])
            else:
                preds = self.model(inputs, boxes)

            # post processing
            preds = preds.cpu().detach().numpy()
            pred_masks = preds > .1
            label_ids = [np.nonzero(pred_mask)[0] for pred_mask in pred_masks]
            pred_labels = [
                [self.labels[label_id] for label_id in perbox_label_ids]
                for perbox_label_ids in label_ids
            ]
            
    def run(self):
        read_frames_thread = Thread(target=self.read_frames)
        process_frames_thread = Thread(target=self.process)

        read_frames_thread.start()
        process_frames_thread.start()

        read_frames_thread.join()
        process_frames_thread.join()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training, testing, and demo pipeline."
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "--backbone",
        help="help backbone",
        default="yolo",
        type=str,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def main():

    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.cfg_file)
    slowfast = SlowFast.get_instance(cfg, args.backbone)
    slowfast.run()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
    