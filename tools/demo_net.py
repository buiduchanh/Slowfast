import os, sys
from time import time

import numpy as np
import pandas as pd
import cv2
import torch

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import slowfast.utils.checkpoint as cu
import slowfast.utils.distributed as du
from slowfast.utils import logging
from slowfast.utils import misc
from slowfast.datasets import cv2_transform
from slowfast.models import model_builder
from slowfast.datasets.cv2_transform import scale
import math


logger = logging.get_logger(__name__)
np.random.seed(20)


def demo(cfg):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Setup logging format.
    logging.setup_logging()

    # Print config.
    logger.info("Run demo with config:")
    logger.info(cfg)
    # Build the video model and print model statistics.
    model = model_builder.build_model(cfg)
    model.eval()
    misc.log_model_info(model)

   # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        ckpt = cfg.TEST.CHECKPOINT_FILE_PATH
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        ckpt = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpoint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        ckpt = cfg.TRAIN.CHECKPOINT_FILE_PATH
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    cu.load_checkpoint(
        ckpt,
        model,
        cfg.NUM_GPUS > 1,
        None,
        inflation=False,
        convert_from_caffe2= "caffe2" in [cfg.TEST.CHECKPOINT_TYPE, cfg.TRAIN.CHECKPOINT_TYPE],
    )

    if cfg.DETECTION.ENABLE:
        # Load object detector from detectron2
        dtron2_cfg_file = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_CFG
        dtron2_cfg = get_cfg()
        dtron2_cfg.merge_from_file(model_zoo.get_config_file(dtron2_cfg_file))
        dtron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
        dtron2_cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_WEIGHTS
        object_predictor = DefaultPredictor(dtron2_cfg)
        # Load the labels of AVA dataset
        with open(cfg.DEMO.LABEL_FILE_PATH) as f:
            labels = f.read().split('\n')[:-1]
        palette = np.random.randint(64, 128, (len(labels), 3)).tolist()
        boxes = []
    else:
        # Load the labels of Kinectics-400 dataset
        labels_df = pd.read_csv(cfg.DEMO.LABEL_FILE_PATH)
        labels = labels_df['name'].values

    count_xxx = 0
    # frame_provider = VideoReader(cfg)
    seq_len = cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE
    frames = []
    org_frames = []
    mid_frame = None
    pred_labels = []


    cap  = cv2.VideoCapture(cfg.DEMO.DATA_SOURCE)
    was_read, frame = cap.read()
    display_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    display_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter('test.mp4',fourcc, fps, (display_width,display_height))
    
    while was_read :
        was_read, frame = cap.read()
        if not was_read:
            videowriter.release()

        if len(frames) != seq_len:
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
            frames.append(frame_processed)
            org_frames.append(frame)

        else:
            print(count_xxx)
            count_xxx += 1
            start = time()
            # if cfg.DETECTION.ENABLE and len(frames) == seq_len//2 - 1:
            mid_frame = org_frames[seq_len//2 - 2]
            draw_imgs = []
            for idx in range(seq_len//2 - 1):
                img = org_frames[idx]
                outputs = object_predictor(img)
                fields = outputs["instances"]._fields
                pred_classes = fields["pred_classes"]
                selection_mask = pred_classes == 0
                # acquire person boxes
                pred_classes = pred_classes[selection_mask]
                pred_boxes = fields["pred_boxes"].tensor[selection_mask]
                scores = fields["scores"][selection_mask]
                boxes = cv2_transform.scale_boxes(cfg.DATA.TEST_CROP_SIZE,
                                                    pred_boxes,
                                                    display_height,
                                                    display_width)
                boxes = torch.cat([torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes], axis=1)
                boxes = boxes.cpu().detach().numpy()
                ratio = np.min(
                    [display_height, display_width]
                ) / cfg.DATA.TEST_CROP_SIZE
                boxes = boxes[:, 1:] * ratio

                for box in boxes:
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(img, (xmin, ymin), (xmax , ymax), (0, 255, 0), thickness=2)
                    draw_imgs.append(img)
            start = time()
            if cfg.DETECTION.ENABLE:
                outputs = object_predictor(mid_frame)
                fields = outputs["instances"]._fields
                pred_classes = fields["pred_classes"]
                selection_mask = pred_classes == 0
                # acquire person boxes
                pred_classes = pred_classes[selection_mask]
                pred_boxes = fields["pred_boxes"].tensor[selection_mask]
                scores = fields["scores"][selection_mask]
                boxes = cv2_transform.scale_boxes(cfg.DATA.TEST_CROP_SIZE,
                                                    pred_boxes,
                                                    display_height,
                                                    display_width)
                boxes = torch.cat([torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes], axis=1)
                
            inputs = torch.as_tensor(frames).float()
            inputs = inputs / 255.0
            # Perform color normalization.
            inputs = inputs - torch.tensor(cfg.DATA.MEAN)
            inputs = inputs / torch.tensor(cfg.DATA.STD)
            # T H W C -> C T H W.
            inputs = inputs.permute(3, 0, 1, 2)

            # 1 C T H W.
            inputs = inputs.unsqueeze(0)

            # Sample frames for the fast pathway.
            index = torch.linspace(0, inputs.shape[2] - 1, cfg.DATA.NUM_FRAMES).long()
            fast_pathway = torch.index_select(inputs, 2, index)
            
            logger.info('fast_pathway.shape={}'.format(fast_pathway.shape))

            # Sample frames for the slow pathway.
            index = torch.linspace(0, fast_pathway.shape[2] - 1, 
                                    fast_pathway.shape[2]//cfg.SLOWFAST.ALPHA).long()
            slow_pathway = torch.index_select(fast_pathway, 2, index)
            logger.info('slow_pathway.shape={}'.format(slow_pathway.shape))
            inputs = [slow_pathway, fast_pathway]

            # Transfer the data to the current GPU device.
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)

            # Perform the forward pass.
            if cfg.DETECTION.ENABLE:
                # When there is nothing in the scene, 
                #   use a dummy variable to disable all computations below.
                if not len(boxes):
                    preds = torch.tensor([])
                else:
                    preds = model(inputs, boxes)
            else:
                preds = model(inputs)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds = du.all_gather(preds)[0]
                
            if cfg.DETECTION.ENABLE:
                # This post processing was intendedly assigned to the cpu since my laptop GPU
                #   RTX 2080 runs out of its memory, if your GPU is more powerful, I'd recommend
                #   to change this section to make CUDA does the processing.
                preds = preds.cpu().detach().numpy()
                pred_masks = preds > .1
                label_ids = [np.nonzero(pred_mask)[0] for pred_mask in pred_masks]
                pred_labels = [
                    [labels[label_id] for label_id in perbox_label_ids]
                    for perbox_label_ids in label_ids
                ]
                # I'm unsure how to detectron2 rescales boxes to image original size, so I use
                #   input boxes of slowfast and rescale back it instead, it's safer and even if boxes
                #   was not rescaled by cv2_transform.rescale_boxes, it still works.
                boxes = boxes.cpu().detach().numpy()
                ratio = np.min(
                    [display_height, display_width]
                ) / cfg.DATA.TEST_CROP_SIZE
                boxes = boxes[:, 1:] * ratio
            else:
                ## Option 1: single label inference selected from the highest probability entry.
                # label_id = preds.argmax(-1).cpu()
                # pred_label = labels[label_id]
                # Option 2: multi-label inferencing selected from probability entries > threshold
                label_ids = torch.nonzero(preds.squeeze() > .1).reshape(-1).cpu().detach().numpy()
                pred_labels = labels[label_ids]
                logger.info(pred_labels)
                if not list(pred_labels):
                    pred_labels = ['Unknown']

            # # option 1: remove the oldest frame in the buffer to make place for the new one.
            # frames.pop(0)
            # option 2: empty the buffer
            s = (time() - start) / len(frames)
            
            if cfg.DETECTION.ENABLE and pred_labels and boxes.any():
                for box, box_labels in zip(boxes.astype(int), pred_labels):
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(mid_frame, (xmin, ymin), (xmax , ymax), (0, 255, 0), thickness=2)
                
                    label_origin = box[:2]
                    for label in box_labels:
                        label_origin[-1] -= 5
                        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 2)
                        cv2.rectangle(
                            mid_frame, 
                            (label_origin[0], label_origin[1] + 5), 
                            (label_origin[0] + label_width, label_origin[1] - label_height - 5),
                            palette[labels.index(label)], -1
                        )
                        cv2.putText(
                            mid_frame, label, tuple(label_origin), 
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
                        )
                        label_origin[-1] -= label_height + 5

            draw_imgs.append(mid_frame)

            for img_ in draw_imgs:
                videowriter.write(img_)
                # cv2.imwrite('out_image/test_%04d_%04d.jpg' % (count_xxx, i) ,frame )

            frames = frames[seq_len//2 - 1:]
            org_frames = org_frames[seq_len//2 - 1:]

            # if count_xxx == 2:
            #     videowriter.release()
            #     exit()