
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

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def detector(object_predictor , image, backbone, cfg , display_height, display_width ):
    if backbone == 'yolo':
        boxes = object_predictor.detect_image(image)
        boxes = torch.as_tensor(boxes).float().cuda()
        return boxes
    else:
        outputs = object_predictor(image)
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
        return boxes

def demo(cfg, backbone):
    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

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
    
    darknetlib_path = '/home/ubuntu/hanhbd/SlowFast/detector/libdarknet.so'
    config_path = '/home/ubuntu/hanhbd/SlowFast/detector/yolov4.cfg'
    meta_path = '/home/ubuntu/hanhbd/SlowFast/detector/coco.data'
    classes_path = '/home/ubuntu/hanhbd/SlowFast/detector/coco.names'
    weight_path = '/home/ubuntu/hanhbd/SlowFast/detector/yolov4.weights'
    
    if backbone == 'yolo':
        object_predictor =  YOLO.get_instance(darknetlib_path, config_path, meta_path, classes_path, weight_path)
    else:
        dtron2_cfg_file = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_CFG
        dtron2_cfg = get_cfg()
        dtron2_cfg.merge_from_file(model_zoo.get_config_file(dtron2_cfg_file))
        dtron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = .5
        dtron2_cfg.MODEL.WEIGHTS = cfg.DEMO.DETECTRON2_OBJECT_DETECTION_MODEL_WEIGHTS
        object_predictor = DefaultPredictor(dtron2_cfg)

    with open(cfg.DEMO.LABEL_FILE_PATH) as f:
        labels = f.read().split('\n')[:-1]
    palette = np.random.randint(64, 128, (len(labels), 3)).tolist()
    count_xxx = 0
    seq_len = cfg.DATA.NUM_FRAMES*cfg.DATA.SAMPLING_RATE

    frames = []
    org_frames = []
    mid_frame = None
    pred_labels = []
    draw_imgs = []

    cap  = cv2.VideoCapture(cfg.DEMO.DATA_SOURCE)
    was_read, frame = cap.read()
    display_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    display_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter('./result/testset_fighting_05.avi',fourcc, fps, (display_width,display_height))

    while was_read :
        was_read, frame = cap.read()
        if not was_read:
            videowriter.release()
            break

        if len(frames) != seq_len:
            frame_processed = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_processed = scale(cfg.DATA.TEST_CROP_SIZE, frame_processed)
            frames.append(frame_processed)
            org_frames.append(frame)

        else:
            #predict all person box in all frame 
            start = time()
            mid_frame = org_frames[seq_len//2 - 2]
            # just draw half of number frame because we will use slide = 1/2 length of sequence
            if cfg.DETECTION.ENABLE and len(draw_imgs) == 0:
                for idx in range(seq_len//2 - 1):
                    image = org_frames[idx]
                    boxes = detector(object_predictor , image, backbone, cfg , display_height, display_width )
                    # boxes = object_predictor.detect_image(img)
                    # boxes = torch.as_tensor(boxes).float().cuda()

                    boxes = torch.cat([torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes], axis=1)
                    boxes = boxes.cpu().detach().numpy()
                    if backbone == 'yolo':
                        boxes = boxes[:, 1:]
                    else:
                        ratio = np.min(
                            [display_height, display_width]
                        ) / cfg.DATA.TEST_CROP_SIZE
                        boxes = boxes[:, 1:] * ratio

                    for box in boxes:
                        xmin, ymin, xmax, ymax = box
                        cv2.rectangle(image, (xmin, ymin), (xmax , ymax), (0, 255, 0), thickness=2)

                    draw_imgs.append(image)

            # detect box in mid frame
            if cfg.DETECTION.ENABLE:
                boxes = detector(object_predictor , mid_frame, backbone, cfg , display_height, display_width )
                boxes = torch.cat([torch.full((boxes.shape[0], 1), float(0)).cuda(), boxes], axis=1)

            inputs = torch.from_numpy(np.array(frames)).float()
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
            

            # Sample frames for the slow pathway.
            index = torch.linspace(0, fast_pathway.shape[2] - 1, 
                                    fast_pathway.shape[2]//cfg.SLOWFAST.ALPHA).long()
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
                preds = model(inputs, boxes)

            # Gather all the predictions across all the devices to perform ensemble.
            if cfg.NUM_GPUS > 1:
                preds = du.all_gather(preds)[0]
                
            # post processing
            preds = preds.cpu().detach().numpy()
            pred_masks = preds > .1
            label_ids = [np.nonzero(pred_mask)[0] for pred_mask in pred_masks]
            pred_labels = [
                [labels[label_id] for label_id in perbox_label_ids]
                for perbox_label_ids in label_ids
            ]
            print(pred_labels)
            boxes = boxes.cpu().detach().numpy()
            if backbone == 'yolo':
                boxes = boxes[:, 1:]
            else:
                ratio = np.min(
                    [display_height, display_width]
                ) / cfg.DATA.TEST_CROP_SIZE
                boxes = boxes[:, 1:] * ratio


            # draw result on mid frame
            if pred_labels and boxes.any():
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

            # append mid frame to the draw array
            draw_imgs.append(mid_frame)

            # write image to videos
            for img_ in draw_imgs:
                videowriter.write(img_)
            print("time process", (time() - start) /64 )
            # clean the buffer of frames and org_frames with slide 1/2 seq_len
            # frames = frames[seq_len//2 - 1:]
            # org_frames = org_frames[seq_len//2 - 1:]

            frames = frames[1:]
            org_frames = org_frames[1:]
            draw_imgs = draw_imgs[-1:]

            count_xxx += 1

def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training, testing, and demo pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
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
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_config()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg

def main():

    args = parse_args()
    cfg = load_config(args)

    if cfg.NUM_GPUS > 1:
        torch.multiprocessing.spawn(
            mpu.run,
            nprocs=cfg.NUM_GPUS,
            args=(
                cfg.NUM_GPUS,
                demo,
                args.init_method,
                cfg.SHARD_ID,
                cfg.NUM_SHARDS,
                cfg.DIST_BACKEND,
                cfg,
            ),
            daemon=False,
        )
    else:
        demo(cfg=cfg, backbone = args.backbone)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("forkserver")
    main()
    