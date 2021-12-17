# -*- coding: utf-8 -*-

from detectron2.config import CfgNode as CN

def add_swint_config(cfg):
    # SwinT backbone
    cfg.MODEL.SWINT = CN()
    cfg.MODEL.SWINT.EMBED_DIM = 192
    cfg.MODEL.SWINT.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
    cfg.MODEL.SWINT.DEPTHS = [ 2, 2, 18, 2 ]
    cfg.MODEL.SWINT.NUM_HEADS = [ 6, 12, 24, 48 ]
    cfg.MODEL.SWINT.WINDOW_SIZE = 12
    cfg.MODEL.SWINT.MLP_RATIO = 4
    cfg.MODEL.SWINT.DROP_PATH_RATE = 0.2
    cfg.MODEL.SWINT.APE = False
    cfg.MODEL.BACKBONE.FREEZE_AT = -1

    # addation
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"

    # RCNN
    cfg.DATASETS.TRAIN = (f"sartorius_train",)
    cfg.DATASETS.TEST = (f"sartorius_val",)
    cfg.INPUT.MIN_SIZE_TRAIN = (440, 480, 520, 560, 580, 620)
    cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 320
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.INPUT.MASK_FORMAT='bitmask'

    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000

    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [128,128,128]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [13.235,13.235,13.235]

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2




