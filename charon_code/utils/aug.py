# -*- coding: utf-8 -*-

import cv2
import copy
import pycocotools.mask as mask_util
import numpy as np
from albumentations import *
import copy
import numpy as np
from PIL import Image
from detectron2.data import MetadataCatalog, DatasetCatalog

import detectron2.data.transforms as T
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import detection_utils as utils

class CopyPasteAugmentator:
  """Copy-paste cells from another image in the dataset
  """
  def __init__(self, d2_dataset,
               paste_same_class=True,
               paste_density=[0.3, 0.6],
               filter_area_thresh=0.1,
               p=1.0):
    
    self.data = d2_dataset
    self.n_samples = len(d2_dataset)
    self.paste_same_class = paste_same_class
    if paste_same_class:
      self.cls_indices = [
        [
          i for i, item in enumerate(d2_dataset)
          if item['annotations'][0]['category_id'] == cls_index
        ]
        for cls_index in range(3)
      ]
    self.filter_area_thresh = filter_area_thresh
    self.paste_density = paste_density
    self.p = p
  
  def __call__(self, dataset_dict):
    # print(dataset_dict)
    orig_img = cv2.imread(dataset_dict["file_name"])
    if 'LIVECell_dataset_2021' in dataset_dict["file_name"]:
      return orig_img, dataset_dict

    if np.random.uniform() < self.p:

      # Choose a sample to copy-paste from
      if self.paste_same_class:
        cls_id = dataset_dict['annotations'][0]['category_id']
        random_idx = np.random.randint(0, len(self.cls_indices[cls_id]))
        random_ds_dict = self.data[self.cls_indices[cls_id][random_idx]]
      else:
        random_idx = np.random.randint(0, self.n_samples)
        random_ds_dict = self.data[random_idx]
      # Load chosen sample
      random_img = cv2.imread(random_ds_dict['file_name'])
      if isinstance(self.paste_density, list):
        paste_density = np.random.uniform(self.paste_density[0], self.paste_density[1])
      else:
        paste_density = self.paste_density

      # Selection indices
      selected_cell_ids = np.random.choice(
        len(random_ds_dict['annotations']),
        size=round(paste_density * len(random_ds_dict['annotations'])),
        replace=False)
      # Select annotations (we deepcopy only selected ones, not the whole dict)
      selected_annos = [copy.deepcopy(random_ds_dict['annotations'][i])
                        for i in selected_cell_ids]
      copypaste_mask = mask_util.decode(selected_annos[0]['segmentation']).astype(np.bool)
      for anno in selected_annos[1:]:
        copypaste_mask |= mask_util.decode(anno['segmentation']).astype(np.bool)
      
      # Copy cells over
      neg_mask = ~copypaste_mask
      filtered_annos = []
      for anno in dataset_dict['annotations']:
        mask = mask_util.decode(anno['segmentation']).astype(np.bool)
        ocluded_mask = (mask & neg_mask)
        if (round(self.filter_area_thresh * mask.sum()) < ocluded_mask.sum()):
          anno['segmentation'] = mask_util.encode(np.asfortranarray(ocluded_mask))
          filtered_annos.append(anno)

      # Form output
      orig_img[copypaste_mask] = random_img[copypaste_mask]
      dataset_dict['annotations'] = filtered_annos + selected_annos

    return orig_img, dataset_dict

class MyMapper:
  def __init__(self, cfg):
    train_ds = DatasetCatalog.get("sartorius_train")

    # self.copy_generator = CopyPasteAugmentator(train_ds,
    #                                         paste_same_class=True,
    #                                                       paste_density=0.1,
    #                                                       filter_area_thresh=0.1,
    #                                                       p=0.3)
    self.min_size_train = cfg.INPUT.MIN_SIZE_TRAIN
    self.max_size_train = cfg.INPUT.MAX_SIZE_TRAIN

    # See "Data Augmentation" tutorial for details usage
    # 几何增强
    self.augs = T.AugmentationList([
      T.ResizeShortestEdge(
          short_edge_length=self.min_size_train,
          max_size=self.max_size_train,
          sample_style=cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
          , interp=Image.BICUBIC), #for fine-tune
      
      T.RandomFlip(prob=0.5, vertical=True, horizontal=False), #for fine-tune
      T.RandomFlip(prob=0.5, vertical=False, horizontal=True), #for fine-tune
      T.RandomRotation(angle=[0,90,180,270],sample_style="choice"), #for fine-tune
      #T.RandomApply(T.RandomExtent((0.8,1.2), (0.1, 0.1)),0.7),
      T.RandomApply(T.RandomRotation(angle=[-15,15],sample_style="range"),0.7),
    ]) 

    print(f"[MyMapper] Augmentations used in training!")

  # Show how to implement a minimal mapper, similar to the default DatasetMapper
  def __call__(self, dataset_dict):
      dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
      # can use other ways to read image      
      #image, dataset_dict = self.copy_generator(dataset_dict)   
      image = utils.read_image(dataset_dict["file_name"], format="BGR")
      auginput = T.AugInput(image)
      transform = self.augs(auginput)
      image = auginput.image
      #image = self.albu_transform(image=image)["image"]
      
      # bboxes = np.array([obj["bbox"] for obj in dataset_dict["annotations"]], dtype=np.float32)
      # category_id = np.arange(len(dataset_dict["annotations"]))

      image_shape = image.shape[:2]  #  h,w,b
      annos = [
          utils.transform_instance_annotations(annotation, transform, image_shape)
          for annotation in dataset_dict.pop("annotations")
      ]

      dataset_dict['image'] = torch.as_tensor(image.transpose(2, 0, 1).astype(np.float32))
      instances = utils.annotations_to_instances(
          annos, image_shape, mask_format="bitmask")

      instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
      dataset_dict["instances"] = utils.filter_empty_instances(instances)
      
      return dataset_dict

