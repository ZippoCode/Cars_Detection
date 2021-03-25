import os
import sys
import json
import cv2
import torch
import random
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer

# Custom importing
from parameters import *


def get_cars_dictionaries(directory: str):
    annotation_filename = os.path.join(directory, NAME_ANNOTATION_FILE)
    if not os.path.isfile(annotation_filename):
        print("[ERROR] File not found!")
        return []
    with open(annotation_filename) as file:
        image_annotations = json.load(file)
    image_annotations = image_annotations['annotations']
    dataset_dictionary = []
    for idx, value in enumerate(image_annotations):
        record_found = False
        if value['category_id'] == 3 and 'counts' not in value['segmentation']:
            image_filename = os.path.abspath(os.path.join(directory, value['file_name']))
            for record in dataset_dictionary:
                if image_filename in record.values():
                    record['annotations'].append({
                        "bbox": [int(v) for v in value['bbox']],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": value['segmentation'],
                        "category_id": 0
                    })
                    record_found = True
                    break
            if not record_found:
                height, width = cv2.imread(filename=image_filename).shape[:2]
                record = dict()
                record['file_name'] = image_filename
                record['image_id'] = value['image_id']
                record["height"] = height
                record['width'] = width
                record['iscrowd'] = value['iscrowd']
                record['annotations'] = [{
                    "bbox": [int(v) for v in value['bbox']],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": value['segmentation'],
                    "category_id": 0
                }]
                dataset_dictionary.append(record)
    return dataset_dictionary


for d in ["train", "val"]:
    DatasetCatalog.register("cars_" + d, lambda d=d: get_cars_dictionaries("cars/" + d))
    MetadataCatalog.get("cars_" + d).set(thing_classes=["cars"])
cars_metadata = MetadataCatalog.get("cars_train")

# Training
cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cars_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 300
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()