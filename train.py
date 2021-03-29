import os
import json
import cv2
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode

# Custom importing
from parameters import *


def get_cars_dictionaries(directory: str):
    annotation_filename = os.path.join(directory, NAME_ANNOTATION_FILE)
    if not os.path.isfile(annotation_filename):
        print(f"[ERROR] File {annotation_filename} not found!")
        return []
    with open(annotation_filename) as file:
        image_annotations = json.load(file)
    image_annotations = image_annotations['annotations']
    dataset_dictionary = []
    for idx, value in enumerate(image_annotations):
        if value['category_id'] == 3 and 'counts' not in value['segmentation']:
            image_filename = os.path.abspath(os.path.join(directory, value['file_name']))
            record = next((value for value in dataset_dictionary if value['file_name'] == image_filename), None)
            if record is None:
                height, width = cv2.imread(filename=image_filename).shape[:2]
                record = dict()
                record['file_name'] = image_filename
                record['image_id'] = value['image_id']
                record["height"] = height
                record['width'] = width
                record['iscrowd'] = value['iscrowd']
                record['annotations'] = [{
                    "bbox": value['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": value['segmentation'],
                    "category_id": 0
                }]
                dataset_dictionary.append(record)
            else:
                record["annotations"].append({
                    "bbox": value['bbox'],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "segmentation": value['segmentation'],
                    "category_id": 0
                })
    return dataset_dictionary


for d in ["train", "val"]:
    DatasetCatalog.register("cars_" + d, lambda d=d: get_cars_dictionaries("cars/" + d))
    MetadataCatalog.get("cars_" + d).set(thing_classes=["cars"])
cars_metadata = MetadataCatalog.get("cars_train")

# Training
cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.merge_from_file(model_zoo.get_config_file(MODEL_ZOO_CONFIGURATION_FILE))
cfg.DATASETS.TRAIN = ("cars_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ZOO_CONFIGURATION_FILE)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
try:
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
except KeyboardInterrupt:
    print("Training stopped!")
