import cv2
import glob
import matplotlib.pyplot as plt
import os
import pandas
import argparse
import random
import torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode

# Custom importing
from parameters import *
from utils import str2bool

# Configuration Parser
parser = argparse.ArgumentParser(description="Detection cars in a set of images")
parser.add_argument('--source', dest='source_folder', default=VAL_FOLDER, type=str,
                    help=f"Folder which contains the set of folder. (DEFAULT: {VAL_FOLDER})")
parser.add_argument('--show', dest='show_result', default=True, type=str2bool,
                    help="If True plot the result, (DEFAULT: True)")

args = parser.parse_args()
folder = args.source_folder
show = args.show_result

# Configuration Model
cfg = get_cfg()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.merge_from_file(model_zoo.get_config_file(MODEL_ZOO_CONFIGURATION_FILE))
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.WEIGHTS = "model_final.pth"
predictor = DefaultPredictor(cfg)


def get_images(folder_source: str):
    image_values = []
    for filename in glob.glob(os.path.abspath(os.path.join(folder_source, '*.jpg'))):
        obj = dict()
        obj['file_name'] = filename
        image_values.append(obj)
    return image_values


DatasetCatalog.register("cars_test", lambda: get_images(folder))
MetadataCatalog.get("cars_test").set(thing_classes=["cars"])
metadata = MetadataCatalog.get("cars_test")

if __name__ == '__main__':
    results = []
    try:
        images = get_images(folder)
        count = 0
        for file in tqdm(images):
            im = cv2.imread(file['file_name'])
            outputs = predictor(im)
            boxes = outputs["instances"].get('pred_boxes')
            boxes_centers = boxes.get_centers()

            name_image = os.path.basename(file['file_name'])
            for index, bbox in enumerate(boxes):
                width = int((bbox[2] - bbox[0]).item())
                height = int((bbox[3] - bbox[1]).item())
                x_center = int((boxes_centers[index][0]).item())
                y_center = int((boxes_centers[index][1]).item())
                results.append((name_image, [width, height, x_center, y_center]))

            # Visualization
            if show:
                v = Visualizer(im[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE_BW)
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                # plt.imshow(out.get_image())
                # plt.show()
                count += 1
                cv2.imwrite(f"results/Result{count}.jpg", out.get_image()[:, :, ::-1])
    except KeyboardInterrupt:
        print("Process ended!")

    df = pandas.DataFrame(results, columns=["Name_image", "Bounding_box"])
    df = df.sort_values(by="Name_image")
    df.reset_index(drop=True, inplace=True)
    df.to_csv("results.csv", sep=",")
