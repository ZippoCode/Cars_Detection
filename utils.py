import argparse
from detectron2.data import build_detection_train_loader, transforms, DatasetMapper
from detectron2.engine import DefaultTrainer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        pass

    @classmethod
    def build_train_loader(cls, cfg):
        augmentation_list = [
            transforms.RandomBrightness(0.5, 1.5),
            transforms.RandomFlip(prob=0.5),
            transforms.RandomCrop("absolute", (250, 250)),
            transforms.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN,
                                          cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING),
        ]
        mapper = DatasetMapper(cfg, is_train=True, augmentations=augmentation_list)
        return build_detection_train_loader(cfg, mapper=mapper)
