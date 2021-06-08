import csv
import io
import math
import os.path as op
import random

import cv2
import mmcv
import numpy as np
from mmcv.utils import build_from_cfg
from torch.utils.data import Dataset

from mmdet.core import eval_map, eval_recalls
from .builder import DATASETS, PIPELINES
from .pipelines import Compose


@DATASETS.register_module()
class ImageNetDataset(Dataset):
    CLASSES = None

    def __init__(
        self,
        ann_file,
        pipeline,
        preprocess,
        data_root=None,
        img_prefix='',
        seg_prefix=None,
    ):

        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix

        self.data_infos = self.load_annotations(ann_file)
        self.flag = np.ones(len(self), dtype=np.uint8)

        self.preprocess_pipeline = build_from_cfg(preprocess, PIPELINES)
        self.pipeline = Compose(pipeline)

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        txt_ptr = open(ann_file)
        txt = txt_ptr.readlines()
        all_imgs = [
            op.join(self.img_prefix,
                    each.strip().split()[0]) for each in txt
        ]
        txt_ptr.close()
        return all_imgs

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = None
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def __getitem__(self, idx):
        return self.prepare_train_img(idx)

    def prepare_train_img(self, idx):
        # Randomly choose a background image from the whole dataset
        base_idx = int(np.random.randint(0, len(self.data_infos), 1))
        base_info = self.data_infos[base_idx]

        base_idx2 = int(np.random.randint(0, len(self.data_infos), 1))
        base_info2 = self.data_infos[base_idx2]
        base_info = [base_info, base_info2]

        # Load the current image as the foreground image
        current_info = self.data_infos[idx]
        instance_info = [current_info]
        instance_idx = [idx]

        # Compose data
        results = dict(img_info=base_info, instance_info=instance_info)
        results['ins_idx'] = instance_idx

        # Copy and paste foreground image onto two background images
        results0, results1 = self.preprocess_pipeline(results)

        # Augment two synthetic images
        results = self.pipeline(results0)
        results1 = self.pipeline(results1)
        results['target_data'] = results1
        return results
