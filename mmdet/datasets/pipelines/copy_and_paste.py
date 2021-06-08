import base64
import math
import random

import cv2
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class CopyAndPaste(object):

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 base_scale=(1333, 800),
                 scale=(0.2, 1.),
                 ratio=(3. / 4., 4. / 3.),
                 w_range=(100, 1000),
                 h_range=(100, 1000),
                 feed_bytes=False,
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None if feed_bytes else mmcv.FileClient(
            **self.file_client_args)
        self.base_scale = base_scale
        self.scale = scale
        self.ratio = ratio
        self.w_range = w_range  # TODO: modify the range
        self.h_range = h_range
        self.feed_bytes = feed_bytes

    @staticmethod
    def get_rescale_param(img, ratio, scale=(0.2, 1.0)):
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                xmin = random.randint(0, height - target_height)
                ymin = random.randint(0, width - target_width)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        xmin = (height - target_height) // 2
        ymin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    @staticmethod
    def get_position_param(crop, size=(256, 256)):
        W, H = size[0], size[1]
        w, h, _ = crop.shape
        sampled_w = int(np.random.randint(0, W - w, 1))
        sampled_h = int(np.random.randint(0, H - h, 1))
        return sampled_w, sampled_h

    def _load_(self, filename):
        img_bytes = self.file_client.get(filename)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def _load_bytes(self, img_str):
        img_bytes = base64.b64decode(img_str)
        img = mmcv.imfrombytes(img_bytes, flag='color', backend='cv2')
        if self.to_float32:
            img = img.astype(np.float32)
        return img

    def load_imgs(self, results):
        if self.feed_bytes:
            self.file_client = None
            base_path = results['img_bytes']
            instance_path_list = results['instance_bytes']
            load_func = self._load_bytes
        else:
            base_path = results['img_info']
            instance_path_list = results['instance_info']
            load_func = self._load_

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        base_img0 = load_func(base_path[0])
        base_img1 = load_func(base_path[1])
        base_img0 = mmcv.imresize(base_img0, self.base_scale)
        base_img1 = mmcv.imresize(base_img1, self.base_scale)

        foreground_img = load_func(instance_path_list[0])
        return base_img0, base_img1, foreground_img

    def pack(self, q_img, bboxes0, results, background_mask=None):
        # organize results0
        results0 = dict()
        results0['img_prefix'] = ''
        results0['seg_prefix'] = None
        results0['proposal_file'] = None
        results0['bbox_fields'] = []
        results0['mask_fields'] = []
        results0['seg_fields'] = []
        results0['filename'] = results['img_info']
        results0['ins_filename'] = results['instance_info']
        results0['ori_filename'] = results['img_info']
        results0['img'] = q_img
        results0['img_shape'] = q_img.shape
        results0['ori_shape'] = q_img.shape
        # Set initial values for default meta_keys
        results0['pad_shape'] = q_img.shape
        results0['scale_factor'] = 1.0
        num_channels = 1 if len(q_img.shape) < 3 else q_img.shape[2]
        results0['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results0['img_fields'] = ['img']
        results0['gt_bboxes'] = np.array(bboxes0, dtype=np.float32)
        results0['bbox_fields'].append('gt_bboxes')
        results0['gt_labels'] = results['ins_idx']
        return results0

    def get_crop(self, img, ratio):
        xmin, ymin, target_height, target_width = self.get_rescale_param(
            img, ratio)
        crop = mmcv.imcrop(
            img,
            np.array([
                ymin, xmin, ymin + target_width - 1, xmin + target_height - 1
            ]))
        return crop

    def get_scale(self):
        w = int(np.random.randint(
            self.w_range[0], self.w_range[1],
            1)) if self.w_range[0] < self.w_range[1] else self.w_range[0]
        h = int(np.random.randint(
            self.h_range[0], self.h_range[1],
            1)) if self.h_range[0] < self.h_range[1] else self.h_range[0]
        return (w, h)

    def __call__(self, results):
        # load img from given paths
        base_img0, base_img1, chosen_img = self.load_imgs(results)

        W, H, C = base_img0.shape
        BASE_SIZE = (W, H)
        q_img = base_img0.copy()
        k_img = base_img1.copy()

        num_imgs = len(chosen_img)

        # Get crop
        crop1 = self.get_crop(chosen_img, self.ratio)
        crop2 = self.get_crop(chosen_img, self.ratio)

        # Get scale
        sampled_scale1 = self.get_scale()
        sampled_scale2 = self.get_scale()

        # Rescale foreground images
        try:
            crop1 = mmcv.imrescale(crop1, sampled_scale1)
            crop2 = mmcv.imrescale(crop2, sampled_scale2)
        except:
            crop1 = mmcv.imresize(crop1, sampled_scale1)
            crop2 = mmcv.imresize(crop2, sampled_scale2)

        # Sample Location
        sampled_w, sampled_h, _ = crop1.shape
        position_w, position_h = self.get_position_param(crop1, BASE_SIZE)
        bbox0 = [
            position_h, position_w, sampled_h + position_h,
            sampled_w + position_w
        ]
        bboxes0 = [bbox0]
        q_img[position_w:position_w + sampled_w,
              position_h:position_h + sampled_h, :] = crop1

        sampled_w, sampled_h, _ = crop2.shape
        position_w, position_h = self.get_position_param(crop2, BASE_SIZE)
        bbox1 = [
            position_h, position_w, sampled_h + position_h,
            sampled_w + position_w
        ]
        bboxes1 = [bbox1]
        k_img[position_w:position_w + sampled_w,
              position_h:position_h + sampled_h, :] = crop2

        results0 = self.pack(q_img, bboxes0, results)
        results1 = self.pack(k_img, bboxes1, results)
        return [results0, results1]
