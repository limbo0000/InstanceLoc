import random

import cv2
import mmcv
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageFilter

from ..builder import PIPELINES


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


@PIPELINES.register_module()
class PixelAugPil(object):
    """ Apply the same augmentation as the MoCoV2.
    """

    def __init__(self, to_rgb=False):
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomApply(
                [
                    transforms.ColorJitter(0.4, 0.4, 0.4,
                                           0.1)  # not strengthened
                ],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        ]
        self.transforms = transforms.Compose(augmentation)
        self.to_rgb = to_rgb

    def __call__(self, results):
        bgr_img = results['img']

        pil_img = Image.fromarray(bgr_img[:, :, ::-1])  # BGR2RGB first
        out_pil_img = self.transforms(pil_img)
        out_rgb_img = np.array(out_pil_img)

        if self.to_rgb:
            results['img'] = out_rgb_img
        else:
            results['img'] = out_rgb_img[:, :, ::-1]  # RGB2BGR
        return results
