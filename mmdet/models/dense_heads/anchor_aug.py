import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import build_anchor_generator, build_bbox_coder
from mmdet.ops import batched_nms
from ..builder import HEADS


@HEADS.register_module()
class AnchorAugHead(nn.Module):

    def __init__(
        self,
        anchor_generator,
        train_cfg=None,
        test_cfg=None,
    ):
        super(AnchorAugHead, self).__init__()
        self.anchor_generator = build_anchor_generator(anchor_generator)

    def init_weights(self, pretrained=None):
        pass

    def forward_train(
        self,
        x,
        img_metas,
        gt_bboxes,
        proposal_cfg=None,
    ):
        proposal_list = self.get_fixed_bboxes(
            x, img_metas, cfg=proposal_cfg, gt_bboxes=gt_bboxes)
        return proposal_list

    def get_fixed_bboxes(self,
                         featmaps,
                         img_metas,
                         cfg=None,
                         rescale=False,
                         gt_bboxes=None):
        if cfg is not None and cfg.get('generate_from_single_level',
                                       None) is not None:
            featmaps = tuple(
                [featmaps[cfg.get('generate_from_single_level', 2)]])
        num_levels = len(featmaps)
        num_imgs = len(gt_bboxes)
        device = featmaps[0].device
        featmap_sizes = [featmaps[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        proposals = torch.cat(mlvl_anchors, 0)
        if not hasattr(self, 'iou_calculator'):
            from mmdet.core.bbox.iou_calculators import build_iou_calculator
            self.iou_calculator = build_iou_calculator(
                dict(type='BboxOverlaps2D'))

        overlaps = self.iou_calculator(torch.cat(gt_bboxes, 0), proposals)
        iou_thr = cfg.get('iou_thr', 0.5)

        nms_cfg = dict(type='nms', iou_thr=cfg.nms_thr)
        pos_box_all = []
        pos_scores_all = []
        pos_idx_all = []
        for i in range(num_imgs):
            pos_proposals = proposals[overlaps[i] > iou_thr]
            if pos_proposals.shape[0] > 0:
                pass
            else:
                ranked_overlaps, ranked_idx = overlaps[i].sort(descending=True)
                pos_proposals = proposals[ranked_idx[:cfg.nms_pre]]
            scores = torch.rand(pos_proposals.shape[0], device=device)
            IDX = torch.ones(scores.shape[0], dtype=torch.long) * i
            pos_box_all.append(pos_proposals)
            pos_scores_all.append(scores)
            pos_idx_all.append(IDX)

        # cat all bboxes across batch to perform nms
        pos_box_all = torch.cat(pos_box_all, 0)
        pos_scores_all = torch.cat(pos_scores_all, 0)
        pos_idx_all = torch.cat(pos_idx_all)
        cat_det, cat_keep = batched_nms(pos_box_all, pos_scores_all,
                                        pos_idx_all, nms_cfg)
        cat_dets = []
        for i in range(num_imgs):
            cat_dets_i = cat_det[pos_idx_all[cat_keep] == i]
            cat_dets.append(cat_dets_i)
        return cat_dets
