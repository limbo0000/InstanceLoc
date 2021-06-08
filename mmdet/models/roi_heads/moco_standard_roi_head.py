#from .bbox_heads import MultiConvFCBBoxInsClsHead
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmdet
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin


@HEADS.register_module()
class MomentumRoIPool(BaseRoIHead, BBoxTestMixin, MaskTestMixin):

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        num_extractor = bbox_roi_extractor.get('num_extractor', 1)
        ops = nn.ModuleList()
        featmap_strides = bbox_roi_extractor.get('featmap_strides', [-1])

        if num_extractor == 1:
            if 'num_extractor' in bbox_roi_extractor.keys():
                bbox_roi_extractor.pop('num_extractor')
            ops.append(build_roi_extractor(bbox_roi_extractor))
        else:
            bbox_roi_extractor_copy = bbox_roi_extractor.deepcopy()
            assert (num_extractor == len(
                bbox_roi_extractor_copy['featmap_strides']))
            out_size = bbox_roi_extractor_copy['roi_layer'].get('out_size', 7)
            out_channels = bbox_roi_extractor['out_channels']
            for i in range(num_extractor):
                if 'num_extractor' in bbox_roi_extractor_copy.keys():
                    bbox_roi_extractor_copy.pop('num_extractor')
                bbox_roi_extractor_copy['featmap_strides'] = [
                    featmap_strides[i]
                ]
                bbox_roi_extractor_copy[
                    'roi_layer']['out_size'] = out_size if not isinstance(
                        out_size, list) else out_size[i]
                if isinstance(out_channels, list):
                    bbox_roi_extractor_copy['out_channels'] = out_channels[i]
                ops.append(build_roi_extractor(bbox_roi_extractor_copy))
        self.bbox_roi_extractor = ops
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self):
        pass

    def init_assigner_sampler(self):
        pass

    def init_weights(self, pretrained):
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            for i, bbox_roi_extractor in enumerate(self.bbox_roi_extractor):
                bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()

    def forward_train(
        self,
        x,
        img_metas,
        proposal_list,
        gt_bboxes,
        gt_labels,
        box_replaced_with_gt=None,
    ):
        logits = self._bbox_forward_train(
            x,
            proposal_list,
            gt_bboxes,
            gt_labels,
            img_metas,
            box_replaced_with_gt,
        )
        return logits

    def _bbox_forward(self, x, rois, gt_rois=None, box_replaced_with_gt=None):
        if len(self.bbox_roi_extractor) > 1:
            bbox_feats = []
            for i, roi_extractor in enumerate(self.bbox_roi_extractor):
                if box_replaced_with_gt is None:
                    bbox_feat = roi_extractor((x[i], ), rois)
                else:
                    current_rois = gt_rois if box_replaced_with_gt[i] else rois
                    bbox_feat = roi_extractor((x[i], ), current_rois)
                bbox_feats.append(bbox_feat)
            bbox_feats = torch.stack(bbox_feats, 1)
            bs, num_level, dim, shape_w, shape_h = bbox_feats.shape
            bbox_feats = bbox_feats.view(bs * num_level, dim, shape_w, shape_h)
        else:
            bbox_feats = self.bbox_roi_extractor[0](
                x[:self.bbox_roi_extractor[0].num_inputs], rois)

        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        logits, _ = self.bbox_head(bbox_feats)
        return logits

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            img_metas, box_replaced_with_gt):
        rois = bbox2roi(sampling_results)
        # GT for other level feats
        if box_replaced_with_gt is not None:
            gt_rois = bbox2roi(gt_bboxes)
        else:
            gt_rois = None

        logits = self._bbox_forward(x, rois, gt_rois, box_replaced_with_gt)
        return logits
