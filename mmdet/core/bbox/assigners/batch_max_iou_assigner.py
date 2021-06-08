import numpy as np
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from .assign_result import AssignResult
from .base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class BatchMaxIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonetrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 batch_size=1,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.batch_size = batch_size
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def assign(self,
               bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               return_raw=False):
        assert (isinstance(bboxes, tuple) or isinstance(bboxes, list))
        assert (isinstance(gt_bboxes, tuple) or isinstance(gt_bboxes, list))
        assert (len(bboxes) == self.batch_size)
        assert (len(gt_bboxes) == self.batch_size)

        overlaps = self.iou_calculator(
            torch.cat(gt_bboxes, 0), torch.cat(bboxes, 0))
        overlaps = overlaps.view(self.batch_size, self.batch_size, -1)
        idx = np.arange(self.batch_size)
        overlaps = overlaps[idx, idx, :].unsqueeze(1)

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels,
                                                 return_raw)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_labels=None, return_raw=False):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_bs, num_gts, num_bboxes = overlaps.size(0), overlaps.size(
            1), overlaps.size(2)
        assert (num_bs == self.batch_size)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((
            num_bs,
            num_bboxes,
        ),
                                             -1,
                                             dtype=torch.long)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        #max_overlaps, argmax_overlaps = overlaps[:, 0, :], torch.zeros_like(overlaps[:, 0, :])
        #max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=2)
        #gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            #assigned_gt_inds[(assigned_gt_inds >= 0) & (assigned_gt_inds < self.neg_iou_thr)] = 0
            assigned_gt_inds = torch.where(
                (overlaps[:, 0, :] >= 0) &
                (overlaps[:, 0, :] < self.neg_iou_thr),
                torch.zeros_like(assigned_gt_inds), assigned_gt_inds)
            #assigned_gt_inds[(max_overlaps >= 0)
            #                 & (max_overlaps < self.neg_iou_thr)] = 0

        # 3. assign positive: above positive IoU threshold
        #pos_inds = max_overlaps >= self.pos_iou_thr
        #assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1
        assigned_gt_inds = torch.where((overlaps[:, 0, :] > self.pos_iou_thr),
                                       torch.ones_like(assigned_gt_inds),
                                       assigned_gt_inds)

        if self.match_low_quality:
            # Low-quality matching will overwirte the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.

            assert (self.gt_max_assign_all)
            # find the index of larger iou
            max_iou_idx = overlaps[:, 0, :] == gt_max_overlaps
            # find the index of iou > min_pos_iou (0.3)
            iou_over_th_idx = overlaps[:, 0, :] > self.min_pos_iou
            replaced_idx = max_iou_idx & iou_over_th_idx
            assigned_gt_inds = torch.where(replaced_idx,
                                           torch.ones_like(assigned_gt_inds),
                                           assigned_gt_inds)

        assigned_labels = None
        assigned_result = [
            AssignResult(
                num_gts,
                assigned_gt_inds[i],
                overlaps[i, 0, :],
                labels=assigned_labels) for i in range(num_bs)
        ]
        if return_raw:
            return tuple([
                num_gts, assigned_gt_inds, overlaps[:, 0, :], assigned_labels
            ])

        return assigned_result
