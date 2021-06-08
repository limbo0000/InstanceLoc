import time

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.cnn import normal_init

from mmdet.core.bbox.iou_calculators import build_iou_calculator
# from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from ..losses import accuracy
from .base import BaseDetector


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


@DETECTORS.register_module()
class InsLocFPN(BaseDetector):

    def __init__(
        self,
        backbone,
        neck=None,
        rpn_head=None,
        roi_head=None,
        pool_with_gt=[True, True],
        shuffle_data=['img'],
        drop_rpn_k=True,
        momentum_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        num_levels=4,
        level_loss_weights=[1.0, 1.0, 1.0, 1.0],
        box_replaced_with_gt=None,
        num_pos_per_instance=1,
        num_region_neg=-1,
    ):
        super(InsLocFPN, self).__init__()
        #nn.Module.__init__(self)
        self.backbone = build_backbone(backbone)
        self.backbone_k = build_backbone(backbone)
        self.shuffle_data = shuffle_data
        shuffle_set = set(['idx', 'bbox', 'img'])
        assert (shuffle_set.intersection(
            set(shuffle_data)) == set(shuffle_data))
        self.drop_rpn_k = drop_rpn_k
        self.num_levels = num_levels
        self.level_loss_weights = level_loss_weights
        self.num_pos_per_instance = num_pos_per_instance
        self.box_replaced_with_gt = box_replaced_with_gt
        self.create_neck(neck)
        self.create_rpn_head(rpn_head, train_cfg, test_cfg)
        self.create_roi_head(roi_head, train_cfg, test_cfg)
        self.num_region_neg = num_region_neg

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pool_with_gt = pool_with_gt
        self.init_weights(pretrained=pretrained)
        self.create_momentum(momentum_cfg)

    def create_momentum(
        self,
        momentum_cfg,
    ):
        self.momentum_cfg = Config(momentum_cfg)
        self.K = self.momentum_cfg.K
        self.m = self.momentum_cfg.m
        self.T = self.momentum_cfg.T
        assert (len(self.level_loss_weights) == self.num_levels)

        if self.momentum_cfg is not None:
            queues = []
            for i in range(self.num_levels):
                # Create queue
                self.register_buffer(
                    "queue",
                    torch.randn(self.momentum_cfg.dim, self.momentum_cfg.K))
                self.queue = nn.functional.normalize(self.queue, dim=0)
                queues.append(self.queue)
            self.register_buffer("queues", torch.stack(queues, 0))
            #self.queues = torch.stack(queues)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Create momentum net
        self.generate_momentum_net()

    def create_roi_head(self, roi_head, train_cfg, test_cfg):
        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            self.roi_head = build_head(roi_head)
            roi_head_copy = roi_head.deepcopy()
            self.roi_head_k = build_head(roi_head_copy)
        else:
            self.roi_head = None
            self.roi_head_k = None

    def create_rpn_head(self, rpn_head, train_cfg, test_cfg):
        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
            self.rpn_head_k = build_head(
                rpn_head_) if not self.drop_rpn_k else None
        else:
            self.rpn_head = None
            self.rpn_head_k = None

    def create_neck(self, neck):
        if neck is not None:
            self.neck = build_neck(neck)
            self.neck_k = build_neck(neck)
        else:
            self.neck = None
            self.neck_k = None

    def _init_momentum_net(self, net_q, net_k):
        if net_q is not None and net_k is not None:
            for param_q, param_k in zip(net_q.parameters(),
                                        net_k.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient
        else:
            pass

    def generate_momentum_net(self):
        self._init_momentum_net(self.backbone, self.backbone_k)
        #if self.with_neck:
        self._init_momentum_net(self.neck, self.neck_k)
        #if self.with_rpn and not self.drop_rpn_k:
        self._init_momentum_net(self.rpn_head, self.rpn_head_k)
        #if self.with_roi_head:
        self._init_momentum_net(self.roi_head, self.roi_head_k)

    @torch.no_grad()
    def _momentum_update_net(self, net_q, net_k):
        """
        Momentum update of the key net
        """
        if net_q is not None and net_k is not None:
            for param_q, param_k in zip(net_q.parameters(),
                                        net_k.parameters()):
                param_k.data = param_k.data * self.m + param_q.data * (1. -
                                                                       self.m)
        else:
            pass

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        self._momentum_update_net(self.backbone, self.backbone_k)
        #if self.with_neck:
        self._momentum_update_net(self.neck, self.neck_k)
        #if self.with_rpn and not self.drop_rpn_k:
        self._momentum_update_net(self.rpn_head, self.rpn_head_k)
        #if self.with_roi_head:
        self._momentum_update_net(self.roi_head, self.roi_head_k)

    def init_weights(self, pretrained=None):
        super(InsLocFPN, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        #if self.with_neck:
        if self.neck is not None:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        #if self.with_rpn:
        if self.rpn_head is not None:
            self.rpn_head.init_weights()
        #if self.with_roi_head:
        if self.roi_head is not None:
            self.roi_head.init_weights(pretrained)

    def fwd(
        self,
        img,
        img_metas,
        gt_labels,
        gt_bboxes,
        backbone,
        roi_head,
        neck=None,
        pool_with_gt=True,
        rpn_head=None,
        query_encoder=False,
    ):

        losses = dict()
        x = backbone(img)

        if neck is not None:
            x = neck(x)

        if pool_with_gt:
            proposal_list = gt_bboxes
        else:
            proposal_cfg = self.train_cfg.get('rpn_proposal', None)
            proposal_list = rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                proposal_cfg=proposal_cfg,
            )
            # implement your strategy for getting 2nd stage bboxes
            proposal_list = self.get_stage2_bboxes(proposal_list, gt_bboxes)

        if self.box_replaced_with_gt is not None and query_encoder:
            box_replaced_with_gt = self.box_replaced_with_gt
        else:
            box_replaced_with_gt = None

        logits = roi_head.forward_train(x, img_metas, proposal_list, gt_bboxes,
                                        gt_labels, box_replaced_with_gt)
        return logits

    def get_stage2_bboxes(self, proposal_list, gt_bboxes):
        assert (len(proposal_list) == len(gt_bboxes))
        num_imgs = len(gt_bboxes)
        # currently, we random choose one proposal for stage 2
        outs = []
        for i in range(num_imgs):
            proposal_i = proposal_list[i]
            idx = np.random.randint(0, proposal_i.shape[0], 1)
            outs.append(proposal_i[int(idx), :4][None, :])
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      binary_masks=None,
                      proposals=None,
                      shifted_bboxes=None,
                      target_data=None,
                      **kwargs):
        losses = dict()
        batch_size, num_ins_per_img, _ = gt_bboxes.shape
        assert (num_ins_per_img == gt_labels.shape[-1])
        assert (num_ins_per_img == 1)

        if not isinstance(gt_bboxes, list):
            gt_bboxes = [each for each in gt_bboxes]

        if shifted_bboxes is not None:
            if not isinstance(shifted_bboxes, list):
                shifted_bboxes = [each for each in shifted_bboxes]

        if not isinstance(gt_labels, list):
            gt_labels = [each for each in gt_labels]

        # compute query features
        logits_q = self.fwd(
            img=img,
            img_metas=img_metas,
            gt_labels=gt_labels,
            gt_bboxes=gt_bboxes,
            backbone=self.backbone,
            roi_head=self.roi_head,
            neck=self.neck,
            pool_with_gt=self.pool_with_gt[0],
            rpn_head=self.rpn_head,
            query_encoder=True)
        logits_q = nn.functional.normalize(logits_q, dim=1)

        num_feat_levels = logits_q.shape[0] // (
            batch_size * self.num_pos_per_instance)
        assert (self.num_levels == num_feat_levels)
        logits_q = logits_q.view(batch_size * self.num_pos_per_instance,
                                 num_feat_levels, -1)

        img_k = target_data['img']
        img_metas_k = target_data['img_metas']
        gt_bboxes_k = target_data['gt_bboxes']
        gt_labels_k = target_data['gt_labels']

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            shuffle_idx = 'idx' in self.shuffle_data
            shuffle_bbox = 'bbox' in self.shuffle_data
            img_k, shuffle_gt_labels_k, shuffle_gt_bboxes_k, idx_unshuffle = self._batch_shuffle_ddp(
                img_k, gt_labels_k if shuffle_idx else None,
                gt_bboxes_k if shuffle_bbox else None)

            if shuffle_idx:
                gt_labels_k = [each for each in shuffle_gt_labels_k]
            else:
                gt_labels_k = [each for each in gt_labels_k]

            if shuffle_bbox:
                gt_bboxes_k = [each for each in shuffle_gt_bboxes_k]
            else:
                gt_bboxes_k = [each for each in gt_bboxes_k]

            logits_k = self.fwd(
                img=img_k,
                img_metas=img_metas_k,
                gt_labels=gt_labels_k,
                gt_bboxes=gt_bboxes_k,
                backbone=self.backbone_k,
                roi_head=self.roi_head_k,
                neck=self.neck_k,
                pool_with_gt=self.pool_with_gt[1],
                rpn_head=self.rpn_head_k,
                query_encoder=False)
            logits_k = nn.functional.normalize(logits_k, dim=1)
            logits_k = logits_k.view(batch_size, num_feat_levels, -1)
            # undo shuffle
            logits_k = self._batch_unshuffle_ddp(logits_k, idx_unshuffle).view(
                batch_size, num_feat_levels, -1)
            if self.num_pos_per_instance > 1:
                batch_size_k = logits_k.shape[0]
                batch_size_q = logits_q.shape[0]
                assert (batch_size_q //
                        batch_size_k == self.num_pos_per_instance)
                repeated_logits_k = torch.repeat_interleave(
                    logits_k.unsqueeze(1), self.num_pos_per_instance,
                    1).view(-1, num_feat_levels, logits_q.shape[-1])
            else:
                repeated_logits_k = logits_k

        for level_idx in range(num_feat_levels):
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [
                logits_q[:, level_idx, :], repeated_logits_k[:, level_idx, :]
            ]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [
                logits_q[:, level_idx, :],
                self.queues[level_idx].clone().detach()
            ])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

            # ce loss
            loss_cls = self.level_loss_weights[
                level_idx] * nn.functional.cross_entropy(logits, labels)
            acc = accuracy(logits, labels)

            losses.update({
                f'loss_cls_{level_idx}': loss_cls,
                f'acc_{level_idx}': acc
            })
            #loss_cls=loss_cls, acc=acc)

        # dequeue and enqueue
        self._dequeue_and_enqueue(logits_k)

        return losses

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        self.queues[:, :, ptr:ptr + batch_size] = keys.permute(1, 2, 0)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x, y=None, z=None):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        if y is not None:
            y_gather = concat_all_gather(y)
            assert (y_gather.shape[0] == x_gather.shape[0])
        else:
            y_gather = None

        if z is not None:
            z_gather = concat_all_gather(z)
            assert (z_gather.shape[0] == x_gather.shape[0])
        else:
            z_gather = None
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], None if y_gather is None else y_gather[
            idx_this], None if z_gather is None else z_gather[
                idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
