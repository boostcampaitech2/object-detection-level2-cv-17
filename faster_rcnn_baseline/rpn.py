import warnings
warnings.filterwarnings(action='ignore')

import six
import numpy as np
from torchvision.ops import nms

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import loc2bbox, bbox2loc, normal_init, unmap, bbox_iou, get_inside_index

def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    """ 
    Args:
        ratios: 비율
        anchor_scales: 스케일
    Returns: basic anchor boxes, shape=(R, 4)
        R: len(ratio) * len(anchor_scales) = anchor 개수 = 9
        4: anchor box 좌표 값
    """

    py = base_size / 2. # center y
    px = base_size / 2. # center x

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32) # anchor_box
    
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            # offset of anchor box
            anchor_base[index, 0] = py - h / 2. # y_min
            anchor_base[index, 1] = px - w / 2. # x_min
            anchor_base[index, 2] = py + h / 2. # y_max
            anchor_base[index, 3] = px + w / 2. # x_max
            
    return anchor_base # (9,4)

class ProposalCreator:
    def __init__(self, parent_model,
                 nms_thresh=0.7, # nms threshold
                 n_train_pre_nms=12000, # train시 nms 전 roi 개수
                 n_train_post_nms=2000, # train시 nms 후 roi 개수
                 n_test_pre_nms=6000,   # test시 nms 전 roi 개수
                 n_test_post_nms=300,   # test시 nms 후 roi 개수
                 min_size=16            
                 ):
        self.parent_model = parent_model # 해당 모델이 train중인지 test중인지 나타냄
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):    
        if self.parent_model.training: # train중일 때
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else: # test중일 때
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        roi = loc2bbox(anchor, loc) # anchor의 좌표값과 predicted bounding bounding box offset(y,x,h,w)를 통해 bounding box 좌표값(y_min, x_min, y_max, x_max) 생성

        # Clip predicted boxes to image.
        roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
        roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

        # min_size 보다 작은 box들은 제거
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]
        
        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN 
        order = score.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        roi = roi[order, :]
        score = score[order]

        # nms 적용
        keep = nms(
            torch.from_numpy(roi).cuda(),
            torch.from_numpy(score).cuda(),
            self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        roi = roi[keep.cpu().numpy()]
        
        return roi 

class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16, proposal_creator_params=dict(),):
        
        super(RegionProposalNetwork, self).__init__()

        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios) # 9개의 anchorbox 생성
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params) # proposal_creator_params : 해당 네트워크가 training인지 testing인지 알려준다.
        n_anchor = self.anchor_base.shape[0] # anchor 개수
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)    # kernel_size=3, stride=1, padding=0
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)  # 9*2
        self.loc = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)   # 9*4
        normal_init(self.conv1, 0, 0.01) # weight initalizer
        normal_init(self.score, 0, 0.01) # weight initalizer
        normal_init(self.loc, 0, 0.01)   # weight initalizer

    def forward(self, x, img_size, scale=1.):
        # x(feature map)
        n, _, hh, ww = x.shape

        # 전체 (h*w*9)개 anchor의 좌표값 # anchor_base:(9, 4)
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, hh, ww) 
        n_anchor = anchor.shape[0] // (hh * ww) # anchor 개수
        
        middle = F.relu(self.conv1(x))
        
        # predicted bounding box offset
        rpn_locs = self.loc(middle)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4) 

        # predicted scores for anchor (foreground or background)
        rpn_scores = self.score(middle)  
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous() 
        
        # scores for foreground
        rpn_softmax_scores = F.softmax(rpn_scores.view(n, hh, ww, n_anchor, 2), dim=4) 
        rpn_fg_scores = rpn_softmax_scores[:, :, :, :, 1].contiguous()    
        rpn_fg_scores = rpn_fg_scores.view(n, -1)    
        
        rpn_scores = rpn_scores.view(n, -1, 2) 

        # proposal생성 (ProposalCreator)
        rois = list()        # proposal의 좌표값이 있는 bounding box array
        roi_indices = list() # roi에 해당하는 image 인덱스
        for i in range(n):
            roi = self.proposal_layer(rpn_locs[i].cpu().data.numpy(),rpn_fg_scores[i].cpu().data.numpy(),anchor, img_size,scale=scale) 
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = np.concatenate(rois, axis=0)
        roi_indices = np.concatenate(roi_indices, axis=0)
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    # anchor_base는 하나의 pixel에 9개 종류의 anchor box를 나타냄
    # 이것을 enumerate시켜 전체 이미지의 pixel에 각각 9개의 anchor box를 가지게 함
    # 32x32 feature map에서는 32x32x9=9216개 anchor box가짐

    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel(), shift_x.ravel()), axis=1)

    A = anchor_base.shape[0]
    K = shift.shape[0]
    anchor = anchor_base.reshape((1, A, 4)) + \
             shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor # (9216, 4)

class AnchorTargetCreator(object):

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):

        img_H, img_W = img_size

        n_anchor = len(anchor) # 9216
        inside_index = get_inside_index(anchor, img_H, img_W) # (2272,)
        anchor = anchor[inside_index] # (2272, 4)
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious]) # (2272, 4)

        # map up to original set of anchors
        label = unmap(label, n_anchor, inside_index, fill=-1) # (9216,)
        loc = unmap(loc, n_anchor, inside_index, fill=0) # (9216, 4)

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label) 1 :positive, 0 : negative, -1 : dont care
        label = np.empty((len(inside_index),), dtype=np.int32)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = self._calc_ious(anchor, bbox, inside_index)

        label[max_ious < self.neg_iou_thresh] = 0 # 0.3

        # 가장 iou가 큰 것은 positive label
        label[gt_argmax_ious] = 1

        # positive label
        label[max_ious >= self.pos_iou_thresh] = 1 # 0.7

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)
        pos_index = np.where(label == 1)[0]
        if len(pos_index) > n_pos:
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)
            label[disable_index] = -1

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1)
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)
        argmax_ious = ious.argmax(axis=1)
        max_ious = ious[np.arange(len(inside_index)), argmax_ious]
        gt_argmax_ious = ious.argmax(axis=0)
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
        gt_argmax_ious = np.where(ious == gt_max_ious)[0]

        return argmax_ious, max_ious, gt_argmax_ious

class ProposalTargetCreator:
    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh # positive iou threshold
        self.neg_iou_thresh_hi = neg_iou_thresh_hi # negitave iou threshold = (neg_iou_thresh_hi ~ neg_iou_thresh_lo)
        self.neg_iou_thresh_lo = neg_iou_thresh_lo 

    def __call__(self, roi, bbox, label,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        n_bbox, _ = bbox.shape

        roi = np.concatenate((roi, bbox), axis=0)

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio) # positive image 갯수 = 32
        iou = bbox_iou(roi, bbox) # RoI와 bounding box IoU
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        gt_roi_label = label[gt_assignment] + 1 # class label [0, n_fg_class - 1] -> [1, n_fg_class].

        # positive sample 선택 (>= pos_iou_thresh IoU)
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)

        # Negative sample 선택 [neg_iou_thresh_lo, neg_iou_thresh_hi)
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0]
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index)
        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative sample의 label = 0
        sample_roi = roi[keep_index] # (128, 4)

        # sample roi와 gt_bbox를 이용해 bbox regression에서 regression해야할 ground truth loc값(t_x, t_y, t_w, t_h) 계산
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]]) # (128, 4)
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)) / np.array(loc_normalize_std, np.float32))

        return sample_roi, gt_roi_loc, gt_roi_label
