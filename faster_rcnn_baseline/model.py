import warnings
warnings.filterwarnings(action='ignore')

import os
import numpy as np

from torchvision.models import vgg16
from torchvision.ops import RoIPool
from torchvision.ops import nms

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import namedtuple

from utils import scalar, totensor, tonumpy
from rpn import AnchorTargetCreator, ProposalTargetCreator
from torchnet.meter import ConfusionMeter, AverageValueMeter

from utils import loc2bbox, normal_init, totensor, tonumpy
from rpn import RegionProposalNetwork


use_drop = False

LossTuple = namedtuple('LossTuple', ['rpn_loc_loss', 'rpn_cls_loss',
                                     'roi_loc_loss', 'roi_cls_loss',
                                     'total_loss'])

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    # Localization loss 구할 때는 positive example에 대해서만 계산
    in_weight = torch.zeros(gt_loc.shape).cuda()
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    loc_loss /= ((gt_label >= 0).sum().float())
    return loc_loss

def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    model = vgg16(pretrained=True)
    # model = vgg16()
    # model.load_state_dict(torch.load('./checkpoints/vgg16-397923af.pth'))
    
    features = list(model.features)[:30]
    classifier = model.classifier

    classifier = list(classifier)
    del classifier[6]
    if not use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)

    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False

    return nn.Sequential(*features), classifier

class VGG16RoIHead(nn.Module):
    """
    Faster R-CNN head
    RoI pool 후에 classifier, regressior 통과
    """

    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()

        self.classifier = classifier  
        self.cls_loc = nn.Linear(4096, n_class * 4) # bounding box regressor
        self.score = nn.Linear(4096, n_class) # Classifier

        normal_init(self.cls_loc, 0, 0.001)  # weight initialize
        normal_init(self.score, 0, 0.01)     # weight initialize

        self.n_class = n_class # 배경 포함한 class 수
        self.roi_size = roi_size # RoI-pooling 후 feature map의  높이, 너비
        self.spatial_scale = spatial_scale # roi resize scale
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        # in case roi_indices is  ndarray
        roi_indices = totensor(roi_indices).float()
        rois = totensor(rois).float()
        indices_and_rois = torch.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous() 

        # 각 이미지 roi pooling 
        pool = self.roi(x, indices_and_rois) 
        # flatten 
        pool = pool.view(pool.size(0), -1)
        # fully connected
        fc7 = self.classifier(pool)
        # regression 
        roi_cls_locs = self.cls_loc(fc7)
        # softmax
        roi_scores = self.score(fc7)

        
        return roi_cls_locs, roi_scores

def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)
    return new_f

class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, head, learning_rate, weight_decay,
                loc_normalize_mean = (0., 0., 0., 0.),
                loc_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor  # extractor : vgg
        self.rpn = rpn              # rpn : region proposal network
        self.head = head            # head : RoiHead
        self.lr = learning_rate
        self.weight_decay = weight_decay

        # mean and std
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset()

    @property
    def n_class(self): # 최종 class 개수 (배경 포함)
        return self.head.n_class
        
    # predict 시 사용하는 forward
    # train 시 FasterRCNNTrainer을 사용하여 FasterRcnn에 있는 extractor, rpn, head를 모듈별로 불러와서 forward
    def forward(self, x, scale=1.):
        img_size = x.shape[2:]

        h = self.extractor(x) # extractor 통과
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(h, img_size, scale) # rpn 통과
        roi_cls_locs, roi_scores = self.head(h, rois, roi_indices) # head 통과
        return roi_cls_locs, roi_scores, rois, roi_indices 

    def use_preset(self): # prediction 과정 쓰이는 threshold 정의
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = nms(cls_bbox_l, prob_l,self.nms_thresh)
            bbox.append(cls_bbox_l[keep].cpu().numpy())
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep].cpu().numpy())
        
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    @nograd
    def predict(self, imgs,sizes=None):
        """
        이미지에서 객체 검출
        Input : images
        Output : bboxes, labels, scores
        """
        self.eval()
        prepared_imgs = imgs
                
        bboxes = list()
        labels = list()
        scores = list()
        for img, size in zip(prepared_imgs, sizes):
            img = totensor(img[None]).float()
            scale = img.shape[3] / size[1]
            roi_cls_loc, roi_scores, rois, _ = self(img, scale=scale) # self = FasterRCNN
            # We are assuming that batch size is 1.
            roi_score = roi_scores.data
            roi_cls_loc = roi_cls_loc.data
            roi = totensor(rois) / scale

            # Convert predictions to bounding boxes in image coordinates.
            # Bounding boxes are scaled to the scale of the input images.
            mean = torch.Tensor(self.loc_normalize_mean).cuda(). repeat(self.n_class)[None]
            std = torch.Tensor(self.loc_normalize_std).cuda(). repeat(self.n_class)[None]

            roi_cls_loc = (roi_cls_loc * std + mean)
            roi_cls_loc = roi_cls_loc.view(-1, self.n_class, 4)
            roi = roi.view(-1, 1, 4).expand_as(roi_cls_loc)
            cls_bbox = loc2bbox(tonumpy(roi).reshape((-1, 4)),tonumpy(roi_cls_loc).reshape((-1, 4)))
            cls_bbox = totensor(cls_bbox)
            cls_bbox = cls_bbox.view(-1, self.n_class * 4)
            # clip bounding box
            cls_bbox[:, 0::2] = (cls_bbox[:, 0::2]).clamp(min=0, max=size[0])
            cls_bbox[:, 1::2] = (cls_bbox[:, 1::2]).clamp(min=0, max=size[1])

            prob = (F.softmax(totensor(roi_score), dim=1))

            bbox, label, score = self._suppress(cls_bbox, prob)
            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        self.use_preset()
        self.train()
        return bboxes, labels, scores

    def get_optimizer(self):
        '''
        Optimizer 선언
        '''
        lr = self.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': self.weight_decay}]
        self.optimizer = torch.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer

class FasterRCNNVGG16(FasterRCNN):

    feat_stride = 16  # downsample 16x for output of conv5 in vgg16

    def __init__(self, learning_rate, weight_decay, n_fg_class=10, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32] ): # n_fg_class : 배경포함 하지 않은 class 개수        
        extractor, classifier = decom_vgg16()
        
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )
        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
            learning_rate,
            weight_decay,
        )

class FasterRCNNTrainer(nn.Module):

    def __init__(self, faster_rcnn, rpn_sigma, roi_sigma):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

        # training 상태 보여주는 지표
        self.rpn_cm = ConfusionMeter(2) # confusion matrix for classification
        self.roi_cm = ConfusionMeter(11)  # confusion matrix for classification
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, imgs, bboxes, labels, scale):
        n = bboxes.shape[0]
        
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        # VGG (features extractor)
        features = self.faster_rcnn.extractor(imgs)
        
        # RPN (region proposal)
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        """
        sample roi =  rpn에서 nms 거친 2000개의 roi들 중 positive/negative 비율 고려해 최종 sampling한 roi
        """
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            tonumpy(bbox),
            tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        
        # NOTE it's all zero because now it only support for batch=1 now
        # Faster R-CNN head (prediction head)
        sample_roi_index = torch.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(features,sample_roi,sample_roi_index) 

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(tonumpy(bbox),anchor,img_size) 
        gt_rpn_label = totensor(gt_rpn_label).long() 
        gt_rpn_loc = totensor(gt_rpn_loc) 
        
        # rpn bounding box regression loss
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_loc,gt_rpn_loc,gt_rpn_label.data,self.rpn_sigma)
        # rpn classification loss
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = tonumpy(rpn_score)[tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0] 
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long().cuda(), \
                              totensor(gt_roi_label).long()]
        gt_roi_label = totensor(gt_roi_label).long() 
        gt_roi_loc = totensor(gt_roi_loc) 

        # faster rcnn bounding box regression loss
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        # faster rcnn classification loss
        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())
        
        self.roi_cm.add(totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)] # total_loss == sum(losses)

        return LossTuple(*losses)
    
    # training
    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses
    
    # checkpoint 만들기
    def save(self, save_optimizer=False, save_path=None):
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            save_path = './checkpoints/faster_rcnn_scratch_checkpoints.pth'

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path
    
    # checkpoint load
    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

