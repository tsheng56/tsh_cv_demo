
# Loss functions
import numpy as np
import torch
import torch.nn as nn

from detect.utils.general import bbox_iou
from detect.utils.torch_utils import is_parallel
from detect.loss.yolo_v3 import compute_loss_v3
from detect.loss.yolo_v5 import compute_loss_v5
from detect.loss.yolo_v4 import compute_loss_v4


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss

class ComputeLoss:
    # Compute losses
    def __init__(self, model, cfg_train):
        super(ComputeLoss, self).__init__()

        self.version = cfg_train['loss_type']
        self.autobalance = cfg_train['autobalance']
        device = next(model.parameters()).device  # get model device
        self.hyp = model.hyp  # hyperparameters

        try:
            det = model.module.model[-1] if is_parallel(model) else model.model[-1]  # Detect() module
        except:
            det = model.module.model.detect if is_parallel(model) else model.model.detect  # Detect() module
        nc = det.nc

        if self.version != 'v4':

            # Define criteria
            BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['cls_pw']], device=device))
            BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.hyp['obj_pw']], device=device))

            # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
            self.cp, self.cn = smooth_BCE(eps=self.hyp.get('label_smoothing', 0.0))  # positive, negative BCE targets

            # Focal loss
            g = self.hyp['fl_gamma']  # focal loss gamma
            if g > 0:
                BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

            
            self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])  # P3-P7
            self.ssi = list(det.stride).index(16) if self.autobalance else 0  # stride 16 index
            self.BCEcls, self.BCEobj, self.gr = BCEcls, BCEobj, model.gr
            for k in 'na', 'nc', 'nl', 'anchors':
                setattr(self, k, getattr(det, k))
        else:

            anchors = []
            [anchors.extend(ans) for ans in cfg_train['anchors']]
            anchors = np.array(anchors).reshape(-1, 2).tolist()

            anchors_mask = []
            ans_index = 0
            for n, ans in enumerate(cfg_train['anchors']):
                anchors_mask.append([])
                for an in ans[::2]:
                    anchors_mask[n].append(ans_index)
                    ans_index += 1

            self.net_w, self.net_h = cfg_train['width'], cfg_train['height']
            self.num_classes = nc
            
            self.anchors_org = anchors
            self.scaled_anchors_org = [(1.0 * a_w / self.net_w, 1.0 * a_h / self.net_h) for a_w, a_h in self.anchors_org]
            self.anchors_masks = anchors_mask
            self.layer_stride = cfg_train['stride']
            self.use_all_anchors = cfg_train['use_all_anchors']
            
            self.object_scale = cfg_train['object_scale']
            self.noobject_scale = cfg_train['noobject_scale']
            self.class_scale = cfg_train['class_scale']
            self.coord_scale = cfg_train['coord_scale']
            
            self.cls_normalizer = cfg_train['cls_normalizer']
            self.iou_normalizer = cfg_train['iou_normalizer']
            self.ignore_thresh = cfg_train['ignore_thresh']
            self.iou_thresh = cfg_train['iou_thresh']

            self.label_smooth_eps = self.hyp.get('label_smoothing', 0.0) # cfg_train['label_smooth_eps']
            self.y_true = 1.0 - 0.5 * self.label_smooth_eps
            self.y_false = 0.5 * self.label_smooth_eps
            
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
            
            if cfg_train['focalloss']:
                self.bce_loss = FocalLoss(self.bce_loss)
            
            self.box_loss_type = cfg_train['box_loss_type']
            if self.box_loss_type == 'mse':
                self.box_loss_func = nn.MSELoss(reduction='none')

            elif self.box_loss_type == 'smooth':
                self.box_loss_func = nn.SmoothL1Loss(reduction='none')
            
            elif 'iou' in self.box_loss_type:
                self.box_loss_func = None
    
    def compute_loss(self, p, targets):

        if self.version == 'v3':
            return compute_loss_v3(self, p, targets)
        elif self.version == 'v4':
            return compute_loss_v4(self, p, targets)
        elif self.version == 'v5':
            return compute_loss_v5(self, p, targets)
        else:
            raise Exception(f'{self.version} does not exist')