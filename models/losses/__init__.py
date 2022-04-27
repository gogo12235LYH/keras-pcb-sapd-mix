from .focal_loss import FocalLoss, compute_focal, compute_focal_v2
from .iou_loss import IoULoss, compute_iou, compute_iou_v2
from .fsn_loss import FSNLoss


def loss():
    def loss_(y_true, y_pred):
        return y_pred

    return loss_


def model_loss(cls='cls_loss', reg='loc_loss', fsn='fsn_loss'):
    return {
        cls: loss(),
        reg: loss(),
        fsn: loss(),
    }


Total_Loss = model_loss()
