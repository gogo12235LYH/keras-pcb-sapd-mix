from .focal_loss import FocalLoss, compute_focal
from .iou_loss import IoULoss, compute_iou
from .fsn_loss import FSNLoss


def loss():
    def loss_(y_true, y_pred):
        return y_pred

    return loss_


def model_loss(cls='cls_loss', reg='reg_loss', fsn='feature_select_loss'):
    return {
        cls: loss(),
        reg: loss(),
        fsn: loss(),
    }


Total_Loss = model_loss()
