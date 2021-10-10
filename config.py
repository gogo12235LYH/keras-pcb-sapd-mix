import numpy as np


""" Dataset Setting """
_dataset_path = {
    'VOC': r'../VOCdevkit/VOC2012+2007',
    'DPCB': r'../DeepPCB_voc',
    'rPCB': r'../PCB_DATASET_Random_Crop'
}
DATASET = 'DPCB'
DATABASE_PATH = _dataset_path[DATASET]
# DATABASE_PATH = r'../VOCdevkit/VOC2012+2007'
# DATABASE_PATH = r'../PCB_DATASET_Random_Crop'
# DATABASE_PATH = r'../DeepPCB_voc'
# DATABASE_PATH = r'../MixPCB'
# DATABASE_PATH = r'../MixPCB_MixUp_Fix'

""" Hyper-parameter setting """
MODE = 1                # MODE = 1: Stage One; MODE = 2: Stage Two; MODE = 3: Top-1 Weight; MODE = 4: Top-5 Weight.
EPOCHs = 10
STEPs_PER_EPOCH = 10  # steps in one epoch
EPOCHs_STAGE_ONE = int(EPOCHs * 0.5)
BATCH_SIZE = 2          # Global Batch size
NUM_CLS = 6
PHI = 0                 # B0:(512, 512), B1:(640, 640), B2:(768, 768), B3:(896, 896), B4:(1024, 1024) ~ B7(1048, 1048)
MULTI_GPU = 0

EPOCHs_STAGE_TWO = 12
LRF_MAX_LR = 1e-2
LRF_MIN_LR = 1e-8


""" Optimizer Setting """
OPTIMIZER = 'SGDW'      # SGDW, AdamW, SGD, Adam
USE_NESTEROV = True     # For SGD and SGDW
BASE_LR = 2.5e-3        # SGDW: (2.5e-3), AdamW: (4e-5, 2.5e-6)
MIN_LR = 2.5e-6         # Adam, AdamW: (2.5e-6)
MOMENTUM = 0.9          # SGDW: (0.9)
# DECAY = BASE_LR * 0.1 * ((BATCH_SIZE/EPOCHs) ** (1/2))
DECAY = 1e-4

""" Callback Setting """
LR_Scheduler = 1        # 1 for Cosine Decay, 2 for Cosine Decay with Restart
USING_HISTORY = 1       # IF Optimizer has weight decay and weight decay can be decayed, must set to be 1 or 2.
EVALUATION = 0          # AP.5, Return training and Inference model when creating model function is called.
EPOCHs_RESTART = 25     # Initial restart epochs
RS_RATIO = 0.1          # For restart Cosine Decay Scheduler initial learning rate

""" Warm Up """
USING_WARMUP = 0
WP_EPOCHs = int(0.1 * EPOCHs)
WP_RATIO = 0.1

""" Cosine Decay learning rate scheduler setting """
# ALPHA = np.round(MIN_LR / BASE_LR, 4)
ALPHA = 0.              # Cosine Decay's alpha


""" Augmentation Setting """
MISC_AUG = 1
VISUAL_AUG = 0
MixUp_AUG = 0


""" Backbone: Feature Extractor """
BACKBONE_TYPE = 'ResNetV1'  # ResNetV1, SEResNet
BACKBONE = 50           # 50 101
FREEZE_BN = False
FREEZE_BACKBONE = False
FREEZE_HALF_BACKBONE = False
PRETRAIN = 1            # 1 for Imagenet, 2 for weights path
PRETRAIN_WEIGHT = './pretrain_weights/resnet50-keras-dpcb-v2.h5'


""" Head: Subnetwork """
SHRINK_RATIO = 0.2      # Bounding Box shrunk ratio
HEAD = 'Std'          # 'Std', 'Mix', 'Align'
HEAD_ALIGN_C = 0.5      # Align Layer factor of centerness fusion. Default = 0.5
HEAD_ALIGN_B = 0.0      # Align Layer bias of centerness fusion. Default = 0.
HEAD_WS = 0             # '1' with WS, '0' without WS
HEAD_GROUPS = 16        # In GroupNormalization's setting
SUBNET_DEPTH = 4        # Depth of Head Subnetworks


""" Neck: Feature Pyramid Network """
FPN = None
FPN_DEPTH = 1
FPN_BN = 0
# TODO : FPN, FPG,...


""" Neck: Feature Selection Network """
FSN = 'V3'              # Feature Selection Network
FSN_WS = 1              # '1' is with WS, '0' is without WS
FSN_TOP_K = 3           # In training session, choosing top 3 weights
FSN_POOL_SIZE = 7       # Feature Selection Network's Input Size
TOP1_TRAIN = 1          # In Stage 1, the weight will choose Top-1 level of FPN.
FSN_FACTOR = 0.1


""" Model: Classification and Regression Loss """
USING_QFL = 0           # Classification Loss: Quality Focal Loss
MAX_BBOXES = 100
IOU_LOSS = 'giou'       # Regression Loss: iou, giou, ciou, fciou
IOU_FACTOR = 1.0


""" Model Name: Date-Dataset-MixUp-HEAD-FSN-Optimizer """
DATE = '20210921-'
D_NAME = f'{DATASET}{MISC_AUG}{VISUAL_AUG}{MixUp_AUG}-'
H_NAME = f'H{HEAD[0]}{HEAD_WS}{HEAD_GROUPS}F{FSN}-'
O_NAME = f'{OPTIMIZER}'
T_NAME = f"E{EPOCHs}BS{BATCH_SIZE}B{PHI}R{BACKBONE}D{SUBNET_DEPTH}"
NAME = DATE + D_NAME + H_NAME + O_NAME + T_NAME


""" Model Detections: NMS, Proposal setting """
NMS = 1                 # 1 for NMS, 2 for Soft-NMS
NMS_TH = 0.5            # intersect of union threshold in same detections
DETECTIONS = 1000       # detecting proposals
