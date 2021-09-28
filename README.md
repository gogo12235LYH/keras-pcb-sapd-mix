# PCB Defect Detection Based on SAPD with Mix Subnetwork

_此貢獻為記錄原始研究程式碼及數據成果，初始環境為 Ubuntu 18.04, Tensorflow 2.3.0_ :
moon: [LINK](http://etheses.lib.ntust.edu.tw/cgi-bin/gs32/gsweb.cgi?o=dstdcdr&s=id=%22G0M10812020%22.&searchmode=basic)

:heart: _目前即將更新為 Tensorflow 2.6.0_ :heart:

~~身邊只有筆電，且 TF2.3 on Windows 使用 GPU 有不少毛病 eo4...~~

## :fire: To Do List :fire:

1. 完成 README.md
2. 上傳 由server訓練完畢之權重
3. 資料集轉換 (tfds 建立) 
4. 透過 tf.data 取代 keras.Sequence 來增加 GPU使用率

## 目錄

1. [安裝-Installation](#1-安裝-Installation)
2. [訓練-Training](#2-訓練-Training)
3. [評估-Evaluation](#3-評估-Evaluation)
4. [推論-Inference](#4-推論-Inference)
5. [參考-Reference](#5-參考-Reference)

## 1. 安裝-Installation

使用的包有 cython, opencv, pillow, keras-resnet, keras, tensorflow,
tensorflow-addons等，其餘必要安裝請參考 [Tensorflow](https://www.tensorflow.org/install/gpu#software_requirements) .

:point_right:可參考如下 :

```
pip install -r requirements.txt
```

:point_right: 注意需要安裝特殊在地包.

```
python setup.py build_ext --inplace
```

:point_right: 到目前為止，可使用 test.py 確認是否可運行.

```
python test.py
```

:point_right: 以下為輸出結果，基本上看最後一行 _[INFO] Testing ... Done._ 就差不多了.

```
[INFO] Initializing...
[INFO] Stage 1 : Creating Model...
2021-09-24 00:52:25.415912: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operati
ons:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-09-24 00:52:25.994526: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2778 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1050 Ti, pci bus id: 0000:01:
00.0, compute capability: 6.1
[INFO] Stage 1 : Loading Weight...  Imagenet Pretrain ... OK.
[INFO] Stage 1 : Model Compiling...
[INFO] Testing ... Done.
```

## 2. 訓練-Training

在 [SAPD](https://arxiv.org/abs/1911.12448) 研究中說明到訓練時， 可在前期訓練將 Feature Selection Network (FSN) 預測權重設定為擁有最低損失的特徵金字塔層(FSN 的
ground truth)， 後續訓練則選擇預測結果前三高來進行訓練時的抑制。

### 2.1 MODE 及 基本超參數

1. MODE 可分為2大類，分別為 "FSN Top-1 與 Top-k 的 Stage 1 及 Stage 2 訓練模式" 及
   "只有 FSN Top-k 的訓練模式"，細分四種模式。
2. 訓練影像解析度可由 PHI 來調整。
3. ~~若使用單機多卡的訓練策略，MULTI_GPU = 1 (建議在非 Windows 平台使用)。~~
(目前多卡訓練，餵資料待須將 Keras.Sequential 轉換為 tf.data)

```python
MODE = 1  # MODE = 1: Stage One; MODE = 2: Stage Two; MODE = 3: Top-1 Weight; MODE = 4: Top-5 Weight.
EPOCHs = 50
STEPs_PER_EPOCH = None  # steps in one epoch
EPOCHs_STAGE_ONE = int(EPOCHs * 0.5)
BATCH_SIZE = 1  # Global Batch size
NUM_CLS = 6
PHI = 0  # B0:(512, 512), B1:(640, 640), B2:(768, 768), B3:(896, 896), B4:(1024, 1024) ~ B7(1048, 1048)
MULTI_GPU = 0
```

### 2.2 Optimizer 及 Callback

建議使用的優化器有: SGDW (Nesterov), Adam 及 AdamW， 初始學習率可先利用 Learning Rate range test 方式找尋。我找到的學習率可如下:

| Optimizer|  base_lr |  min_lr  | momentum |   decay  |
| -------- | -------- | -------- | -------- | -------- |
| SGDW(Nesterov) | 2.5e-3 | none | 0.9 | 1e-4 |
| Adam | 4e-5 | 2.5e-6 | none | none |
| AdamW | 4e-5 | 2.5e-6 | none | base_lr * 0.1 * ((epoch/batch)**0.5) |

callback 預設使用 early stopping，並使用 Cosine-Decay 調整 學習率。

### 2.3 Subnetworks ( Head )

目前提供 標準, 混合 及 對準 等子網路:

1. Std (標準): 基本常見的子網路。
2. Mix (混合): 透過 WS 及 GN(groups=16) 來正規化特徵的輸入與輸出，並將回歸子網路融合進分類子網路達到更佳的效能。
3. MixV2 : 基於 Mix 將自兩個子網路的捲積核改為 1*1 ，且融合方式為直接相加，最後使用 1*1 捲積加強學習效果。
4. Align (對準): 除了透過正規化輸入與輸出，加入 centerness融合 增強分類子網路的預測。

```python
""" Head: Subnetwork """
SHRINK_RATIO = 0.2  # Bounding Box shrunk ratio
HEAD = 'Std'  # 'Std', 'Mix', 'Align'
HEAD_ALIGN_C = 0.5  # Align Layer factor of centerness fusion. Default = 0.5
HEAD_ALIGN_B = 0.0  # Align Layer bias of centerness fusion. Default = 0.
HEAD_WS = 0  # '1' with WS, '0' without WS
HEAD_GROUPS = 16  # In GroupNormalization's setting
SUBNET_DEPTH = 4  # Depth of Head Subnetworks
```

### 2.4 損失函數

分類問題透過 Focal Loss (FL) 進行損失值評估及學習，目前也在測試 QFL 中。輸出預選框則透過 IoU Loss 也就是重疊度損失， 目前提供 IoU, GIoU 及 CIoU 選用。都採用 XLA 做加速。

CIoU 方面提供 2 種重現 :

1. 去除中心位置偏移問題，只保留長寬比之神奇版本。
2. 透過 ground truth 的左上角座標為依據，亦可推算中心偏移之完整版本。

```python
""" Model: Classification and Regression Loss """
USING_QFL = 0  # Classification Loss: Quality Focal Loss
IOU_LOSS = 'giou'  # Regression Loss: iou, giou, ciou, fciou
IOU_FACTOR = 1.0
```

### 2.5 :point_right: 開始~~煉丹~~訓練

```
python train.py
```

```
[INFO] Initializing...
[INFO] Stage 1 : Creating Optimizer...
[INFO] Stage 1 : Creating Generators...
[INFO] Stage 1 : Creating Model...
2021-09-25 11:06:31.083194: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operati
ons:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-09-25 11:06:32.157288: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1510] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 2778 MB memory:  -> device: 0, name: NVIDIA GeForce GTX 1050 Ti, pci bus id: 0000:01:
00.0, compute capability: 6.1
[INFO] Stage 1 : Loading Weight...  Imagenet Pretrain ... OK.
[INFO] Stage 1 : Model Compiling...
[INFO] Stage 1 : Model Name : 20210921-DPCB100-HS016FV3-SGDW-E50BS1B0R50D4
[INFO] Stage 1 : Creating Callbacks... Cosine-Decay, HistoryEarlyStopping,
[INFO] Stage 1 : Start Training...
2021-09-25 11:06:39.512493: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)
Epoch 1/25
-- [INFO] Cosine-Decay LR : 0.0025
-- [INFO] Cosine-Decay WD : 0.0001
[INFO] From History Callback: EP:0 LR: 0.0024999999441206455, WD: 9.999999747378752e-05
2021-09-25 11:06:51.568779: I tensorflow/compiler/xla/service/service.cc:171] XLA service 0x2430051c620 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2021-09-25 11:06:51.568930: I tensorflow/compiler/xla/service/service.cc:179]   StreamExecutor device (0): NVIDIA GeForce GTX 1050 Ti, Compute Capability 6.1
2021-09-25 11:06:51.615357: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:210] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_DIRECTORY` to enable.
2021-09-25 11:06:52.730175: I tensorflow/stream_executor/cuda/cuda_dnn.cc:369] Loaded cuDNN version 8100
2021-09-25 11:06:54.429547: I tensorflow/compiler/jit/xla_compilation_cache.cc:363] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
   8/1000 [..............................] - ETA: 16:00 - loss: 1.8903 - cls_loss_loss: 1.1315 - reg_loss_loss: 0.5946 - feature_select_loss_loss: 0.1642
```

## 3. 評估-Evaluation

透過 mean Average Precision (mAP) 評估模型預測效果，
其中 mAP 為重疊度由 0.5 至 0.95 間隔 0.05 共 10 組 AP 所算得知平均值。

### 3.1 模型輸出設定

可設定提案數量(上限值為依據影像解析度為定，Ex: 使用 640 * 640 解析度之輸入影像， 
依據每層特徵金字塔得出 80 * 80, 40 * 40, 20 * 20, 10 * 10, 5 * 5 
各層面積累加可得 8525 之上限提案數量。)

非極大值抑制(NMS)，目前提供 NMS，Soft-NMS 。

```python
""" Model Detections: NMS, Proposal setting """
NMS = 1  # 1 for NMS, 2 for Soft-NMS
NMS_TH = 0.5  # intersect of union threshold in same detections
DETECTIONS = 1000  # detecting proposals
```

_待_

### 3.2 Deep PCB:
似乎 Align 對於 小目標瑕疵 及 Binary影像 效果非常顯著。

| Subnetworks | setting | mAP | AP.5 | AP.75 | AP.9 |
| ------ |------ |------ |------ |------ |------ |
| Std | x1 | 0.753 | 0.9751 | 0.8881 | 0.3518 |
| Mix | x1, gn | 0.7629 | 0.9802 | 0.8950 | 0.3861 |
| Align | x2, ws + gn | 0.8293 | 0.9862 | 0.9415 | 0.5247 |

### 3.3 PCB-Defect:

| Subnetworks | setting | mAP | AP.5 | AP.75 | AP.9 |
| ------ |------ |------ |------ |------ |------ |
| Std | x1 | 0.6891 | 0.9991 | 0.8346 | 0.1142 |
| Mix | x1, gn | 0.7163 | 0.9995 | 0.8731 | 0.1641 |
| Align | - | - | - | - | - |

## 4. 推論-Inference

### Deep PCB 範例
![image](https://github.com/gogo12235LYH/keras-pcb-sapd-mix/tree/master/fig/dpcb_temp.png)

### PCB-Defect 範例
![image](https://github.com/gogo12235LYH/keras-pcb-sapd-mix/tree/master/fig/pcbdd_temp_1.png)

![image](https://github.com/gogo12235LYH/keras-pcb-sapd-mix/tree/master/fig/pcbdd_temp_2.png)

## 5. 參考-Reference

1. [https://github.com/fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet)
2. [https://keras.io/examples/vision/retinanet/](https://keras.io/examples/vision/retinanet/)
3. [https://github.com/xuannianz/SAPD](https://github.com/xuannianz/SAPD)