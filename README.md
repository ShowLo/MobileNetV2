# MobileNetV2
Implementation of MobileNetV2 with pyTorch, adapted from [MobileNetV2-PyTorch](https://github.com/miraclewkf/MobileNetV2-PyTorch) and [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2).
# Theory
&emsp;You can find the paper of MobileNetV2 at [Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segment](https://arxiv.org/abs/1801.04381).

# Usage

&emsp;This project uses Python 3.7.3 and PyTorch 1.0.1.

## Prepare data

&emsp;The ImageNet dataset is used in this project and is put as follows (Copied from [miraclewkf/MobileNetV2-PyTorch](https://github.com/miraclewkf/MobileNetV2-PyTorch) where you can find the files `ILSVRC2012_img_train` and `ILSVRC2012_img_val`).

```
├── train.py # train script
├── MobileNetV2.py # network of MobileNetV2
├── read_ImageNetData.py
├── ImageData
  ├── ILSVRC2012_img_train
    ├── n01440764
    ├──    ...
    ├── n15075141
  ├── ILSVRC2012_img_val
  ├── ILSVRC2012_dev_kit_t12
    ├── data
      ├── ILSVRC2012_validation_ground_truth.txt
      ├── meta.mat
```

## Train

* Train from scratch:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py --batch-size 128
```

* Train from one checkpoint(for example, train from `epoch_200.pth`, the `--start-epoch` parameter is corresponding to the epoch of the checkpoint):

```
CUDA_VISIBLE_DEVICES=2,3 python train.py --batch-size 256 --resume /media/data2/chenjiarong/MobileNetV2/output/epoch_200.pth --start-epoch 200 --num-epochs 300
```

## Pretrained models

&emsp;In `pretrained`, achieving an accuracy of 71.62%.

# Experiments

## training setting:

1. number of epochs: 400
2. learning rate schedule: learning rate decay rate of 0.98 per epoch, initial lr=0.045
3. weight decay: 4e-5
4. remove dropout
5. batch size: 256
6. optimizer: SGD

### MobileNetV3 large
|              | Madds     | Parameters | Top1-acc  |
| -----------  | --------- | ---------- | --------- |
| Offical 1.0  | 300 M     | 3.4  M     | 72.0%     |
| Ours    1.0 (Madds&Parameters calculated by [thop](https://github.com/Lyken17/pytorch-OpCounter)) | 328.78 M     | 3.5 M     | 71.62%     |
| Ours    1.0 (Madds&Parameters calculated by [torchsummaryX](https://github.com/nmhkahn/torchsummaryX)) | 300.79 M     | 3.5 M     | 71.62%     |
