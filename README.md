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

* Train from one checkpoint(for example, train from `epoch_4.pth.tar`, the `--start-epoch` parameter is corresponding to the epoch of the checkpoint):

```
CUDA_VISIBLE_DEVICES=0,1 python train.py --batch-size 128 --resume output/epoch_4.pth.tar --start-epoch 4
```

## Pretrained models

&emsp;To be added...

# Experiments

&emsp;To be added...
