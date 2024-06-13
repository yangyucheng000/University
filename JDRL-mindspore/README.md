# JDRL-mindspore

## Introduction

The transplantation of pytorch implementation of "Learning Single Image Defocus Deblurring with Misaligned Training Pairs" on huawei mindspore platform. Original project is [here](https://github.com/liyucs/JDRL). Due to some platform limits, this project only transplants single device training and testing codes with original torch weights converted to measure precisions.

## Environment

this transplantation based on huaweicloud ModelArts platform with following environment:

- python 3.7
- mindspore 1.7.0
- cuda 10.1

Main third party packages:

- numpy 1.19.5
- opencv-python 4.6.0
- natsort 8.1.0
- scikit-image 0.19.3
- yacs 0.1.8
- tqdm 4.64.1

## Datasets

Test precisions on [SDD test](https://drive.google.com/file/d/1f6WQmBPNp3StdQZVahq9JA5J_5u1h9SN/view?usp=sharing) and [DPDD test](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel). 

Test training scripts on [SDD train](https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel).

## Precisions

| model | PSNR | SSIM |
| ----- | ---- | ---- |
| mprnet-torch-sdd | 26.88 | 0.81 |
| mprnet-mindspore-sdd* | 24.8 | 0.753 |
| mprnet-torch-dpdd | 25.73 | 0.792 |
| mprnet-mindspore-dpdd | 25.64 | 0.788 |

\* In SDD testing usage of mindspore implementation of [PWC-Net](https://gitee.com/mindspore/models/tree/master/official/cv/PWCNet) which is different from official torch implementation causes the most of precisions lost.

Mindspore implementation suffers a precisions lost from the different implementation of the [Bilinear Interpolation operator](https://www.mindspore.cn/docs/zh-CN/r1.10/note/api_mapping/pytorch_diff/ResizeBilinear.html), which should not be a problem with completely retraining on the new platform.

## Train

1. Ajust configs in `training.yml`, including datasets path, epochs, batchsize, etc.
2. run `train.sh`

## Test

1. Edit the checkpoint path and dataset path in `test.sh`, and choose dataset
2. run `test.sh`

## Device

Any GPUs support mindspore can be used for training or testing, but make sure you have at least 22GB of memory on GPU in total.
Not tested in Ascend devices, but it should work.