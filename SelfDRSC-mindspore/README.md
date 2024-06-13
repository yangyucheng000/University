# Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive (ICCV2023)
---
[[arXiv](https://arxiv.org/abs/2305.19862)]

This repository is the Mindspore implementation of SelfDRSC: Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive.
The original implementation in pytorch is at [SelfDRSC](https://github.com/shangwei5/selfdrsc). 

### Introduction
To correct RS distortions, existing methods adopt a fully supervised learning manner, where high framerate global shutter (GS) images should be collected as ground-truth supervision. In this paper, we propose a Self-supervised learning framework for Dual reversed RS distortions Correction (SelfDRSC), where a DRSC network can be learned to generate a high framerate GS video only based on dual RS images with reversed distortions. In particular, a bidirectional distortion warping module is proposed for reconstructing dual reversed RS images, and then a self-supervised loss can be deployed to train DRSC network by enhancing the cycle consistency between input and reconstructed dual reversed RS images. Besides start and end RS scanning time, GS images at arbitrary intermediate scanning time can also be supervised in SelfDRSC, thus enabling the learned DRSC network to generate a high framerate GS video. Moreover, a simple yet effective self-distillation strategy is introduced in self-supervised loss for mitigating boundary artifacts in generated GS images.


### Prerequisites
- Python >= 3.8, Mindspore >= 2.0.1
- Requirements: opencv-python, numpy, matplotlib, imageio, scikit-image, tqdm, mindcv


### Datasets
Please download the RS-GOPRO datasets from [link](https://drive.google.com/u/0/uc?id=1DuJphkVpvsNjgPs73y_sm4WZ8tzfxOZf&export=download).


## Download Pre-trained Model of SelfDRSC
Please download the pre-trained pwcnet from [link](https://pan.baidu.com/s/1_Cu-8MInef6U9Mh7ypxOkw?pwd=zwfg)(password: zwfg). Please put these models to `./pretrained`.

Please download the stage two checkpoint from [Stage two](https://pan.baidu.com/s/1-F8weUSY76_gboGotWpmmg?pwd=f4dl)(password: f4dl). Please put these models to `./experiments`.

## Getting Started
### 1) Testing
1.Testing on RS-GOPRO dataset:
```
python main_test_srsc_rsflow_multi_distillv2.py --opt options/test_srsc_rsflow_multi_distillv2_psnr.json
```
or use script `run_test.sh`

Please change `data_root` and `pretrained_netG` in options according to yours.

Using pretrained stage two checkpoint in test would get the same PSNR as original pytorch version implementation.

### 2) Training
1.Training the first stage:
```
python main_train_srsc_rsflow_multi.py --opt options/train_srsc_rsflow_multi_psnr.json
```
Please change `data_root` and `pretrained_rsg` in options according to yours.


2.Training the second stage (adding self-distillation):
```
python main_train_srsc_rsflow_multi_distillv2.py --opt options/train_srsc_rsflow_multi_distillv2_psnr.json
```
Please change `data_root`, `pretrained_rsg` and `pretrained_netG` in options according to yours.

Due to the difference between Mindspore and pytorch, and the unstable performance of implementation of PWC-Net in Mindspore, there maybe a difference between this version and original pytorch implementation.

## Cite
If you use any part of our code, or SelfDRSC is useful for your research, please consider citing:
```
@article{shang2023selfsupervised,
  title={Self-supervised Learning to Bring Dual Reversed Rolling Shutter Images Alive}, 
  author={Wei Shang and Dongwei Ren and Chaoyu Feng and Xiaotao Wang and Lei Lei and Wangmeng Zuo},
  journal={arXiv preprint arXiv:2305.19862},
  year={2023}
}
```

## Contact
If you have any questions on technics, please check the original pytorch version and the paper, or contact original author csweishang@gmail.com .
if you have questions about this implementation, contact with csshihao@outlook.com .