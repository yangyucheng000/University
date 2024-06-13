import numpy as np
from models.MPRNet import MPRNet
from PIL import Image
import os
from models.warp import get_backwarp
import cv2
from utils import index

import mindspore
from mindspore.dataset import GeneratorDataset
import mindspore.ops as O
from mindspore.dataset.vision.c_transforms import CenterCrop as tf_CenterCrop
from mindspore.dataset.vision.py_transforms import ToTensor as tf_ToTensor

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

norm_val = (2**8)-1
src_mean = 0


def creat_list(path):
    gt_list = []
    im_list = []
    im_path = path + 'blur/'
    gt_path = path + 'gt/'

    for _, _, fnames in sorted(os.walk(gt_path)):
        for fname in fnames:
            gt_list.append(gt_path+fname)
            im_list.append(im_path+fname)

    return gt_list, im_list


to_tensor = tf_ToTensor()


class TestDatasetGen:
    def __init__(self, args, gt_list, im_list):
        self.gt_list = gt_list
        self.im_list = im_list
        self.args = args

    def __getitem__(self, index):
        if self.args.model == 'UNet':
            gt = cv2.imread(self.gt_list[index])
            im = cv2.imread(self.im_list[index])

            gt = np.float32((gt-src_mean)/norm_val)
            im = np.float32((im-src_mean)/norm_val)

            gt = np.rollaxis(gt, 2)
            im = np.rollaxis(im, 2)
            #gt = O.transpose(mindspore.Tensor(gt),(2,0,1))
            #im = O.transpose(mindspore.Tensor(im),(2,0,1))

            c, h, w = gt.shape
            H = h//16*16
            W = w//16*16

            gt = gt[:, 0:H, 0:W]
            im = im[:, 0:H, 0:W]

            return gt, im

        if self.args.model == "MPRNet":
            inp = Image.open(self.im_list[index])
            tar = Image.open(self.gt_list[index])

            h, w = inp.size

            tf_center_crop = tf_CenterCrop((w//8*8, h//8*8))
            #tf_to_tensor = tf_ToTensor()

            inp = tf_center_crop(inp)
            tar = tf_center_crop(tar)

            inp = to_tensor(inp)
            tar = to_tensor(tar)
            return tar, inp

    def __len__(self):
        return len(self.gt_list)


def test_dataset(args, deblur_model, test_loader, save_path=None):
    ssim_sum = 0
    mse_sum = 0
    psnr_sum = 0

    test_iter = test_loader.create_tuple_iterator()
    for j, (gt, im) in enumerate(test_iter):
        gt = mindspore.Tensor(gt)
        im = mindspore.Tensor(im)

        if args.model == 'UNet':
            output = deblur_model(im)
        if args.model == 'MPRNet':
            restored = deblur_model(im)
            z_tensor = mindspore.Tensor(0, dtype=restored[0].dtype)
            o_tensor = mindspore.Tensor(1, dtype=restored[0].dtype)
            output = O.clip_by_value(restored[0], z_tensor, o_tensor)
        wrap_gt, flowl = get_backwarp(output, gt)
        wrap_gt = O.clip_by_value(wrap_gt, z_tensor, o_tensor)

        gt = O.transpose(gt[0, ...], (1, 2, 0)).asnumpy()
        output = O.transpose(output[0, ...], (1, 2, 0)).asnumpy()
        wrap_gt = O.transpose(wrap_gt[0, ...], (1, 2, 0)).asnumpy()

        im = O.transpose(im[0, ...], (1, 2, 0)).asnumpy()

        if args.dataset == 'SDD':
            mse, psnr, ssim = index.MSE_PSNR_SSIM(wrap_gt.astype(np.float64), output.astype(np.float64))
        if args.dataset == 'DPDD':
            mse, psnr, ssim = index.MSE_PSNR_SSIM(gt.astype(np.float64), output.astype(np.float64))

        ssim_sum += ssim
        mse_sum += mse
        psnr_sum += psnr
        print('SSIM:', ssim, 'PSNR:', psnr, 'MSE:', mse)

        if save_path:
            if not os.path.exists(save_path):
                os.mkdir(save_path) 
            if args.model == 'UNet':
                cv2.imwrite("%s/%s.png" % (save_path, j), np.uint8(im*norm_val))
                cv2.imwrite("%s/%s_wgt.png" % (save_path, j), np.uint8(wrap_gt*norm_val))
                cv2.imwrite("%s/%s_gt.png" % (save_path, j), np.uint8(gt*norm_val))
                cv2.imwrite("%s/%s_output.png" % (save_path, j), np.uint8(output*norm_val))
            if args.model == "MPRNet":
                cv2.imwrite("%s/%s.png" % (save_path, j), cv2.cvtColor(np.uint8(im*norm_val), cv2.COLOR_RGB2BGR))
                cv2.imwrite("%s/%s_wgt.png" % (save_path, j), cv2.cvtColor(np.uint8(wrap_gt*norm_val), cv2.COLOR_RGB2BGR)) 
                cv2.imwrite("%s/%s_gt.png" % (save_path, j), cv2.cvtColor(np.uint8(gt*norm_val), cv2.COLOR_RGB2BGR)) 
                cv2.imwrite("%s/%s_output.png" % (save_path, j), cv2.cvtColor(np.uint8(output*norm_val), cv2.COLOR_RGB2BGR)) 

    data_size = test_loader.get_dataset_size()
    print(data_size, 'SSIM:', ssim_sum/data_size, 'PSNR:', psnr_sum/data_size, 'MSE:', mse_sum/data_size)
    return data_size, ssim_sum/data_size, psnr_sum/data_size, mse_sum/data_size


def add(num_, ssim_sum_, psnr_sum_, lmse_sum_, ncc_sum_, num_test, ssim_sum_test, psnr_sum_test, lmse_sum_test, ncc_sum_test):
    return num_+num_test, ssim_sum_+ssim_sum_test, psnr_sum_+psnr_sum_test, lmse_sum_+lmse_sum_test, ncc_sum_+ncc_sum_test


def test_state(args, state_dict):
    if args.model == 'MPRNet':
        deblur_model = MPRNet()
    gt_list, im_list = creat_list(args.test_path)
    test_dpdd_dataset = TestDatasetGen(args, gt_list, im_list)
    test_loader_dpdd = GeneratorDataset(test_dpdd_dataset, column_names=['gt', 'img'], shuffle=False, 
                                        num_parallel_workers=args.num_workers).batch(batch_size=1)
    mindspore.load_param_into_net(deblur_model, state_dict)
    del (state_dict)
    save_path = None

    if args.save_result_path is True:
        save_path = './result_sdd_' + args.model
        if args.dataset == 'DPDD':
            save_path = './result_dpdd_' + args.model
    num, ssim_av, psnr_av, mse_av = test_dataset(args, deblur_model, test_loader_dpdd, save_path)

    return ssim_av, psnr_av, mse_av

