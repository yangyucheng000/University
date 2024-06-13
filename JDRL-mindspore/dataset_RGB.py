import os
import numpy as np
from PIL import Image
import random
from skimage import exposure

from mindspore.dataset.vision.c_transforms import CenterCrop, RandomColorAdjust
from mindspore.dataset.vision.py_transforms import ToTensor

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

to_tensor = ToTensor()

class DataLoaderTrain:
    def __init__(self, rgb_dir, img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'blur')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.inp_filenames = [os.path.join(rgb_dir, 'blur', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'gt', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4*np.random.rand())
            adjust_sat = RandomColorAdjust(saturation=sat_factor)
            inp_img = adjust_sat(inp_img)
            tar_img = adjust_sat(tar_img)

        inp_img = np.array(inp_img)
        tar_img = np.array(tar_img)

        w, h, _ = tar_img.shape
        padw = ps-w if w<ps else 0
        padh = ps-h if h<ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw!=0 or padh!=0:
            inp_img = np.pad(inp_img, (0,0,padw,padh), 'reflect')
            tar_img = np.pad(tar_img, (0,0,padw,padh), 'reflect')

        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = exposure.adjust_gamma(inp_img, 1)
            tar_img = exposure.adjust_gamma(tar_img, 1)

        inp_img = to_tensor(inp_img)
        tar_img = to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr     = random.randint(0, hh-ps)
        cc     = random.randint(0, ww-ps)
        aug    = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr+ps, cc:cc+ps]
        tar_img = tar_img[:, rr:rr+ps, cc:cc+ps]

        # Data Augmentations
        if aug==1:
            inp_img = np.flip(inp_img,1)
            tar_img = np.flip(tar_img,1)
        elif aug==2:
            inp_img = np.flip(inp_img,2)
            tar_img = np.flip(tar_img,2)
        elif aug==3:
            inp_img = np.rot90(inp_img,axes=(1,2))
            tar_img = np.rot90(tar_img,axes=(1,2))
        elif aug==4:
            inp_img = np.rot90(inp_img,axes=(1,2), k=2)
            tar_img = np.rot90(tar_img,axes=(1,2), k=2)
        elif aug==5:
            inp_img = np.rot90(inp_img,axes=(1,2), k=3)
            tar_img = np.rot90(tar_img,axes=(1,2), k=3)
        elif aug==6:
            inp_img = np.rot90(np.flip(inp_img,1),axes=(1,2))
            tar_img = np.rot90(np.flip(tar_img,1),axes=(1,2))
        elif aug==7:
            inp_img = np.rot90(np.flip(inp_img,2),axes=(1,2))
            tar_img = np.rot90(np.flip(tar_img,2),axes=(1,2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img


class DataLoaderVal:
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'blur')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'gt')))

        self.inp_filenames = [os.path.join(rgb_dir, 'blur', x)  for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'gt', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # Validate on center crop
        if self.ps is not None:
            center_crop=CenterCrop((ps,ps))
            inp_img = center_crop(inp_img)
            tar_img = center_crop(tar_img)

        inp_img = to_tensor(inp_img)
        tar_img = to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img


class DataLoaderTest:
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = to_tensor(inp)
        return inp
