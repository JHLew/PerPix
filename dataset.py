import torch
from torch.utils import data
from glob import glob
import random
from random import randint
from PIL import Image
import os
from torchvision.transforms.functional import to_tensor
import numpy as np
from kernel_encoding import get_cov, reconstruct_from_cov, encode_cov, decode_to_cov
from utils import downsample_via_kcode, make_kmap
from torch.nn.functional import interpolate


class dataset(data.Dataset):
    def __init__(self, dirs, patch_size=None, scale=4, kernel_size=49, is_train=True):
        self.patch_size = patch_size
        self.ksize = kernel_size
        self.scale = scale

        self.img_list = []
        self.is_train = is_train
        for d in dirs:
            self.img_list = self.img_list + glob(os.path.join(d, '*.png'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = os.path.basename(img_path)

        img = Image.open(img_path).convert('RGB')

        # if for training, crop patches
        if self.is_train:
            img, _ = crop_img(img, size=(self.patch_size, self.patch_size))
            gt_img, _ = augmentation(img)
            lr_size = (self.patch_size - self.ksize + self.scale) // self.scale
            lr_size = (lr_size, lr_size)

        # if for evaluation, use the full image
        else:
            gt_img = img
            w, h = gt_img.size

            lr_size = ((w - self.ksize + self.scale) // self.scale,
                       (h - self.ksize + self.scale) // self.scale)

        ref_size = (lr_size[0] * self.scale, lr_size[1] * self.scale)
        hr_img = gt_img
        gt_img = discard_boundary(gt_img, ref_size=ref_size, k_size=self.ksize)

        hr_img = to_tensor(hr_img)
        gt_img = to_tensor(gt_img)

        if self.is_train:
            n_k = 1
            # ratio between batch size : number of kernels to use in Kernel Collage.
            # Since batch size is 16 and number of kernels used in Kernel Collage n is 16, n_k is 1 (16 / 16)

            k_list = []
            code_list = []
            count = 0
            while True:
                # generate kernel via code
                size = np.clip(np.random.normal(25, 8), 2.5, 47.5)

                w_ratio = np.clip(np.random.normal(0, 0.5), -1, 1)
                w_ratio = 10 ** w_ratio

                v_ratio = randint(1, 1000)
                v_ratio /= 1000

                code = [size, w_ratio, v_ratio]
                kernel = reconstruct_from_cov(decode_to_cov(code), mean=(self.ksize // 2, self.ksize // 2), size=(self.ksize, self.ksize))
                kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
                code = torch.FloatTensor(code)

                k_list.append(kernel)
                code_list.append(code)
                count += 1
                if count >= n_k:
                    break

            kernel = torch.stack(k_list)
            code = torch.stack(code_list)

        else:
            n_k = 32  # number of kernels to use in Kernel Collage for each image in validation.
            count = 0
            kernels = []
            codes = []
            while True:
                size = np.clip(np.random.normal(25, 8), 2.5, 47.5)

                w_ratio = np.clip(np.random.normal(0, 0.5), -1, 1)
                w_ratio = 10 ** w_ratio

                v_ratio = randint(1, 1000)
                v_ratio /= 1000

                code = [size, w_ratio, v_ratio]
                kernel = reconstruct_from_cov(decode_to_cov(code), mean=(self.ksize // 2, self.ksize // 2), size=(self.ksize, self.ksize))
                kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
                code = torch.FloatTensor(code)

                kernels.append(kernel)
                codes.append(code)
                count += 1
                if count >= n_k:
                    break
            kernel = torch.stack(kernels)
            code = torch.stack(codes)

        return hr_img, gt_img, kernel, code, img_name


# crop a part of image
def crop_img(img, size, custom=None):
    width, height = size
    if custom is None:
        left = randint(0, img.size[0] - width)
        top = randint(0, img.size[1] - height)
    else:
        left, top = custom

    cropped_img = img.crop((left, top, left + width, top + height))

    return cropped_img, (left, top)


# data augmentation by flipping and rotating
def augmentation(img, custom=None, do_rot=True):
    if custom is None:
        flip_flag = randint(0, 1)
        rot = randint(0, 359)
    else:
        flip_flag, rot = custom
        if rot is None:
            do_rot = False

    # flipping
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # rotation
    if do_rot:
        if rot < 90:
            rot = 45
            img = img.rotate(90)
        elif rot < 180:
            rot = 135
            img = img.rotate(180)
        elif rot < 270:
            rot = 225
            img = img.rotate(270)
        else:
            rot = 315
    else:
        rot = None

    return img, (flip_flag, rot)


def discard_boundary(img, ref_size, k_size=None):
    w, h = img.size
    syn_w, syn_h = ref_size  # reference size is given in (W, H)

    if k_size is None:
        w_discard = (w - syn_w) // 2
        h_discard = (h - syn_h) // 2
    else:
        w_discard = k_size // 2
        h_discard = k_size // 2

    w_discard -= 1
    h_discard -= 1
    img = img.crop((w_discard, h_discard, w_discard + syn_w, h_discard + syn_h))

    return img


def get_kernels(n):
    sqrt_n = int(np.sqrt(n))
    if not (sqrt_n ** 2) == n:
        raise ValueError('Wrong number of kernels configured!')

    code_list = []
    count = 0

    while True:
        # generate via code
        size = np.clip(np.random.normal(25, 8), 2.5, 47.5)

        w_ratio = np.clip(np.random.normal(0, 0.5), -1, 1)
        w_ratio = 10 ** w_ratio

        v_ratio = randint(1, 1000)
        v_ratio /= 1000

        code = [size, w_ratio, v_ratio]
        code = torch.FloatTensor(code)

        code_list.append(code)
        count += 1
        if count >= n:
            break

    code = torch.stack(code_list)
    return code


