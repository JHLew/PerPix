import torch
import numpy as np
import os
import random
import torch.nn.functional as F
from kernel_encoding import reconstruct_from_cov, decode_to_cov
from models import preprocess, postprocess


def multiple_downsample(hr, kernels, scale):
    with torch.no_grad():
        hr = preprocess(hr)

        # downsample with each kernels
        b, c, h, w = hr.shape
        hr = hr.view((b * c, 1, h, w))
        fake_lr = batch_forward(hr, kernels, scale)
        n, _, _, h, w = fake_lr.shape
        fake_lr = fake_lr.view((n, b, c, h, w))

        fake_lr = postprocess(fake_lr)
        fake_lr = fake_lr.detach().clone()

    return fake_lr


def conv_downsample(img, kernel, scale):
    return F.conv2d(img, kernel, bias=None, stride=scale)


def batch_forward(batch_img, batch_kernel, scale):
    batch = []
    batch_size = batch_kernel.shape[0]
    for b in range(batch_size):
        fake_lr = conv_downsample(batch_img, batch_kernel[b], scale)
        batch.append(fake_lr)

    batch = torch.stack(batch)

    return batch


def kernel_collage(lr_imgs, kernels=None, ratio=(0.33, 0.8)):
    with torch.no_grad():
        n = len(lr_imgs)
        base = lr_imgs[0]
        batch_size = base.shape[0]
        n_pixels = base.shape[-2:]

        _min, _max = int(ratio[0] * 1000), int(ratio[1] * 1000)
        ratios_h = [random.choice(range(_min, _max)) / 1000 for _ in range(n - 1)]
        ratios_w = [random.choice(range(_min, _max)) / 1000 for _ in range(n - 1)]

        # priority in descending orders
        ratios_h = sorted(ratios_h)
        ratios_w = sorted(ratios_w)

        # if kernels is not None:
        kernel_map = kernels[0].expand((batch_size, n_pixels[0], n_pixels[1], kernels.shape[1]))
        kernels = kernels.view(n, 1, 1, 1, kernels.shape[1]).expand((n, batch_size, n_pixels[0], n_pixels[1], kernels.shape[1]))

        # compose locations to cut & mix
        # crop & paste to base image
        # from lower priority.
        # higher priority patches can overwrite on lower priority patches
        for i in range(n - 1, 0, -1):
            cur_crop_h = int(ratios_h[i - 1] * min(n_pixels))
            cur_crop_w = int(ratios_w[i - 1] * min(n_pixels))
            _pos_h = [random.randint(0, n_pixels[0] - cur_crop_h) for _ in range(batch_size)]
            _pos_w = [random.randint(0, n_pixels[1] - cur_crop_w) for _ in range(batch_size)]
            crop_mask = torch.zeros_like(base)
            kmap_mask = torch.zeros_like(kernel_map)

            for b in range(batch_size):
                crop_mask[b, :, _pos_h[b]: _pos_h[b] + cur_crop_h, _pos_w[b]: _pos_w[b] + cur_crop_w] = 1
                kmap_mask[b, _pos_h[b]: _pos_h[b] + cur_crop_h, _pos_w[b]: _pos_w[b] + cur_crop_w, :] = 1

            crop = crop_mask * lr_imgs[i]
            base_mask = torch.abs(crop_mask - 1)
            base = base_mask * base
            base = base + crop

            if kernels is not None:
                kmap_crop = kmap_mask * kernels[i]
                kmap_mask = torch.abs(kmap_mask - 1)
                kernel_map = kernel_map * kmap_mask
                kernel_map = kernel_map + kmap_crop

        if kernels is not None:
            return base, kernel_map.permute(0, 3, 1, 2)
        else:
            return base


def pixel_mix(lr_imgs, kernels):
    with torch.no_grad():
        fake_lr = lr_imgs
        n, b, c, h, w = fake_lr.shape
        kernels = kernels.view(n, 1, 1, 1, kernels.shape[1]).expand((n, b, h, w, kernels.shape[1]))  # n, b, h, w, c

        fake_lr = fake_lr.permute((1, 3, 4, 0, 2))  # b, h, w, n, c
        kernels = kernels.permute((1, 2, 3, 0, 4))
        _b = np.repeat(range(0, b), h * w)
        i = np.tile(np.repeat(range(0, h), w), b)
        j = np.tile(np.tile(range(0, w), h), b)
        k = np.random.randint(n, size=b * h * w)
        fake_lr = fake_lr[_b, i, j, k, :].view(b, h, w, c).permute(0, 3, 1, 2).detach().clone()
        kernels = kernels[_b, i, j, k, :].view(b, h, w, c).permute(0, 3, 1, 2).detach().clone()

    return fake_lr, kernels


def mask_mix(lr_imgs, kernels, mask):
    mask = mask.permute(1, 0, 2, 3)
    # mask = mask.permute(1, 0, 3, 2)
    n_mask = mask.shape[0]

    final_lr = torch.zeros([1, 3, mask.shape[2], mask.shape[3]]).cuda()
    final_kernel_map = torch.zeros([1, 3, mask.shape[2], mask.shape[3]]).cuda()

    with torch.no_grad():
        for i in range(n_mask):
            cur_mask = mask[i]
            cur_lr = lr_imgs[i][0]
            cur_kernel = kernels[i]
            cur_kernel = cur_kernel.view(-1, 3, 1, 1).expand_as(final_kernel_map)[0]

            masked_lr = cur_mask * cur_lr
            masked_kmap = cur_mask * cur_kernel
            masked_lr = masked_lr.unsqueeze(0)
            masked_kmap = masked_kmap.unsqueeze(0)

            final_lr = final_lr + masked_lr
            final_kernel_map = final_kernel_map + masked_kmap

    return final_lr, final_kernel_map



from torch.nn.functional import interpolate


_max_bound = torch.FloatTensor([[[[50]], [[10]], [[1]]]])
_min_bound = torch.FloatTensor([[[[2.5]], [[0.1]], [[1e-4]]]])


def make_kmap(k_code, size=(64, 64)):
    # interpolate to make kernel map
    # size is in (H, W)
    n = int(np.sqrt(len(k_code)))

    k_code = k_code.permute(1, 0)
    k_code = k_code.view(1, 3, n, n)
    k_map = interpolate(k_code, size, mode='bicubic', align_corners=True)
    k_map = torch.max(torch.min(k_map, _max_bound), _min_bound)

    return k_code, k_map[0]


def downsample_via_kcode(hr, kmap, scale=4):
    ksize = (49, 49)
    kmap = kmap.permute(2, 1, 0)
    # kmap = kmap.permute(1, 2, 0)
    cols = []
    for h in range(0, hr.shape[1] - ksize[0] + 1, scale):
        row = []
        for w in range(0, hr.shape[2] - ksize[1] + 1, scale):
            kernel = reconstruct_from_cov(decode_to_cov(kmap[h // scale, w // scale]), mean=(24, 24), size=ksize)
            patch = hr[:, h: h + ksize[0], w: w + ksize[1]]
            pixel = patch * kernel
            pixel = pixel.sum(dim=(1, 2))
            row.append(pixel)
        row = torch.stack(row)
        cols.append(row)
    cols = torch.stack(cols)
    cols = torch.clamp(cols.permute(2, 0, 1), 0, 1)

    return cols


def visualize_kmap(kmap, s=1, out_dir='./', tag=None, ksize=49, each=False):
    import cv2
    out_dir = os.path.join(out_dir, 'visualized_kmap')
    os.makedirs(out_dir, exist_ok=True)
    if each:
        out_dir = os.path.join(out_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
    ksize = (ksize, ksize)
    mean = (ksize[0] // 2, ksize[1] // 2)
    cols = []
    for i in range(kmap.shape[0]):
        # parameter s corresponds to the stride of kernel visualization.
        # if s = 1, kernels are visualized for every pixel.
        if not i % s == 0:
           continue
        row = []
        for j in range(kmap.shape[1]):
            if not j % s == 0:
                continue
            kernel = reconstruct_from_cov(decode_to_cov(kmap[i, j]), mean=mean, size=ksize).astype(np.float32)
            if kernel.max() > 0:
                kernel /= kernel.max()
            kernel *= 255
            if each:
                cv2.imwrite(os.path.join(out_dir, '{}_{}.png'.format(i, j)), kernel)
            else:
                row.append(kernel)
        if not each:
            row = cv2.hconcat(row)
            cols.append(row)
    if not each:
        cols = cv2.vconcat(cols)
        if tag is None:
            cv2.imwrite(os.path.join(out_dir, '{}.png'.format(s)), cols)
        else:
            cv2.imwrite(os.path.join(out_dir, '{}_{}.png'.format(tag, s)), cols)


def init_Kmap(img, random=False):
    _size = img[:, 0, :, :].shape
    if random:
        rand_n = torch.clamp(torch.randn(1).normal_(25, 8), 2.5, 47.5)[0]
        rand_w = torch.clamp(torch.randn(1).normal_(0, 0.5), -1, 1)[0]
        rand_w = 10 ** rand_w
        rand_v = torch.clamp(torch.rand(1), 1e-3, 1)[0]

        norm = torch.zeros(_size).fill_(rand_n)
        w = torch.zeros(_size).fill_(rand_w)
        v = torch.zeros(_size).fill_(rand_v)
    else:
        norm = torch.zeros(_size).fill_(45.0)
        w = torch.zeros(_size).fill_(1.0)
        v = torch.zeros(_size).fill_(0.5)

    init_kmap = torch.stack([norm, w, v], dim=1)

    return init_kmap


