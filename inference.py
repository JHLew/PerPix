import torch
from models import PerPix_SFTMD, Predictor
from config import config
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import os
import numpy as np
from glob import glob
from utils import visualize_kmap
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='path to pretrained model')
    parser.add_argument('--in_dir', type=str, required=True, help='directory of images to test on')
    parser.add_argument('--out_dir', type=str, default='./output', help='directory to save output results')
    parser.add_argument('--scale', type=int, default=4, help='scale for SR')
    parser.add_argument('--tuned', action='store_true', help='use the finetuned SR model')
    parser.add_argument('--kmap', action='store_true', help='save the estimated kernel map')

    args = parser.parse_args()

    save_path_G = './ckpt/{}.pth'.format(args.ckpt)
    save_path_P = './ckpt/{}_Predictor.pth'.format(args.ckpt)

    print('loading model...')
    generator = PerPix_SFTMD(args.scale, config['model']['code_len']).cuda()
    predictor = Predictor(config['model']['code_len']).cuda()
    if args.tuned:
        save_path_G = save_path_G[:-4] + '_tuned.pth'

    print(save_path_P)
    print(save_path_G)
    predictor.load_state_dict(torch.load(save_path_P))
    generator.load_state_dict(torch.load(save_path_G))

    print('loaded.')

    input_list = sorted(glob(os.path.join(args.in_dir, '*')))

    os.makedirs(args.out_dir, exist_ok=True)
    if args.kmap:
        map_dir = os.path.join(args.out_dir, 'kmap')
        os.makedirs(map_dir, exist_ok=True)

    for img in input_list:
        name = os.path.basename(img)
        img = Image.open(img).convert('RGB')
        print(name)

        img = to_tensor(img).cuda().unsqueeze(0)
        with torch.no_grad():
            pred_k_map = predictor(img)
            sr = generator(img, pred_k_map)

        if args.kmap:
            pred_k_map = np.array(pred_k_map[0].cpu()).transpose((1, 2, 0))
            np.save(os.path.join(map_dir, name[:-4] + '.npy'), pred_k_map)
            visualize_kmap(pred_k_map, s=16, out_dir=args.out_dir, tag=name[:-4], each=False)

        sr = to_pil_image(sr[0].cpu())
        sr.save(os.path.join(args.out_dir, name))
