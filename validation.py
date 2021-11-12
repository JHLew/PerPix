import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import os
from config import config
from utils import multiple_downsample, kernel_collage


class G_validation:
    mse = nn.MSELoss()

    def __init__(self, network, loader, writer, save_path, kmap=True):
        self.generator = network
        self.loader = loader
        self.writer = writer
        self.n = loader.dataset.__len__()
        self.save_path = save_path
        self.kmap = kmap
        self.best = 100

    def run(self, epoch):
        generator = self.generator.eval()
        val_mse_loss = 0

        self.lr_list = []
        self.valid_outputs = []
        self.img_names = []

        for _, val_data in enumerate(self.loader):
            # lr, gt, gt_k_map, img_name = val_data
            # lr = lr.cuda()
            # gt = gt.cuda()
            # gt_k_map = gt_k_map.cuda()

            hr, gt, kernels, k_code, img_name = val_data
            hr = hr.cuda()
            gt = gt.cuda()
            kernels = kernels.cuda()
            k_code = k_code.cuda()
            kernels = kernels.view(-1, 1, 1, config['model']['kernel_size'], config['model']['kernel_size'])
            k_code = k_code.view(-1, config['model']['code_len'])

            # downsample via kernel collage
            lr = multiple_downsample(hr, kernels, config['model']['scale'])
            lr, kernel_map = kernel_collage(lr, k_code)

            with torch.no_grad():
                if self.kmap:
                    sr = generator(lr, kernel_map)
                else:
                    sr = generator(lr)

                self.lr_list.append(lr[0].cpu())
                self.valid_outputs.append(sr[0].cpu())
                self.img_names.append(img_name[0])

                val_mse_loss += self.mse(sr, gt).item()

        val_mse_loss /= self.n
        print("Validation loss(MSE) at %2d:\t==>\t%.6f" % (epoch, val_mse_loss))
        self.writer.add_scalar('G Loss/Total_G_Loss', val_mse_loss, (epoch + 1))
        self.writer.add_scalar('G Loss/HR_loss', val_mse_loss, (epoch + 1))
        self.generator.train()
        if self.best >= val_mse_loss:
            self.best = val_mse_loss
            return True
        else:
            return False


    def save(self, tag):
        save_dir = os.path.join(self.save_path, str(tag))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(self.n):
            lr = self.lr_list[i]
            img = self.valid_outputs[i]
            name = self.img_names[i]

            F.to_pil_image(lr).save(os.path.join(save_dir, 'LR_' + name))
            F.to_pil_image(img).save(os.path.join(save_dir, name))


class P_validation:
    mse = nn.MSELoss()

    def __init__(self, generator, predictor, loader, writer, save_path):
        self.generator = generator
        self.predictor = predictor
        self.loader = loader
        self.writer = writer
        self.n = loader.dataset.__len__()
        self.save_path = save_path + '_P'
        self.best = 100

    def run(self, epoch):
        generator = self.generator.eval()
        predictor = self.predictor.eval()
        val_mse_loss = 0

        self.lr_list = []
        self.valid_outputs = []
        self.img_names = []

        for _, val_data in enumerate(self.loader):
            # lr, gt, gt_k_map, img_name = val_data
            # lr = lr.cuda()
            # gt = gt.cuda()

            hr, gt, kernels, k_code, img_name = val_data
            hr = hr.cuda()
            gt = gt.cuda()
            kernels = kernels.cuda()
            k_code = k_code.cuda()
            kernels = kernels.view(-1, 1, 1, config['model']['kernel_size'], config['model']['kernel_size'])
            k_code = k_code.view(-1, config['model']['code_len'])

            # downsample via kernel collage
            lr = multiple_downsample(hr, kernels, config['model']['scale'])
            lr, kernel_map = kernel_collage(lr, k_code)

            with torch.no_grad():
                pred_k_map = predictor(lr)
                sr = generator(lr, pred_k_map)

                self.lr_list.append(lr[0].cpu())
                self.valid_outputs.append(sr[0].cpu())
                self.img_names.append(img_name[0])

                val_mse_loss += self.mse(sr, gt).item()

        val_mse_loss /= self.n
        print("Validation loss(MSE) at %2d:\t==>\t%.6f" % (epoch, val_mse_loss))
        self.writer.add_scalar('C Loss/Total_G_Loss', val_mse_loss, (epoch + 1))
        self.writer.add_scalar('C Loss/HR_loss', val_mse_loss, (epoch + 1))
        self.generator.train()
        self.predictor.train()
        if self.best >= val_mse_loss:
            self.best = val_mse_loss
            return True
        else:
            return False


    def save(self, tag):
        save_dir = os.path.join(self.save_path, str(tag))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(self.n):
            lr = self.lr_list[i]
            img = self.valid_outputs[i]
            name = self.img_names[i]

            F.to_pil_image(lr).save(os.path.join(save_dir, 'LR_' + name))
            F.to_pil_image(img).save(os.path.join(save_dir, name))
