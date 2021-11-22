import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tensorboardX import SummaryWriter
from config import config as _config
from dataset import dataset
from validation import P_validation

from models import PerPix_SFTMD as Generator
from models import Predictor
from utils import multiple_downsample, kernel_collage


# proj_directory = '/project'
# data_directory = '/dataset'


def train(config, epoch_from=0):
    dataParallel = True
    kernel_size = config['model']['kernel_size']

    print('process before training...')
    train_dataset = dataset(config['path']['dataset']['train'], patch_size=config['train']['patch size'],
                            scale=config['model']['scale'], kernel_size=kernel_size)
    train_data = DataLoader(
        dataset=train_dataset, batch_size=config['train']['batch size'],
        shuffle=True, num_workers=16
    )

    valid_dataset = dataset(config['path']['dataset']['valid'], patch_size=config['train']['patch size'],
                            scale=config['model']['scale'], is_train=False)
    valid_data = DataLoader(dataset=valid_dataset, batch_size=config['valid']['batch size'], num_workers=4)

    # training details - epochs & iterations
    iterations_per_epoch = len(train_dataset) // config['train']['batch size'] + 1
    n_epoch = config['train']['iterations_P'] // iterations_per_epoch + 1
    print('epochs scheduled: %d , iterations per epoch %d...' % (n_epoch, iterations_per_epoch))

    # define main SR network as generator
    generator = Generator(scale=config['model']['scale'], code_len=config['model']['code_len'])
    save_path_G = config['path']['ckpt']
    generator.load_state_dict(torch.load(save_path_G))
    for param in generator.parameters():
        param.requires_grad = False

    predictor = Predictor(config['model']['code_len']).cuda()
    save_path_P = save_path_G[:-4] + '_Predictor.pth'

    # optimizer
    learning_rate = config['train']['lr_P']
    P_optimizer = optim.Adam(predictor.parameters(), lr=learning_rate)
    save_path_Opt = save_path_P[:-4] + '_Opt.pth'
    lr_scheduler = optim.lr_scheduler.StepLR(P_optimizer, config['train']['decay']['every'],
                                             config['train']['decay']['by'])
    lr_scheduler.last_epoch = epoch_from * iterations_per_epoch

    # if training from scratch, remove all validation images and logs
    if epoch_from == 0:
        if os.path.exists(config['path']['validation']):
            _old = os.listdir(config['path']['validation'])
            for f in _old:
                if os.path.isfile(os.path.join(config['path']['validation'], f)):
                    os.remove(os.path.join(config['path']['validation'], f))
        if os.path.exists(config['path']['logs']):
            _old = os.listdir(config['path']['logs'])
            for f in _old:
                if os.path.isfile(os.path.join(config['path']['logs'], f)):
                    os.remove(os.path.join(config['path']['logs'], f))

    # if training not from scratch, load weights
    else:
        if os.path.exists(save_path_P):
            print('reading predictor checkpoints...')
            predictor.load_state_dict(torch.load(save_path_P))
            print('reading optimizer checkpoints...')
            P_optimizer.load_state_dict(torch.load(save_path_Opt))
        else:
            raise FileNotFoundError('Pretrained weight not found.')

    if not os.path.exists(config['path']['validation']):
        os.makedirs(config['path']['validation'])
    if not os.path.exists(os.path.dirname(config['path']['ckpt'])):
        os.makedirs(os.path.dirname(config['path']['ckpt']))
    if not os.path.exists(config['path']['logs']):
        os.makedirs(config['path']['logs'])
    writer = SummaryWriter(config['path']['logs'])

    # loss functions
    loss = nn.L1Loss().cuda()
    if dataParallel:
        generator = nn.DataParallel(generator)
        predictor = nn.DataParallel(predictor)
    generator = generator.cuda()
    predictor = predictor.cuda()

    # validation
    valid = P_validation(generator, predictor, valid_data, writer, config['path']['validation'])

    # training
    print('start training...')
    for epoch in range(epoch_from, n_epoch):
        generator = generator.eval()
        predictor = predictor.train()
        epoch_loss = 0
        for i, data in enumerate(train_data):
            # lr, gt, gt_k_map, _ = data
            # lr = lr.cuda()
            # gt_k_map = gt_k_map.cuda()
            hr, gt, kernels, k_code, _ = data
            hr = hr.cuda()
            kernels = kernels.cuda()
            k_code = k_code.cuda()
            kernels = kernels.view(-1, 1, 1, kernel_size, kernel_size)
            k_code = k_code.view(-1, config['model']['code_len'])

            # downsample via kernel collage
            lr = multiple_downsample(hr, kernels, config['model']['scale'])
            lr, gt_k_map = kernel_collage(lr, k_code)

            # if direct loss on kernel map
            #     pred_k_map = predictor(lr)
            #     p_loss = loss(pred_k_map, gt_k_map)
            #
            #     # back propagation
            #     P_optimizer.zero_grad()
            #     p_loss.backward()
            #     P_optimizer.step()
            #     epoch_loss += p_loss.item()
            #
            # if reconstruction with GT HR image
            #     gt = gt.cuda()
            #     pred_k_map = predictor(lr)
            #     sr = generator(lr, pred_k_map)
            #     p_loss = loss(sr, gt)
            #
            #     # back propagation
            #     P_optimizer.zero_grad()
            #     p_loss.backward()
            #     P_optimizer.step()
            #     epoch_loss += p_loss.item()
            #

            # with our proposed indirect reconstruction loss
            with torch.no_grad():
                gt = generator(lr, gt_k_map)

            pred_k_map = predictor(lr)
            sr = generator(lr, pred_k_map)
            p_loss = loss(sr, gt)

            # back propagation
            P_optimizer.zero_grad()
            p_loss.backward()
            P_optimizer.step()
            lr_scheduler.step()
            epoch_loss += p_loss.item()

        print('Training loss at {:d} : {:.8f}'.format(epoch, epoch_loss))

        # validation after epoch
        if (epoch + 1) % config['valid']['every'] == 0:
            is_best = valid.run(epoch + 1)

            # save validation image
            valid.save(tag='latest')
            if is_best:
                if dataParallel:
                    torch.save(predictor.module.state_dict(), save_path_P)
                else:
                    torch.save(predictor.state_dict(), save_path_P)
            torch.save(P_optimizer.state_dict(), save_path_Opt)

    # training process finished.
    # final validation and save checkpoints
    is_best = valid.run(n_epoch)
    valid.save(tag='final')
    writer.close()
    if is_best:
        if dataParallel:
            torch.save(predictor.module.state_dict(), save_path_P)
        else:
            torch.save(predictor.state_dict(), save_path_P)
    torch.save(P_optimizer.state_dict(), save_path_Opt)

    print('training finished.')


if __name__ == '__main__':
    train(_config, epoch_from=0)
