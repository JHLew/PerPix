name = 'PerPix_x4'  # configurate name for the model: used for saving ckpt, validation, and logs
config = {
  'train': {
    'patch size': 301,  # HR size 301 ==> LR size 64. downsampling is done via strided convolution w/o padding (303 or 175 for x2)
    'batch size': 16,
    'iterations_G': 300000,  # for G: 300000, for C: 200000
    'iterations_P': 250000,  # for G: 300000, for C: 200000
    'iterations_T': 100000,  # for G: 300000, for C: 200000
    'lr_G': 2e-4,
    'lr_P': 1e-4,
    'lr_T': 1e-4,
    'decay': {
      'every': 200000,
      'by': 0.1
    },
  },

  'valid': {
    'batch size': 1,
    'every': 50,
  },

  'model': {
    'scale': 4,
    'code_len': 3,
    'kernel_size': 49,
  },

  'path': {
    'project': '/project',
    'ckpt': './ckpt/{}.pth'.format(name),
    'dataset': {
      'train': ['/dataset/DIV2K/train_HR',
                '/dataset/Flickr2K/Flickr2K_HR'],
      'valid': ['/dataset/DIV2K/valid_HR'],
    },

    'validation': './validation/{}'.format(name),
    'logs': './logs/{}'.format(name)
  }
}