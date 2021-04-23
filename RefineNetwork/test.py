import os

import numpy as np

from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util import util
from util.visualizer import save_images
from util import html
from data import refine_cloth_test_dataset
from torch.utils.data import DataLoader
import torchvision
import torch
from PIL import Image


if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.no_dropout = True
    opt.display_id = -1   # no visdom display
    # data_loader = CreateDataLoader(opt)
    # dataset = data_loader.load_data()
    test_set = refine_cloth_test_dataset.RefineClothTestDataset(opt)
    dataset = DataLoader(test_set, batch_size=1, shuffle=True)
    dataset_size = len(dataset)
    print('#test images = %d' % dataset_size)
    model = create_model(opt)
    model.setup(opt)
    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        # visuals = model.get_current_visuals()
        # img_path = model.get_image_paths()
        # if i % 5 == 0:
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        test = model.fake_image.cpu().squeeze_(0)
        print(test.shape)
        img_path = os.path.join('/home/ubuntu/Desktop/data-conversion/RefineNetwork/results/test_latest/images', f'test_{i}_fakeimage.png')
        tensor_to_pil = torchvision.transforms.ToPILImage()(test)
        tensor_to_pil.save(img_path)

    # save the website
    # webpage.save()
