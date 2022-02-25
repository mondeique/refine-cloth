import os

import torch
from PIL import Image
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.gramMatrix import StyleLoss
import torchvision
import numpy as np
from util.wasserstein_loss import calc_gradient_penalty


class AndreAIModel(BaseModel):
    def name(self):
        return 'AndreAImodel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['content_vgg_real', 'content_vgg_white', 'perceptual_matched', 'L1_matched']
        # specify the images G_A'you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['source_image', 'white_source_image', 'hist_real_image', 'input_cloth_image', 'fake_image']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # load/define networks
        self.netWhite = networks.define_G(opt.input_nc_warp, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        load_filename = '%s_net_%s.pth' % ('latest', 'White')
        load_path = os.path.join(self.save_dir, load_filename)
        net = getattr(self, 'net' + 'White')
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # if you are using PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        print(state_dict.keys())
        ## patch InstanceNorm checkpoints prior to 0.4
        #for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
        #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
        from collections import OrderedDict
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():
            if 'module' not in k:
                k = 'module.' + k
            else:
                k = k.replace('features.module.', 'module.features.')
            new_state_dict[k] = v
        self.netWhite.load_state_dict(new_state_dict)

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                         not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.VGG19 = networks.VGG19(requires_grad=False).cuda()
        use_sigmoid = opt.no_lsgan

        if self.isTrain:
            # define loss functions
            self.criterionContent = networks.ContentLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionPerceptual = networks.PerceptualLoss().to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.source_image = input['source_image'].to(self.device)
        self.source_image_mask = input['source_image_mask'].to(self.device)
        self.source_cloth_image = input['source_cloth_image'].to(self.device)
        self.source_cloth_image_mask = input['source_cloth_image_mask'].to(self.device)
        self.input_cloth_image = input['input_cloth_image'].to(self.device)
        self.input_cloth_image_mask = input['input_cloth_image_mask'].to(self.device)
        self.matched_mask = input['matched_image'].to(self.device)
        self.hist_real_image = input['matched_real_image'].to(self.device)


    def get_vgg_loss(self):
        source_features = self.VGG19(self.white_source_image)
        source_image_features = self.VGG19(self.source_image)
        fake_features = self.VGG19(self.fake_image)
        return self.criterionContent(source_image_features, fake_features), self.criterionContent(source_features, fake_features)

    def get_perceptual_loss(self):
        matched_features = self.VGG19(self.hist_real_image)
        fake_features = self.VGG19(self.fake_image)
        return self.criterionPerceptual(matched_features, fake_features)

    def forward(self):
        a = torch.ones([1, 1, 256, 256]).to(self.device)
        self.input_cloth_image_mask = torch.sub(a, self.input_cloth_image_mask).type(torch.uint8)
        self.source_cloth_image_mask = torch.sub(a, self.source_cloth_image_mask).type(torch.uint8)

        self.source_image = self.source_image.mul(self.source_image_mask)
        self.input_cloth_image = self.input_cloth_image.mul(self.input_cloth_image_mask)
        self.source_cloth_image = self.source_cloth_image.mul(self.source_cloth_image_mask)
        self.matched_mask = self.matched_mask.mul(self.source_cloth_image_mask)
        self.white_source_image = self.netWhite(torch.cat([self.source_image, self.source_image_mask], dim=1))
        self.white_source_image = self.white_source_image.mul(self.source_image_mask)

        # white source image 에 input cloth color 입히고 content 만 따오도록 설정

        self.hist_real_image = self.hist_real_image.mul(self.source_image_mask)

        self.fake_image = self.netG_A(torch.cat([self.hist_real_image, self.white_source_image], dim=1))

        self.fake_image = self.fake_image.mul(self.source_image_mask)

    def backward_G(self):

        # get content loss + get style loss
        self.loss_content_vgg_real, self.loss_content_vgg_white = self.get_vgg_loss()

        # get perceptual loss
        self.loss_perceptual_matched = self.get_perceptual_loss()

        # get L1 loss
        self.loss_L1_matched = self.criterionL1(self.hist_real_image, self.fake_image)

        # combined loss
        self.loss_G = 10 * self.loss_content_vgg_real + 5 * self.loss_content_vgg_white + 8 * self.loss_perceptual_matched + 30 * self.loss_L1_matched
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
