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


class AndreAIWhiteModel(BaseModel):
    def name(self):
        return 'AndreAIWhiteModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['content_vgg', 'L1', 'perceptual']
        # specify the images G_A'you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['source_image', 'source_image_mask_multi', 'white_source_image']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['White']
        else:  # during test time, only load Gs
            self.model_names = ['White']

        # load/define networks
        self.netWhite = networks.define_G(opt.input_nc_warp, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.VGG19 = networks.VGG19(requires_grad=False).cuda()
        use_sigmoid = opt.no_lsgan

        if self.isTrain:
            # define loss functions
            self.criterionContent = networks.ContentLoss().to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionPerceptual = networks.PerceptualLoss().to(self.device)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netWhite.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.source_image = input['source_image'].to(self.device)
        self.source_image_mask = input['source_image_mask'].to(self.device)
        self.source_image_mask_multi = self.source_image_mask.expand(2,3,512,512)

    def get_vgg_loss(self):
        source_features = self.VGG19(self.source_image)
        fake_features = self.VGG19(self.white_source_image)
        return self.criterionContent(source_features, fake_features)

    def get_perceptual_loss(self):
        input_features = self.VGG19(self.source_image_mask_multi)
        fake_features = self.VGG19(self.white_source_image)
        return self.criterionPerceptual(input_features, fake_features)

    def forward(self):
        self.white_source_image = self.netWhite(torch.cat([self.source_image, self.source_image_mask], dim=1))
        self.white_source_image = self.white_source_image.mul(self.source_image_mask)

    def backward_G(self):

        # get content loss + get style loss
        self.loss_content_vgg = self.get_vgg_loss()

        # get perceptual loss
        self.loss_perceptual = self.get_perceptual_loss()

        # get L1 loss
        self.loss_L1 = self.criterionL1(self.source_image_mask_multi, self.white_source_image)

        # combined loss
        self.loss_G = 2 * self.loss_content_vgg + self.loss_perceptual + 10 * self.loss_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
