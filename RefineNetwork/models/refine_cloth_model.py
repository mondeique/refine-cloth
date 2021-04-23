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


class RefineClothmodel(BaseModel):
    def name(self):
        return 'RefineClothmodel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['L1', 'content_vgg', 'perceptual']
        # specify the images G_A'you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['matched_mask', 'source_image', 'real_image', 'fake_image']

        self.visual_names = visual_names_A
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A']
        else:  # during test time, only load Gs
            self.model_names = ['G_A']

        # load/define networks
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
        self.real_image = input['real_image'].to(self.device)
        self.real_image_mask = input['real_image_mask'].to(self.device)
        self.source_image = input['source_image'].to(self.device)
        self.source_image_mask = input['source_image_mask'].to(self.device)
        self.matched_mask = input['matched_image'].to(self.device)

    def get_vgg_loss(self):
        source_features = self.VGG19(self.source_mask)
        fake_features = self.VGG19(self.fake_image)
        return self.criterionContent(source_features, fake_features)

    def get_perceptual_loss(self):
        image_features = self.VGG19(self.image_mask)
        fake_features = self.VGG19(self.fake_image)
        return self.criterionPerceptual(image_features, fake_features)

    def forward(self):
        a = torch.ones([4,1,256,256]).to(self.device)
        self.real_image_mask = torch.sub(a, self.real_image_mask).type(torch.uint8)
        self.source_image_mask = torch.sub(a, self.source_image_mask).type(torch.uint8)
        self.image_mask = self.real_image.mul(self.real_image_mask)
        self.source_mask = self.source_image.mul(self.source_image_mask)
        self.matched_mask = self.matched_mask.mul(self.source_image_mask)
        self.fake_image = self.netG_A(torch.cat([self.matched_mask, self.source_mask], dim=1))

        self.fake_image = self.fake_image.mul(self.source_image_mask)

    def backward_G(self):

        # get content loss + get style loss
        self.loss_content_vgg = self.get_vgg_loss()

        # get perceptual loss
        self.loss_perceptual = self.get_perceptual_loss()

        # get L1 loss
        self.loss_L1 = 2 * self.criterionL1(self.image_mask, self.fake_image)

        # combined loss
        self.loss_G = self.loss_content_vgg + self.loss_perceptual + self.loss_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
