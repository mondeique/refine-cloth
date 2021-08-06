from .base_model import BaseModel
from .refine_cloth_model import RefineClothmodel
from . import networks
import torch


class TestModel(RefineClothmodel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser.set_defaults(dataset_mode='refine_cloth_test')
        parser.add_argument('--model_suffix', type=str, default='_A',
                            help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will'
                            ' be loaded as the generator of TestModel')

        return parser

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_image_mask', 'cloth_mask', 'warped_cloth']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['G' + opt.model_suffix]

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        self.real_image = input['real_image'].to(self.device)
        self.real_image_mask = input['real_image_mask'].to(self.device)
        self.source_image = input['source_image'].to(self.device)
        self.source_image_mask = input['source_image_mask'].to(self.device)
        self.matched_mask = input['matched_image'].to(self.device)

    def forward(self):
        a = torch.ones([1, 1, 512, 512]).to(self.device)
        self.real_image_mask_sub = torch.sub(a, self.real_image_mask).type(torch.uint8)
        self.source_image_mask_sub = torch.sub(a, self.source_image_mask).type(torch.uint8)
        self.image_mask = self.real_image.mul(self.real_image_mask_sub)
        self.source_mask = self.source_image.mul(self.source_image_mask_sub)
        self.matched_mask = self.matched_mask.mul(self.source_image_mask_sub)
        self.fake_image = self.netG_A(torch.cat([self.matched_mask, self.source_mask], dim=1))

        self.fake_image = self.fake_image.mul(self.source_image_mask_sub)

        self.fake_image = self.fake_image.add(self.source_image_mask)
