from .base_model import BaseModel
from .refine_cloth_model import RefineClothmodel
from .andre_ai_model import AndreAIModel
from . import networks
import torch
import os


class TestModel(AndreAIModel):
    def name(self):
        return 'TestModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        assert not is_train, 'TestModel cannot be used in train mode'
        parser.set_defaults(dataset_mode='andre_ai_test')
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
        # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
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

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see BaseModel.load_networks
        setattr(self, 'netG' + opt.model_suffix, self.netG)

    def set_input(self, input):
        self.source_image = input['source_image'].to(self.device)
        self.source_image_mask = input['source_image_mask'].to(self.device)
        self.input_cloth_image = input['input_cloth_image'].to(self.device)
        self.input_cloth_image_mask = input['input_cloth_image_mask'].to(self.device)
        self.hist_real_image = input['matched_real_image'].to(self.device)

    def forward(self):
        a = torch.ones([1, 1, 256, 256]).to(self.device)
        self.input_cloth_image_mask = torch.sub(a, self.input_cloth_image_mask).type(torch.uint8)

        self.original_image = self.source_image
        self.source_image = self.source_image.mul(self.source_image_mask)
        self.input_cloth_image = self.input_cloth_image.mul(self.input_cloth_image_mask)
        self.white_source_image = self.netWhite(torch.cat([self.source_image, self.source_image_mask], dim=1))
        self.white_source_image = self.white_source_image.mul(self.source_image_mask)

        # white source image 에 input cloth color 입히고 content 만 따오도록 설정

        self.hist_real_image = self.hist_real_image.mul(self.source_image_mask)

        self.fake_image = self.netG_A(torch.cat([self.hist_real_image, self.white_source_image], dim=1))

        self.fake_image = self.fake_image.mul(self.source_image_mask)

        # make final image
        self.other_image = torch.sub(self.original_image, self.source_image)
        self.final_image = torch.add(self.other_image, self.fake_image)
