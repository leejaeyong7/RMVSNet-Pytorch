import torch
import torch.nn as nn
import numpy as np
from os import path

from .gru import GRU
from .unet_ds2gn import UNetDS2GN

from .warping import *

class RMVSNet(nn.Module):
    def __init__(self, train=False):
        super(RMVSNet, self).__init__()
        # setup network modules

        self.feature_extractor = UNetDS2GN()
        
        gru_input_size = self.feature_extractor.output_size
        gru1_output_size = 16
        gru2_output_size = 4
        gru3_output_size = 2
        self.gru1 = GRU(gru_input_size, gru1_output_size, 3)
        self.gru2 = GRU(gru1_output_size, gru2_output_size, 3)
        self.gru3 = GRU(gru2_output_size, gru3_output_size, 3)

        self.prob_conv = nn.Conv2d(2, 1, 3, 1, 1)

        file_path = path.dirname(path.abspath(__file__))
        pretrained_weights_file = path.join(file_path,
                                            'RMVSNet-pretrained.pth')
        pretrained_weights = torch.load(pretrained_weights_file)
        self.load_state_dict(pretrained_weights)

    def compute_cost_volume(self, warped):
        '''
        Warped: N x C x M x H x W

        returns: 1 x C x M x H x W
        '''
        warped_sq = warped ** 2
        av_warped = warped.mean(0)
        av_warped_sq = warped_sq.mean(0)
        cost = av_warped_sq - (av_warped ** 2)

        return cost.unsqueeze(0)


    def forward(self, images, intrinsics, extrinsics, depth_start, depth_interval, depth_num):
        '''
        Takes all entry and outputs probability volume

        N x D x H x W probability map
        '''
        f = self.feature_extractor(images)

        Hs = get_homographies(f, intrinsics, extrinsics, depth_start, depth_interval, depth_num)

        warped = warp_homographies(f, Hs)
        cost = self.compute_cost_volume(warped)


        N, C, D, H, W = warped.shape
        cost_1 = None
        cost_2 = None
        cost_3 = None
        depth_costs = []
        for d in range(depth_num):
            cost_d = cost[:, :, d]

            cost_1 = self.gru1(-cost_d, cost_1)
            cost_2 = self.gru2(cost_1, cost_2)
            cost_3 = self.gru3(cost_2, cost_3)

            reg_cost = self.prob_conv(cost_3)
            depth_costs.append(reg_cost)

        prob_volume = torch.cat(depth_costs, 1)
        return torch.softmax(prob_volume, 1)

