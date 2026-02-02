#!/usr/bin/env python
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision



def rot_mat(theta, device):
    mat = torch.tensor([[
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0]
    ]], dtype=torch.float, device=device)
    return mat

class GraspNet(nn.Module):

    def __init__(self, use_cuda, num_rotations):
        super(GraspNet, self).__init__()
        self.use_cuda = use_cuda
        self.device = 'cuda' if self.use_cuda else 'cpu'

        # Initialize network trunks with DenseNet pre-trained on ImageNet
        self.grasp_color_trunk = torchvision.models.densenet.densenet121(
            weights='DEFAULT'
        )
        self.grasp_depth_trunk = torchvision.models.densenet.densenet121(
            weights='DEFAULT'
        )

        self.num_rotations = num_rotations

        # Construct network branches for pushing and grasping
        self.graspnet = nn.Sequential(OrderedDict([
            ('grasp-norm0', nn.BatchNorm2d(2048)),
            ('grasp-relu0', nn.ReLU(inplace=True)),
            ('grasp-conv0', nn.Conv2d(
                2048, 64, kernel_size=1, stride=1, bias=False
            )),
            ('grasp-norm1', nn.BatchNorm2d(64)),
            ('grasp-relu1', nn.ReLU(inplace=True)),
            ('grasp-conv1', nn.Conv2d(
                64, 1, kernel_size=1, stride=1, bias=False
            ))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'grasp-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal_(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

        # Initialize output variable (for backprop)
        self.interm_feat = []
        self.output_prob = []

    def forward(self, input_color_data, input_depth_data, specific_rotation=-1):
        output_prob = []
        interm_feat = []

        # Apply rotations to images
        if specific_rotation < 0:
            cicles = self.num_rotations
            start_rot = 0
        else:
            cicles = 1
            start_rot = int(specific_rotation)
            
        for rotate_idx in range(start_rot, cicles + start_rot):
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE neural network
            affine_mat_before = rot_mat(-rotate_theta, self.device)
            flow_grid_before = F.affine_grid(
                affine_mat_before, input_color_data.size(), align_corners=False
            )
            
            # Rotate images clockwises
            rotate_color = F.grid_sample(
                input_color_data, flow_grid_before, mode='nearest',
                align_corners=False
            )
            rotate_depth = F.grid_sample(
                input_depth_data, flow_grid_before, mode='nearest',
                align_corners=False
            )

            # Compute intermediate features 
            interm_grasp_color_feat = self.grasp_color_trunk.features(
                rotate_color
            )
            interm_grasp_depth_feat = self.grasp_depth_trunk.features(
                rotate_depth
            )
            interm_grasp_feat = torch.cat(
                (interm_grasp_color_feat, interm_grasp_depth_feat), dim=1
            )
            interm_feat.append(interm_grasp_feat)

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = rot_mat(rotate_theta, self.device)
            flow_grid_after = F.affine_grid(
                affine_mat_after, interm_grasp_feat.size(), align_corners=False
            )
            unrotate_feat = F.grid_sample(
                self.graspnet(interm_grasp_feat), flow_grid_after,
                mode='nearest', align_corners=False
            )

            # Forward pass through branches, undo rotation on output
            # predictions, upsample results
            output_prob.append(nn.Upsample(
                scale_factor=16, mode='bilinear'
            ).forward(unrotate_feat))
            
        return output_prob, interm_feat
