import numpy as np
import cv2
import torch
from .models import GraspNet
from scipy import ndimage
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, force_cpu, num_rotations):
        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Fully convolutional Q network for deep reinforcement learning
        self.model = GraspNet(self.use_cuda, num_rotations)
        self.resize = torchvision.transforms.Resize((448, 448))
        self.normalize_rgb = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.normalize_depth = torchvision.transforms.Normalize(
            mean=[0.01, 0.01, 0.01], std=[0.03, 0.03, 0.03]
        )

        # Initialize Huber loss
        self.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        
        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer Adam
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-4, weight_decay=2e-5,
            betas=(0.9,0.99)
        )
        self.iteration = 0

    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, depth_heightmap, specific_rotation=-1):
        # (224, 224, 3)
        # Apply 2x scale to input heightmaps
        # The third dimension is rgb
        color = color_heightmap.permute([2, 0, 1])[None]
        depth = depth_heightmap[None, None].to(torch.float)
        
        color_2x = self.resize(color)
        depth_2x = self.resize(depth)

        # Add extra padding (to handle rotations inside network)
        diag_length = color_2x.shape[-1] * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_2x.shape[-1]) / 2)
        
        color_2x = F.pad(color_2x, (padding_width, ) * 4, "constant", 0)
        depth_2x = F.pad(depth_2x, (padding_width, ) * 4, "constant", 0) 

        # Pre-process color image (scale and normalize)
        input_color = color_2x.to(torch.float) / 255
        input_color = self.normalize_rgb(input_color)
        
        # Pre-process depth image (normalize)
        input_depth = self.normalize_depth(
            torch.cat([depth_2x, depth_2x, depth_2x], dim=1)
        )
        
        # Pass input data through model
        output_prob, state_feat = self.model.forward(
            input_color, input_depth, specific_rotation
        )

        output_prob = torch.cat(output_prob, dim=0).squeeze(dim=1)
        grasp_predictions = output_prob[
            :,
            int(padding_width / 2): int(color_2x.shape[-1] / 2 - padding_width / 2),
            int(padding_width / 2): int(color_2x.shape[-1] / 2 - padding_width / 2)
        ]
        return grasp_predictions, state_feat, output_prob

    # Compute labels and backpropagate
    def backprop(
        self, color_heightmap, depth_heightmap, best_pix_ind, label_value
    ):

        # Compute labels
        label = torch.zeros((1, 320, 320), device=self.model.device)
        action_area = torch.zeros((224, 224), device=self.model.device)
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1

        tmp_label = torch.zeros((224, 224), device=self.model.device)
        tmp_label[action_area > 0] = label_value
        label[0,48:(320-48),48:(320-48)] = tmp_label

        # Compute label mask
        label_weights = torch.zeros(label.shape, device=self.model.device)
        label_weights[0, 48:(320-48), 48:(320-48)] = action_area

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0

        # Do forward pass with specified rotation (to save gradients)
        grasp_pred, _, output_prob = self.forward(
            color_heightmap, depth_heightmap, specific_rotation=best_pix_ind[0]
        )
        loss = torch.sum(
            self.criterion(output_prob.view(1, 320, 320), label) * label_weights
        )
        loss.backward()
        loss_value1 = loss.detach().cpu().numpy()

        opposite_rotate_idx = (
            best_pix_ind[0] + self.model.num_rotations / 2
        ) % self.model.num_rotations

        *_, output_prob = self.forward(
            color_heightmap, depth_heightmap,
            specific_rotation=opposite_rotate_idx
        )

        loss = torch.sum(
            self.criterion(output_prob.view(1, 320, 320), label) * label_weights
        )
        loss.backward()
        loss_value2 = loss.detach().cpu().numpy()

        loss_value = loss_value1 + loss_value2

        print('Training loss: %f' % (loss_value))
        self.optimizer.step()
        return loss_value, grasp_pred


    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        state_dict = torch.load(path, weights_only=True)
        self.model.load_state_dict(state_dict)
