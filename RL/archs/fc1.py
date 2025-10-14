import torch
import torch.nn as nn

class FC1(nn.Module):
    def __init__(self, 
                 final_output_dim=10, 
                 image_input_shape=(3, 16, 16)):
        super(FC1, self).__init__()
        self.image_input_shape = image_input_shape
        height, width, depth = image_input_shape

        self.final_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(height * width * depth, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, final_output_dim),
            nn.LeakyReLU(),
        )

    def forward(self, imgs):
        # assert imgs.shape == self.image_input_shape, f"expected shape {self.image_input_shape}, got shape {imgs.shape}"

        return self.final_net(imgs)