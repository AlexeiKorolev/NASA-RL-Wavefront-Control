import torch
import torch.nn as nn

class CNN1(nn.Module):
    def __init__(self, 
                 image_output_dim=32, 
                 dm_input_dim=10, 
                 dm_hidden_dim=32, 
                 final_output_dim=10, 
                 image_input_shape=(3, 40, 40)):
        super(CNN1, self).__init__()
        height, width = image_input_shape[-2:]

        # Shared encoder for both images
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * height * width, 256),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(64, image_output_dim)
        )

        # Encoder for the list of elements
        self.list_encoder = nn.Sequential(
            nn.Linear(dm_input_dim, dm_hidden_dim),
            nn.ReLU(),
            nn.Linear(dm_hidden_dim, image_output_dim),
            nn.ReLU()
        )

        # Final fusion and output
        self.final_net = nn.Sequential(
            nn.Linear(image_output_dim * 2 + image_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, final_output_dim)
        )

    def forward(self, img1, img2, elem_list):
        # img1, img2: (batch, 1, H, W)
        # elem_list: (batch, list_input_dim)
        img1_feat = self.image_encoder(img1)
        # print(f"img1_feat shape: {img1_feat.shape}")
        img2_feat = self.image_encoder(img2)
        list_feat = self.list_encoder(elem_list)
        # print(f"list_feat shape: {list_feat.shape}")
        concat = torch.cat([img1_feat, img2_feat, list_feat], dim=1)
        return self.final_net(concat)