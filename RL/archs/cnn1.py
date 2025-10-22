import torch
from torchviz import make_dot
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
            # nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(128, image_output_dim),
            nn.ReLU()
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
            nn.Linear(image_output_dim * 2 + image_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
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
        out = self.final_net(concat)

        return out
    
if __name__ == "__main__":
    torch.set_grad_enabled(True)

    model = CNN1()
    model.eval()

    batch_size = 2
    H, W = 40, 40
    img1 = torch.randn(batch_size, 1, H, W)
    img2 = torch.randn(batch_size, 1, H, W)
    dm_input_dim = model.list_encoder[0].in_features
    elem_list = torch.randn(batch_size, dm_input_dim)

    out = model(img1, img2, elem_list)

    # One-time graph visualization (saves cnn1_arch.png in CWD if torchviz is available)
    if torch.is_grad_enabled() and not getattr(model, "_viz_done", False):
        try:
            dot = make_dot(out, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
            dot.format = "png"
            dot.render("cnn1_arch", cleanup=True)
            model._viz_done = True
        except Exception:
            pass