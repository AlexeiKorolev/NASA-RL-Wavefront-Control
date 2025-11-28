import torch
import torch.nn as nn
from typing import List, Optional

class FC1(nn.Module):
    def __init__(
        self,
        final_output_dim: int = 10,
        image_input_shape: tuple = (3, 16, 16),
        hidden_layers: Optional[List[int]] = None,
        activation: str = "leaky_relu",
        final_activation: Optional[str] = "leaky_relu",
        dropout: float = 0.0,
    ):
        super(FC1, self).__init__()
        self.image_input_shape = image_input_shape
        height, width, depth = image_input_shape

        in_features = height * width * depth
        hidden_layers = hidden_layers if hidden_layers is not None else [128, 64]

        def act(name: str):
            name = (name or "").lower()
            if name == "relu":
                return nn.ReLU()
            if name == "gelu":
                return nn.GELU()
            if name == "tanh":
                return nn.Tanh()
            if name == "none":
                return nn.Identity()
            # default
            return nn.LeakyReLU()

        layers: List[nn.Module] = [nn.Flatten()]
        prev = in_features
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(act(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, final_output_dim))
        if final_activation:
            layers.append(act(final_activation))

        self.final_net = nn.Sequential(*layers)

    def forward(self, imgs):
        # assert imgs.shape == self.image_input_shape, f"expected shape {self.image_input_shape}, got shape {imgs.shape}"

        return self.final_net(imgs)