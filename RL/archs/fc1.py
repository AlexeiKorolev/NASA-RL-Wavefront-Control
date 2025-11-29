import torch
import torch.nn as nn
from typing import List, Optional

class FC1(nn.Module):
    """Simple MLP for coronagraph images with optional Conv2d encoder front-end."""
    def __init__(
        self,
        final_output_dim: int = 10,
        image_input_shape: tuple = (3, 16, 16),
        hidden_layers: Optional[List[int]] = None,
        activation: str = "leaky_relu",
        final_activation: Optional[str] = "none",
        dropout: float = 0.0,
        encoder_enabled: bool = False,
        filter_sizes: Optional[List[int]] = None,
        filter_channels: Optional[List[int]] = None,
        final_embedding_size: Optional[int] = None,
        final_embedding_channels: Optional[int] = None,
    ):
        super(FC1, self).__init__()
        self.image_input_shape = image_input_shape
        self.encoder_enabled = encoder_enabled
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
        
        encoder_out_features = in_features
        self.encoder: Optional[nn.Sequential] = None
        if encoder_enabled:
            if not filter_sizes or not filter_channels:
                raise ValueError("encoder_enabled=True requires filter_sizes and filter_channels")
            if len(filter_sizes) != len(filter_channels):
                raise ValueError("filter_sizes and filter_channels must have the same length")

            target_size = final_embedding_size or height
            target_channels = final_embedding_channels or filter_channels[-1]
            if target_size <= 0:
                raise ValueError("final_embedding_size must be positive")
            if target_channels <= 0:
                raise ValueError("final_embedding_channels must be positive")

            conv_layers: List[nn.Module] = []
            in_channels = depth
            for kernel, out_channels in zip(filter_sizes, filter_channels):
                # Keep padding so H/W remain constant, simplifying the first Linear layer sizing
                conv_layers.append(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel,
                        padding=kernel // 2,
                    )
                )
                conv_layers.append(act(activation))
                in_channels = out_channels

            if in_channels != target_channels:
                conv_layers.append(nn.Conv2d(in_channels, target_channels, kernel_size=1))
                conv_layers.append(act(activation))
                in_channels = target_channels

            # force spatial dimensions to requested embedding size
            conv_layers.append(nn.AdaptiveAvgPool2d((target_size, target_size)))
            conv_layers.append(nn.Flatten())
            self.encoder = nn.Sequential(*conv_layers)
            encoder_out_features = in_channels * target_size * target_size

        layers: List[nn.Module] = []
        if not encoder_enabled:
            layers.append(nn.Flatten())

        prev = encoder_out_features
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

    def forward(self, imgs: torch.Tensor):
        # Optional CNN encoder expects channel-first tensors
        if self.encoder is not None:
            if imgs.ndim != 4:
                raise ValueError("Encoder expects 4D input (N, H, W, C) or (N, C, H, W)")

            if imgs.shape[1] == self.image_input_shape[2]:  # already NCHW
                x = imgs
            elif imgs.shape[-1] == self.image_input_shape[2]:  # NHWC -> NCHW
                x = imgs.permute(0, 3, 1, 2).contiguous()
            else:
                raise ValueError("Input channel dimension does not match image_input_shape")

            feats = self.encoder(x)
        else:
            feats = imgs

        return self.final_net(feats)