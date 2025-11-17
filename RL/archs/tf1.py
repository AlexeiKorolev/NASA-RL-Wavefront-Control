import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def _act(name: str) -> nn.Module:
    name = (name or "").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    # default
    return nn.LeakyReLU()


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=attn_dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


class TF1(nn.Module):
    """
    Vision Transformer with an FC-style customizable head.

    Keeps FC1-like customizability for the prediction head via:
    - hidden_layers: list of MLP head sizes
    - activation: nonlinearity for head layers
    - final_activation: optional final activation after output
    - dropout: dropout used in head and transformer MLPs

    Additional ViT params (reasonable defaults provided):
    - patch_size: size of square patches
    - dim: token embedding dimension
    - depth: number of transformer blocks
    - heads: attention heads
    - mlp_dim: transformer MLP hidden dimension
    - attn_dropout, emb_dropout: attention / embedding dropout
    - use_cls_token: use a [CLS] token or mean pool tokens
    """

    def __init__(
        self,
        final_output_dim: int = 10,
        image_input_shape: Tuple[int, int, int] = (3, 16, 16),
        hidden_layers: Optional[List[int]] = None,
        activation: str = "leaky_relu",
        final_activation: Optional[str] = "leaky_relu",
        dropout: float = 0.0,
        # ViT specifics
        patch_size: int = 8,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        mlp_dim: int = 256,
        attn_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.image_input_shape = image_input_shape  # (C, H, W) or (H, W, C) depending on convention
        self.patch_size = patch_size
        self.use_cls_token = use_cls_token

        # Normalize input shape to (C, H, W)
        if len(image_input_shape) != 3:
            raise ValueError("image_input_shape must be a 3-tuple")
        c, h, w = image_input_shape if image_input_shape[0] in (1, 3) else (image_input_shape[2], image_input_shape[0], image_input_shape[1])
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(f"H and W must be divisible by patch_size={patch_size}; got {(h, w)}")

        num_patches = (h // patch_size) * (w // patch_size)

        # Patch embedding via Conv2d
        self.patch_embed = nn.Conv2d(in_channels=c, out_channels=dim, kernel_size=patch_size, stride=patch_size)

        # Positional embeddings and optional CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, (num_patches + (1 if use_cls_token else 0)), dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(dim=dim, heads=heads, mlp_dim=mlp_dim, attn_dropout=attn_dropout, dropout=dropout)
            for _ in range(depth)
        ])
        self.ln_final = nn.LayerNorm(dim)

        # Head: preserve FC1-like customizability
        hidden_layers = hidden_layers if hidden_layers is not None else [128, 64]
        layers: List[nn.Module] = []
        prev = dim
        for hsize in hidden_layers:
            layers.append(nn.Linear(prev, hsize))
            layers.append(_act(activation))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = hsize
        layers.append(nn.Linear(prev, final_output_dim))
        if final_activation:
            layers.append(_act(final_activation))
        self.head = nn.Sequential(*layers)

        # Parameter init
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        # Patch embed conv is Kaiming by default; keep

    def _to_nchw(self, imgs: torch.Tensor) -> torch.Tensor:
        # Accept (N, H, W, C) or (N, C, H, W)
        if imgs.ndim != 4:
            raise ValueError("Expected 4D tensor: (N,H,W,C) or (N,C,H,W)")
        if imgs.shape[1] in (1, 3):
            return imgs  # already NCHW
        # assume NHWC -> NCHW
        return imgs.permute(0, 3, 1, 2).contiguous()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        x = self._to_nchw(imgs)
        # Patchify and embed: (N, C, H, W) -> (N, dim, H/ps, W/ps)
        x = self.patch_embed(x)
        n, d, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)  # (N, num_patches, dim)

        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(n, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.emb_dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_final(x)
        if self.use_cls_token:
            x = x[:, 0]  # CLS
        else:
            x = x.mean(dim=1)

        out = self.head(x)
        return out


if __name__ == "__main__":
    # Simple test
    model = TF1(
        final_output_dim=10,
        image_input_shape=(3, 20, 20),
        hidden_layers=[512, 256, 128],
        activation="relu",
        final_activation=None,
        dropout=0.1,
        patch_size=4,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        attn_dropout=0.1,
        emb_dropout=0.1,
        use_cls_token=True,
    )
    dummy_input = torch.randn(2, 20, 20, 3)  # (N, H, W, C)
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (2, 10)
    print("Model:", model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")