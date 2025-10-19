#!/usr/bin/env python
"""
train_job.py

Training entrypoint for neural network models (FC1, CNN1) on coronagraph datasets.

Features:
- CLI with dataset path, model type, normalization (minmax|zscore|log), epochs, batch size, learning rate.
- Architecture params pass-through for FC1/CNN1.
- Fine-tune from existing checkpoint (freeze or not).
- Saves best model by validation loss and simple metrics JSON.

Dataset expectations (pickle):
{
  "images": List[np.ndarray]  # each shaped (3, H, W) or list/tuple of 3 arrays
  "dm_settings": List[np.ndarray]  # each shaped (num_modes,)
}

This script will transform images to (N, C=3, H, W) for FC1 input, or (N, 1, H, W) stacks with two frames for CNN1 if selected.
By default we will feed FC1 with flattened 3-channel-like input shape (3, H, W) -> (H, W, 3) expectation in FC1 is image_input_shape=(3,H,W) with its internal flatten order.
"""
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from archs.fc1 import FC1
from archs.cnn1 import CNN1


# ---------------------------
# Normalization utilities
# ---------------------------

def norm_minmax(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    x_min = np.min(x)
    x_max = np.max(x)
    x_mm = (x - x_min) / (x_max - x_min + 1e-12)
    return x_mm, {"type": "minmax", "min": float(x_min), "max": float(x_max)}


def norm_zscore(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    mu = np.mean(x)
    sd = np.std(x)
    x_zn = (x - mu) / (sd + 1e-12)
    return x_zn, {"type": "zscore", "mean": float(mu), "std": float(sd)}


def norm_log(x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    # log1p handles zeros; assume intensities >= 0
    x_log = np.log1p(x)
    mu = np.mean(x_log)
    sd = np.std(x_log)
    x_zn = (x_log - mu) / (sd + 1e-12)
    return x_zn, {"type": "log+zscore", "mean": float(mu), "std": float(sd)}


NORMALIZERS = {
    "minmax": norm_minmax,
    "zscore": norm_zscore,
    "log": norm_log,
}


# ---------------------------
# Data loading
# ---------------------------

def load_dataset(pkl_path: str) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, Any]]]:
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    images = np.array([np.array(x) for x in data["images"]], dtype=np.float32)  # (N,3,H,W)
    y = np.array(data["dm_settings"], dtype=np.float32)  # (N,num_modes)
    meta = data.get("meta") if isinstance(data, dict) else None
    return images, y, meta


# ---------------------------
# Build model by type
# ---------------------------

def build_model(model_type: str,
                image_shape: Tuple[int, int, int],
                output_dim: int,
                arch_args: Dict[str, Any]) -> nn.Module:
    if model_type.lower() == "fc1":
        # FC1 expects image_input_shape=(3,H,W) but inside uses height,width,depth order
        h, w, c = image_shape
        model = FC1(
            final_output_dim=output_dim,
            image_input_shape=(h, w, c),
            hidden_layers=arch_args.get("hidden_layers"),
            activation=arch_args.get("activation", "leaky_relu"),
            final_activation=arch_args.get("final_activation", "leaky_relu"),
            dropout=arch_args.get("dropout", 0.0),
        )
        return model
    elif model_type.lower() == "cnn1":
        # CNN1: We'll process two images + a list. For simplicity, encode three images by summing first two as pair.
        # But our simpler path: adapt CNN1 by feeding two frames and a zeros list the correct size.
        # We'll wrap later in training loop.
        h, w, c = image_shape
        model = CNN1(
            image_output_dim=arch_args.get("image_output_dim", 32),
            dm_input_dim=output_dim,
            dm_hidden_dim=arch_args.get("dm_hidden_dim", 32),
            final_output_dim=output_dim,
            image_input_shape=(c, h, w)
        )
        return model
    else:
        raise ValueError("Unsupported model_type. Use 'fc1' or 'cnn1'.")


# ---------------------------
# Training
# ---------------------------

def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          epochs: int,
          lr: float,
          device: torch.device) -> Dict[str, List[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    model.to(device)
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            bx, by = batch
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)

            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running += loss.item() * by.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for batch in val_loader:
                bx, by = batch
                bx, by = bx.to(device), by.to(device)
                out = model(bx)

                vloss = criterion(out, by)
                val_running += vloss.item() * by.size(0)
        val_loss = val_running / len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Epoch {epoch+1}/{epochs} - train {train_loss:.6f} - val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ---------------------------
# CLI main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Train a neural net on coronagraph dataset")
    ap.add_argument("--datapath", required=True, help="Path to dataset .pkl")
    ap.add_argument("--model_type", choices=["fc1", "cnn1"], default="fc1")
    ap.add_argument("--norm", choices=["minmax", "zscore", "log"], default="minmax")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    # Architecture params
    ap.add_argument("--fc1_hidden", type=int, nargs="+", default=[128, 64], help="Hidden layer sizes for FC1 (space-separated)")
    ap.add_argument("--fc1_activation", choices=["leaky_relu", "relu", "gelu", "tanh"], default="leaky_relu")
    ap.add_argument("--fc1_final_activation", choices=["leaky_relu", "relu", "gelu", "tanh", "none"], default="leaky_relu")
    ap.add_argument("--fc1_dropout", type=float, default=0.0)
    ap.add_argument("--cnn1_img_out", type=int, default=32)
    ap.add_argument("--cnn1_dm_hidden", type=int, default=32)

    # Fine-tuning / checkpoint
    ap.add_argument("--resume", type=str, default=None, help="Path to model checkpoint to resume")
    ap.add_argument("--freeze", action="store_true", help="Freeze all layers except final output for fine-tune")

    # Output
    ap.add_argument("--out_dir", type=str, default="models/AutoTrain", help="Directory to save model and metrics")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load
    X, y, dataset_meta = load_dataset(args.datapath)  # X: (N,3,H,W)

    # Normalization for X
    X_norm, x_meta = NORMALIZERS[args.norm](X)

    # Normalize y (zscore is common for regressions)
    y_mu = np.mean(y)
    y_sd = np.std(y)
    y_norm = (y - y_mu) / (y_sd + 1e-12)

    # Build tensors and dataset according to model
    N, C, H, W = X_norm.shape
    output_dim = y.shape[1]

    # Model
    if args.model_type == "fc1":
        # FC1 expects image_input_shape=(H,W,C) internally stored as (h,w,depth)
        # We'll provide in training a tensor shaped (N, H, W, C) flattened by FC1 via Flatten
        # But FC1's forward expects (N, H, W, C) already. Its code uses nn.Flatten() -> correct.
        # Prepare tensor accordingly by permuting to (N, H, W, C)
        X_fc = np.transpose(X_norm, (0, 2, 3, 1)).astype(np.float32)
        X_t = torch.tensor(X_fc, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        input_shape = (H, W, C)
        fa = None if args.fc1_final_activation == "none" else args.fc1_final_activation
        model = build_model("fc1", input_shape, output_dim, arch_args={
            "hidden_layers": args.fc1_hidden,
            "activation": args.fc1_activation,
            "final_activation": fa,
            "dropout": args.fc1_dropout,
        })
    else:
        # CNN1 takes (img1, img2, list) -> We'll map (3,H,W) into two frames and a zero-vector list
        img1 = torch.tensor(X_norm[:, 0:1, :, :], dtype=torch.float32)
        img2 = torch.tensor(X_norm[:, 1:2, :, :], dtype=torch.float32)
        vec = torch.zeros((N, output_dim), dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)
        dataset = TensorDataset(img1, img2, vec, y_t)
        input_shape = (H, W, 1)
        model = build_model("cnn1", (H, W, 1), output_dim, arch_args={
            "image_output_dim": args.cnn1_img_out,
            "dm_hidden_dim": args.cnn1_dm_hidden,
        })

    # Split
    torch.manual_seed(args.seed)
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # Resume / fine-tune
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        if args.freeze:
            for name, p in model.named_parameters():
                if "final" not in name.lower():
                    p.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"using device: {device}")

    history = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)

    # Save
    model_path = os.path.join(args.out_dir, f"{args.model_type}_best.pth")
    torch.save(model.state_dict(), model_path)

    # Build training job metadata and aggregate into a single meta JSON
    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu_name = None

    train_job_meta = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": gpu_name,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
    }

    meta = {
        "args": vars(args),
        "x_norm": x_meta,
        "y_norm": {"mean": float(y_mu), "std": float(y_sd)},
        "history": history,
        "input_shape": input_shape,
        "output_dim": int(output_dim),
        "dataset_meta": dataset_meta,
        "train_job_meta": train_job_meta,
    }
    # Ensure JSON-serializable content (convert numpy arrays/scalars and tensors)
    def to_jsonable(o):
        import numpy as _np
        if isinstance(o, dict):
            return {k: to_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [to_jsonable(v) for v in o]
        if isinstance(o, _np.ndarray):
            return o.tolist()
        if isinstance(o, (_np.floating, _np.integer, _np.bool_)):
            return o.item()
        if isinstance(o, _np.generic):
            return o.item()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        return o

    meta_safe = to_jsonable(meta)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(meta_safe, f, indent=2)

    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
