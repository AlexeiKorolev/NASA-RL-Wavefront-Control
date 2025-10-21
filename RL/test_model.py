#!/usr/bin/env python
r"""
test_model.py

Given a training run directory containing metrics.json and a model .pth file,
this script:
  1) Reconstructs the coronagraph environment from dataset_meta.environment_config
  2) Generates N samples using the same delta_t, noise_enabled, and nudge settings
  3) Normalizes the images exactly as in training (per metrics.x_norm)
  4) Rebuilds the model architecture from args and loads weights
  5) Runs inference and optionally saves predictions (both normalized and unnormalized)

Usage (PowerShell):
    python a:\\Projects\\DM + RL\\RL\\test_model.py --run_dir "a:\\Projects\\DM + RL\\RL\\models\\AutoTrain\\exp-fc1..." --N 1000
"""
from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, Tuple

import numpy as np
import torch

from environment import CoronagraphEnvironment
from archs.fc1 import FC1
from archs.cnn1 import CNN1


def load_metrics(run_dir: str) -> Dict[str, Any]:
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_model_path(run_dir: str, model_type: str) -> str:
    # Prefer the conventional naming used by train_job.py
    expected = os.path.join(run_dir, f"{model_type}_best.pth")
    if os.path.isfile(expected):
        return expected
    # Fallback: first .pth in directory
    for fn in os.listdir(run_dir):
        if fn.endswith(".pth"):
            return os.path.join(run_dir, fn)
    raise FileNotFoundError("No .pth model file found in run_dir")


def build_env_from_meta(dataset_meta: Dict[str, Any]) -> CoronagraphEnvironment:
    if dataset_meta is None or "environment_config" not in dataset_meta:
        raise ValueError("dataset_meta.environment_config missing in metrics.json; cannot rebuild environment.")
    ec = dataset_meta["environment_config"]
    # Map to CoronagraphEnvironment args
    env = CoronagraphEnvironment(
        num_modes=int(ec["num_modes"]),
        pixels=int(ec["pixel_resolution"]),
        oversizing_factor=float(ec["oversizing_factor"]),
        num_airy=int(ec["num_airy"]),
        coronagraph_charge=int(ec["coronagraph_charge"]),
        pixels_per_spacial_res=int(ec["ppsr"]),
    )
    return env


def generate_samples(env: CoronagraphEnvironment,
                     N: int,
                     dataset_cfg: Dict[str, Any],
                     nudge_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Generate N triplets of images as in data generation.
    Returns array X with shape (N, 3, H, W).
    """
    delta_t = float(dataset_cfg.get("delta_t", 1e-3))
    noise_enabled = bool(dataset_cfg.get("noise_enabled", False))
    dm_random_noise = float(dataset_cfg.get("dm_random_noise", 1e-7))

    images = []
    dms = []
    for _ in range(N):
        env.deformable_mirror.flatten()
        env.set_random_dm(noise=dm_random_noise)
        original = env.deformable_mirror.actuators.copy()
        dms.append(original.astype(np.float32))

        img1 = env.get_camera_image(delta_t=delta_t, noise_enabled=noise_enabled)

        env.deformable_mirror.flatten()
        env.deformable_mirror.actuators = original + nudge_vec
        img2 = env.get_camera_image(delta_t=delta_t, noise_enabled=noise_enabled)

        env.deformable_mirror.flatten()
        env.deformable_mirror.actuators = original - nudge_vec
        img3 = env.get_camera_image(delta_t=delta_t, noise_enabled=noise_enabled)

        images.append(np.stack([img1, img2, img3], axis=0).astype(np.float32))

    X = np.stack(images, axis=0)  # (N,3,H,W)
    return X, np.stack(dms, axis=0)


def compute_contrast(env: CoronagraphEnvironment, delta_t: float, noise_enabled: bool) -> float:
    """Compute contrast using explicit images to honor noise_enabled setting."""
    corona = env.get_camera_image(delta_t=delta_t, noise_enabled=noise_enabled, coronagraph_enabled=True, crop=False)
    clear = env.get_camera_image(delta_t=delta_t, noise_enabled=noise_enabled, coronagraph_enabled=False, crop=False)
    return float(env.get_contrast(corona_image=corona, clear_image=clear))


def apply_x_normalization(X: np.ndarray, x_norm_meta: Dict[str, Any]) -> np.ndarray:
    norm_type = x_norm_meta.get("type")
    if norm_type == "minmax":
        x_min = float(x_norm_meta["min"])
        x_max = float(x_norm_meta["max"])
        return (X - x_min) / (x_max - x_min + 1e-12)
    elif norm_type == "zscore":
        mu = float(x_norm_meta["mean"])
        sd = float(x_norm_meta["std"])
        return (X - mu) / (sd + 1e-12)
    elif norm_type == "log+zscore":
        mu = float(x_norm_meta["mean"])
        sd = float(x_norm_meta["std"])
        X_log = np.log1p(X)
        return (X_log - mu) / (sd + 1e-12)
    else:
        raise ValueError(f"Unknown x_norm type: {norm_type}")


def build_model_from_args(args_meta: Dict[str, Any], input_shape: Tuple[int, int, int], output_dim: int) -> torch.nn.Module:
    model_type = args_meta.get("model_type", "fc1").lower()
    if model_type == "fc1":
        hidden_layers = args_meta.get("fc1_hidden", [128, 64])
        if isinstance(hidden_layers, str):
            # Handle possible string entries if saved differently
            hidden_layers = [int(x) for x in hidden_layers.split()] if hidden_layers else [128, 64]
        activation = args_meta.get("fc1_activation", "leaky_relu")
        final_activation = args_meta.get("fc1_final_activation", "leaky_relu")
        if final_activation == "none":
            final_activation = None
        dropout = float(args_meta.get("fc1_dropout", 0.0))
        H, W, C = input_shape
        return FC1(
            final_output_dim=output_dim,
            image_input_shape=(H, W, C),
            hidden_layers=hidden_layers,
            activation=activation,
            final_activation=final_activation,
            dropout=dropout,
        )
    elif model_type == "cnn1":
        img_out = int(args_meta.get("cnn1_img_out", 32))
        dm_hidden = int(args_meta.get("cnn1_dm_hidden", 32))
        H, W, C = input_shape
        return CNN1(
            image_output_dim=img_out,
            dm_input_dim=output_dim,
            dm_hidden_dim=dm_hidden,
            final_output_dim=output_dim,
            image_input_shape=(C, H, W),
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def main():
    ap = argparse.ArgumentParser(description="Test a trained model on newly generated samples")
    ap.add_argument("--run_dir", required=True, help="Directory containing metrics.json and model .pth")
    ap.add_argument("--N", type=int, default=1000, help="Number of samples to generate")
    ap.add_argument("--device", type=str, default=None, help="cpu or cuda; defaults to training device if available")
    ap.add_argument("--save_preds", type=str, default=None, help="Optional path (.npy) to save predictions (unnormalized)")
    args = ap.parse_args()

    metrics = load_metrics(args.run_dir)
    args_meta = metrics.get("args", {})
    x_norm_meta = metrics.get("x_norm", {})
    y_norm_meta = metrics.get("y_norm", {})
    dataset_meta = metrics.get("dataset_meta")
    input_shape = tuple(metrics.get("input_shape"))  # (H,W,C) or (H,W,1)
    output_dim = int(metrics.get("output_dim"))

    # Build environment from dataset meta
    env = build_env_from_meta(dataset_meta)

    # Rebuild nudge vector
    if dataset_meta and "nudge_vector" in dataset_meta:
        nudge_vec = np.array(dataset_meta["nudge_vector"], dtype=np.float32)
    else:
        # Fallback to dataset_config fields
        dcfg = dataset_meta.get("dataset_config", {}) if dataset_meta else {}
        num_modes = int(env.num_modes)
        nudge_vec = np.zeros(num_modes, dtype=np.float32)
        idx = int(dcfg.get("nudge_mode_index", 0))
        amp = float(dcfg.get("nudge_amplitude", 3e-7))
        if 0 <= idx < num_modes:
            nudge_vec[idx] = amp

    # Generate samples according to dataset configuration
    dcfg = dataset_meta.get("dataset_config", {}) if dataset_meta else {}
    X, dm_list = generate_samples(env, args.N, dcfg, nudge_vec)  # X: (N,3,H,W), dm_list: (N,num_modes)

    # Validate shape vs training input
    H, W, C = input_shape
    hX, wX = X.shape[-2], X.shape[-1]
    if (H, W) != (hX, wX):
        raise ValueError(f"Generated image shape {(hX,wX)} does not match training input shape {(H,W)}. Adjust environment config.")

    # Normalize X per training
    X_norm = apply_x_normalization(X, x_norm_meta)

    # Build model and load weights
    model_type = args_meta.get("model_type", "fc1")
    model = build_model_from_args(args_meta, input_shape=input_shape, output_dim=output_dim)
    model_path = find_model_path(args.run_dir, model_type)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state, strict=False)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        if model_type == "fc1":
            # Permute to (N,H,W,C)
            X_fc = np.transpose(X_norm, (0, 2, 3, 1)).astype(np.float32)
            X_t = torch.tensor(X_fc, dtype=torch.float32, device=device)
            y_pred_norm = model(X_t)
        else:
            # CNN1 consumes first two frames and a zero vector
            img1 = torch.tensor(X_norm[:, 0:1, :, :], dtype=torch.float32, device=device)
            img2 = torch.tensor(X_norm[:, 1:2, :, :], dtype=torch.float32, device=device)
            vec = torch.zeros((X_norm.shape[0], output_dim), dtype=torch.float32, device=device)
            y_pred_norm = model(img1, img2, vec)

    y_pred_norm = y_pred_norm.detach().cpu().numpy()
    # Unnormalize predictions back to actuator scale
    y_mu = float(y_norm_meta.get("mean", 0.0))
    y_sd = float(y_norm_meta.get("std", 1.0))
    y_pred = y_pred_norm * y_sd + y_mu

    # Evaluate contrast before and after applying correction (-y_pred)
    delta_t = float(dcfg.get("delta_t", 1e-3))
    noise_enabled = bool(dcfg.get("noise_enabled", False))
    before_contrasts = []
    after_contrasts = []
    for i in range(args.N):
        # Before
        env.deformable_mirror.flatten()
        env.deformable_mirror.actuators = dm_list[i]
        c_before = compute_contrast(env, delta_t=delta_t, noise_enabled=noise_enabled)
        before_contrasts.append(c_before)

        # After applying correction
        correction = -y_pred[i]
        env.deformable_mirror.flatten()
        env.deformable_mirror.actuators = dm_list[i] + correction
        c_after = compute_contrast(env, delta_t=delta_t, noise_enabled=noise_enabled)
        after_contrasts.append(c_after)

    before_contrasts = np.array(before_contrasts, dtype=float)
    after_contrasts = np.array(after_contrasts, dtype=float)

    summary = {
        "run_dir": args.run_dir,
        "model_type": model_type,
        "N": int(args.N),
        "pred_shape": y_pred.shape,
        "first_pred_sample": y_pred[0].tolist() if y_pred.shape[0] > 0 else None,
        "contrast_before_mean": float(before_contrasts.mean()) if before_contrasts.size else None,
        "contrast_after_mean": float(after_contrasts.mean()) if after_contrasts.size else None,
        "contrast_before_median": float(np.median(before_contrasts)) if before_contrasts.size else None,
        "contrast_after_median": float(np.median(after_contrasts)) if after_contrasts.size else None,
    }
    print(summary)

    if args.save_preds:
        os.makedirs(os.path.dirname(args.save_preds), exist_ok=True) if os.path.dirname(args.save_preds) else None
    np.save(args.save_preds, y_pred)
    # Also save contrasts alongside
    base = os.path.splitext(args.save_preds)[0]
    np.save(base + "_contrast_before.npy", before_contrasts)
    np.save(base + "_contrast_after.npy", after_contrasts)
    print(f"Saved predictions to {args.save_preds} and contrasts to {base}_contrast_*.npy")


if __name__ == "__main__":
    main()
