#!/usr/bin/env python
"""
data_generator.py

Rewritten from the original Jupyter notebook `data_generator.ipynb`.

Generates deformable mirror (DM) settings and corresponding coronagraph images
(with optional positive / negative DM mode nudges) for supervised learning tasks.

Default parameters (can be overridden via CLI):
  Data generation:
    N=100
    delta_t=1e-3
    noise_enabled=False
    dm_random_noise=1e-7
    nudge_amplitude=3e-7
    nudge_mode_index=0
  Environment:
    num_modes=10
    pixel_resolution=64
    oversizing_factor=1
    num_airy=5
    coronagraph_charge=6
    ppsr=2

Output:
  A pickle file containing:
    {
      "dm_settings": List[np.ndarray]              # Baseline DM actuator vectors
      "images": List[Tuple[np.ndarray, ...]]       # (baseline, +nudge, -nudge) images
      "meta": { ... }                              # Metadata & parameters
    }

Example:
  python data_generator.py --N 5000 --delta-t 1e-3 --output data/dataset.pkl
"""

from __future__ import annotations
import argparse
import os
import pickle
import sys
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

# External dependency: assumes CoronagraphEnvironment is available in PYTHONPATH
try:
    from environment import CoronagraphEnvironment
except ImportError as exc:
    raise ImportError(
        "Could not import CoronagraphEnvironment. Ensure environment.py is on PYTHONPATH."
    ) from exc


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    num_modes: int = 10
    pixel_resolution: int = 64
    oversizing_factor: int = 1
    num_airy: int = 5
    coronagraph_charge: int = 6
    ppsr: int = 2  # pixels_per_spacial_res
    basis: str = "zernike"  # New field for mode type

@dataclass
class DatasetConfig:
    N: int = 100
    delta_t: float = 1e-3
    noise_enabled: bool = False
    dm_random_noise: float = 1e-7
    nudge_amplitude: float = 3e-7
    nudge_mode_index: int = 0
    save_checkpoints: Optional[List[int]] = None  # e.g. [1000, 5000, 10000]


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def build_environment(ecfg: EnvironmentConfig) -> CoronagraphEnvironment:
    env = CoronagraphEnvironment(
        num_modes=ecfg.num_modes,
        pixels=ecfg.pixel_resolution,
        oversizing_factor=ecfg.oversizing_factor,
        num_airy=ecfg.num_airy,
        coronagraph_charge=ecfg.coronagraph_charge,
        pixels_per_spacial_res=ecfg.ppsr
    )
    return env


def generate_dataset(
    env: CoronagraphEnvironment,
    dcfg: DatasetConfig,
) -> Dict[str, Any]:
    """
    Generate dataset of DM settings and corresponding images.

    For each sample:
      1. Randomize DM (Gaussian with std=dm_random_noise).
      2. Capture baseline image.
      3. Apply +nudge to a single mode -> image2
      4. Apply -nudge to the same mode -> image3

    Returns dict with dm settings, images, and metadata.
    """
    images: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    dm_settings: List[np.ndarray] = []

    # Pre-compute nudge vector
    nudge = np.zeros(env.deformable_mirror.actuators.shape, dtype=float)
    if dcfg.nudge_mode_index >= len(nudge):
        raise ValueError(f"nudge_mode_index {dcfg.nudge_mode_index} out of range (num_modes={len(nudge)})")
    nudge[dcfg.nudge_mode_index] = dcfg.nudge_amplitude

    checkpoints = set(dcfg.save_checkpoints or [])
    for i in range(dcfg.N):
        # Random baseline DM
        env.deformable_mirror.flatten()
        env.set_random_dm(noise=dcfg.dm_random_noise)
        baseline_dm = env.deformable_mirror.actuators.copy()

        image1 = env.get_camera_image(delta_t=dcfg.delta_t, noise_enabled=dcfg.noise_enabled)

        # + nudge
        env.deformable_mirror.flatten()
        env.deformable_mirror.actuators = baseline_dm + nudge
        image2 = env.get_camera_image(delta_t=dcfg.delta_t, noise_enabled=dcfg.noise_enabled)

        # - nudge
        env.deformable_mirror.flatten()
        env.deformable_mirror.actuators = baseline_dm - nudge
        image3 = env.get_camera_image(delta_t=dcfg.delta_t, noise_enabled=dcfg.noise_enabled)

        images.append((np.array(image1), np.array(image2), np.array(image3)))
        dm_settings.append(baseline_dm)

        if (i + 1) in checkpoints:
            print(f"[Checkpoint] Generated {i + 1} / {dcfg.N} samples")

    meta = {
        "environment_config": asdict(ecfg_from_env(env)),
        "dataset_config": asdict(dcfg),
        "nudge_vector": nudge,
        "num_samples": dcfg.N,
        "image_triplet_description": "(baseline, +nudge, -nudge)"
    }

    return {
        "dm_settings": dm_settings,
        "images": images,
        "meta": meta
    }


def ecfg_from_env(env: CoronagraphEnvironment) -> EnvironmentConfig:
    """
    Attempt to reconstruct EnvironmentConfig from an environment instance.
    """
    # Attribute names may differ; adjust if your CoronagraphEnvironment differs.
    return EnvironmentConfig(
        num_modes=getattr(env, "num_modes"),
        pixel_resolution=getattr(env, "pixels"),
        oversizing_factor=getattr(env, "oversizing_factor"),
        num_airy=getattr(env, "num_airy"),
        coronagraph_charge=getattr(env, "coronagraph_charge"),
        ppsr=getattr(env, "pixels_per_spacial_res"),
        basis=getattr(env, "basis", "zernike")  # Default; adjust if environment exposes this
    )


def save_dataset(data: Dict[str, Any], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[Saved] {output_path}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate coronagraph DM-image dataset.")
    # Dataset parameters
    p.add_argument("--N", type=int, default=100, help="Number of samples to generate.")
    p.add_argument("--delta-t", type=float, default=1e-3, dest="delta_t", help="Exposure/integration time for images.")
    p.add_argument("--noise-enabled", action="store_true", dest="noise_enabled", help="Enable camera noise.")
    p.add_argument("--dm-random-noise", type=float, default=1e-7, help="Std dev for random DM initialization.")
    p.add_argument("--nudge-amplitude", type=float, default=3e-7, help="Amplitude of single-mode nudge.")
    p.add_argument("--nudge-mode-index", type=int, default=0, help="Index of mode to nudge.")
    p.add_argument("--checkpoints", type=int, nargs="*", default=None, help="Optional sample counts at which to print progress.")
    # Environment parameters
    p.add_argument("--num-modes", type=int, default=10, dest="num_modes", help="Number of DM modes.")
    p.add_argument("--pixel-resolution", type=int, default=64, dest="pixel_resolution", help="Image pixel resolution.")
    p.add_argument("--oversizing-factor", type=int, default=1, dest="oversizing_factor", help="Oversizing factor.")
    p.add_argument("--num-airy", type=int, default=5, dest="num_airy", help="Number of Airy rings (if applicable).")
    p.add_argument("--coronagraph-charge", type=int, default=6, dest="coronagraph_charge", help="Coronagraph topological charge.")
    p.add_argument("--ppsr", type=int, default=2, help="Pixels per spatial resolution element.")
    # Output
    p.add_argument("--output", type=str, default="data/dataset.pkl", help="Path to output pickle file.")
    p.add_argument("--basis-type", type=str, default="zernike", choices=["zernike", "harmonic"])  # New argument for mode type

    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    ecfg = EnvironmentConfig(
        num_modes=args.num_modes,
        pixel_resolution=args.pixel_resolution,
        oversizing_factor=args.oversizing_factor,
        num_airy=args.num_airy,
        coronagraph_charge=args.coronagraph_charge,
        ppsr=args.ppsr,
        basis=args.basis_type,
    )
    dcfg = DatasetConfig(
        N=args.N,
        delta_t=args.delta_t,
        noise_enabled=args.noise_enabled,
        dm_random_noise=args.dm_random_noise,
        nudge_amplitude=args.nudge_amplitude,
        nudge_mode_index=args.nudge_mode_index,
        save_checkpoints=args.checkpoints
    )

    print("[Info] Building environment...")
    env = build_environment(ecfg)

    print("[Info] Generating dataset...")
    dataset = generate_dataset(env, dcfg)

    print("[Info] Saving dataset...")
    save_dataset(dataset, args.output)

    print("[Done]")


if __name__ == "__main__":
    main(sys.argv[1:])