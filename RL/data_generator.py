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
    telescope_diameter: float = 8.0
    oversizing_factor: float = 16/15
    wavelength_sci: float = 2.2e-6
    num_modes: int = 10
    zero_magnitude_flux: float = 3.9e10
    stellar_magnitude: float = 5.0
    env_delta_t: float = 1e-3  # internal loop delta_t for environment
    pixels: int = 64
    num_iterations: int = 10
    coronagraph_charge: int = 6
    num_airy: int = 5
    pixels_per_spacial_res: int = 2  # ppsr
    num_noise_modes: int = 10
    diversity_enabled: bool = True
    nudge_magnitude: float = 3e-7
    nudge_mode_indices: Optional[List[int]] = None
    num_diversity_pairs: int = 1
    obs_noise_enabled: bool = False
    obs_delta_t: Optional[float] = None
    include_slopes: bool = True
    action_scale: float = 1e-8
    dm_clip: Optional[float] = None
    lyot_fraction: float = 0.8
    basis: str = "harmonic"  # 'harmonic' or 'zernike'

@dataclass
class DatasetConfig:
    N: int = 100
    delta_t: float = 1e-3
    noise_enabled: bool = False
    dm_random_noise: float = 1e-7
    dm_random_noise_std: float = 0.0
    nudge_amplitude: float = 3e-7
    nudge_mode_index: int = 0
    save_checkpoints: Optional[List[int]] = None  # e.g. [1000, 5000, 10000]

# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------

def build_environment(ecfg: EnvironmentConfig) -> CoronagraphEnvironment:
    env = CoronagraphEnvironment(
        telescope_diameter=ecfg.telescope_diameter,
        oversizing_factor=ecfg.oversizing_factor,
        wavelength_sci=ecfg.wavelength_sci,
        num_modes=ecfg.num_modes,
        zero_magnitude_flux=ecfg.zero_magnitude_flux,
        stellar_magnitude=ecfg.stellar_magnitude,
        delta_t=ecfg.env_delta_t,
        pixels=ecfg.pixels,
        num_iterations=ecfg.num_iterations,
        coronagraph_charge=ecfg.coronagraph_charge,
        num_airy=ecfg.num_airy,
        pixels_per_spacial_res=ecfg.pixels_per_spacial_res,
        num_noise_modes=ecfg.num_noise_modes,
        diversity_enabled=ecfg.diversity_enabled,
        nudge_magnitude=ecfg.nudge_magnitude,
        nudge_mode_indices=ecfg.nudge_mode_indices,
        num_diversity_pairs=ecfg.num_diversity_pairs,
        obs_noise_enabled=ecfg.obs_noise_enabled,
        obs_delta_t=ecfg.obs_delta_t,
        include_slopes=ecfg.include_slopes,
        action_scale=ecfg.action_scale,
        dm_clip=ecfg.dm_clip,
        lyot_fraction=ecfg.lyot_fraction,
        basis=ecfg.basis,
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
    slopes: List[np.ndarray] = []

    # Pre-compute nudge vector
    nudge = np.zeros(env.deformable_mirror.actuators.shape, dtype=float)
    if dcfg.nudge_mode_index >= len(nudge):
        raise ValueError(f"nudge_mode_index {dcfg.nudge_mode_index} out of range (num_modes={len(nudge)})")
    nudge[dcfg.nudge_mode_index] = dcfg.nudge_amplitude

    checkpoints = set(dcfg.save_checkpoints or [])
    for i in range(dcfg.N):
        # Random baseline DM
        env.deformable_mirror.flatten()
        if dcfg.dm_random_noise_std and dcfg.dm_random_noise_std > 0:
            sampled_noise = float(np.random.normal(loc=dcfg.dm_random_noise, scale=dcfg.dm_random_noise_std))
            noise_level = sampled_noise # max(0.0, sampled_noise)
            while np.abs(noise_level) < 1e-20: # keep retrying to avoid small noise values
                sampled_noise = float(np.random.normal(loc=dcfg.dm_random_noise, scale=dcfg.dm_random_noise_std))
                noise_level = sampled_noise
        else:
            noise_level = float(dcfg.dm_random_noise)

        env.set_random_dm(noise=noise_level)
        baseline_dm = env.deformable_mirror.actuators.copy()
        cur_slope = np.array(env.get_slopes())

        imgs = env.generate_diversity_images(delta_t=dcfg.delta_t, noise_enabled=dcfg.noise_enabled)

        # image1 = env.get_camera_image(delta_t=dcfg.delta_t, noise_enabled=dcfg.noise_enabled)

        # # + nudge
        # env.deformable_mirror.flatten()
        # env.deformable_mirror.actuators = baseline_dm + nudge
        # image2 = env.get_camera_image(delta_t=dcfg.delta_t, noise_enabled=dcfg.noise_enabled)

        # # - nudge
        # env.deformable_mirror.flatten()
        # env.deformable_mirror.actuators = baseline_dm - nudge
        # image3 = env.get_camera_image(delta_t=dcfg.delta_t, noise_enabled=dcfg.noise_enabled)

        images.append(imgs)
        dm_settings.append(baseline_dm)
        slopes.append(cur_slope)

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
        "slopes": slopes,
        "meta": meta
    }


def ecfg_from_env(env: CoronagraphEnvironment) -> EnvironmentConfig:
    """
    Attempt to reconstruct EnvironmentConfig from an environment instance.
    """
    # Attribute names may differ; adjust if your CoronagraphEnvironment differs.
    return EnvironmentConfig(
        telescope_diameter=getattr(env, "telescope_diameter"),
        oversizing_factor=getattr(env, "oversizing_factor"),
        wavelength_sci=getattr(env, "wavelength_sci"),
        num_modes=getattr(env, "num_modes"),
        zero_magnitude_flux=getattr(env, "wf", None).total_power if hasattr(env, "wf") else 3.9e10,
        stellar_magnitude=getattr(env, "stellar_magnitude", 5.0),
        env_delta_t=getattr(env, "delta_t", 1e-3),
        pixels=getattr(env, "pixels"),
        num_iterations=getattr(env, "num_iterations", 10),
        coronagraph_charge=getattr(env, "coronagraph_charge"),
        num_airy=getattr(env, "num_airy"),
        pixels_per_spacial_res=getattr(env, "pixels_per_spacial_res"),
        num_noise_modes=getattr(env, "num_noise_modes", 0),
        diversity_enabled=getattr(env, "diversity_enabled", True),
        nudge_magnitude=getattr(env, "nudge_magnitude", 3e-7),
        nudge_mode_indices=getattr(env, "nudge_mode_indices", None),
        num_diversity_pairs=getattr(env, "num_diversity_pairs", 1),
        obs_noise_enabled=getattr(env, "obs_noise_enabled", False),
        obs_delta_t=getattr(env, "obs_delta_t", None),
        include_slopes=getattr(env, "include_slopes", True),
        action_scale=getattr(env, "action_scale", 1e-8),
        dm_clip=getattr(env, "dm_clip", None),
        lyot_fraction=getattr(env, "lyot_fraction", 0.8) if hasattr(env, "lyot_fraction") else 0.8,
        basis=getattr(env, "basis", "harmonic"),
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
    p.add_argument("--dm-random-noise", type=float, default=1e-7, help="Mean std dev for random DM initialization.")
    p.add_argument("--dm-random-noise-std", type=float, default=0.0, help="Std deviation around --dm-random-noise; sampled per sample.")
    p.add_argument("--nudge-amplitude", type=float, default=3e-7, help="Amplitude of single-mode nudge.")
    p.add_argument("--nudge-mode-index", type=int, default=0, help="Index of mode to nudge.")
    p.add_argument("--checkpoints", type=int, nargs="*", default=None, help="Optional sample counts at which to print progress.")
    # Environment parameters
    # Environment parameters
    p.add_argument("--telescope-diameter", type=float, default=8.0)
    p.add_argument("--oversizing-factor", type=float, default=16/15)
    p.add_argument("--wavelength-sci", type=float, default=2.2e-6)
    p.add_argument("--num-modes", type=int, default=10)
    p.add_argument("--zero-magnitude-flux", type=float, default=3.9e10)
    p.add_argument("--stellar-magnitude", type=float, default=5.0)
    p.add_argument("--env-delta-t", type=float, default=1e-3, help="Environment internal loop delta_t")
    p.add_argument("--pixels", type=int, default=64)
    p.add_argument("--num-iterations", type=int, default=10)
    p.add_argument("--coronagraph-charge", type=int, default=6)
    p.add_argument("--num-airy", type=int, default=5)
    p.add_argument("--pixels-per-spacial-res", type=int, default=2, dest="pixels_per_spacial_res")
    p.add_argument("--num-noise-modes", type=int, default=10)
    p.add_argument("--diversity-enabled", action="store_true", dest="diversity_enabled")
    p.add_argument("--no-diversity", action="store_false", dest="diversity_enabled")
    p.set_defaults(diversity_enabled=True)
    p.add_argument("--nudge-magnitude", type=float, default=3e-7)
    p.add_argument("--nudge-mode-indices", type=int, nargs="*", default=None, help="List of mode indices for diversity nudges.")
    p.add_argument("--num-diversity-pairs", type=int, default=1)
    p.add_argument("--obs-noise-enabled", action="store_true", dest="obs_noise_enabled")
    p.add_argument("--no-obs-noise", action="store_false", dest="obs_noise_enabled")
    p.set_defaults(obs_noise_enabled=False)
    p.add_argument("--obs-delta-t", type=float, default=None)
    p.add_argument("--include-slopes", action="store_true", dest="include_slopes")
    p.add_argument("--no-slopes", action="store_false", dest="include_slopes")
    p.set_defaults(include_slopes=True)
    p.add_argument("--action-scale", type=float, default=1e-8)
    p.add_argument("--dm-clip", type=float, default=None)
    p.add_argument("--lyot-fraction", type=float, default=0.8)
    p.add_argument("--basis-type", type=str, default="harmonic", choices=["harmonic", "zernike"], dest="basis_type")
    # Output
    p.add_argument("--output", type=str, default="data/dataset.pkl", help="Path to output pickle file.")

    return p.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)

    ecfg = EnvironmentConfig(
        telescope_diameter=args.telescope_diameter,
        oversizing_factor=args.oversizing_factor,
        wavelength_sci=args.wavelength_sci,
        num_modes=args.num_modes,
        zero_magnitude_flux=args.zero_magnitude_flux,
        stellar_magnitude=args.stellar_magnitude,
        env_delta_t=args.env_delta_t,
        pixels=args.pixels,
        num_iterations=args.num_iterations,
        coronagraph_charge=args.coronagraph_charge,
        num_airy=args.num_airy,
        pixels_per_spacial_res=args.pixels_per_spacial_res,
        num_noise_modes=args.num_noise_modes,
        diversity_enabled=args.diversity_enabled,
        nudge_magnitude=args.nudge_magnitude,
        nudge_mode_indices=args.nudge_mode_indices,
        num_diversity_pairs=args.num_diversity_pairs,
        obs_noise_enabled=args.obs_noise_enabled,
        obs_delta_t=args.obs_delta_t,
        include_slopes=args.include_slopes,
        action_scale=args.action_scale,
        dm_clip=args.dm_clip,
        lyot_fraction=args.lyot_fraction,
        basis=args.basis_type,
    )
    dcfg = DatasetConfig(
        N=args.N,
        delta_t=args.delta_t,
        noise_enabled=args.noise_enabled,
        dm_random_noise=args.dm_random_noise,
        dm_random_noise_std=args.dm_random_noise_std,
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