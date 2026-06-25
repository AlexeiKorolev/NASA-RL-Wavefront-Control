"""Standalone evaluation for a trained PO4NCPA checkpoint.

Loads a saved policy, replays it deterministically on a held-out set of aberrations,
and characterizes it: Strehl + dark-hole contrast distributions, per-step refinement
over the episode, and the gap to the *ideal* modal correction (the achievable floor
for this geometry: exact projection of the aberration onto the corrector).

Usage:
    python src/eval_po4ncpa.py --checkpoint logs/po4ncpa_corona/po4ncpa_best.pt \
        --episodes 150 --seed-offset 900000
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from po4ncpa import Config, make_env, PolicyNet, calibrate_per_mode_scale


def preprocess(raw_2d: np.ndarray, ideal_2d: np.ndarray) -> np.ndarray:
    return np.cbrt(raw_2d - ideal_2d).astype(np.float32)


def ideal_floor(env) -> tuple[float, float]:
    """Contrast/Strehl after the exact ideal modal correction of the current draw."""
    opt = env.optics
    coeffs = opt.ideal_modal_correction_coeffs()
    opt.set_correction_modes(coeffs)
    sci = opt.normalized_intensity(coronagraph=env.use_coronagraph, lyot=env.use_coronagraph)
    c = opt.dark_hole_contrast(intensity=sci)
    s = float(np.max(sci)) if not env.use_coronagraph else opt.strehl()
    opt.clear_correction()
    return c, s


def run_policy_episode(env, policy, ideal, device, max_steps, seed):
    """Deterministic rollout; returns per-step contrast and Strehl (incl. the start)."""
    obs, info = env.reset(seed=seed)
    o_cur = preprocess(obs["image"][0], ideal)
    o_prev = o_cur.copy()
    a_prev = obs["command"].astype(np.float32).copy()
    contrasts = [info["contrast"]]
    strehls = [info["strehl"]]
    for _ in range(max_steps):
        pair = torch.as_tensor(np.stack([o_cur, o_prev])[None], device=device)
        ap = torch.as_tensor(a_prev[None], device=device)
        with torch.no_grad():
            a = policy(pair, ap)[0].cpu().numpy().astype(np.float32)
        obs, _r, _term, _trunc, info = env.step(a)
        o_prev, o_cur = o_cur, preprocess(obs["image"][0], ideal)
        a_prev = obs["command"].astype(np.float32).copy()
        contrasts.append(info["contrast"])
        strehls.append(info["strehl"])
    return np.array(contrasts), np.array(strehls)


def main():
    p = argparse.ArgumentParser(description="Evaluate a PO4NCPA checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--seed-offset", type=int, default=900000,
                   help="held-out seeds (disjoint from training/in-loop eval)")
    p.add_argument("--max-steps", type=int, default=None, help="override episode length")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    print(f"[eval] checkpoint {args.checkpoint}")
    print(f"[eval] run={cfg.run_name} coronagraph={cfg.use_coronagraph} modes={cfg.num_modes} "
          f"start_mode={cfg.aberration_start_mode} rms={cfg.rms_min}-{cfg.rms_max} device={device}")

    env = make_env(cfg)
    if cfg.per_mode_scale:
        scale = calibrate_per_mode_scale(env.optics, cfg)
        env.action_scale = scale
        env.max_abs_actuator = scale
    h, w = env.image_shape
    action_dim = int(env.action_space.shape[0])
    ideal = env.ideal_image().astype(np.float32)
    max_steps = args.max_steps if args.max_steps is not None else env.max_steps

    policy = PolicyNet(h, w, action_dim, ch=cfg.ch).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    start_c, final_c, start_s, final_s = [], [], [], []
    floor_c, floor_s = [], []
    per_step_c, per_step_s = [], []
    for e in range(args.episodes):
        seed = args.seed_offset + e
        env.reset(seed=seed)                 # same draw -> ideal floor
        fc, fs = ideal_floor(env)
        cs, ss = run_policy_episode(env, policy, ideal, device, max_steps, seed)
        start_c.append(cs[0]); final_c.append(cs[-1])
        start_s.append(ss[0]); final_s.append(ss[-1])
        floor_c.append(fc); floor_s.append(fs)
        per_step_c.append(cs); per_step_s.append(ss)

    start_c, final_c = np.array(start_c), np.array(final_c)
    start_s, final_s = np.array(start_s), np.array(final_s)
    floor_c, floor_s = np.array(floor_c), np.array(floor_s)
    per_step_c = np.array(per_step_c); per_step_s = np.array(per_step_s)

    def q(a): return np.median(a), np.min(a), np.max(a)

    print(f"\n=== PO4NCPA eval: {args.episodes} held-out episodes, {max_steps} steps ===")
    print("\n-- Strehl (higher better) --")
    print(f"  start    median {np.median(start_s):.4f}")
    print(f"  FINAL    median {np.median(final_s):.4f}  best {final_s.max():.4f}  worst {final_s.min():.4f}")
    print(f"  ideal    median {np.median(floor_s):.4f}  (exact modal correction floor)")
    print(f"  frac final Strehl > 0.99: {np.mean(final_s > 0.99):.2f}")

    print("\n-- Dark-hole contrast (lower better) --")
    cm, cmin, cmax = q(final_c)
    print(f"  start    median {np.median(start_c):.3e}")
    print(f"  FINAL    median {cm:.3e}  best {cmin:.3e}  worst {cmax:.3e}")
    print(f"  ideal    median {np.median(floor_c):.3e}  (exact modal correction floor)")
    print(f"  policy/ideal contrast ratio (median): {np.median(final_c) / max(np.median(floor_c), 1e-30):.1f}x")

    print("\n-- Per-step median trajectory (refinement check) --")
    med_c = np.median(per_step_c, axis=0)
    med_s = np.median(per_step_s, axis=0)
    for t in range(len(med_c)):
        print(f"  step {t:2d}: contrast {med_c[t]:.3e}  strehl {med_s[t]:.4f}")

    print("\n[eval] done.")


if __name__ == "__main__":
    main()
