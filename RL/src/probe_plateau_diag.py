"""Diagnose the learned-probe plateau (notes/PLAN_active_probing.md, M2).

The learned probe stalls at ~1.4e-6 while classical EFC reaches 1.18e-10 from the SAME
pairwise-probe information (M1). This script asks *why*, empirically, before we design a
fix. For a loaded probe policy it runs deterministic episodes EXTENDED far past the 8-20
training steps and records, per step:

  * true (noiseless) dark-hole contrast          -> does it keep digging or stall?
  * ||correction_t - correction_{t-1}||          -> does the policy converge to a fixed
                                                    point (stops acting) above the floor?
  * ||correction_t||                             -> is it saturating the stroke budget?

Reading:
  - contrast flattens AND the correction stops changing  => the policy converged to a
    non-floor fixed point: a control/optimization limit (reward gives no pressure to refine,
    or the fixed point is all the CNN can sense). Fix: reward shaping / refinement head.
  - contrast keeps dropping with more steps               => it just needs more iterations;
    cheap fix (longer episodes / iterate at eval).
  - correction keeps changing but contrast does not       => sensing limit: the policy moves
    but cannot null further from the difference image it extracts.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from po4ncpa_probe import ProbeConfig, ProbePolicyNet, make_probe_env
from po4ncpa import calibrate_per_mode_scale


def _pre(image_2ch, ideal):
    o = np.cbrt(image_2ch[0] - ideal)
    d = np.cbrt(image_2ch[1])
    return np.stack([o, d], axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="logs/po4ncpa_probe_r2/po4ncpa_probe_best.pt")
    ap.add_argument("--steps", type=int, default=40, help="extended episode length")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--seed-offset", type=int, default=500000)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ProbeConfig(**ckpt["cfg"])
    env = make_probe_env(cfg)
    scale = calibrate_per_mode_scale(env.optics, cfg)
    probe_scale = cfg.probe_frac * scale
    env.action_scale = scale
    env.max_abs_actuator = scale
    env.probe_scale = probe_scale
    env.max_steps = args.steps
    ideal = env.ideal_image().astype(np.float32)
    h, w = env.image_shape
    M = env.control_dim
    A = int(env.action_space.shape[0])

    policy = ProbePolicyNet(h, w, A, ch=cfg.ch)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    print(f"[diag] ckpt={args.checkpoint} modes={M} action={A} steps={args.steps} "
          f"draws={args.draws}", flush=True)

    C = np.zeros((args.draws, args.steps + 1))     # contrast incl. step 0 (passive)
    DC = np.zeros((args.draws, args.steps))         # ||corr_t - corr_{t-1}|| (modal, scaled)
    NRM = np.zeros((args.draws, args.steps))        # ||corr_t|| (modal, scaled)
    for e in range(args.draws):
        obs, info = env.reset(seed=args.seed_offset + e)
        C[e, 0] = env.optics.dark_hole_contrast()
        img = _pre(obs["image"], ideal)
        a_prev = np.zeros(A, np.float32)
        corr_prev = np.zeros(M)
        for t in range(args.steps):
            with torch.no_grad():
                a = policy(torch.as_tensor(img[None]), torch.as_tensor(a_prev[None]))[0].numpy()
            corr = a[M:] * scale                    # absolute modal correction (meters RMS)
            obs, _r, _term, _trunc, info = env.step(a.astype(np.float32))
            C[e, t + 1] = env.optics.dark_hole_contrast()
            DC[e, t] = np.linalg.norm(corr - corr_prev)
            NRM[e, t] = np.linalg.norm(corr)
            img = _pre(obs["image"], ideal)
            a_prev = a.astype(np.float32)
            corr_prev = corr

    cmed = np.median(C, axis=0)
    dcmed = np.median(DC, axis=0)
    nrmed = np.median(NRM, axis=0)
    print("\n step |   median contrast | ||dcorr|| (Δ vs prev) | ||corr||")
    print(f"    0 | {cmed[0]:.3e}         |          -           |    -")
    for t in range(args.steps):
        mark = "  <- training horizon" if t + 1 == cfg.max_steps else ""
        print(f" {t+1:4d} | {cmed[t+1]:.3e}         | {dcmed[t]:.3e}            | "
              f"{nrmed[t]:.3e}{mark}")

    best_step = int(np.argmin(cmed))
    print(f"\n[diag] best median contrast {cmed.min():.3e} at step {best_step} "
          f"(training horizon = {cfg.max_steps})")
    print(f"[diag] contrast at training horizon: {cmed[min(cfg.max_steps, args.steps)]:.3e}")
    print(f"[diag] contrast at {args.steps} steps: {cmed[-1]:.3e}")
    tail = dcmed[cfg.max_steps:] if args.steps > cfg.max_steps else dcmed[-1:]
    print(f"[diag] median ||dcorr|| in the extended tail: {np.median(tail):.3e} "
          f"(vs step-1 {dcmed[0]:.3e}) -- near-zero => converged to a fixed point")
    print("[diag] done.")


if __name__ == "__main__":
    main()
