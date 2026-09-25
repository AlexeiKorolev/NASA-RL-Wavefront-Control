"""Render PO4NCPA performance figures (no Jupyter required).

Produces the same four views as ``notebooks/po4ncpa_viz.ipynb`` as PNGs, reusing the
exact training/eval pipeline so it can't drift:

  view1_image_evolution.png  - focal-plane (dark) image over steps, dark-hole annulus
  view2_trajectory.png       - contrast & Strehl vs step against the ideal floor
  view3_phase_maps.png       - aberration / correction / residual pupil phase
  view4_distribution.png     - held-out final contrast & Strehl histograms

Usage:
    python src/viz_po4ncpa.py --checkpoint logs/po4ncpa_corona/po4ncpa_best.pt \
        --out notebooks/figs --demo-seed 900000 --episodes 60
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import torch

from po4ncpa import Config, make_env, PolicyNet, calibrate_per_mode_scale
from eval_po4ncpa import preprocess, ideal_floor


def build(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = Config(**ckpt["cfg"])
    env = make_env(cfg)
    if cfg.per_mode_scale:
        scale = calibrate_per_mode_scale(env.optics, cfg)
        env.action_scale = scale
        env.max_abs_actuator = scale
    h, w = env.image_shape
    action_dim = int(env.action_space.shape[0])
    policy = PolicyNet(h, w, action_dim, ch=cfg.ch).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()
    ideal = env.ideal_image().astype(np.float32)
    return cfg, env, policy, ideal


def rollout(env, policy, ideal, device, seed):
    """Deterministic episode; returns images, contrast, strehl, final phase maps."""
    obs, info = env.reset(seed=seed)
    imgs = [obs["image"][0].copy()]
    contr = [info["contrast"]]
    strehl = [info["strehl"]]
    o_cur = preprocess(obs["image"][0], ideal)
    o_prev = o_cur.copy()
    a_prev = obs["command"].astype(np.float32).copy()
    for _ in range(env.max_steps):
        pair = torch.as_tensor(np.stack([o_cur, o_prev])[None], device=device)
        ap = torch.as_tensor(a_prev[None], device=device)
        with torch.no_grad():
            a = policy(pair, ap)[0].cpu().numpy().astype(np.float32)
        obs, _r, _t, _tr, info = env.step(a)
        imgs.append(obs["image"][0].copy())
        contr.append(info["contrast"]); strehl.append(info["strehl"])
        o_prev, o_cur = o_cur, preprocess(obs["image"][0], ideal)
        a_prev = obs["command"].astype(np.float32).copy()
    opt = env.optics
    sup = opt._aperture_support
    side = int(np.sqrt(opt.aberration_phase.size))

    def pup(ph):
        ph = np.asarray(ph).astype(float).copy(); ph[~sup] = np.nan
        return ph.reshape(side, side)

    phases = dict(aberration=pup(opt.aberration_phase),
                  correction=pup(opt.correction_phase),
                  residual=pup(np.asarray(opt.aberration_phase) + np.asarray(opt.correction_phase)))
    return np.array(imgs), np.array(contr), np.array(strehl), phases


def view1(env, cfg, imgs, contr, strehl, out):
    h, w = env.image_shape
    dh = np.asarray(env.optics.dark_hole_mask).reshape(h, w)
    steps = sorted(set([0, 1, 2, 3, 4, env.max_steps]))
    vmin = max(min(im[im > 0].min() for im in imgs), 1e-12)
    vmax = float(np.max(imgs))
    fig, axes = plt.subplots(1, len(steps), figsize=(3.0 * len(steps), 3.4))
    for ax, t in zip(axes, steps):
        m = ax.imshow(np.clip(imgs[t], vmin, None), norm=LogNorm(vmin=vmin, vmax=vmax),
                      cmap="inferno", origin="lower")
        ax.contour(dh.astype(float), levels=[0.5], colors="cyan", linewidths=0.8)
        ax.set_title(f"step {t}\nC={contr[t]:.2e}  S={strehl[t]:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(m, ax=axes, shrink=0.7, label="normalized intensity (log)")
    fig.suptitle(f"Focal-plane image - {cfg.run_name} (coronagraph={env.use_coronagraph})", y=1.02)
    p = os.path.join(out, "view1_image_evolution.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); return p


def view2(env, contr, strehl, floor_c, floor_s, seed, out):
    steps = np.arange(len(contr))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.semilogy(steps, contr, "o-", color="tab:red", label="policy")
    a1.axhline(floor_c, ls="--", color="k", label=f"ideal floor {floor_c:.2e}")
    a1.set_xlabel("step"); a1.set_ylabel("dark-hole contrast"); a1.set_title("Contrast")
    a1.legend(); a1.grid(alpha=0.3)
    a2.plot(steps, strehl, "o-", color="tab:blue", label="policy")
    a2.axhline(floor_s, ls="--", color="k", label=f"ideal floor {floor_s:.4f}")
    a2.set_xlabel("step"); a2.set_ylabel("Strehl"); a2.set_title("Strehl")
    a2.legend(); a2.grid(alpha=0.3)
    fig.suptitle(f"Refinement trajectory - seed {seed}")
    p = os.path.join(out, "view2_trajectory.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); return p


def view3(phases, out):
    vlim = np.nanmax(np.abs(phases["aberration"]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ["aberration", "correction", "residual"]):
        m = ax.imshow(phases[key], cmap="RdBu_r", vmin=-vlim, vmax=vlim, origin="lower")
        rms = np.sqrt(np.nanmean(phases[key] ** 2))
        ax.set_title(f"{key}\nRMS = {rms:.3f} rad", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(m, ax=ax, shrink=0.75)
    fig.suptitle("Pupil-plane phase (radians)")
    p = os.path.join(out, "view3_phase_maps.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig); return p


def view4(env, policy, ideal, device, cfg, n, seed_offset, out):
    fin_c, fin_s, flo_c = [], [], []
    for e in range(n):
        seed = seed_offset + e
        env.reset(seed=seed); fc, _ = ideal_floor(env)
        _, c, s, _ = rollout(env, policy, ideal, device, seed)
        fin_c.append(c[-1]); fin_s.append(s[-1]); flo_c.append(fc)
    fin_c, fin_s, flo_c = map(np.array, (fin_c, fin_s, flo_c))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(np.log10(fin_c), bins=20, color="tab:red", alpha=0.8)
    a1.axvline(np.log10(np.median(fin_c)), color="k", label=f"median {np.median(fin_c):.2e}")
    a1.axvline(np.log10(np.median(flo_c)), color="k", ls="--", label=f"ideal floor {np.median(flo_c):.2e}")
    a1.set_xlabel("log10(final contrast)"); a1.set_ylabel("count"); a1.set_title("Final contrast")
    a1.legend(fontsize=8)
    a2.hist(fin_s, bins=20, color="tab:blue", alpha=0.8)
    a2.axvline(np.median(fin_s), color="k", label=f"median {np.median(fin_s):.4f}")
    a2.set_xlabel("final Strehl"); a2.set_ylabel("count")
    a2.set_title(f"Final Strehl ({np.mean(fin_s > 0.99) * 100:.0f}% > 0.99)"); a2.legend(fontsize=8)
    fig.suptitle(f"{n} held-out episodes - {cfg.run_name}")
    p = os.path.join(out, "view4_distribution.png")
    fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
    return p, fin_c, fin_s, flo_c


def main():
    ap = argparse.ArgumentParser(description="Render PO4NCPA performance figures")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="notebooks/figs")
    ap.add_argument("--demo-seed", type=int, default=900000)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed-offset", type=int, default=900000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, env, policy, ideal = build(args.checkpoint, device)
    print(f"[viz] run={cfg.run_name} coronagraph={env.use_coronagraph} modes={int(env.action_space.shape[0])} "
          f"image={env.image_shape} device={device}")

    imgs, contr, strehl, phases = rollout(env, policy, ideal, device, args.demo_seed)
    env.reset(seed=args.demo_seed); floor_c, floor_s = ideal_floor(env)
    print(f"[viz] demo seed {args.demo_seed}: final C={contr[-1]:.3e} S={strehl[-1]:.4f} "
          f"| ideal floor C={floor_c:.3e}")

    print("[viz]", view1(env, cfg, imgs, contr, strehl, args.out))
    print("[viz]", view2(env, contr, strehl, floor_c, floor_s, args.demo_seed, args.out))
    print("[viz]", view3(phases, args.out))
    p4, fin_c, fin_s, flo_c = view4(env, policy, ideal, device, cfg, args.episodes, args.seed_offset, args.out)
    print("[viz]", p4)
    print(f"[viz] {args.episodes} held-out: contrast median {np.median(fin_c):.3e} "
          f"(best {fin_c.min():.3e}, worst {fin_c.max():.3e}) | Strehl median {np.median(fin_s):.4f} "
          f"| ideal floor median {np.median(flo_c):.3e}")
    print("[viz] done.")


if __name__ == "__main__":
    main()
