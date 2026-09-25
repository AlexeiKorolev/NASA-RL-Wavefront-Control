"""M0 diagnostic for learned active probing (notes/PLAN_active_probing.md).

Verifies the physics that the whole active-probing thesis rests on, in the coronagraph
plateau regime where the passive policy is stuck (~1e-5 contrast):

  1. Linear identity:  I(+p) - I(-p) == 4 * Re(E* . dE_p)   (pairwise-probe sensing)
  2. Reconstruction:   from K known probes, recover the complex dark-hole residual
     field E per pixel by least squares, and compare to the true field focal_field().

Both must hold for learned probing to be able to break the sensing wall. Sweeps the
probe amplitude so a single job finds the linear/SNR sweet spot without manual tuning.
Config (geometry / modes) is taken from a trained PO4NCPA corona checkpoint so the
diagnostic matches the real setup exactly.

Usage:
    python src/probe_check.py --checkpoint logs/po4ncpa_corona/po4ncpa_best.pt \
        --n-draws 20 --n-probes 8
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from po4ncpa import Config, make_env, calibrate_per_mode_scale


def reconstruct_field(dE, d):
    """Per-pixel LS for the complex residual field E from probe responses.

    dE : (K, M) complex   known focal-field response of each probe (dark-hole pixels)
    d  : (K, M) real      measured pairwise differences  I(+p)-I(-p)
    Model per pixel/probe:  d_k = 4 (E_r Re(dE_k) + E_i Im(dE_k)).  Solve 2x2 normal eqs.
    """
    a0 = 4.0 * dE.real            # (K, M) coefficient of E_r
    a1 = 4.0 * dE.imag            # (K, M) coefficient of E_i
    n00 = np.sum(a0 * a0, axis=0)
    n01 = np.sum(a0 * a1, axis=0)
    n11 = np.sum(a1 * a1, axis=0)
    b0 = np.sum(a0 * d, axis=0)
    b1 = np.sum(a1 * d, axis=0)
    det = n00 * n11 - n01 * n01
    det = np.where(np.abs(det) > 0, det, np.inf)
    er = (n11 * b0 - n01 * b1) / det
    ei = (-n01 * b0 + n00 * b1) / det
    return er + 1j * ei


def main():
    ap = argparse.ArgumentParser(description="M0: probe linearization + field reconstruction")
    ap.add_argument("--checkpoint", default="logs/po4ncpa_corona/po4ncpa_best.pt",
                    help="only its cfg (geometry/modes) is used")
    ap.add_argument("--n-draws", type=int, default=20)
    ap.add_argument("--n-probes", type=int, default=8, help="number of modal probes (low-order modes)")
    ap.add_argument("--seed-offset", type=int, default=700000)
    ap.add_argument("--residual-frac", type=float, default=0.1,
                    help="std of the residual perturbation as a fraction of per-mode correction scale "
                         "(sets how far from the floor the test state sits)")
    ap.add_argument("--amps", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.4],
                    help="probe amplitudes as a fraction of per-mode correction scale, swept")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["cfg"])
    if not cfg.use_coronagraph:
        print("[warn] checkpoint is non-coronagraphic; probing is meant for the corona regime")
    env = make_env(cfg)
    opt = env.optics
    scale = calibrate_per_mode_scale(opt, cfg)          # per-mode correction stroke (m of surface RMS coeff)
    dh = np.asarray(opt.dark_hole_mask)                 # boolean over focal grid
    corr_modes = np.asarray(opt._correction_modes)      # (num_modes, n_pupil), col k = 2k*Z_k phase
    n_modes = corr_modes.shape[0]
    K = min(args.n_probes, n_modes)
    corona, lyot = env.use_coronagraph, env.use_coronagraph

    # per-mode unit pupil phase for the probed (low-order) modes
    probe_unit_phase = [corr_modes[k] for k in range(K)]   # each (n_pupil,)

    print(f"[m0] run={cfg.run_name} corona={corona} modes={n_modes} probes={K} "
          f"dark-hole pixels={int(dh.sum())} draws={args.n_draws}")
    print(f"[m0] residual_frac={args.residual_frac}  amp sweep (x per-mode scale)={args.amps}")

    rng = np.random.default_rng(cfg.seed + 4242)

    # accumulate over the amplitude sweep
    lin_corr = {a: [] for a in args.amps}     # linear-identity correlation
    rec_err = {a: [] for a in args.amps}      # relative reconstruction error
    contrasts = []

    for e in range(args.n_draws):
        seed = args.seed_offset + e
        env.reset(seed=seed)
        # land in the plateau regime: ideal correction + a small random residual
        ideal = opt.ideal_modal_correction_coeffs()
        resid = args.residual_frac * scale * rng.standard_normal(n_modes)
        opt.set_correction_modes(ideal + resid)
        c_here = opt.dark_hole_contrast(
            intensity=opt.normalized_intensity(coronagraph=corona, lyot=lyot))
        contrasts.append(c_here)

        E = opt.focal_field(coronagraph=corona, lyot=lyot)[dh]          # true residual field (dark hole)

        for a in args.amps:
            dE = np.empty((K, E.size), dtype=complex)
            d = np.empty((K, E.size), dtype=float)
            for k in range(K):
                ph = a * scale[k] * probe_unit_phase[k]                 # probe pupil phase
                Ep = opt.focal_field(coronagraph=corona, lyot=lyot, extra_phase=+ph)[dh]
                dE[k] = Ep - E                                          # known probe field response
                ip = opt.normalized_intensity(coronagraph=corona, lyot=lyot, extra_phase=+ph)[dh]
                im = opt.normalized_intensity(coronagraph=corona, lyot=lyot, extra_phase=-ph)[dh]
                d[k] = ip - im                                         # measured pairwise difference
            # (1) linear identity: d ?= 4 Re(E* dE)
            pred = 4.0 * np.real(np.conj(E)[None, :] * dE)
            cc = np.corrcoef(d.ravel(), pred.ravel())[0, 1]
            lin_corr[a].append(cc)
            # (2) reconstruction of E from (dE, d)
            E_est = reconstruct_field(dE, d)
            rec_err[a].append(np.linalg.norm(E_est - E) / max(np.linalg.norm(E), 1e-30))

        opt.clear_correction()

    contrasts = np.array(contrasts)
    print(f"\n[m0] test-state dark-hole contrast: median {np.median(contrasts):.3e} "
          f"(min {contrasts.min():.3e}, max {contrasts.max():.3e})  <- plateau regime\n")

    print("  amp(xscale) | linear-identity corr | field reconstruction rel-err")
    print("  ------------+----------------------+------------------------------")
    best = None
    for a in args.amps:
        lc = np.median(lin_corr[a]); re = np.median(rec_err[a])
        print(f"     {a:6.3f}   |       {lc:6.4f}        |   {re*100:7.3f}%  (median)")
        if best is None or re < best[1]:
            best = (a, re, lc)
    a, re, lc = best
    print(f"\n[m0] best amplitude {a}x: reconstruction rel-err {re*100:.3f}%, linear corr {lc:.4f}")
    gate = (re < 0.05) and (lc > 0.99)
    print(f"[m0] GATE (rel-err < 5% and corr > 0.99): {'PASS' if gate else 'FAIL'}")
    print("[m0] done.")


if __name__ == "__main__":
    main()
