"""Model-mismatch stress test for classical probe-EFC (notes/PLAN_active_probing.md).

The premise of the whole learned-probe program is that EFC's power in our simulation is
its *perfect analytic model* -- we hand it the exact control Jacobian and probe response,
built from the same optics that generates the truth. On a real coronagraph that model is
never exact (DM actuator-gain uncertainty, drift, NCPA), and EFC can only null as well as
its model is accurate. This script measures exactly that: it builds EFC's model at the
NOMINAL corrector gain (gain=1) and runs the loop on a TRUTH optics whose per-mode gain is
``1 + sigma * z`` (z a fixed unit-normal pattern). Sweeping sigma traces how EFC degrades
as its static calibration drifts away from the instrument.

Two sensing paths, both closing the loop with the same *model* (nominal-gain) Jacobian G:
  - "perfect": reads the true field. Isolates the CONTROL-Jacobian error.
  - "probe":   estimates the field from pairwise probes using the *model* probe response dE
    (nominal amplitude) against the *measured* intensity difference (true, gained
    amplitude) -- adds the SENSING error on top of the control error.

EMPIRICAL FINDING (job 3296740): the two curves are nearly identical, so the damage is
dominated by CONTROL, not sensing. I originally expected the closed loop to self-correct a
diagonal gain error with perfect sensing -- it does not. A per-mode gain rotates the
commanded correction in modal space (truth applies ``gain*dc`` not ``dc``), which a scalar
step / rcond line-search cannot undo, so even perfect-field EFC converges to a residual
floor set by the mismatch in the ill-conditioned modal directions. Probe sensing adds only
a secondary bias at large sigma. The point stands and is in fact stronger: EFC's *entire*
operation is built on a static model, and when that model is wrong neither iteration nor
perfect field access rescues it -- the commands land in the wrong place. An RL agent trained
end-to-end on the true (gained) system never builds a nominal-gain model to be wrong about.
Run under SLURM on Adroit (env dmrl2); never on the login node.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from po4ncpa import Config, make_env, calibrate_per_mode_scale
from probe_check import reconstruct_field
from efc_probe import build_modal_jacobian


def estimate_field_probe_mismatch(opt, corr_modes, scale, probe_ids, probe_amp, dh, probe_gain):
    """Pairwise-probe field estimate under corrector-gain mismatch.

    EFC's calibration ``dE`` is the probe response it *believes* it commands (nominal
    amplitude ``probe_amp``); the measured pairwise difference ``d`` is what the true optics
    actually produces (amplitude scaled by the unknown per-mode gain). The reconstruction
    inverts the true measurement with the wrong model -> a biased field estimate."""
    E_state = opt.focal_field()[dh]                       # true current field (operating point)
    K = len(probe_ids)
    dE = np.empty((K, E_state.size), dtype=complex)
    d = np.empty((K, E_state.size), dtype=float)
    for i, k in enumerate(probe_ids):
        base = probe_amp * scale[k] * corr_modes[k]
        ph_model = base                                   # EFC's assumed probe (gain=1)
        ph_true = probe_gain[k] * base                    # what the corrector actually applies
        Ep = opt.focal_field(extra_phase=+ph_model)[dh]   # model probe response (calibration)
        Em = opt.focal_field(extra_phase=-ph_model)[dh]
        dE[i] = Ep - E_state
        ip = opt.normalized_intensity(coronagraph=True, lyot=True, extra_phase=+ph_true)[dh]
        im = opt.normalized_intensity(coronagraph=True, lyot=True, extra_phase=-ph_true)[dh]
        d[i] = ip - im                                    # measured on the TRUE optics
    return reconstruct_field(dE, d)


def efc_loop_mismatch(opt, U, S, Vt, corr_modes, scale, rconds, n_iters, max_abs,
                      sensing, probe_ids, probe_amp, probe_gain):
    """EFC iteration with a nominal-gain model G but a gain-mismatched truth optics.

    The truth's per-mode gain is applied inside ``set_correction_modes`` (via
    opt.correction_gain), so commanding ``dc`` realizes ``gain*dc`` -- EFC steps in
    commanded space and the loop reads back the true contrast each line-search trial."""
    dh = np.asarray(opt.dark_hole_mask)
    smax = S[0]
    curve = [opt.dark_hole_contrast()]
    for _ in range(n_iters):
        if sensing == "perfect":
            E0 = opt.focal_field()[dh]
        else:
            E0 = estimate_field_probe_mismatch(opt, corr_modes, scale, probe_ids,
                                               probe_amp, dh, probe_gain)
        Utb = U.T @ np.concatenate([E0.real, E0.imag])
        c_save = opt.correction_coeffs.copy()
        best = (np.inf, c_save)
        for rc in rconds:
            keep = S >= rc * smax
            sinv = np.where(keep, 1.0 / np.where(keep, S, 1.0), 0.0)
            dc = -(Vt.T @ (sinv * Utb))
            opt.set_correction_modes(np.clip(c_save + dc, -max_abs, max_abs))
            c = opt.dark_hole_contrast()
            if c < best[0]:
                best = (c, opt.correction_coeffs.copy())
        opt.set_correction_modes(best[1])
        curve.append(best[0])
    return curve


def run_sensing(opt, U, S, Vt, corr_modes, scale, rconds, n_iters, max_abs,
                sensing, probe_ids, probe_amp, probe_gain, n_draws, seed_offset):
    """Median final contrast over n_draws aberration realizations at a fixed gain pattern."""
    fins = []
    for e in range(n_draws):
        opt_env_reset(opt, seed_offset + e)
        curve = efc_loop_mismatch(opt, U, S, Vt, corr_modes, scale, rconds, n_iters,
                                  max_abs, sensing, probe_ids, probe_amp, probe_gain)
        fins.append(min(curve))
    return float(np.median(fins)), fins


def opt_env_reset(opt, seed):
    """Draw the seed's aberration and clear the corrector (gain is left in place)."""
    # env.reset re-seeds the aberration; the corrector gain is a persistent instrument
    # property, so we clear only the commanded correction here.
    opt._efc_env.reset(seed=seed)
    opt.clear_correction()


def main():
    ap = argparse.ArgumentParser(description="EFC under corrector-gain model mismatch")
    ap.add_argument("--checkpoint", default="logs/po4ncpa_corona/po4ncpa_best.pt",
                    help="only its cfg (geometry/modes) is used")
    ap.add_argument("--n-draws", type=int, default=15)
    ap.add_argument("--efc-iters", type=int, default=10)
    ap.add_argument("--n-probes", type=int, default=8)
    ap.add_argument("--probe-amp", type=float, default=0.05)
    ap.add_argument("--poke-frac", type=float, default=0.05)
    ap.add_argument("--max-abs-frac", type=float, default=50.0)
    ap.add_argument("--seed-offset", type=int, default=800000)
    ap.add_argument("--gain-sigma-list", default="0,0.02,0.05,0.1,0.2,0.3",
                    help="comma-separated per-mode gain-error std (fraction). 0 = perfect model")
    ap.add_argument("--gain-seed", type=int, default=12345,
                    help="seed for the fixed unit-normal gain pattern (scaled by each sigma)")
    ap.add_argument("--out-csv", default="logs/efc_mismatch/efc_mismatch.csv")
    args = ap.parse_args()

    rconds = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["cfg"])
    env = make_env(cfg)
    opt = env.optics
    opt._efc_env = env                                   # for opt_env_reset
    scale = calibrate_per_mode_scale(opt, cfg)
    corr_modes = np.asarray(opt._correction_modes)       # (num_modes, n_pupil)
    n_modes = corr_modes.shape[0]
    probe_ids = list(range(min(args.n_probes, n_modes)))
    max_abs = args.max_abs_frac * scale

    # Model Jacobian: built once at flat state at NOMINAL gain (focal_field pokes bypass
    # the corrector gain), so it is exactly EFC's gain=1 belief regardless of the truth.
    opt.set_correction_gain(1.0)
    G = build_modal_jacobian(opt, corr_modes, scale, args.poke_frac)
    A = np.vstack([G.real, G.imag])
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Fixed unit-normal gain-error pattern; each sigma scales it so the sweep is monotone
    # (same modes miscalibrated, deeper each step) rather than a fresh random draw per point.
    z = np.random.default_rng(args.gain_seed).standard_normal(n_modes)

    sigmas = [float(x) for x in args.gain_sigma_list.split(",")]
    print(f"[mismatch] run={cfg.run_name} modes={n_modes} probes={len(probe_ids)} "
          f"dark-hole px={int(opt.dark_hole_mask.sum())} draws={args.n_draws} "
          f"iters={args.efc_iters} | Jacobian cond {S[0]/S[-1]:.2e}", flush=True)

    rows = []
    for sigma in sigmas:
        gain = np.clip(1.0 + sigma * z, 0.05, None)      # keep gains positive
        opt.set_correction_gain(gain)
        perfect_med, _ = run_sensing(opt, U, S, Vt, corr_modes, scale, rconds,
                                     args.efc_iters, max_abs, "perfect", probe_ids,
                                     args.probe_amp, gain, args.n_draws, args.seed_offset)
        probe_med, _ = run_sensing(opt, U, S, Vt, corr_modes, scale, rconds,
                                   args.efc_iters, max_abs, "probe", probe_ids,
                                   args.probe_amp, gain, args.n_draws, args.seed_offset)
        rows.append((sigma, float(np.std(sigma * z)), perfect_med, probe_med))
        print(f"  sigma={sigma:5.3f}  gain rms={sigma*np.std(z):6.3f}  "
              f"perfect-EFC={perfect_med:.3e}  probe-EFC={probe_med:.3e}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("gain_sigma,gain_rms,perfect_efc,probe_efc\n")
        for sigma, grms, pm, prm in rows:
            f.write(f"{sigma},{grms},{pm},{prm}\n")

    base_probe = rows[0][3]
    print("\n=== EFC under corrector-gain model mismatch ===")
    print(f"  ideal-model floor (sigma=0) probe-EFC: {base_probe:.3e}")
    print(f"  learned probe (logdeep3, true system) : 3.70e-08")
    print(f"\n  {'sigma':>7} | {'perfect-EFC':>13} | {'probe-EFC':>13} | {'probe degrade x':>15}")
    for sigma, grms, pm, prm in rows:
        print(f"  {sigma:>7.3f} | {pm:>13.3e} | {prm:>13.3e} | {prm/base_probe:>14.1f}x")
    print(f"\n[mismatch] wrote {args.out_csv}")
    print("[mismatch] done.")


if __name__ == "__main__":
    main()
