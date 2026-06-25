"""Diagnostic: deepest dark-hole contrast reachable under optimal (EFC) control.

Answers the strategic question -- is 1e-10 contrast physically reachable for these
aberrations, and how much control authority (actuator count) does it need? -- by
running Electric Field Conjugation with *perfect* (simulator-truth) field
knowledge over the full actuator set, for several DM sizes.

EFC model: the focal-plane field is ~linear in the DM command, E ~ E0 + G @ da.
We build the complex Jacobian G once by poking each actuator, SVD it, then
iteratively solve the regularized least-squares command that drives the dark-hole
field toward zero. Because we read the *true* field (not a probe estimate), the
result is an upper bound on achievable contrast -- the floor a learned controller
could aspire to. Regularization (singular-value cutoff) is line-searched each
iteration against the real measured contrast.

Run under SLURM on Adroit (conda env dmrl2); never on the login node.
"""
from __future__ import annotations
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import numpy as np

from optics import CoronagraphOptics


def focal_field(opt: CoronagraphOptics) -> np.ndarray:
    """Complex focal field in contrast-amplitude units (|field|^2 = normalized intensity)."""
    wf = opt.dm.forward(opt._wavefront())
    wf = opt.coronagraph.forward(wf)
    wf = opt.lyot_stop.forward(wf)
    foc = opt.prop.forward(wf)
    return np.asarray(foc.electric_field) / np.sqrt(opt.reference_peak)


def build_jacobian(opt: CoronagraphOptics, poke: float) -> np.ndarray:
    """Complex dE_darkhole/d(actuator), finite-differenced at flat DM / no aberration."""
    dh = opt.dark_hole_mask
    opt.flatten_dm()
    opt.clear_aberration()
    E_ref = focal_field(opt)[dh]
    n_act = opt.num_actuators
    G = np.empty((int(dh.sum()), n_act), dtype=complex)
    a = np.zeros(n_act)
    for j in range(n_act):
        a[:] = 0.0
        a[j] = poke
        opt.set_actuators(a)
        G[:, j] = (focal_field(opt)[dh] - E_ref) / poke
        if (j + 1) % 128 == 0:
            print(f"    jacobian col {j + 1}/{n_act}", flush=True)
    opt.flatten_dm()
    return G


def efc_correct(opt, U, S, Vt, rconds, n_iters, max_abs):
    """Iterative EFC from the optics' current aberrated state.

    Returns (best_contrast, curve, max_abs_actuator) where curve[i] is the best
    contrast after iteration i (curve[0] is the starting contrast)."""
    dh = opt.dark_hole_mask
    smax = S[0]
    best = opt.dark_hole_contrast()
    curve = [best]
    for _ in range(n_iters):
        E0 = focal_field(opt)[dh]
        Utb = U.T @ np.concatenate([E0.real, E0.imag])
        a_save = opt.get_actuators()
        iter_best = (np.inf, a_save)
        for rc in rconds:
            keep = S >= rc * smax
            sinv = np.where(keep, 1.0 / np.where(keep, S, 1.0), 0.0)
            da = -(Vt.T @ (sinv * Utb))
            opt.set_actuators(np.clip(a_save + da, -max_abs, max_abs))
            c = opt.dark_hole_contrast()
            if c < iter_best[0]:
                iter_best = (c, opt.get_actuators())
        opt.set_actuators(iter_best[1])
        curve.append(iter_best[0])
        best = min(best, iter_best[0])
    return best, curve, float(np.max(np.abs(opt.get_actuators())))


def parse_args():
    p = argparse.ArgumentParser(description="EFC achievable-contrast-floor diagnostic")
    p.add_argument("--num-actuators", type=str, default="16,24,32")
    p.add_argument("--pupil-pixels", type=int, default=256)
    p.add_argument("--dark-hole-iwa", type=float, default=3.0)
    p.add_argument("--dark-hole-owa", type=float, default=12.0)
    p.add_argument("--n-aberrations", type=int, default=6)
    p.add_argument("--efc-iters", type=int, default=12)
    p.add_argument("--poke", type=float, default=1e-9)
    p.add_argument("--rms-min", type=float, default=0.02)
    p.add_argument("--rms-max", type=float, default=0.08)
    p.add_argument("--max-abs", type=float, default=5e-7)
    p.add_argument("--seed-base", type=int, default=1000)
    return p.parse_args()


def main():
    args = parse_args()
    rconds = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    n_acts = [int(x) for x in args.num_actuators.split(",") if x.strip()]

    summary = []
    for n_across in n_acts:
        print(f"\n===== DM {n_across}x{n_across} =====", flush=True)
        opt = CoronagraphOptics(pupil_pixels=args.pupil_pixels,
                                num_actuators_across=n_across,
                                dark_hole_iwa=args.dark_hole_iwa,
                                dark_hole_owa=args.dark_hole_owa)
        # Controllable spatial frequency vs. OWA (N/2 cycles/pupil -> N/2 lambda/D).
        ctrl_lod = n_across / 2.0
        opt.flatten_dm(); opt.clear_aberration()
        floor = opt.dark_hole_contrast()
        print(f"  actuators={opt.num_actuators}  controllable<= {ctrl_lod:.0f} lambda/D "
              f"(OWA={args.dark_hole_owa:.0f})  unaberrated floor={floor:.3e} "
              f"(log10 {np.log10(floor):.2f})", flush=True)

        print("  building Jacobian ...", flush=True)
        G = build_jacobian(opt, args.poke)
        A = np.vstack([G.real, G.imag])
        U, S, Vt = np.linalg.svd(A, full_matrices=False)

        c0s, bests, strokes = [], [], []
        for i in range(args.n_aberrations):
            opt.flatten_dm()
            rng = np.random.default_rng(args.seed_base + i)
            rms = rng.uniform(args.rms_min, args.rms_max)
            opt.set_random_aberration(rms, rng=rng)
            c0 = opt.dark_hole_contrast()
            best, curve, stroke = efc_correct(opt, U, S, Vt, rconds, args.efc_iters, args.max_abs)
            c0s.append(c0); bests.append(best); strokes.append(stroke)
            print(f"  ab {i} (rms={rms:.3f}w): {c0:.2e} -> {best:.2e} "
                  f"(log10 {np.log10(best):.2f}, {c0 / max(best, 1e-300):.0f}x, "
                  f"max|act|={stroke:.1e})", flush=True)

        med_c0 = float(np.median(c0s)); med_best = float(np.median(bests))
        print(f"  MEDIAN: start {med_c0:.2e} (log10 {np.log10(med_c0):.2f}) -> "
              f"EFC floor {med_best:.2e} (log10 {np.log10(med_best):.2f}), "
              f"{med_c0 / max(med_best, 1e-300):.0f}x  max|act| median {np.median(strokes):.1e}", flush=True)
        summary.append((n_across, opt.num_actuators, ctrl_lod, floor, med_c0, med_best,
                        float(np.median(strokes))))

    print("\n===== SUMMARY =====", flush=True)
    print(f"{'DM':>6} {'n_act':>6} {'ctrl(l/D)':>10} {'floor':>10} {'start':>10} "
          f"{'EFC floor':>11} {'log10':>7} {'max|act|':>9}", flush=True)
    for n_across, n_act, ctrl, floor, c0, best, stroke in summary:
        print(f"{n_across:>4}x{n_across:<1} {n_act:>6} {ctrl:>10.0f} {floor:>10.2e} "
              f"{c0:>10.2e} {best:>11.2e} {np.log10(best):>7.2f} {stroke:>9.1e}", flush=True)


if __name__ == "__main__":
    main()
