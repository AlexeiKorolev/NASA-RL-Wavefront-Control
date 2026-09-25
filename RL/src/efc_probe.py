"""M1 baseline for learned active probing (notes/PLAN_active_probing.md).

Classical pairwise-probe Electric Field Conjugation (EFC) in the ideal-corrector modal
space, driven through the same corona setup as PO4NCPA. This is the *honest* baseline the
learned probe (M2) must match or beat: it uses only intensity measurements + known probes
(no simulator-truth field), the same information a learned agent has.

Two field-estimation paths, both closing the loop with the same modal control Jacobian:
  - "perfect": read the true field focal_field() -> upper bound (like efc_floor, in modal
    space).
  - "probe":   estimate the complex dark-hole field each iteration by pairwise probing
    (the M0 reconstruction) -> the realistic baseline.

EFC model: dark-hole field ~ E0 + G @ dc, G = d(field)/d(correction mode) built once at
the flat state. Each iteration solve the regularized LS command that drives E toward zero,
line-searching the singular-value cutoff against the *measured* contrast.

Run under SLURM on Adroit (env dmrl2); never on the login node.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from po4ncpa import Config, make_env, calibrate_per_mode_scale
from probe_check import reconstruct_field


def build_modal_jacobian(opt, corr_modes, scale, poke_frac):
    """Complex d(dark-hole field)/d(correction-mode coeff), finite-differenced at flat state."""
    dh = np.asarray(opt.dark_hole_mask)
    opt.clear_aberration(); opt.clear_correction()
    E_ref = opt.focal_field()[dh]
    n_modes = corr_modes.shape[0]
    G = np.empty((int(dh.sum()), n_modes), dtype=complex)
    for k in range(n_modes):
        dk = poke_frac * scale[k]
        ph = dk * corr_modes[k]
        G[:, k] = (opt.focal_field(extra_phase=ph)[dh] - E_ref) / dk
    return G


def estimate_field_probe(opt, corr_modes, scale, probe_ids, probe_amp, dh, flux=0.0, rng=None):
    """Reconstruct the complex dark-hole field at the current corrector state via pairwise
    probing (measured intensities + model-known probe responses). Same math as M0.

    flux>0 (M3): the pairwise-difference *measurement* d is built from independently shot-
    noise-limited intensity exposures (Poisson at `flux` photons/peak), while the probe
    response dE stays noiseless -- it is a calibration, known a priori. This is the realistic
    degradation that raises EFC's achievable floor as photons run out."""
    E_state = opt.focal_field()[dh]                       # only used for the model probe response
    K = len(probe_ids)
    dE = np.empty((K, E_state.size), dtype=complex)
    d = np.empty((K, E_state.size), dtype=float)
    for i, k in enumerate(probe_ids):
        ph = probe_amp * scale[k] * corr_modes[k]
        Ep = opt.focal_field(extra_phase=+ph)[dh]         # model-known probe response (calibration)
        Em = opt.focal_field(extra_phase=-ph)[dh]
        dE[i] = Ep - E_state
        if flux and flux > 0:
            ip = opt.normalized_intensity(coronagraph=True, lyot=True, extra_phase=+ph,
                                          flux=flux, rng=rng)[dh]
            im = opt.normalized_intensity(coronagraph=True, lyot=True, extra_phase=-ph,
                                          flux=flux, rng=rng)[dh]
            d[i] = ip - im                                # NOISY measured pairwise difference
        else:
            d[i] = np.abs(Ep) ** 2 - np.abs(Em) ** 2      # noiseless measurement
    return reconstruct_field(dE, d)


def efc_loop(opt, U, S, Vt, corr_modes, scale, rconds, n_iters, max_abs,
             sensing, probe_ids, probe_amp, flux=0.0, rng=None,
             drift_rho=None, drift_rng=None, lag_match=False):
    """lag_match=True reproduces the RL env's actuation ordering exactly: environment.py's
    step() evolves the aberration BEFORE applying the action, and that action was computed
    from the PREVIOUS step's observation -- so the correction landing this frame is always
    one full drift step stale. Plain EFC (lag_match=False) senses AFTER this iteration's
    drift and reacts immediately -- zero lag, a fundamentally easier problem under drift.
    This isolates whether the RL-vs-EFC gap under drift is a learning gap or a fairness
    artifact of who gets to react to whose measurement."""
    dh = np.asarray(opt.dark_hole_mask)
    smax = S[0]
    curve = [opt.dark_hole_contrast()]

    def sense():
        if sensing == "perfect":
            return opt.focal_field()[dh]
        return estimate_field_probe(opt, corr_modes, scale, probe_ids, probe_amp, dh,
                                    flux=flux, rng=rng)

    # RL-equivalent stale measurement: taken before any drift, like the reset() observation
    # that drives the episode's first action.
    E_stale = sense() if lag_match else None
    for _ in range(n_iters):
        if drift_rho is not None:
            # M4: the aberration drifts one AR(1) step per control iteration, so this
            # iteration senses (and corrects) a field the previous solve never saw --
            # static EFC has to re-chase the disturbance every frame.
            opt.evolve_aberration(drift_rho, drift_rng)
        E0 = E_stale if lag_match else sense()
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
        if lag_match:
            # Measurement taken AFTER this iteration's correction -- becomes the stale
            # input the NEXT iteration's decision is limited to, exactly like the RL
            # env's obs_t driving action_{t+1}.
            E_stale = sense()
    return curve


def main():
    ap = argparse.ArgumentParser(description="M1: pairwise-probe EFC baseline")
    ap.add_argument("--checkpoint", default="logs/po4ncpa_corona/po4ncpa_best.pt",
                    help="only its cfg (geometry/modes) is used")
    ap.add_argument("--n-draws", type=int, default=15)
    ap.add_argument("--efc-iters", type=int, default=10)
    ap.add_argument("--n-probes", type=int, default=8, help="pairwise sensing probes (low-order modes)")
    ap.add_argument("--probe-amp", type=float, default=0.05, help="probe amp x per-mode scale (M0 sweet spot)")
    ap.add_argument("--poke-frac", type=float, default=0.05, help="Jacobian finite-diff poke x per-mode scale")
    ap.add_argument("--max-abs-frac", type=float, default=50.0, help="corrector coeff clip x per-mode scale")
    ap.add_argument("--seed-offset", type=int, default=800000)
    ap.add_argument("--flux-list", default="0",
                    help="comma-separated photons/peak for probe sensing; 0=noiseless (M3 sweep)")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="M4: AR(1) aberration drift correlation time in EFC iterations; 0=static")
    ap.add_argument("--match-rl-lag", action="store_true",
                    help="one-iteration-stale sensing, matching the RL env's actuation "
                         "ordering exactly (isolate lag-fairness from a real learning gap)")
    args = ap.parse_args()

    rconds = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["cfg"])
    env = make_env(cfg)
    opt = env.optics
    scale = calibrate_per_mode_scale(opt, cfg)
    corr_modes = np.asarray(opt._correction_modes)          # (num_modes, n_pupil)
    n_modes = corr_modes.shape[0]
    probe_ids = list(range(min(args.n_probes, n_modes)))
    max_abs = args.max_abs_frac * scale

    drift_rho = float(np.exp(-1.0 / args.tau)) if args.tau > 0 else None
    # With drift the converged min is a transient; the honest figure is the steady-state
    # TRACKING contrast, taken as the mean of the last 3 iterations.
    final_of = (min if drift_rho is None
                else (lambda curve: float(np.mean(curve[-3:]))))

    print(f"[m1] run={cfg.run_name} corona={env.use_coronagraph} modes={n_modes} "
          f"probes={len(probe_ids)} dark-hole pixels={int(opt.dark_hole_mask.sum())} "
          f"draws={args.n_draws} iters={args.efc_iters} tau={args.tau} "
          f"lag_match={args.match_rl_lag}")

    print("[m1] building modal control Jacobian ...", flush=True)
    G = build_modal_jacobian(opt, corr_modes, scale, args.poke_frac)
    A = np.vstack([G.real, G.imag])
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"[m1] Jacobian {G.shape}, condition number {S[0] / S[-1]:.2e}")

    flux_list = [float(x) for x in args.flux_list.split(",")]
    rng = np.random.default_rng(args.seed_offset)

    # Perfect-knowledge EFC (flux-independent: reads the true field) -- the noiseless
    # controllability floor, computed once as the common reference for every flux.
    perfect_final = []
    for e in range(args.n_draws):
        env.reset(seed=args.seed_offset + e)
        opt.clear_correction()
        # Per-draw drift seed, re-created identically for every sensing mode / flux so
        # all variants chase the exact same disturbance trajectory.
        drng = np.random.default_rng(args.seed_offset + 5000 + e) if drift_rho else None
        curve = efc_loop(opt, U, S, Vt, corr_modes, scale, rconds, args.efc_iters,
                         max_abs, "perfect", probe_ids, args.probe_amp,
                         drift_rho=drift_rho, drift_rng=drng, lag_match=args.match_rl_lag)
        perfect_final.append(final_of(curve))
    perfect_med = float(np.median(perfect_final))

    # Probe-EFC at each photon budget: intensity-only sensing, shot-noise-limited.
    probe_med = {}
    for flux in flux_list:
        fins = []
        for e in range(args.n_draws):
            env.reset(seed=args.seed_offset + e)            # identical aberration across flux
            opt.clear_correction()
            drng = np.random.default_rng(args.seed_offset + 5000 + e) if drift_rho else None
            curve = efc_loop(opt, U, S, Vt, corr_modes, scale, rconds, args.efc_iters,
                             max_abs, "probe", probe_ids, args.probe_amp, flux=flux, rng=rng,
                             drift_rho=drift_rho, drift_rng=drng, lag_match=args.match_rl_lag)
            fins.append(final_of(curve))
        probe_med[flux] = float(np.median(fins))
        tag = "noiseless" if flux <= 0 else f"{flux:.1e} ph/peak"
        print(f"  probe-EFC @ {tag:>16}: median final {probe_med[flux]:.3e}", flush=True)

    print("\n=== M3: pairwise-probe EFC vs photon budget ===")
    print(f"  perfect-knowledge EFC (noiseless floor): {perfect_med:.3e}")
    print(f"  ideal 55-mode floor                    : 1.18e-10")
    print(f"  passive PO4NCPA (to beat)              : 1.5e-05")
    print(f"\n  {'flux (ph/peak)':>16} | {'median true contrast':>20}")
    for flux in flux_list:
        tag = "noiseless" if flux <= 0 else f"{flux:.1e}"
        print(f"  {tag:>16} | {probe_med[flux]:>20.3e}")
    print("[m3-efc] done.")


if __name__ == "__main__":
    main()
