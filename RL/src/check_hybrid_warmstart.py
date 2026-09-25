"""Decisive cheap check that the EFC warm-start + residual-offset hybrid is mechanically
correct (before committing a long training run). For a few resets it prints:

  * contrast right after reset  -> should be ~EFC's floor (deep), not the passive ~1e-5;
  * contrast after a ZERO residual action -> must STAY at the EFC floor. If it jumps back
    to passive, the residual offset is NOT being applied (the old overwrite bug);
  * contrast after a small random residual -> perturbs around the EFC floor.

Also compares against a plain (no-warmstart) env reset at the same seed so the EFC gain is
visible. Runs optics -> SLURM only, never the login node."""
from __future__ import annotations

import numpy as np

from po4ncpa_probe import ProbeConfig, make_probe_env


def main():
    sigma = 0.20
    base_cfg = dict(correction_gain_sigma=sigma, correction_gain_seed=12345)
    plain = make_probe_env(ProbeConfig(efc_warmstart=False, **base_cfg))
    hyb = make_probe_env(ProbeConfig(efc_warmstart=True, efc_warmstart_iters=6, **base_cfg))
    M = hyb.control_dim
    zero_act = np.zeros(2 * M)                       # [probe=0, residual=0]

    print(f"[check] sigma={sigma} modes={M} | comparing plain vs EFC-warmstart envs\n")
    print(f"  {'seed':>6} | {'plain reset C':>13} | {'hybrid reset C':>14} | "
          f"{'after 0-residual':>16} | {'after rand-resid':>16}")
    for seed in range(6):
        plain.reset(seed=seed)
        c_plain = plain.optics.dark_hole_contrast()

        hyb.reset(seed=seed)
        c_reset = hyb.optics.dark_hole_contrast()     # should be ~EFC floor
        hyb.step(zero_act)
        c_zero = hyb.optics.dark_hole_contrast()      # MUST stay ~EFC floor
        hyb.reset(seed=seed)
        rnd = np.concatenate([np.zeros(M), 0.3 * np.random.default_rng(seed).standard_normal(M)])
        hyb.step(np.clip(rnd, -1, 1))
        c_rand = hyb.optics.dark_hole_contrast()

        print(f"  {seed:>6} | {c_plain:>13.3e} | {c_reset:>14.3e} | "
              f"{c_zero:>16.3e} | {c_rand:>16.3e}")

    print("\n[check] PASS criteria: hybrid reset C ~ EFC floor (<<1e-5, ~1-3e-7 at sigma=0.2), "
          "and after-0-residual stays ~equal to reset C (offset applied, not overwritten).")


if __name__ == "__main__":
    main()
