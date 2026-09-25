"""Plot classical EFC degradation vs corrector-gain model mismatch, from the CSV written by
efc_mismatch.py. Shows the two sensing paths (perfect vs pairwise-probe) against the ideal
1.18e-10 floor and the learned-probe result on the *true* system (3.70e-8). The point of the
figure: BOTH EFC curves climb steeply with model mismatch (the damage is dominated by the
wrong control Jacobian, not sensing -- perfect-field EFC degrades too), while the learned
probe on the true system holds flat. Past ~10% gain error the learned probe wins. Pure-CSV,
safe on the login node."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL"
CSV = os.path.join(REPO, "logs/efc_mismatch/efc_mismatch.csv")
OUT = os.path.join(REPO, "notebooks/figs"); os.makedirs(OUT, exist_ok=True)

EFC_IDEAL = 1.18e-10   # classical probe-EFC floor at a perfect model
RL_TRUE = 3.70e-8      # learned probe (logdeep3) on the true system, no model needed

d = np.genfromtxt(CSV, delimiter=",", names=True)
sigma = np.atleast_1d(d["gain_sigma"])
perfect = np.atleast_1d(d["perfect_efc"])
probe = np.atleast_1d(d["probe_efc"])

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.semilogy(sigma * 100, probe, "-o", color="tab:red", lw=2, ms=6,
            label="probe-EFC (realistic: intensity + model)")
ax.semilogy(sigma * 100, perfect, "-o", color="tab:blue", lw=2, ms=6,
            label="perfect-sensing EFC (reads true field)")
ax.axhline(EFC_IDEAL, color="k", ls="--", lw=1.2, label=f"ideal EFC floor ({EFC_IDEAL:.0e})")
ax.axhline(RL_TRUE, color="darkgreen", ls="--", lw=1.5,
           label=f"learned probe on true system ({RL_TRUE:.1e})")

# Shade where the learned probe already beats realistic EFC.
cross = np.where(probe > RL_TRUE)[0]
if cross.size:
    ax.axvspan(sigma[cross[0]] * 100, sigma[-1] * 100, color="green", alpha=0.07)
    ax.text(sigma[cross[0]] * 100, RL_TRUE * 1.3,
            "  learned probe wins", color="darkgreen", fontsize=9, va="bottom")

ax.set_xlabel("corrector-gain model mismatch  (per-mode RMS, %)")
ax.set_ylabel("median dark-hole contrast (true)")
ax.set_title("EFC degrades with model mismatch; the learned probe does not")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8.5, loc="upper left")

p = os.path.join(OUT, "efc_mismatch.png")
fig.savefig(p, dpi=140, bbox_inches="tight")
print("wrote", p)
for s, pf, pr in zip(sigma, perfect, probe):
    print(f"  sigma={s:5.3f}  perfect={pf:.3e}  probe={pr:.3e}  probe/ideal={pr/EFC_IDEAL:.1f}x")
