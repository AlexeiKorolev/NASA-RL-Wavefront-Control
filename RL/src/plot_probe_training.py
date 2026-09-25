"""Plot the full learned-probe campaign as one continuous curve: the original M2 runs
(raw reward), the resumed deep run (raw reward + parallel/max_steps=8 speedup), then the
log-reward deep runs that broke the 1.4e-6 fixed point. Segments are stitched on a
cumulative-episode axis and colored by phase, against the passive-PO4NCPA plateau and the
classical-EFC floor. Pure CSV plotting (no optics), safe on the login node."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/scratch/network/ak9088/NASA-RL-Wavefront-Control/RL"
OUT = os.path.join(REPO, "notebooks/figs"); os.makedirs(OUT, exist_ok=True)

PASSIVE = 1.5e-5      # passive PO4NCPA plateau
EFC = 1.18e-10        # classical probe-EFC floor (M1)

# (run dir, label, color, reward-phase) in resume order.
PHASES = [
    ("po4ncpa_probe",          "M2 initial (raw reward)",   "tab:blue",   "raw"),
    ("po4ncpa_probe_r2",       "M2 resume (raw)",           "tab:cyan",   "raw"),
    ("po4ncpa_probe_deep",     "deep resume (raw, fast)",   "tab:orange", "raw"),
    ("po4ncpa_probe_logdeep",  "log-reward deep",           "tab:green",  "log"),
    ("po4ncpa_probe_logdeep2", "log-reward deep (cont.)",   "darkgreen",  "log"),
    ("po4ncpa_probe_logdeep3", "log-reward deep (cont. 2)", "purple",     "log"),
]


def load(path):
    d = np.genfromtxt(path, delimiter=",", names=True)
    return (np.atleast_1d(d["episode"]), np.atleast_1d(d["eval_contrast"]),
            np.atleast_1d(d["eval_strehl"]))


fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.5))

offset = 0.0
log_boundary = None
best_c, best_x = np.inf, 0
for run, label, color, phase in PHASES:
    path = os.path.join(REPO, "logs", run, "train.csv")
    if not os.path.exists(path):
        continue
    try:
        ep, c, s = load(path)
    except Exception:
        continue
    if ep.size == 0:
        continue
    x = ep + offset
    a1.semilogy(x, c, "-o", ms=2.5, color=color, label=label)
    a2.plot(x, s, "-o", ms=2.5, color=color, label=label)
    if phase == "log" and log_boundary is None:
        log_boundary = offset
    if c.min() < best_c:
        best_c, best_x = c.min(), x[np.argmin(c)]
    offset += ep.max()

a1.axhline(PASSIVE, color="tab:red", ls="--", lw=1.2, label=f"passive PO4NCPA ({PASSIVE:.0e})")
a1.axhline(EFC, color="k", ls="--", lw=1.2, label=f"classical EFC floor ({EFC:.0e})")
if log_boundary is not None:
    a1.axvline(log_boundary, color="gray", ls=":", lw=1.2)
    a1.text(log_boundary, a1.get_ylim()[1], "  log reward on", va="top", fontsize=8, color="gray")
    a2.axvline(log_boundary, color="gray", ls=":", lw=1.2)

a1.annotate(f"best {best_c:.2e}", xy=(best_x, best_c),
            xytext=(best_x * 0.55, best_c * 0.5),
            arrowprops=dict(arrowstyle="->", color="darkgreen"), fontsize=9, color="darkgreen")
a1.set_xlabel("cumulative training episode"); a1.set_ylabel("eval dark-hole contrast (median, true)")
a1.set_title("Learned active probing — full campaign")
a1.grid(alpha=0.3, which="both"); a1.legend(fontsize=7.5, loc="lower left")

a2.set_xlabel("cumulative training episode"); a2.set_ylabel("eval Strehl (median)")
a2.set_title("Strehl vs training"); a2.grid(alpha=0.3); a2.legend(fontsize=7.5, loc="lower right")

fig.suptitle(f"PO4NCPA learned probe: passive 1.5e-5 -> M2 1.39e-6 -> log-reward {best_c:.1e} "
             f"(EFC floor {EFC:.0e})", y=1.01)
p = os.path.join(OUT, "probe_campaign.png")
fig.savefig(p, dpi=140, bbox_inches="tight")
print("wrote", p)
print(f"best eval contrast {best_c:.3e} at cumulative episode {best_x:.0f}")
print(f"gap to EFC floor: {best_c / EFC:.0f}x")
