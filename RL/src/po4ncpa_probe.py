"""PO4NCPA + learned active probing (M2 -- notes/PLAN_active_probing.md).

Extends the model-based PO4NCPA controller so the policy outputs BOTH a probe command
and a DM correction each step, and learns the probe jointly with the correction.

Why this can break the sensing wall (M0/M1): a coronagraph image is intensity-only and
sign-blind at high contrast, so the passive PO4NCPA policy plateaus (~1.5e-5). A *pairwise
probe* difference image d = I(+p) - I(-p) ~ 4 Re(E* E_p) is LINEAR in the residual field
and reveals its phase -- classical probe-EFC uses this to reach the 1.18e-10 floor. Here
the agent *learns* the probe end-to-end instead of hand-deriving it.

Mechanism. Observation = [dark image o, probe-difference image d] (env probe_mode). The
policy maps (o, d, prev action) -> action = [probe p, correction dc]. The dynamics model
predicts BOTH next channels [o', d'] from (obs, prev action, action). In the imagined
rollout the reward -||o'||^2 is applied to the correction, and -- crucially -- the probe p
affects the predicted d', which the policy reads at the next step to choose its correction;
back-propagating the multi-step reward therefore flows a gradient into p, shaping it to be
maximally informative for nulling. The differentiable learned model is the field-response
operator classical EFC builds by hand.

Given M1 (classical EFC already reaches the floor noiselessly), the bar here is to reach
the floor WITHOUT a hand-built Jacobian and in few exposures; the payoff regimes are photon
noise (M3) and dynamic NCPA (M4).
"""
from __future__ import annotations

import os
import argparse
import time
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import CoronagraphEnv
from po4ncpa import calibrate_per_mode_scale, _conv_out


# ---------------------------------------------------------------------------
# Networks (2-channel image [o, d]; action = [probe, correction] of length A=2M)
# ---------------------------------------------------------------------------
class ProbeDynamicsNet(nn.Module):
    """Predicts the next observation [o', d'] (2 channels) from the current observation
    [o, d], the previous action and the current action. U-Net with skips."""

    def __init__(self, h: int, w: int, action_dim: int, ch: int = 32):
        super().__init__()
        self.ch = ch
        self.c1 = nn.Conv2d(2, ch, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        h2, w2 = _conv_out(_conv_out(h)), _conv_out(_conv_out(w))
        self.h2, self.w2 = h2, w2
        self.fc1 = nn.Linear(ch * h2 * w2 + 2 * action_dim, 256)
        self.fc2 = nn.Linear(256, ch * h2 * w2)
        self.d2 = nn.Conv2d(2 * ch, ch, 3, padding=1)
        self.d1 = nn.Conv2d(2 * ch, ch, 3, padding=1)
        self.out = nn.Conv2d(ch, 2, 3, padding=1)          # -> [o', d']

    def forward(self, obs, cmd_prev, cmd_cur):
        e1 = F.relu(self.c1(obs))
        e2 = F.relu(self.c2(e1))
        e3 = F.relu(self.c3(e2))
        z = torch.cat([e3.flatten(1), cmd_prev, cmd_cur], dim=1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z)).view(-1, self.ch, self.h2, self.w2)
        d = F.interpolate(z, size=e2.shape[-2:], mode="nearest")
        d = F.relu(self.d2(torch.cat([d, e2], dim=1)))
        d = F.interpolate(d, size=e1.shape[-2:], mode="nearest")
        d = F.relu(self.d1(torch.cat([d, e1], dim=1)))
        return self.out(d)                                 # (B, 2, H, W)


class ProbePolicyNet(nn.Module):
    """Maps (obs [o, d], previous action) -> action [probe, correction] in [-1, 1]^A."""

    def __init__(self, h: int, w: int, action_dim: int, ch: int = 32):
        super().__init__()
        self.ch = ch
        self.c1 = nn.Conv2d(2, ch, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        h2, w2 = _conv_out(_conv_out(h)), _conv_out(_conv_out(w))
        self.fc1 = nn.Linear(ch * h2 * w2 + action_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, obs, cmd_prev):
        x = F.relu(self.c1(obs))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = torch.cat([x.flatten(1), cmd_prev], dim=1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class ProbeBuffer:
    """One-step transitions: obs_t [2,H,W], obs_{t+1} [2,H,W], a_{t-1} [A], a_t [A]."""

    def __init__(self, capacity, h, w, action_dim):
        self.cap = capacity
        self.it = np.zeros((capacity, 2, h, w), np.float32)
        self.itn = np.zeros((capacity, 2, h, w), np.float32)
        self.ap = np.zeros((capacity, action_dim), np.float32)
        self.ac = np.zeros((capacity, action_dim), np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, it, itn, ap, ac):
        i = self.ptr
        self.it[i], self.itn[i], self.ap[i], self.ac[i] = it, itn, ap, ac
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, size=n)
        t = lambda a: torch.as_tensor(a[idx], device=device)
        return t(self.it), t(self.itn), t(self.ap), t(self.ac)

    def save(self, path):
        """Persist the filled portion so a resumed run can reload the same data
        distribution (avoids the dynamics-model re-fit transient on resume). Writes to a
        file object so the exact name is honored (np.savez auto-appends .npz otherwise)."""
        n = self.size
        with open(path, "wb") as f:
            np.savez(f, it=self.it[:n], itn=self.itn[:n], ap=self.ap[:n], ac=self.ac[:n],
                     ptr=np.int64(self.ptr), size=np.int64(self.size))

    def load(self, path):
        """Refill from a saved buffer. Copies into the preallocated arrays (clamped to this
        buffer's capacity) and restores the ring pointer. Returns the number loaded."""
        with np.load(path) as d:
            n = min(int(d["size"]), self.cap)
            self.it[:n] = d["it"][:n]
            self.itn[:n] = d["itn"][:n]
            self.ap[:n] = d["ap"][:n]
            self.ac[:n] = d["ac"][:n]
            self.size = n
            self.ptr = int(d["ptr"]) % self.cap
        return n


@dataclass
class ProbeConfig:
    run_name: str = "po4ncpa_probe"
    use_coronagraph: bool = True
    num_modes: int = 55
    aberration_start_mode: int = 4
    q: int = 3
    num_airy: float = 5.5
    rms_min: float = 0.026
    rms_max: float = 0.026
    aberration_spectrum: str = "power_law"
    psd_exponent: float = 3.667
    aberration_tau: float = 0.0        # M4: AR(1) drift correlation time in env steps; 0 = static
    photon_flux: float = 0.0
    dark_hole_iwa: float = 1.5
    dark_hole_owa: float = 4.0
    action_scale: float = 1e-7
    per_mode_scale: bool = True
    action_headroom: float = 5.0
    probe_frac: float = 0.1            # probe per-mode scale = probe_frac x correction scale (M0: ~0.05-0.1)
    max_steps: int = 20
    total_episodes: int = 30000
    warmup_episodes: int = 4000
    ensemble: int = 5
    ch: int = 64
    dyn_iters: int = 8
    pol_iters: int = 5
    batch: int = 64
    h_min: int = 2
    h_max: int = 7
    dyn_lr: float = 1e-3
    pol_lr: float = 1e-4
    explore_std: float = 0.1
    buffer: int = 200_000
    eval_every: int = 250
    eval_episodes: int = 30
    seed: int = 0
    device: str = "cuda"
    resume: str = ""                  # checkpoint path to continue from (loads nets + optimizers)
    # --- model-mismatch experiment (efc_mismatch.py) --------------------------
    # Per-mode corrector-gain error on the TRUTH optics: the corrector realizes
    # ``gain_i * coeff_i`` while a nominal-gain controller (EFC) is miscalibrated.
    # The RL agent trains AND evals on the same gained truth, so it learns the real
    # system rather than trusting a static model. 0 -> perfect model (no change).
    correction_gain_sigma: float = 0.0
    correction_gain_seed: int = 12345  # fixed gain pattern, shared with efc_mismatch.py
    # --- EFC warm-start / residual-learning hybrid (grey-box) -----------------
    # If true, each reset first drives the corrector to classical EFC's converged
    # (nominal-model) state on the true gained system, then the RL policy refines
    # the residual EFC's miscalibrated model cannot reach. Perfect-sensing EFC is
    # used as a cheap classical pre-conditioner; the RL refinement stays intensity-only.
    efc_warmstart: bool = False
    efc_warmstart_iters: int = 6


def _corrector_gain_vector(n_modes: int, sigma: float, seed: int) -> np.ndarray:
    """Fixed unit-normal per-mode gain pattern scaled by sigma (matches efc_mismatch.py)."""
    z = np.random.default_rng(seed).standard_normal(n_modes)
    return np.clip(1.0 + sigma * z, 0.05, None)


class EFCWarmStartEnv(CoronagraphEnv):
    """CoronagraphEnv that can pre-condition each episode with classical EFC.

    With efc_warmstart on, reset() drives the ideal corrector to EFC's converged
    (nominal-model) solution on the true (gained) system before handing off to the RL
    policy, which then learns the residual EFC's miscalibrated model cannot reach.
    Perfect-sensing EFC is used as a cheap classical pre-conditioner; the policy's own
    sensing stays intensity-only. With efc_warmstart off it is exactly a CoronagraphEnv."""

    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        self._efc_on = False

    def setup_efc_warmstart(self, cfg):
        """Build EFC's nominal-gain control Jacobian once (no-op unless cfg.efc_warmstart)."""
        self._efc_on = bool(cfg.efc_warmstart)
        if not self._efc_on:
            return
        opt = self.optics
        # Decouple EFC's poke/stroke scale from the RL residual action headroom: EFC always
        # uses the headroom=5 calibration so the warm-start reproduces the efc_mismatch floor
        # regardless of how small the residual action scale is tuned.
        self._efc_scale = calibrate_per_mode_scale(opt, replace(cfg, action_headroom=5.0))
        self._efc_corr_modes = np.asarray(opt._correction_modes)     # (M, n_pupil)
        self._efc_max_abs = 50.0 * self._efc_scale
        self._efc_rconds = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
        self._efc_iters = int(cfg.efc_warmstart_iters)
        dh = np.asarray(opt.dark_hole_mask)
        opt.clear_aberration(); opt.clear_correction()
        E_ref = opt.focal_field()[dh]                                # flat-state reference
        n_modes = self._efc_corr_modes.shape[0]
        G = np.empty((int(dh.sum()), n_modes), dtype=complex)
        for k in range(n_modes):                                     # nominal-gain pokes
            dk = 0.05 * self._efc_scale[k]
            G[:, k] = (opt.focal_field(extra_phase=dk * self._efc_corr_modes[k])[dh] - E_ref) / dk
        A = np.vstack([G.real, G.imag])
        self._efc_U, self._efc_S, self._efc_Vt = np.linalg.svd(A, full_matrices=False)

    def _run_efc_warmstart(self):
        opt = self.optics
        dh = np.asarray(opt.dark_hole_mask)
        U, S, Vt = self._efc_U, self._efc_S, self._efc_Vt
        smax = S[0]
        for _ in range(self._efc_iters):
            E0 = opt.focal_field()[dh]                               # perfect sensing (setup only)
            Utb = U.T @ np.concatenate([E0.real, E0.imag])
            c_save = opt.correction_coeffs.copy()
            best = (np.inf, c_save)
            for rc in self._efc_rconds:                             # line-search the SVD cutoff
                keep = S >= rc * smax
                sinv = np.where(keep, 1.0 / np.where(keep, S, 1.0), 0.0)
                dc = -(Vt.T @ (sinv * Utb))
                opt.set_correction_modes(np.clip(c_save + dc, -self._efc_max_abs, self._efc_max_abs))
                c = opt.dark_hole_contrast()
                if c < best[0]:
                    best = (c, opt.correction_coeffs.copy())
            opt.set_correction_modes(best[1])

    def reset(self, *, seed=None, options=None):
        self._correction_offset = None                            # base only after warm-start
        obs, info = super().reset(seed=seed, options=options)
        if self._efc_on:
            self._run_efc_warmstart()                              # corrector -> EFC floor
            self._correction_offset = self.optics.correction_coeffs.copy()   # EFC base
            obs, contrast, strehl = self._observe_probe(np.zeros(self.control_dim))
            self._prev_contrast = contrast
            info = {"contrast": contrast, "strehl": strehl}
        return obs, info


def make_probe_env(cfg: ProbeConfig) -> CoronagraphEnv:
    env = EFCWarmStartEnv(
        cfg,
        q=cfg.q, num_airy=cfg.num_airy,
        use_coronagraph=cfg.use_coronagraph,
        dark_hole_iwa=cfg.dark_hole_iwa, dark_hole_owa=cfg.dark_hole_owa,
        max_steps=cfg.max_steps,
        rms_min_waves=cfg.rms_min, rms_max_waves=cfg.rms_max,
        num_aberration_modes=cfg.num_modes, aberration_start_mode=cfg.aberration_start_mode,
        aberration_spectrum=cfg.aberration_spectrum, psd_exponent=cfg.psd_exponent,
        aberration_tau=(cfg.aberration_tau if cfg.aberration_tau > 0 else None),
        num_control_modes=cfg.num_modes, ideal_modal_correction=True,
        action_mode="absolute", include_command=True,
        action_scale=cfg.action_scale, max_abs_actuator=cfg.action_scale,
        diversity="none", image_scale="raw", flatten_obs=False,
        photon_flux=(cfg.photon_flux if cfg.photon_flux > 0 else None),
        objective=("strehl" if not cfg.use_coronagraph else "log_contrast"),
        probe_mode=True, probe_scale=cfg.action_scale,
    )
    if cfg.correction_gain_sigma > 0:
        g = _corrector_gain_vector(env.optics.num_correction_modes,
                                   cfg.correction_gain_sigma, cfg.correction_gain_seed)
        env.optics.set_correction_gain(g)
    env.setup_efc_warmstart(cfg)   # no-op unless cfg.efc_warmstart
    return env


class PO4NCPAProbe:
    def __init__(self, cfg: ProbeConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = make_probe_env(cfg)
        self.eval_env = make_probe_env(cfg)
        self.h, self.w = self.env.image_shape
        self.M = self.env.control_dim                          # correction modes
        self.A = int(self.env.action_space.shape[0])           # 2M (probe + correction)
        self._apply_scales()
        self.ideal = self.env.ideal_image().astype(np.float32)  # o-channel reference (2D)

        self.dyn = [ProbeDynamicsNet(self.h, self.w, self.A, ch=cfg.ch).to(self.device)
                    for _ in range(cfg.ensemble)]
        self.dyn_opt = [torch.optim.Adam(m.parameters(), lr=cfg.dyn_lr) for m in self.dyn]
        self.policy = ProbePolicyNet(self.h, self.w, self.A, ch=cfg.ch).to(self.device)
        self.pol_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.pol_lr)

        self.buf = ProbeBuffer(cfg.buffer, self.h, self.w, self.A)

        # Optional resume: load policy + dynamics (+ optimizer moments if present). The
        # replay buffer is not saved (too large), so a resumed run refills it on-policy
        # with the loaded policy before policy updates resume (no random re-warmup).
        self.resumed = False
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ck["policy"])
            for m, sd in zip(self.dyn, ck["dynamics"]):
                m.load_state_dict(sd)
            if ck.get("pol_opt") is not None:
                self.pol_opt.load_state_dict(ck["pol_opt"])
            if ck.get("dyn_opt") is not None:
                for opt, sd in zip(self.dyn_opt, ck["dyn_opt"]):
                    opt.load_state_dict(sd)
            self.resumed = True
            print(f"[probe] RESUMED from {cfg.resume} "
                  f"(optimizers {'restored' if ck.get('pol_opt') is not None else 'reset'})", flush=True)

        self.log_dir = os.path.join("logs", cfg.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv = open(os.path.join(self.log_dir, "train.csv"), "w")
        self._csv.write("episode,train_contrast,train_strehl,eval_contrast,eval_strehl\n")
        self._csv.flush()

    def _apply_scales(self):
        scale = calibrate_per_mode_scale(self.env.optics, self.cfg)   # correction per-mode scale (M)
        probe_scale = self.cfg.probe_frac * scale
        for e in (self.env, self.eval_env):
            e.action_scale = scale
            e.max_abs_actuator = scale
            e.probe_scale = probe_scale
        self.scale = scale
        self.probe_scale = probe_scale
        print(f"[probe] correction scale min {scale.min():.2e} max {scale.max():.2e} | "
              f"probe scale (x{self.cfg.probe_frac}) min {probe_scale.min():.2e} "
              f"max {probe_scale.max():.2e} | modes={self.M}", flush=True)

    # --- preprocessing: [o, d] raw -> [cbrt(o-ideal), cbrt(d)] -------------
    def _pre(self, image_2ch: np.ndarray) -> np.ndarray:
        o = np.cbrt(image_2ch[0] - self.ideal)
        d = np.cbrt(image_2ch[1])
        return np.stack([o, d], axis=0).astype(np.float32)

    def _dyn_rand(self, obs, ap, ac):
        m = self.dyn[np.random.randint(self.cfg.ensemble)]
        return m(obs, ap, ac)

    # --- data collection ---------------------------------------------------
    def collect_episode(self, episode, deterministic, env, seed=None):
        cfg = self.cfg
        obs, info = env.reset(seed=seed)
        img = self._pre(obs["image"])
        a_prev = np.zeros(self.A, np.float32)
        # No random warm-up on a resumed run: the loaded policy already collects good data.
        warmup = (episode < cfg.warmup_episodes) and not deterministic and not self.resumed
        last = info
        for _ in range(env.max_steps):
            if warmup:
                a = np.random.uniform(-1.0, 1.0, self.A).astype(np.float32)
            else:
                it = torch.as_tensor(img[None], device=self.device)
                ap = torch.as_tensor(a_prev[None], device=self.device)
                with torch.no_grad():
                    a = self.policy(it, ap)[0].cpu().numpy().astype(np.float32)
                if not deterministic:
                    a = np.clip(a + np.random.normal(0, cfg.explore_std, self.A), -1, 1).astype(np.float32)
            obs, _r, _t, _tr, info = env.step(a)
            img_next = self._pre(obs["image"])
            if not deterministic:
                self.buf.add(img, img_next, a_prev, a)
            img, a_prev = img_next, a
            last = info
        return last["contrast"], last["strehl"]

    # --- dynamics update (predict both channels) ---------------------------
    def update_dynamics(self):
        cfg = self.cfg
        losses = []
        for m, opt in zip(self.dyn, self.dyn_opt):
            for _ in range(cfg.dyn_iters):
                it, itn, ap, ac = self.buf.sample(cfg.batch, self.device)
                pred = m(it, ap, ac)                            # (B,2,H,W)
                num = torch.sqrt(((itn - pred) ** 2).sum(dim=[1, 2, 3]))
                den = torch.sqrt((itn ** 2).sum(dim=[1, 2, 3])) + 1e-8
                loss = (num / den).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    # --- policy update: backprop reward through imagined rollouts -----------
    def update_policy(self):
        cfg = self.cfg
        for p in (q for m in self.dyn for q in m.parameters()):
            p.requires_grad_(False)
        rewards = []
        for _ in range(cfg.pol_iters):
            it, _itn, ap, _ac = self.buf.sample(cfg.batch, self.device)
            obs_t, a_tm1 = it, ap
            H = int(np.random.randint(cfg.h_min, cfg.h_max + 1))
            total_r = 0.0
            for _k in range(H):
                a = self.policy(obs_t, a_tm1)                  # [probe, correction]
                pred = self._dyn_rand(obs_t, a_tm1, a)         # (B,2,H,W) = [o', d']
                total_r = total_r + (-(pred[:, 0:1] ** 2).sum(dim=[1, 2, 3]))  # reward on dark image only
                obs_t = pred
                a_tm1 = a
            loss = -total_r.mean()
            self.pol_opt.zero_grad(); loss.backward(); self.pol_opt.step()
            rewards.append((total_r.mean() / H).item())
        for p in (q for m in self.dyn for q in m.parameters()):
            p.requires_grad_(True)
        return float(np.mean(rewards)) if rewards else 0.0

    def evaluate(self):
        cs, ss = [], []
        for e in range(self.cfg.eval_episodes):
            c, s = self.collect_episode(0, True, self.eval_env, seed=100_000 + e)
            cs.append(c); ss.append(s)
        return float(np.median(cs)), float(np.median(ss))

    def train(self):
        cfg = self.cfg
        print(f"[probe] device={self.device} image={self.h}x{self.w} modes={self.M} "
              f"action={self.A} coronagraph={cfg.use_coronagraph}", flush=True)
        t0 = time.time()
        best = -np.inf
        for ep in range(cfg.total_episodes):
            c_tr, s_tr = self.collect_episode(ep, False, self.env)
            dyn_loss = pol_r = 0.0
            if self.buf.size >= cfg.batch:
                dyn_loss = self.update_dynamics()
                ready = (ep >= cfg.warmup_episodes) or self.resumed
                # On resume, refill the buffer on-policy before updating the loaded policy.
                if self.resumed and self.buf.size < cfg.batch * 20:
                    ready = False
                if ready:
                    pol_r = self.update_policy()
            if (ep + 1) % cfg.eval_every == 0 or ep == cfg.total_episodes - 1:
                ec, es = self.evaluate()
                dt = time.time() - t0
                metric = es if not cfg.use_coronagraph else -np.log10(max(ec, 1e-12))
                print(f"ep {ep+1:5d} | dynL {dyn_loss:.3f} polR {pol_r:.3e} | "
                      f"train C {c_tr:.2e} S {s_tr:.3f} | EVAL C {ec:.3e} S {es:.4f} | "
                      f"buf {self.buf.size} | {dt:.0f}s", flush=True)
                self._csv.write(f"{ep+1},{c_tr:.6e},{s_tr:.6f},{ec:.6e},{es:.6f}\n")
                self._csv.flush()
                if metric > best:
                    best = metric
                    self.save("best")
        self.save("final")
        self._csv.close()
        print(f"[probe] done in {time.time()-t0:.0f}s", flush=True)

    def save(self, tag):
        path = os.path.join(self.log_dir, f"po4ncpa_probe_{tag}.pt")
        torch.save({"policy": self.policy.state_dict(),
                    "dynamics": [m.state_dict() for m in self.dyn],
                    "pol_opt": self.pol_opt.state_dict(),
                    "dyn_opt": [o.state_dict() for o in self.dyn_opt],
                    "cfg": vars(self.cfg)}, path)


def parse_args() -> ProbeConfig:
    p = argparse.ArgumentParser(description="PO4NCPA + learned active probing (M2)")
    d = ProbeConfig()
    for f in fields(d):
        name = "--" + f.name.replace("_", "-")
        if isinstance(getattr(d, f.name), bool):
            p.add_argument(name, type=lambda x: x.lower() in ("1", "true", "yes"),
                           default=getattr(d, f.name))
        else:
            p.add_argument(name, type=type(getattr(d, f.name)), default=getattr(d, f.name))
    a = p.parse_args()
    return ProbeConfig(**{f.name: getattr(a, f.name) for f in fields(d)})


if __name__ == "__main__":
    PO4NCPAProbe(parse_args()).train()
