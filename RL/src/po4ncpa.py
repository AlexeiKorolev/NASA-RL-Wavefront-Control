"""PO4NCPA -- model-based RL for focal-plane wavefront control.

A from-scratch implementation of "Policy Optimization for NCPAs" (Nousiainen,
Taskin, Kasper, Orban de Xivry & Absil, A&A 709, A267 / arXiv:2604.00993),
running on this repo's `CoronagraphEnv` / `CoronagraphOptics` physics backend.

Unlike the model-free PPO pipeline (`train_ppo.py`), PO4NCPA learns an explicit,
*differentiable* dynamics model of the focal-plane image and trains the policy by
back-propagating the reward through imagined rollouts of that model. This is what
lets it learn where pure PPO plateaued: the learned model supplies an analytic
gradient of (predicted) focal-plane light w.r.t. the DM command even when the
dark image carries almost no model-free exploration signal.

Two ingredients (paper Sec. 3):

  * **Sequential phase diversity.** The state is ``s_t = (o_t, o_{t-1}, a_{t-1})``
    -- the current and previous focal images plus the previous command. The
    natural step-to-step command change acts as an implicit phase probe and
    resolves the even-mode sign ambiguity a single (especially dark) frame cannot.
    No dedicated defocus/probe diversity is used (`diversity="none"`).

  * **Cube-root residual preprocessing (Eq. 7).** Images are preprocessed as
    ``o = cbrt(PSF - PSF_ideal)`` -- ideal-PSF-subtracted, signed cube root --
    which compresses the dynamic range for the network. The reward is the negative
    squared norm of this residual, ``r = -||o_{t+1}||^2`` (total focal-plane light;
    for the coronagraph this *is* the dark-hole contrast objective).

Algorithm 1, per episode:
  1. roll the policy for T steps (random actions during warm-up), store transitions;
  2. improve the dynamics ensemble by supervised one-step prediction (relative MSE);
  3. improve the policy by back-propagating the reward through H-step imagined
     rollouts of the (frozen) dynamics ensemble.
"""
from __future__ import annotations

import os
import argparse
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment import CoronagraphEnv


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
def _conv_out(n: int) -> int:
    """Spatial size after a 3x3 stride-2 pad-1 conv."""
    return (n - 1) // 2 + 1


class DynamicsNet(nn.Module):
    """Predicts the next preprocessed focal image o_{t+1} from (o_t, o_{t-1}) and
    the previous + current commands (a_{t-1}, a_t). A small U-Net with skips."""

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
        self.d2 = nn.Conv2d(2 * ch, ch, 3, padding=1)   # after skip with e2
        self.d1 = nn.Conv2d(2 * ch, ch, 3, padding=1)   # after skip with e1
        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, img_pair, cmd_prev, cmd_cur):
        e1 = F.relu(self.c1(img_pair))          # (B, ch, H, W)
        e2 = F.relu(self.c2(e1))                # (B, ch, h1, w1)
        e3 = F.relu(self.c3(e2))                # (B, ch, h2, w2)
        z = torch.cat([e3.flatten(1), cmd_prev, cmd_cur], dim=1)
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z)).view(-1, self.ch, self.h2, self.w2)
        d = F.interpolate(z, size=e2.shape[-2:], mode="nearest")
        d = F.relu(self.d2(torch.cat([d, e2], dim=1)))
        d = F.interpolate(d, size=e1.shape[-2:], mode="nearest")
        d = F.relu(self.d1(torch.cat([d, e1], dim=1)))
        return self.out(d)                      # (B, 1, H, W)


class PolicyNet(nn.Module):
    """Maps the state (o_t, o_{t-1}, a_{t-1}) to an absolute command a_t in [-1,1]^N."""

    def __init__(self, h: int, w: int, action_dim: int, ch: int = 32):
        super().__init__()
        self.ch = ch
        self.c1 = nn.Conv2d(2, ch, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        h2, w2 = _conv_out(_conv_out(h)), _conv_out(_conv_out(w))
        self.fc1 = nn.Linear(ch * h2 * w2 + action_dim, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, img_pair, cmd_prev):
        x = F.relu(self.c1(img_pair))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = torch.cat([x.flatten(1), cmd_prev], dim=1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


# ---------------------------------------------------------------------------
# Replay buffer (one-step transitions in preprocessed-image space)
# ---------------------------------------------------------------------------
class Buffer:
    def __init__(self, capacity: int, h: int, w: int, action_dim: int):
        self.cap = capacity
        self.op = np.zeros((capacity, h, w), np.float32)   # o_{t-1}
        self.oc = np.zeros((capacity, h, w), np.float32)   # o_t
        self.on = np.zeros((capacity, h, w), np.float32)   # o_{t+1}
        self.ap = np.zeros((capacity, action_dim), np.float32)  # a_{t-1}
        self.ac = np.zeros((capacity, action_dim), np.float32)  # a_t
        self.size = 0
        self.ptr = 0

    def add(self, op, oc, on, ap, ac):
        i = self.ptr
        self.op[i], self.oc[i], self.on[i] = op, oc, on
        self.ap[i], self.ac[i] = ap, ac
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, size=n)
        t = lambda a: torch.as_tensor(a[idx], device=device)
        return t(self.op), t(self.oc), t(self.on), t(self.ap), t(self.ac)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
@dataclass
class Config:
    run_name: str = "po4ncpa_strehl"
    use_coronagraph: bool = False
    num_modes: int = 20
    aberration_start_mode: int = 2     # 4 = exclude tip/tilt (paper convention)
    q: int = 3
    num_airy: float = 5.5
    rms_min: float = 0.02
    rms_max: float = 0.08
    aberration_spectrum: str = "power_law"
    psd_exponent: float = 2.0          # 3.667 (11/3) approximates a Kolmogorov modal spectrum
    photon_flux: float = 0.0           # 0 -> noiseless
    dark_hole_iwa: float = 1.5
    dark_hole_owa: float = 4.0
    action_scale: float = 1e-7         # fallback scalar scale if per_mode_scale=False
    per_mode_scale: bool = True        # scale each mode's action by its expected NCPA
    action_headroom: float = 5.0       # per-mode authority = headroom x per-mode RMS
    #   Per-mode scaling (the paper's "actions scaled by max expected NCPA per mode")
    #   is essential with many modes + a steep (Kolmogorov) spectrum: a uniform scalar
    #   scale gives high-order, ~zero-power modes full authority to inject speckles and
    #   wreck the PSF -- modes the dynamics model predicts worst, so the policy exploits
    #   them. Sizing each mode's stroke to its expected aberration starves that failure.
    max_steps: int = 20
    # MBRL
    total_episodes: int = 3000
    warmup_episodes: int = 200
    ensemble: int = 5
    ch: int = 32                       # conv width of the dynamics/policy nets
    dyn_iters: int = 8
    pol_iters: int = 5
    batch: int = 64
    h_min: int = 2
    h_max: int = 7
    dyn_lr: float = 1e-3
    pol_lr: float = 1e-4
    explore_std: float = 0.1
    buffer: int = 200_000
    eval_every: int = 100
    eval_episodes: int = 30
    seed: int = 0
    device: str = "cuda"


def calibrate_per_mode_scale(optics, cfg) -> np.ndarray:
    """Per-mode action authority ~ headroom x each mode's expected NCPA RMS.

    Monte-Carlo over the aberration distribution: project random draws onto the ideal
    corrector and take the per-mode coefficient std. Deterministic given ``cfg.seed``,
    so training and eval reproduce the same scaling vector."""
    rng = np.random.default_rng(cfg.seed + 777)
    coeffs = []
    for _ in range(200):
        rms = rng.uniform(cfg.rms_min, cfg.rms_max)
        optics.set_random_aberration(rms, rng=rng, spectrum=cfg.aberration_spectrum,
                                     psd_exponent=cfg.psd_exponent)
        coeffs.append(optics.ideal_modal_correction_coeffs())
    optics.clear_aberration()
    per_mode_std = np.std(np.asarray(coeffs), axis=0)
    per_mode_std = np.maximum(per_mode_std, per_mode_std.max() * 1e-3)  # tiny floor, never zero
    return (cfg.action_headroom * per_mode_std).astype(np.float64)


def make_env(cfg: Config, seed_offset: int = 0) -> CoronagraphEnv:
    return CoronagraphEnv(
        q=cfg.q, num_airy=cfg.num_airy,
        use_coronagraph=cfg.use_coronagraph,
        dark_hole_iwa=cfg.dark_hole_iwa, dark_hole_owa=cfg.dark_hole_owa,
        max_steps=cfg.max_steps,
        rms_min_waves=cfg.rms_min, rms_max_waves=cfg.rms_max,
        num_aberration_modes=cfg.num_modes, aberration_start_mode=cfg.aberration_start_mode,
        aberration_spectrum=cfg.aberration_spectrum, psd_exponent=cfg.psd_exponent,
        num_control_modes=cfg.num_modes, ideal_modal_correction=True,
        action_mode="absolute", include_command=True,
        action_scale=cfg.action_scale, max_abs_actuator=cfg.action_scale,
        diversity="none", image_scale="raw", flatten_obs=False,
        photon_flux=(cfg.photon_flux if cfg.photon_flux > 0 else None),
        objective=("strehl" if not cfg.use_coronagraph else "log_contrast"),
    )


class PO4NCPA:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = make_env(cfg)
        self.eval_env = make_env(cfg)
        self.h, self.w = self.env.image_shape
        self.action_dim = int(self.env.action_space.shape[0])
        if cfg.per_mode_scale:
            self._apply_per_mode_scale()
        # ideal reference image for the cube-root residual preprocessing (Eq. 7)
        self.ideal = self.env.ideal_image().astype(np.float32)   # (H, W), raw norm. intensity

        self.dyn = [DynamicsNet(self.h, self.w, self.action_dim, ch=cfg.ch).to(self.device)
                    for _ in range(cfg.ensemble)]
        self.dyn_opt = [torch.optim.Adam(m.parameters(), lr=cfg.dyn_lr) for m in self.dyn]
        self.policy = PolicyNet(self.h, self.w, self.action_dim, ch=cfg.ch).to(self.device)
        self.pol_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.pol_lr)

        self.buf = Buffer(cfg.buffer, self.h, self.w, self.action_dim)

        self.log_dir = os.path.join("logs", cfg.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv = open(os.path.join(self.log_dir, "train.csv"), "w")
        self._csv.write("episode,train_contrast,train_strehl,eval_contrast,eval_strehl\n")
        self._csv.flush()

    # --- per-mode action scaling -------------------------------------------
    def _apply_per_mode_scale(self):
        """Override the env's scalar action_scale/max_abs with the per-mode vector so
        high-order, low-power modes get little stroke (cannot wreck the PSF) while the
        dominant modes keep range. See ``calibrate_per_mode_scale``."""
        scale = calibrate_per_mode_scale(self.env.optics, self.cfg)
        for e in (self.env, self.eval_env):
            e.action_scale = scale
            e.max_abs_actuator = scale
        self.action_scale_vec = scale
        print(f"[po4ncpa] per-mode action scale: min {scale.min():.2e} max {scale.max():.2e} "
              f"(headroom {self.cfg.action_headroom}, {len(scale)} modes)", flush=True)

    # --- preprocessing -----------------------------------------------------
    def _pre(self, raw_img_2d: np.ndarray) -> np.ndarray:
        """Cube-root residual preprocessing (Eq. 7), signed."""
        return np.cbrt(raw_img_2d - self.ideal).astype(np.float32)

    def _dyn_predict(self, img_pair, cmd_prev, cmd_cur):
        """Ensemble-mean next-image prediction (differentiable)."""
        preds = [m(img_pair, cmd_prev, cmd_cur) for m in self.dyn]
        return torch.stack(preds, 0).mean(0)

    def _dyn_predict_rand(self, img_pair, cmd_prev, cmd_cur):
        """Next-image prediction from a randomly chosen ensemble member.

        Used inside policy rollouts: forcing the policy to do well under a randomly
        sampled model each step penalizes exploiting any single model's errors
        (model disagreement is largest exactly where the data is sparse)."""
        m = self.dyn[np.random.randint(self.cfg.ensemble)]
        return m(img_pair, cmd_prev, cmd_cur)

    # --- one episode of data collection ------------------------------------
    def collect_episode(self, episode: int, deterministic: bool, env: CoronagraphEnv,
                        seed: int | None = None):
        cfg = self.cfg
        obs, info = env.reset(seed=seed)
        o_cur = self._pre(obs["image"][0])
        o_prev = o_cur.copy()                      # no previous frame at t=0
        a_prev = obs["command"].astype(np.float32).copy()
        warmup = (episode < cfg.warmup_episodes) and not deterministic
        last = info
        for _ in range(env.max_steps):
            if warmup:
                a = np.random.uniform(-1.0, 1.0, self.action_dim).astype(np.float32)
            else:
                pair = torch.as_tensor(np.stack([o_cur, o_prev])[None], device=self.device)
                ap = torch.as_tensor(a_prev[None], device=self.device)
                with torch.no_grad():
                    a = self.policy(pair, ap)[0].cpu().numpy().astype(np.float32)
                if not deterministic:
                    a = np.clip(a + np.random.normal(0, cfg.explore_std, self.action_dim), -1, 1)
                    a = a.astype(np.float32)
            obs, _r, _term, _trunc, info = env.step(a)
            o_next = self._pre(obs["image"][0])
            if not deterministic:
                self.buf.add(o_prev, o_cur, o_next, a_prev, a)
            o_prev, o_cur = o_cur, o_next
            a_prev = obs["command"].astype(np.float32).copy()
            last = info
        return last["contrast"], last["strehl"]

    # --- dynamics update ---------------------------------------------------
    def update_dynamics(self):
        cfg = self.cfg
        losses = []
        for m, opt in zip(self.dyn, self.dyn_opt):
            for _ in range(cfg.dyn_iters):
                op, oc, on, ap, ac = self.buf.sample(cfg.batch, self.device)
                pair = torch.stack([oc, op], dim=1)          # (B,2,H,W)
                pred = m(pair, ap, ac)                        # (B,1,H,W)
                tgt = on.unsqueeze(1)
                num = torch.sqrt(((tgt - pred) ** 2).sum(dim=[1, 2, 3]))
                den = torch.sqrt((tgt ** 2).sum(dim=[1, 2, 3])) + 1e-8
                loss = (num / den).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    # --- policy update (back-prop through imagined rollouts) ----------------
    def update_policy(self):
        cfg = self.cfg
        for p in (q for m in self.dyn for q in m.parameters()):
            p.requires_grad_(False)
        rewards = []
        for _ in range(cfg.pol_iters):
            op, oc, _on, ap, _ac = self.buf.sample(cfg.batch, self.device)
            o_t, o_tm1, a_tm1 = oc, op, ap
            H = int(np.random.randint(cfg.h_min, cfg.h_max + 1))
            total_r = 0.0
            for _k in range(H):
                pair = torch.stack([o_t, o_tm1], dim=1)
                a = self.policy(pair, a_tm1)
                pred = self._dyn_predict_rand(pair, a_tm1, a)  # o_{t+1}, (B,1,H,W)
                total_r = total_r + (-(pred ** 2).sum(dim=[1, 2, 3]))
                o_tm1, o_t = o_t, pred.squeeze(1)
                a_tm1 = a
            loss = -total_r.mean()
            self.pol_opt.zero_grad(); loss.backward(); self.pol_opt.step()
            rewards.append((total_r.mean() / H).item())
        for p in (q for m in self.dyn for q in m.parameters()):
            p.requires_grad_(True)
        return float(np.mean(rewards)) if rewards else 0.0

    # --- evaluation --------------------------------------------------------
    def evaluate(self):
        cs, ss = [], []
        for e in range(self.cfg.eval_episodes):
            c, s = self.collect_episode(0, deterministic=True, env=self.eval_env,
                                        seed=100_000 + e)
            cs.append(c); ss.append(s)
        return float(np.median(cs)), float(np.median(ss))

    # --- main loop ---------------------------------------------------------
    def train(self):
        cfg = self.cfg
        print(f"[po4ncpa] device={self.device} image={self.h}x{self.w} "
              f"modes={self.action_dim} coronagraph={cfg.use_coronagraph}", flush=True)
        print(f"[po4ncpa] ideal-image residual ref: max={self.ideal.max():.3e} "
              f"sum={self.ideal.sum():.3e}", flush=True)
        t0 = time.time()
        best = -np.inf
        for ep in range(cfg.total_episodes):
            c_tr, s_tr = self.collect_episode(ep, deterministic=False, env=self.env)
            dyn_loss = pol_r = 0.0
            if self.buf.size >= cfg.batch:
                dyn_loss = self.update_dynamics()
                if ep >= cfg.warmup_episodes:
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
        print(f"[po4ncpa] done in {time.time()-t0:.0f}s", flush=True)

    def save(self, tag: str):
        path = os.path.join(self.log_dir, f"po4ncpa_{tag}.pt")
        torch.save({"policy": self.policy.state_dict(),
                    "dynamics": [m.state_dict() for m in self.dyn],
                    "cfg": vars(self.cfg)}, path)


# ---------------------------------------------------------------------------
def parse_args() -> Config:
    p = argparse.ArgumentParser(description="PO4NCPA model-based RL wavefront control")
    d = Config()
    for f in d.__dataclass_fields__.values():
        name = "--" + f.name.replace("_", "-")
        if f.type == bool or isinstance(getattr(d, f.name), bool):
            p.add_argument(name, type=lambda x: x.lower() in ("1", "true", "yes"),
                           default=getattr(d, f.name))
        else:
            p.add_argument(name, type=type(getattr(d, f.name)), default=getattr(d, f.name))
    a = p.parse_args()
    return Config(**{k: getattr(a, k) for k in d.__dataclass_fields__})


if __name__ == "__main__":
    cfg = parse_args()
    PO4NCPA(cfg).train()
