"""PO4NCPA + learned active probing with a reconstruction auxiliary (M2-refine).

The plain learned-probe trainer (`po4ncpa_probe.py`) learns the probe ONLY implicitly,
through the multi-step reward -- a weak, roundabout signal -- and plateaus ~4 orders above
classical probe-EFC. This variant gives the probe a direct sensing objective, the missing
ingredient (notes/PLAN_active_probing.md, M2 risks):

  1. Field-reconstruction head + loss. A small net reconstructs the complex dark-hole field
     E from the probe-difference image d, supervised against the *true* field
     (optics.focal_field(), the M0 ground truth). The reconstruction is fed to the
     correction, so the loop is sense -> reconstruct -> correct (EFC's structure, learned).
     Because the reconstruction is used in the control path, back-prop pushes the probe to
     produce difference images from which the field is genuinely recoverable.

  2. Reweighted dynamics loss. The difference channel is weighted (lambda_dchan) so the
     dynamics model learns the probe -> difference physics faithfully -- making the policy's
     imagined probe gradient trustworthy at probe scale.

Everything else (2-channel obs [dark image, probe-difference], per-mode scaling, imagined
rollouts, corona-aligned config) matches `po4ncpa_probe.py`.
"""
from __future__ import annotations

import os
import argparse
import time
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from po4ncpa import calibrate_per_mode_scale, _conv_out
from po4ncpa_probe import ProbeConfig, ProbeDynamicsNet, make_probe_env


class FieldHead(nn.Module):
    """Reconstruct the complex dark-hole field from the probe-difference image.
    Input d (B,1,H,W) -> (B, 2*n_dh) = [Re, Im] in signed-cube-root space."""

    def __init__(self, h, w, n_dh, ch=32):
        super().__init__()
        self.c1 = nn.Conv2d(1, ch, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        h2, w2 = _conv_out(_conv_out(h)), _conv_out(_conv_out(w))
        self.fc1 = nn.Linear(ch * h2 * w2, 256)
        self.fc2 = nn.Linear(256, 2 * n_dh)

    def forward(self, d):
        x = F.relu(self.c1(d))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)


class ReconPolicyNet(nn.Module):
    """Maps (obs [o, d], previous action, reconstructed field) -> action [probe, correction]."""

    def __init__(self, h, w, action_dim, n_dh, ch=32):
        super().__init__()
        self.c1 = nn.Conv2d(2, ch, 3, stride=1, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
        h2, w2 = _conv_out(_conv_out(h)), _conv_out(_conv_out(w))
        self.fc1 = nn.Linear(ch * h2 * w2 + action_dim + 2 * n_dh, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def forward(self, obs, cmd_prev, field_est):
        x = F.relu(self.c1(obs))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = torch.cat([x.flatten(1), cmd_prev, field_est], dim=1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))


class ReconBuffer:
    def __init__(self, capacity, h, w, action_dim, n_dh):
        self.cap = capacity
        self.it = np.zeros((capacity, 2, h, w), np.float32)
        self.itn = np.zeros((capacity, 2, h, w), np.float32)
        self.ap = np.zeros((capacity, action_dim), np.float32)
        self.ac = np.zeros((capacity, action_dim), np.float32)
        self.fn = np.zeros((capacity, 2 * n_dh), np.float32)   # field target for itn (cbrt space)
        self.size = 0
        self.ptr = 0

    def add(self, it, itn, ap, ac, fn):
        i = self.ptr
        self.it[i], self.itn[i], self.ap[i], self.ac[i], self.fn[i] = it, itn, ap, ac, fn
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, size=n)
        t = lambda a: torch.as_tensor(a[idx], device=device)
        return t(self.it), t(self.itn), t(self.ap), t(self.ac), t(self.fn)


@dataclass
class ReconConfig(ProbeConfig):
    run_name: str = "po4ncpa_probe_recon"
    buffer: int = 150_000
    lambda_dchan: float = 3.0          # weight of the difference-channel dynamics loss
    lambda_recon: float = 1.0          # weight of the field-reconstruction loss
    field_lr: float = 1e-3
    field_iters: int = 8
    resume: str = ""


class PO4NCPAProbeRecon:
    def __init__(self, cfg: ReconConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        self.env = make_probe_env(cfg)
        self.eval_env = make_probe_env(cfg)
        self.env.return_true_field = True          # collection needs the field target
        self.h, self.w = self.env.image_shape
        self.M = self.env.control_dim
        self.A = int(self.env.action_space.shape[0])
        self.n_dh = int(self.env.optics.dark_hole_mask.sum())
        self._apply_scales()
        self.ideal = self.env.ideal_image().astype(np.float32)

        self.dyn = [ProbeDynamicsNet(self.h, self.w, self.A, ch=cfg.ch).to(self.device)
                    for _ in range(cfg.ensemble)]
        self.dyn_opt = [torch.optim.Adam(m.parameters(), lr=cfg.dyn_lr) for m in self.dyn]
        self.field = FieldHead(self.h, self.w, self.n_dh, ch=cfg.ch).to(self.device)
        self.field_opt = torch.optim.Adam(self.field.parameters(), lr=cfg.field_lr)
        self.policy = ReconPolicyNet(self.h, self.w, self.A, self.n_dh, ch=cfg.ch).to(self.device)
        self.pol_opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.pol_lr)

        self.buf = ReconBuffer(cfg.buffer, self.h, self.w, self.A, self.n_dh)

        self.resumed = False
        if cfg.resume:
            ck = torch.load(cfg.resume, map_location=self.device, weights_only=False)
            self.policy.load_state_dict(ck["policy"])
            self.field.load_state_dict(ck["field"])
            for m, sd in zip(self.dyn, ck["dynamics"]):
                m.load_state_dict(sd)
            if ck.get("pol_opt") is not None:
                self.pol_opt.load_state_dict(ck["pol_opt"])
                self.field_opt.load_state_dict(ck["field_opt"])
                for opt, sd in zip(self.dyn_opt, ck["dyn_opt"]):
                    opt.load_state_dict(sd)
            self.resumed = True
            print(f"[recon] RESUMED from {cfg.resume}", flush=True)

        self.log_dir = os.path.join("logs", cfg.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv = open(os.path.join(self.log_dir, "train.csv"), "w")
        self._csv.write("episode,train_contrast,train_strehl,eval_contrast,eval_strehl\n")
        self._csv.flush()

    def _apply_scales(self):
        scale = calibrate_per_mode_scale(self.env.optics, self.cfg)
        probe_scale = self.cfg.probe_frac * scale
        for e in (self.env, self.eval_env):
            e.action_scale = scale
            e.max_abs_actuator = scale
            e.probe_scale = probe_scale
        self.scale = scale
        print(f"[recon] n_dh={int(self.env.optics.dark_hole_mask.sum())} "
              f"correction scale [{scale.min():.2e},{scale.max():.2e}] "
              f"probe x{self.cfg.probe_frac} | modes={self.env.control_dim}", flush=True)

    def _pre(self, image_2ch):
        o = np.cbrt(image_2ch[0] - self.ideal)
        d = np.cbrt(image_2ch[1])
        return np.stack([o, d], axis=0).astype(np.float32)

    def _field_target(self, field_dh):
        """Complex dark-hole field -> signed-cube-root [Re, Im] vector (2*n_dh)."""
        f = np.asarray(field_dh)
        return np.concatenate([np.cbrt(f.real), np.cbrt(f.imag)]).astype(np.float32)

    def _dyn_rand(self, obs, ap, ac):
        return self.dyn[np.random.randint(self.cfg.ensemble)](obs, ap, ac)

    def collect_episode(self, episode, deterministic, env, seed=None):
        cfg = self.cfg
        obs, info = env.reset(seed=seed)
        img = self._pre(obs["image"])
        a_prev = np.zeros(self.A, np.float32)
        warmup = (episode < cfg.warmup_episodes) and not deterministic and not self.resumed
        last = info
        for _ in range(env.max_steps):
            if warmup:
                a = np.random.uniform(-1.0, 1.0, self.A).astype(np.float32)
            else:
                it = torch.as_tensor(img[None], device=self.device)
                ap = torch.as_tensor(a_prev[None], device=self.device)
                dch = it[:, 1:2]
                with torch.no_grad():
                    fe = self.field(dch)
                    a = self.policy(it, ap, fe)[0].cpu().numpy().astype(np.float32)
                if not deterministic:
                    a = np.clip(a + np.random.normal(0, cfg.explore_std, self.A), -1, 1).astype(np.float32)
            obs, _r, _t, _tr, info = env.step(a)
            img_next = self._pre(obs["image"])
            if not deterministic:
                fn = (self._field_target(info["field_dh"]) if info.get("field_dh") is not None
                      else np.zeros(2 * self.n_dh, np.float32))
                self.buf.add(img, img_next, a_prev, a, fn)
            img, a_prev = img_next, a
            last = info
        return last["contrast"], last["strehl"]

    def update_dynamics(self):
        cfg = self.cfg
        losses = []
        for m, opt in zip(self.dyn, self.dyn_opt):
            for _ in range(cfg.dyn_iters):
                it, itn, ap, ac, _fn = self.buf.sample(cfg.batch, self.device)
                pred = m(it, ap, ac)                               # (B,2,H,W)
                # per-channel relative MSE; up-weight the difference channel
                def rel(ch):
                    num = torch.sqrt(((itn[:, ch] - pred[:, ch]) ** 2).sum(dim=[1, 2]))
                    den = torch.sqrt((itn[:, ch] ** 2).sum(dim=[1, 2])) + 1e-8
                    return (num / den).mean()
                loss = rel(0) + cfg.lambda_dchan * rel(1)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    def update_field(self):
        cfg = self.cfg
        losses = []
        for _ in range(cfg.field_iters):
            _it, itn, _ap, _ac, fn = self.buf.sample(cfg.batch, self.device)
            est = self.field(itn[:, 1:2])                          # reconstruct from the d channel
            num = torch.sqrt(((fn - est) ** 2).sum(dim=1))
            den = torch.sqrt((fn ** 2).sum(dim=1)) + 1e-8
            loss = cfg.lambda_recon * (num / den).mean()
            self.field_opt.zero_grad(); loss.backward(); self.field_opt.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else 0.0

    def update_policy(self):
        cfg = self.cfg
        frozen = [q for m in self.dyn for q in m.parameters()] + list(self.field.parameters())
        for p in frozen:
            p.requires_grad_(False)
        rewards = []
        for _ in range(cfg.pol_iters):
            it, _itn, ap, _ac, _fn = self.buf.sample(cfg.batch, self.device)
            obs_t, a_tm1 = it, ap
            H = int(np.random.randint(cfg.h_min, cfg.h_max + 1))
            total_r = 0.0
            for _k in range(H):
                fe = self.field(obs_t[:, 1:2])                     # reconstruct (grad flows to probe)
                a = self.policy(obs_t, a_tm1, fe)
                pred = self._dyn_rand(obs_t, a_tm1, a)             # [o', d']
                total_r = total_r + (-(pred[:, 0:1] ** 2).sum(dim=[1, 2, 3]))
                obs_t = pred
                a_tm1 = a
            loss = -total_r.mean()
            self.pol_opt.zero_grad(); loss.backward(); self.pol_opt.step()
            rewards.append((total_r.mean() / H).item())
        for p in frozen:
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
        print(f"[recon] device={self.device} image={self.h}x{self.w} modes={self.M} "
              f"action={self.A} n_dh={self.n_dh} lambda_dchan={cfg.lambda_dchan}", flush=True)
        t0 = time.time()
        best = -np.inf
        for ep in range(cfg.total_episodes):
            c_tr, s_tr = self.collect_episode(ep, False, self.env)
            dyn_loss = fld_loss = pol_r = 0.0
            if self.buf.size >= cfg.batch:
                dyn_loss = self.update_dynamics()
                fld_loss = self.update_field()
                ready = (ep >= cfg.warmup_episodes) or self.resumed
                if self.resumed and self.buf.size < cfg.batch * 20:
                    ready = False
                if ready:
                    pol_r = self.update_policy()
            if (ep + 1) % cfg.eval_every == 0 or ep == cfg.total_episodes - 1:
                ec, es = self.evaluate()
                dt = time.time() - t0
                metric = es if not cfg.use_coronagraph else -np.log10(max(ec, 1e-12))
                print(f"ep {ep+1:5d} | dynL {dyn_loss:.3f} fldL {fld_loss:.3f} polR {pol_r:.3e} | "
                      f"train C {c_tr:.2e} S {s_tr:.3f} | EVAL C {ec:.3e} S {es:.4f} | "
                      f"buf {self.buf.size} | {dt:.0f}s", flush=True)
                self._csv.write(f"{ep+1},{c_tr:.6e},{s_tr:.6f},{ec:.6e},{es:.6f}\n")
                self._csv.flush()
                if metric > best:
                    best = metric
                    self.save("best")
        self.save("final")
        self._csv.close()
        print(f"[recon] done in {time.time()-t0:.0f}s", flush=True)

    def save(self, tag):
        path = os.path.join(self.log_dir, f"po4ncpa_probe_recon_{tag}.pt")
        torch.save({"policy": self.policy.state_dict(),
                    "field": self.field.state_dict(),
                    "dynamics": [m.state_dict() for m in self.dyn],
                    "pol_opt": self.pol_opt.state_dict(),
                    "field_opt": self.field_opt.state_dict(),
                    "dyn_opt": [o.state_dict() for o in self.dyn_opt],
                    "cfg": vars(self.cfg)}, path)


def parse_args() -> ReconConfig:
    p = argparse.ArgumentParser(description="PO4NCPA learned probing + reconstruction auxiliary")
    d = ReconConfig()
    for f in fields(d):
        name = "--" + f.name.replace("_", "-")
        if isinstance(getattr(d, f.name), bool):
            p.add_argument(name, type=lambda x: x.lower() in ("1", "true", "yes"),
                           default=getattr(d, f.name))
        else:
            p.add_argument(name, type=type(getattr(d, f.name)), default=getattr(d, f.name))
    a = p.parse_args()
    return ReconConfig(**{f.name: getattr(a, f.name) for f in fields(d)})


if __name__ == "__main__":
    PO4NCPAProbeRecon(parse_args()).train()
