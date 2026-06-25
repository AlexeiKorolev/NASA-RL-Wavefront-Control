"""Reinforcement-learning environment for focal-plane wavefront control.

A single deformable mirror (an actuator grid) must be driven to dig a dark hole
in the focal plane of a vortex coronagraph. The agent observes the science image
plus one phase-diversity image and applies *incremental* corrections to the DM.

Design (see optics.py for the physics):
  * Observation: 2-channel log-scaled normalized-intensity image
        channel 0 = science image (current DM state)
        channel 1 = phase-diversity image (a fixed known probe added to the DM)
  * Action: increment in [-1, 1] per control channel -- raw actuators, or a few
        low-order Zernike modes if num_control_modes is set -- scaled to a surface
        step. The DM accumulates these increments over the episode (closed loop).
  * Reward: log10(prev_contrast / contrast) -- decades of contrast improvement
        per step (sums to log10(c_start / c_final) over an episode), minus an
        optional penalty on DM stroke.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from optics import CoronagraphOptics


class CoronagraphEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 # optics
                 telescope_diameter: float = 8.0,
                 wavelength: float = 1.6e-6,
                 pupil_pixels: int = 256,
                 q: int = 2,
                 num_airy: float = 14.0,
                 vortex_charge: int = 2,
                 lyot_fraction: float = 0.95,
                 num_actuators_across: int = 16,
                 dark_hole_iwa: float = 3.0,
                 dark_hole_owa: float = 12.0,
                 # episode / aberration
                 max_steps: int = 20,
                 rms_min_waves: float = 0.02,
                 rms_max_waves: float = 0.08,
                 num_aberration_modes: int = 20,
                 aberration_start_mode: int = 2,       # Noll index of first mode (4 excludes tip/tilt)
                 aberration_spectrum: str = "white",   # "white" or "power_law" (1/f^psd)
                 psd_exponent: float = 2.0,
                 # action / DM
                 action_scale: float = 1e-8,            # meters of surface per unit action
                 max_abs_actuator: float = 5e-7,        # DM stroke limit (meters of surface)
                 stroke_penalty: float = 0.0,           # reward penalty weight on DM stroke
                 num_control_modes: int | None = None,  # if set, act on N Zernike modes not raw actuators
                 ideal_modal_correction: bool = False,  # if set, bypass DM; act on ideal Zernike modes
                 # observation / objective
                 probe_amplitude: float = 2e-8,         # meters of surface for diversity probe
                 photon_flux: float | None = None,      # if set, add shot noise to images
                 log_floor: float = -12.0,
                 log_ceil: float = 0.0,
                 use_coronagraph: bool = True,          # False -> bare PSF + Strehl objective
                 objective: str = "contrast",           # "contrast" or "strehl"
                 diversity: str = "probe",              # "probe" (DM probe) or "defocus"
                 defocus_rad: float = 1.0,              # defocus-diversity amplitude (radians RMS)
                 include_command: bool = False,         # append previous corrector command to obs
                 flatten_obs: bool = False,             # flat vector obs (image+command) for an MLP
                 image_scale: str = "log",              # "log" or "linear" image normalization
                 linear_ceil: float = 1.0,              # clip ceiling for linear image scaling
                 action_mode: str = "incremental"):     # "incremental" (add) or "absolute" (set) command
        super().__init__()

        self.ideal_modal_correction = bool(ideal_modal_correction)
        if self.ideal_modal_correction and num_control_modes is None:
            raise ValueError("ideal_modal_correction=True requires num_control_modes (number of Zernike modes).")

        self.use_coronagraph = bool(use_coronagraph)
        self.objective = str(objective)
        if self.objective not in ("contrast", "strehl", "log_contrast", "neg_contrast"):
            raise ValueError(f"objective must be 'contrast', 'strehl', 'log_contrast', or "
                             f"'neg_contrast', got {objective!r}")
        self.diversity = str(diversity)
        self.include_command = bool(include_command)
        if self.include_command and not self.ideal_modal_correction:
            raise ValueError("include_command=True is only supported with ideal_modal_correction=True.")
        self.aberration_spectrum = str(aberration_spectrum)
        self.psd_exponent = float(psd_exponent)
        self.flatten_obs = bool(flatten_obs)
        self.image_scale = str(image_scale)
        if self.image_scale not in ("log", "linear", "raw"):
            raise ValueError(f"image_scale must be 'log', 'linear', or 'raw', got {image_scale!r}")
        # Single-channel science obs (no diversity probe) for PO4NCPA-style
        # sequential phase diversity, where the temporal frame pair is the probe.
        self.n_img_channels = 1 if str(diversity) == "none" else 2
        self.linear_ceil = float(linear_ceil)
        self.action_mode = str(action_mode)
        if self.action_mode not in ("incremental", "absolute"):
            raise ValueError(f"action_mode must be 'incremental' or 'absolute', got {action_mode!r}")
        if self.action_mode == "absolute" and not bool(ideal_modal_correction):
            raise ValueError("action_mode='absolute' is only supported with ideal_modal_correction=True.")

        self.optics = CoronagraphOptics(
            telescope_diameter=telescope_diameter, wavelength=wavelength,
            pupil_pixels=pupil_pixels, q=q, num_airy=num_airy,
            vortex_charge=vortex_charge, lyot_fraction=lyot_fraction,
            num_actuators_across=num_actuators_across,
            dark_hole_iwa=dark_hole_iwa, dark_hole_owa=dark_hole_owa,
            num_aberration_modes=int(num_aberration_modes),
            aberration_start_mode=int(aberration_start_mode),
            ideal_modal_correction=self.ideal_modal_correction,
            num_correction_modes=int(num_control_modes) if self.ideal_modal_correction else None)

        self._defocus_phase = self.optics.defocus_phase(defocus_rad) if self.diversity == "defocus" else None

        self.num_actuators = self.optics.num_actuators
        self.image_shape = self.optics.image_shape

        # Action space: ideal Zernike corrector, DM modal control, or raw actuators.
        if self.ideal_modal_correction:
            # Action drives the ideal corrector's Zernike modes directly (no DM).
            self.modal_basis = None
            action_dim = int(num_control_modes)
        elif num_control_modes is not None:
            # Action drives N Zernike modes mapped onto the DM actuator grid.
            self.modal_basis = self.optics.build_modal_actuator_basis(int(num_control_modes))
            action_dim = int(num_control_modes)
        else:
            self.modal_basis = None
            action_dim = self.num_actuators
        self.num_control_modes = num_control_modes

        self.max_steps = int(max_steps)
        self.rms_min_waves = float(rms_min_waves)
        self.rms_max_waves = float(rms_max_waves)

        self.action_scale = float(action_scale)
        self.max_abs_actuator = float(max_abs_actuator)
        self.stroke_penalty = float(stroke_penalty)

        self.photon_flux = photon_flux
        self.log_floor = float(log_floor)
        self.log_ceil = float(log_ceil)

        # Fixed, known phase-diversity probe (seeded so it is reproducible).
        probe_rng = np.random.default_rng(0)
        probe = probe_rng.standard_normal(self.num_actuators)
        self.probe_actuators = probe / np.std(probe) * float(probe_amplitude)

        h, w = self.image_shape
        nc = self.n_img_channels
        img_dim = nc * h * w
        img_hi = np.inf if self.image_scale == "raw" else 1.0
        if self.flatten_obs:
            # Flat vector obs (paper-style): flattened images, optionally followed by
            # the previous command. Image part is in [0,1], command part in [-1,1].
            if self.include_command:
                low = np.concatenate([np.zeros(img_dim, np.float32), -np.ones(action_dim, np.float32)])
                high = np.concatenate([np.full(img_dim, img_hi, np.float32), np.ones(action_dim, np.float32)])
                self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
            else:
                self.observation_space = spaces.Box(low=0.0, high=img_hi, shape=(img_dim,), dtype=np.float32)
        elif self.include_command:
            # Multi-input obs: focal images + the previous (normalized) corrector command.
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0.0, high=img_hi, shape=(nc, h, w), dtype=np.float32),
                "command": spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32),
            })
        else:
            self.observation_space = spaces.Box(low=0.0, high=img_hi, shape=(nc, h, w), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)

        self._step_count = 0
        self._ideal_image_cache = None

    # ------------------------------------------------------------------------
    def ideal_image(self) -> np.ndarray:
        """Unaberrated, flat-correction science image (raw normalized intensity, 2D).

        This is the reference PSF that PO4NCPA subtracts before its cube-root
        preprocessing. Independent of the episode's aberration draw, so it is
        computed once (saving/restoring optics state) and cached."""
        if self._ideal_image_cache is None:
            opt = self.optics
            saved_ab = opt.aberration_phase
            saved_corr_phase = opt.correction_phase
            saved_corr_coeffs = (opt.correction_coeffs.copy()
                                 if opt.correction_coeffs is not None else None)
            saved_act = opt.get_actuators()
            opt.clear_aberration()
            if self.ideal_modal_correction:
                opt.clear_correction()
            opt.flatten_dm()
            ideal = opt.normalized_intensity(coronagraph=self.use_coronagraph,
                                             lyot=self.use_coronagraph)  # noiseless reference
            opt.aberration_phase = saved_ab
            opt.correction_phase = saved_corr_phase
            if saved_corr_coeffs is not None:
                opt.correction_coeffs = saved_corr_coeffs
            opt.set_actuators(saved_act)
            self._ideal_image_cache = ideal.reshape(self.image_shape).astype(np.float32)
        return self._ideal_image_cache

    # ------------------------------------------------------------------------
    def _scale_image(self, intensity_1d: np.ndarray) -> np.ndarray:
        if self.image_scale == "raw":
            # Raw normalized intensity (no compression). PO4NCPA does its own
            # ideal-PSF-subtracted cube-root preprocessing on top of this.
            return np.clip(np.asarray(intensity_1d), 0.0, None).reshape(self.image_shape).astype(np.float32)
        if self.image_scale == "linear":
            scaled = np.asarray(intensity_1d) / self.linear_ceil
        else:
            log_img = np.log10(np.clip(intensity_1d, 10.0 ** self.log_floor, None))
            scaled = (log_img - self.log_floor) / (self.log_ceil - self.log_floor)
        return np.clip(scaled, 0.0, 1.0).reshape(self.image_shape).astype(np.float32)

    def _command_vector(self) -> np.ndarray:
        """Previous corrector command, normalized to ~[-1, 1] by the stroke limit."""
        return np.clip(self.optics.correction_coeffs / self.max_abs_actuator,
                       -1.0, 1.0).astype(np.float32)

    def _observe(self):
        """Propagate the science image and the diversity image.

        Returns (obs, contrast, strehl) so the reward reuses these propagations
        instead of recomputing. In non-coronagraphic mode both channels are bare
        PSFs and the Strehl is read off the science peak (no extra propagation).
        """
        corona = self.use_coronagraph
        science = self.optics.normalized_intensity(coronagraph=corona, lyot=corona,
                                                   flux=self.photon_flux, rng=self.np_random)
        if self.diversity == "none":
            # Single-channel science image; the temporal frame pair carries the
            # phase diversity (PO4NCPA sequential diversity).
            image = self._scale_image(science)[None]
        else:
            if self.diversity == "defocus":
                diversity = self.optics.normalized_intensity(coronagraph=corona, lyot=corona,
                                                             extra_phase=self._defocus_phase,
                                                             flux=self.photon_flux, rng=self.np_random)
            else:
                diversity = self.optics.normalized_intensity(coronagraph=corona, lyot=corona,
                                                             probe_actuators=self.probe_actuators,
                                                             flux=self.photon_flux, rng=self.np_random)
            image = np.stack([self._scale_image(science), self._scale_image(diversity)], axis=0)
        if self.flatten_obs:
            flat = image.reshape(-1)
            obs = (np.concatenate([flat, self._command_vector()]).astype(np.float32)
                   if self.include_command else flat.astype(np.float32))
        elif self.include_command:
            obs = {"image": image, "command": self._command_vector()}
        else:
            obs = image

        # Dark-hole annulus mean intensity. In coronagraphic mode this is the dark
        # image; in non-coronagraphic mode it is the bare-PSF annulus (aberration-
        # scattered light), a valid alternative objective to Strehl.
        contrast = self.optics.dark_hole_contrast(intensity=science)
        # Non-corono science IS the bare PSF, so its peak is the Strehl ratio.
        strehl = float(np.max(science)) if not corona else self.optics.strehl()
        return obs, contrast, strehl

    # ------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.optics.flatten_dm()
        if self.ideal_modal_correction:
            self.optics.clear_correction()
        rms = self.np_random.uniform(self.rms_min_waves, self.rms_max_waves)
        if options is not None and "rms_waves" in options:
            rms = float(options["rms_waves"])
        self.optics.set_random_aberration(rms, rng=self.np_random,
                                          spectrum=self.aberration_spectrum,
                                          psd_exponent=self.psd_exponent)
        self._step_count = 0
        obs, contrast, strehl = self._observe()
        self._prev_contrast = contrast
        return obs, {"contrast": contrast, "strehl": strehl}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        command = action * self.action_scale
        if self.ideal_modal_correction:
            if self.action_mode == "absolute":
                # Action IS the absolute corrector command (paper: action = phi_DM).
                coeffs = command
                if self.max_abs_actuator is not None:
                    coeffs = np.clip(coeffs, -self.max_abs_actuator, self.max_abs_actuator)
                self.optics.set_correction_modes(coeffs)
            else:
                # Action increments the ideal corrector's modal coefficients.
                self.optics.add_correction_modes(command)
                if self.max_abs_actuator is not None:
                    self.optics.set_correction_modes(
                        np.clip(self.optics.correction_coeffs, -self.max_abs_actuator, self.max_abs_actuator))
        else:
            if self.modal_basis is not None:
                command = self.modal_basis @ command   # mode coeffs -> actuator deltas
            self.optics.add_actuators(command)
            if self.max_abs_actuator is not None:
                self.optics.set_actuators(
                    np.clip(self.optics.get_actuators(), -self.max_abs_actuator, self.max_abs_actuator))

        self._step_count += 1
        obs, contrast, strehl = self._observe()

        if self.objective == "strehl":
            # Gutierrez et al. reward: r = -(1 - SR)^(2/5); maximized (=0) at SR=1.
            reward = -float((max(0.0, 1.0 - strehl)) ** 0.4)
        elif self.objective == "log_contrast":
            # Absolute dark-hole reward (paper's coronagraphic style): higher (=deeper
            # dark hole) for lower contrast. Suits gamma=0 greedy nulling.
            reward = -float(np.log10(max(contrast, 1e-12))) / 10.0
        elif self.objective == "neg_contrast":
            # Absolute linear dark-hole reward: r = -contrast / linear_ceil. Unlike the
            # log reward this barely distinguishes deep solutions (its gradient vanishes
            # as contrast -> 0), so it tests how much the log shaping matters.
            reward = -float(contrast) / self.linear_ceil
        else:
            # Per-step reward = decades of contrast dug this step; sums to log10(c_start/c_final).
            reward = float(np.log10(max(self._prev_contrast, 1e-12)) - np.log10(max(contrast, 1e-12)))
        self._prev_contrast = contrast
        if self.stroke_penalty > 0.0:
            stroke = (self.optics.correction_coeffs if self.ideal_modal_correction
                      else self.optics.get_actuators())
            reward -= self.stroke_penalty * float(np.mean(np.asarray(stroke) ** 2))

        terminated = False
        truncated = self._step_count >= self.max_steps
        info = {"contrast": contrast, "strehl": strehl}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None


# Backwards-compatible alias for code/notebooks that imported the old name.
CoronagraphEnvironment = CoronagraphEnv


if __name__ == "__main__":
    env = CoronagraphEnv()
    obs, info = env.reset(seed=0)
    print(f"obs shape {obs.shape}, dtype {obs.dtype}, range [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"action dim {env.action_space.shape}, initial contrast {info['contrast']:.2e}")
    total = 0.0
    for _ in range(env.max_steps):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total += r
        if term or trunc:
            break
    print(f"random rollout: final contrast {info['contrast']:.2e}, strehl {info['strehl']:.3f}, "
          f"return {total:.2f}")
