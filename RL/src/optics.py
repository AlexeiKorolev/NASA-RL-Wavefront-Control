"""Clean coronagraph optical model for wavefront-control experiments.

A single deformable mirror sits in the (clear, circular) pupil of a telescope,
ahead of a charge-2 vector-free vortex coronagraph and a Lyot stop. Light is
propagated to a science focal plane where the dark-hole contrast is measured.

The model is intentionally physical and free of the normalization hacks that
accumulated in the previous environment:

  * The aperture is a clean unobstructed disk (required for deep vortex nulling).
  * The DM is a real actuator grid built from Gaussian influence functions, so
    its state is an actual surface map rather than an abstract modal vector.
  * Intensities are reported as *normalized intensity* (a.k.a. raw contrast):
    focal-plane intensity divided by the peak of the unaberrated,
    non-coronagraphic PSF. No per-mode rescaling is applied anywhere.

With the defaults below the dark-hole contrast at zero aberration is ~7e-11,
i.e. below the 1e-10 target, limited only by numerical sampling.
"""
from __future__ import annotations

import numpy as np
from hcipy import (make_pupil_grid, make_focal_grid, FraunhoferPropagator,
                   make_circular_aperture, evaluate_supersampled, Wavefront,
                   VortexCoronagraph, Apodizer, DeformableMirror,
                   make_gaussian_influence_functions, make_zernike_basis,
                   large_poisson)


def _noll_radial_order(j: int) -> int:
    """Radial order n of Noll index j (j=1 piston -> n=0, j=2,3 tip/tilt -> n=1, ...)."""
    n = 0
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    return n


class CoronagraphOptics:
    """Telescope + DM + vortex coronagraph + Lyot stop + science camera."""

    def __init__(self,
                 telescope_diameter: float = 8.0,      # meters
                 wavelength: float = 1.6e-6,            # meters
                 pupil_pixels: int = 256,
                 q: int = 2,                            # science pixels per lambda/D
                 num_airy: float = 14.0,                # focal grid half-extent in lambda/D
                 vortex_charge: int = 2,
                 lyot_fraction: float = 0.95,
                 num_actuators_across: int = 16,
                 dark_hole_iwa: float = 3.0,            # lambda/D
                 dark_hole_owa: float = 12.0,           # lambda/D
                 num_aberration_modes: int = 20,
                 aberration_start_mode: int = 2,        # Noll index of the first mode (2 incl. tip/tilt; 4 excludes it)
                 vortex_q: int = 256,
                 ideal_modal_correction: bool = False,
                 num_correction_modes: int | None = None):
        self.telescope_diameter = float(telescope_diameter)
        self.wavelength = float(wavelength)
        self.pupil_pixels = int(pupil_pixels)
        self.num_actuators_across = int(num_actuators_across)
        self.dark_hole_iwa = float(dark_hole_iwa)
        self.dark_hole_owa = float(dark_hole_owa)
        self.spatial_resolution = self.wavelength / self.telescope_diameter  # lambda/D in focal units

        # --- grids and propagation ------------------------------------------
        self.pupil_grid = make_pupil_grid(self.pupil_pixels, self.telescope_diameter)
        self.focal_grid = make_focal_grid(q=q, num_airy=num_airy,
                                          spatial_resolution=self.spatial_resolution)
        self.prop = FraunhoferPropagator(self.pupil_grid, self.focal_grid)
        self.image_shape = (int(np.sqrt(self.focal_grid.size)),) * 2

        # --- clean unobstructed aperture ------------------------------------
        self.aperture = evaluate_supersampled(
            make_circular_aperture(self.telescope_diameter), self.pupil_grid, 4)
        self._aperture_support = self.aperture > 0.5
        self.aberration_phase = self.pupil_grid.zeros()  # radians, applied at the pupil
        # Ideal modal corrector phase (radians). Always defined so _wavefront and
        # the reference-peak computation below can read it; populated with a real
        # Zernike basis at the end of __init__ when ideal_modal_correction is set.
        self.correction_phase = self.pupil_grid.zeros()

        # --- coronagraph + Lyot stop ----------------------------------------
        self.coronagraph = VortexCoronagraph(self.pupil_grid, vortex_charge, q=vortex_q)
        self.lyot_stop = Apodizer(evaluate_supersampled(
            make_circular_aperture(lyot_fraction * self.telescope_diameter),
            self.pupil_grid, 4))

        # --- deformable mirror (actuator grid) ------------------------------
        self.actuator_spacing = self.telescope_diameter / self.num_actuators_across
        self.dm = DeformableMirror(make_gaussian_influence_functions(
            self.pupil_grid, self.num_actuators_across, self.actuator_spacing))
        self.num_actuators = int(self.dm.num_actuators)
        self.dm.flatten()

        # --- normalization reference ----------------------------------------
        # Peak of the unaberrated, flat-DM, bare-aperture PSF (no coronagraph,
        # no Lyot stop). Used to normalize both contrast and Strehl.
        self.reference_peak = float(self._focal_intensity(coronagraph=False, lyot=False).max())

        # --- dark-hole mask (in lambda/D) -----------------------------------
        r_lod = self.focal_grid.as_('polar').r / self.spatial_resolution
        self.dark_hole_mask = (r_lod >= self.dark_hole_iwa) & (r_lod <= self.dark_hole_owa)

        # --- aberration model (smooth low-order Zernike phase screen) -------
        # starting_mode=2 skips piston (incl. tip/tilt); starting_mode=4 also skips
        # tip/tilt (the paper's convention -- a separate fast loop handles those).
        self.aberration_start_mode = int(aberration_start_mode)
        self._aberration_basis = make_zernike_basis(
            num_aberration_modes, self.telescope_diameter, self.pupil_grid,
            starting_mode=self.aberration_start_mode)
        # Noll radial order of each aberration mode (Noll j = i + start_mode), used as
        # the spatial-frequency proxy for a power-law spectrum (Kolmogorov ~ n^-11/3).
        self._aberration_radial_orders = np.array(
            [_noll_radial_order(i + self.aberration_start_mode) for i in range(num_aberration_modes)],
            dtype=np.float64)

        # --- ideal modal corrector (optional) -------------------------------
        # An idealized "Zernike corrector" that bypasses the DM entirely: a modal
        # coefficient (in meters of surface RMS, same units as the DM modal basis)
        # is turned directly into the pupil phase ``2*k*surface`` the DM *would*
        # produce if it could realize that Zernike shape exactly. Because the
        # correction basis is the same low-order Zernike set as the disturbance,
        # phase conjugation becomes exact and the DM spatial-frequency bandwidth
        # limit disappears -- the residual is set only by the numerical floor.
        self.ideal_modal_correction = bool(ideal_modal_correction)
        if self.ideal_modal_correction:
            n = num_aberration_modes if num_correction_modes is None else int(num_correction_modes)
            self.num_correction_modes = int(n)
            k = 2.0 * np.pi / self.wavelength
            basis = make_zernike_basis(self.num_correction_modes, self.telescope_diameter,
                                       self.pupil_grid, starting_mode=self.aberration_start_mode)
            modes = np.empty((self.num_correction_modes, self.pupil_grid.size), dtype=np.float64)
            for i in range(self.num_correction_modes):
                z = np.asarray(basis[i], dtype=np.float64)
                rms = np.std(z[self._aperture_support])
                z = z / rms if rms > 0 else z          # unit surface-RMS Zernike map
                modes[i] = 2.0 * k * z                 # pupil phase per unit coeff
            self._correction_modes = modes             # (num_correction_modes, num_pixels)
            self.correction_coeffs = np.zeros(self.num_correction_modes)
        else:
            self.num_correction_modes = None
            self._correction_modes = None
            self.correction_coeffs = None

    # ------------------------------------------------------------------------
    # Wavefront construction
    # ------------------------------------------------------------------------
    def _wavefront(self, extra_phase=None) -> Wavefront:
        # Aberration plus the ideal-corrector phase (zero unless that mode is on),
        # plus an optional per-exposure diversity phase (e.g. defocus).
        phase = self.aberration_phase + self.correction_phase
        if extra_phase is not None:
            phase = phase + extra_phase
        field = self.aperture * np.exp(1j * phase)
        return Wavefront(field, self.wavelength)

    def defocus_phase(self, rms_rad: float):
        """A Noll j=4 defocus phase over the aperture, scaled to ``rms_rad`` RMS.

        Used as a known phase-diversity probe for the non-coronagraphic PSF
        observation (breaks the even-aberration sign ambiguity of a single PSF)."""
        basis = make_zernike_basis(1, self.telescope_diameter, self.pupil_grid, starting_mode=4)
        z = np.asarray(basis[0], dtype=np.float64)
        rms = np.std(z[self._aperture_support])
        z = z / rms if rms > 0 else z
        return self.pupil_grid.zeros() + float(rms_rad) * z

    # ------------------------------------------------------------------------
    # Aberration control
    # ------------------------------------------------------------------------
    def set_aberration_phase(self, phase) -> None:
        self.aberration_phase = self.pupil_grid.zeros() + np.asarray(phase)

    def set_random_aberration(self, rms_waves: float, rng: np.random.Generator | None = None,
                              spectrum: str = "white", psd_exponent: float = 2.0) -> None:
        """Set a random smooth aberration with the requested RMS wavefront error (in waves).

        spectrum: "white" draws every Zernike mode with equal expected power;
            "power_law" gives a PSD ~ f^-psd_exponent by scaling each mode's
            coefficient std by ``1/n^(psd_exponent/2)`` (n = Noll radial order),
            concentrating power in low-order modes (Gutierrez et al. use 1/f^2).
        """
        rng = np.random.default_rng() if rng is None else rng
        coeffs = rng.standard_normal(len(self._aberration_basis))
        if spectrum == "power_law":
            n = np.maximum(self._aberration_radial_orders, 1.0)
            coeffs = coeffs * n ** (-0.5 * psd_exponent)
        elif spectrum != "white":
            raise ValueError(f"unknown spectrum {spectrum!r} (expected 'white' or 'power_law')")
        phase = sum(c * m for c, m in zip(coeffs, self._aberration_basis))
        rms = np.std(phase[self._aperture_support])
        if rms > 0:
            phase = phase / rms * (rms_waves * 2 * np.pi)
        self.aberration_phase = phase

    def clear_aberration(self) -> None:
        self.aberration_phase = self.pupil_grid.zeros()

    # ------------------------------------------------------------------------
    # DM control
    # ------------------------------------------------------------------------
    def flatten_dm(self) -> None:
        self.dm.flatten()

    def get_actuators(self) -> np.ndarray:
        return np.asarray(self.dm.actuators, dtype=np.float64).copy()

    def set_actuators(self, actuators) -> None:
        self.dm.actuators = np.asarray(actuators, dtype=np.float64)

    def add_actuators(self, delta) -> None:
        self.dm.actuators = np.asarray(self.dm.actuators, dtype=np.float64) + np.asarray(delta, dtype=np.float64)

    def get_surface(self) -> np.ndarray:
        """Return the DM surface map (meters) on the pupil grid, as a 2D array."""
        return np.asarray(self.dm.surface).reshape(self.pupil_grid.shape)

    # ------------------------------------------------------------------------
    # Ideal modal corrector control (only when ideal_modal_correction=True)
    # ------------------------------------------------------------------------
    def set_correction_modes(self, coeffs) -> None:
        """Set the ideal-corrector modal coefficients (meters of surface RMS).

        Builds ``correction_phase = sum_i coeff_i * (2*k*Z_i)`` directly, bypassing
        the DM. Same units as the DM modal basis, so action_scale / the LS expert
        carry over unchanged."""
        if self._correction_modes is None:
            raise RuntimeError("set_correction_modes requires ideal_modal_correction=True")
        coeffs = np.asarray(coeffs, dtype=np.float64)
        self.correction_coeffs = coeffs.copy()
        self.correction_phase = self.pupil_grid.zeros() + (self._correction_modes.T @ coeffs)

    def add_correction_modes(self, delta) -> None:
        """Increment the ideal-corrector modal coefficients (closed loop)."""
        self.set_correction_modes(self.correction_coeffs + np.asarray(delta, dtype=np.float64))

    def clear_correction(self) -> None:
        """Reset the ideal corrector to zero (flat correction phase)."""
        if self.correction_coeffs is not None:
            self.correction_coeffs = np.zeros(self.num_correction_modes)
        self.correction_phase = self.pupil_grid.zeros()

    def ideal_modal_correction_coeffs(self) -> np.ndarray:
        """Exact ideal-corrector coefficients that cancel the current aberration.

        Least-squares projection of ``-aberration_phase`` onto the correction modes
        (each column of ``_correction_modes`` is the pupil phase ``2k*Z_i`` per unit
        coefficient), restricted to the aperture. Used to calibrate per-mode action
        scaling: a controller's per-mode authority should match the per-mode RMS of
        these coefficients so high-order (low-power) modes get little stroke."""
        if self._correction_modes is None:
            raise RuntimeError("ideal_modal_correction_coeffs requires ideal_modal_correction=True")
        sup = self._aperture_support
        A = self._correction_modes.T[sup]            # (n_support, num_modes)
        b = -np.asarray(self.aberration_phase)[sup]  # target phase to cancel
        coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coeffs

    def build_modal_actuator_basis(self, num_modes: int) -> np.ndarray:
        """Actuator commands (num_actuators x num_modes) reproducing the first
        ``num_modes`` Zernike shapes on the DM surface, each scaled to unit surface
        RMS over the aperture.

        Lets a controller drive a few smooth low-order modes instead of all
        independent actuators: ``actuator_delta = basis @ mode_coeffs``. The modal
        space matches the Zernike disturbance model, so a unit coefficient produces
        1 meter of surface RMS regardless of mode (balanced control authority).
        """
        # Influence matrix M: surface = M @ actuators (columns are influence funcs).
        M = np.array([np.asarray(f) for f in self.dm.influence_functions]).T
        basis = make_zernike_basis(num_modes, self.telescope_diameter,
                                   self.pupil_grid, starting_mode=2)
        targets = np.array([np.asarray(basis[i]) for i in range(num_modes)]).T
        coeffs, *_ = np.linalg.lstsq(M, targets, rcond=None)   # (num_actuators, num_modes)
        surface = M @ coeffs                                   # (num_pixels, num_modes)
        rms = np.std(surface[self._aperture_support], axis=0)
        return coeffs / np.where(rms > 0, rms, 1.0)

    # ------------------------------------------------------------------------
    # Imaging and metrics
    # ------------------------------------------------------------------------
    def _focal_intensity(self, coronagraph: bool, lyot: bool, extra_phase=None) -> np.ndarray:
        wf = self.dm.forward(self._wavefront(extra_phase=extra_phase))
        if coronagraph:
            wf = self.coronagraph.forward(wf)
        if lyot:
            wf = self.lyot_stop.forward(wf)
        return self.prop.forward(wf).intensity

    def normalized_intensity(self, coronagraph: bool = True, lyot: bool | None = None,
                             probe_actuators=None, extra_phase=None, flux: float | None = None,
                             rng: np.random.Generator | None = None) -> np.ndarray:
        """Focal-plane intensity (1D, on the focal grid) in raw-contrast units.

        Normalized by the peak of the unaberrated bare-aperture PSF. This is the
        single propagation routine that both the observation and the reward reuse.

        probe_actuators: optional actuator command added to the DM only for this
            exposure (phase diversity); the DM state is restored afterwards.
        extra_phase: optional pupil phase added only for this exposure (e.g. a
            defocus diversity); leaves DM / corrector state untouched.
        flux: if given, total photons used to add shot noise; otherwise noiseless.
        """
        if lyot is None:
            lyot = coronagraph
        if probe_actuators is not None:
            saved = self.get_actuators()
            self.add_actuators(probe_actuators)
        try:
            intensity = self._focal_intensity(coronagraph=coronagraph, lyot=lyot,
                                              extra_phase=extra_phase)
        finally:
            if probe_actuators is not None:
                self.set_actuators(saved)

        image = np.asarray(intensity) / self.reference_peak
        if flux is not None:
            rng = np.random.default_rng() if rng is None else rng
            image = large_poisson(np.clip(image * flux, 0, None), thresh=1e6) / flux
        return image

    def science_image(self, coronagraph: bool = True, probe_actuators=None,
                      flux: float | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        """Normalized-intensity image reshaped to 2D (convenience wrapper)."""
        return self.normalized_intensity(coronagraph=coronagraph, probe_actuators=probe_actuators,
                                         flux=flux, rng=rng).reshape(self.image_shape)

    def dark_hole_contrast(self, intensity: np.ndarray | None = None) -> float:
        """Mean normalized intensity inside the dark-hole annulus (lower is better).

        Pass a precomputed 1D normalized intensity to avoid an extra propagation.
        """
        if intensity is None:
            intensity = self.normalized_intensity(coronagraph=True)
        return float(np.mean(np.asarray(intensity).ravel()[self.dark_hole_mask]))

    def strehl(self) -> float:
        """Strehl ratio (bare-aperture PSF peak relative to the unaberrated peak)."""
        return float(self.normalized_intensity(coronagraph=False, lyot=False).max())


if __name__ == "__main__":
    opt = CoronagraphOptics()
    print(f"actuators: {opt.num_actuators}, image: {opt.image_shape}, "
          f"dark-hole pixels: {int(opt.dark_hole_mask.sum())}")
    opt.flatten_dm()
    opt.clear_aberration()
    print(f"flat/unaberrated dark-hole contrast: {opt.dark_hole_contrast():.3e}")
    print(f"flat/unaberrated Strehl:              {opt.strehl():.5f}")
    opt.set_random_aberration(0.05, rng=np.random.default_rng(0))
    print(f"0.05-wave aberration contrast:        {opt.dark_hole_contrast():.3e}")
    print(f"0.05-wave aberration Strehl:          {opt.strehl():.5f}")
