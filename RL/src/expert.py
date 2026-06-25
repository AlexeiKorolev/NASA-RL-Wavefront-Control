"""Least-squares (phase-conjugation) expert for the modal CoronagraphEnv.

The deformable mirror adds a phase ``2 * k * surface`` to the wavefront, where the
factor 2 is the reflection optical-path difference and ``k = 2*pi/lambda`` (see
hcipy ``DeformableMirror.opd`` / ``phase_for``). The modal actuator basis ``B``
(``num_actuators x num_modes``) maps mode coefficients -- in *meters of surface
RMS* -- to actuator commands, so the DM phase produced per unit coefficient is

    P[:, i] = 2 * k * surface(B[:, i])              # evaluated over the aperture

The correction that best cancels an aberration phase ``phi`` (also in radians, over
the same pupil support) is the least-squares phase conjugate

    c* = pinv(P_support) @ (-phi_support)

i.e. drive the DM phase toward ``-phi`` across the illuminated pupil. Because the
20 control modes are the same low-order Zernikes used to build the disturbance,
this nulls the aberration almost exactly -- a strong supervisory signal to clone.

Usage:
    env = CoronagraphEnv(num_control_modes=20)
    expert = LeastSquaresExpert(env.optics, env.modal_basis)
    env.reset()                       # sets a random aberration
    c = expert.correction()           # modal coeffs (meters of surface RMS)
    env.optics.set_actuators(env.modal_basis @ c)   # one-shot full correction
"""
from __future__ import annotations

import numpy as np


class LeastSquaresExpert:
    """Phase-conjugation expert that returns the modal correction for the optics'
    *current* aberration. Build once per optics instance (precomputes a pinv)."""

    def __init__(self, optics, modal_basis=None):
        self.optics = optics
        k = 2.0 * np.pi / optics.wavelength
        support = np.asarray(optics._aperture_support)       # bool over pupil pixels
        self._support = support

        if getattr(optics, "ideal_modal_correction", False):
            # Ideal corrector: the per-mode pupil phase is already stored as
            # 2*k*Z_i over all pixels; just restrict it to the aperture support.
            self.B = None
            self.num_modes = int(optics.num_correction_modes)
            P = np.asarray(optics._correction_modes)[:, support].T   # (num_support, num_modes)
        else:
            self.B = np.asarray(modal_basis, dtype=np.float64)   # (num_actuators, num_modes)
            self.num_modes = self.B.shape[1]
            # DM phase produced over the aperture per unit modal coefficient.
            saved = optics.get_actuators()
            try:
                P = np.empty((int(support.sum()), self.num_modes), dtype=np.float64)
                for i in range(self.num_modes):
                    optics.set_actuators(self.B[:, i])
                    surface = np.asarray(optics.dm.surface)
                    P[:, i] = (2.0 * k * surface)[support]
            finally:
                optics.set_actuators(saved)

        # Precompute the least-squares solver: c = A_pinv @ (-phi_support).
        self._A_pinv = np.linalg.pinv(P)                     # (num_modes, num_support)

    def correction(self) -> np.ndarray:
        """Modal coefficients (meters of surface RMS) that cancel the optics'
        current ``aberration_phase``. For the DM modal basis, apply with
        ``optics.set_actuators(B @ c)``; for the ideal corrector, apply with
        ``optics.set_correction_modes(c)``. Either way feed it incrementally to
        the env action as ``c / action_scale``."""
        phi = np.asarray(self.optics.aberration_phase)[self._support]
        return self._A_pinv @ (-phi)
