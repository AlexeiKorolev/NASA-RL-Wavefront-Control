import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hcipy import *
import hcipy
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

"""
- Compare Zernike performance to Neumann performance
- Evaluate out-of-distribution generalization by training on 5 modes and testing on 25 modes
- Make nudges a hyperparameter
- Compare strehl with contrast optimization
- Somehow involve mech interp.
"""


class CoronagraphEnvironment(gym.Env):
    def __init__(self, telescope_diameter = 8., oversizing_factor = 16 / 15, 
                 wavelength_sci = 2.2e-6, num_modes = 500, zero_magnitude_flux = 3.9e10, #3.9e10 photon/s for a mag 0 star
                stellar_magnitude = 5, delta_t = 1e-3, pixels = 240, # sec, so a loop speed of 1kHz.
                num_iterations = 10, coronagraph_charge=4, num_airy=7, pixels_per_spacial_res=4,
                num_noise_modes=500,
                # Diversity / observation configuration
                diversity_enabled: bool = True,
                nudge_magnitude: float = 3e-7,
                nudge_mode_indices: list | None = None,
                num_diversity_pairs: int = 1,
                obs_noise_enabled: bool = False,
                obs_delta_t: float | None = None,
                include_slopes: bool = True,
                # RL stability knobs
                action_scale: float = 1e-8,
                dm_clip: float | None = None,
                lyot_fraction: float = 0.8,
                # Basis type
                basis: str = 'harmonic'
                ):
        super().__init__()

        print(f"initializing coronagraph env. might take a minute.")

        self.telescope_diameter = telescope_diameter
        self.oversizing_factor = oversizing_factor
        self.pixels = pixels
        self.num_airy = num_airy
        self.pixels_per_spacial_res = pixels_per_spacial_res
        self.coronagraph_charge = coronagraph_charge
        self.num_iterations = num_iterations
        self.delta_t = delta_t 
        self.stellar_magnitude = stellar_magnitude
        self.num_modes = num_modes
        self.num_noise_modes = num_noise_modes
        self.wavelength_sci = wavelength_sci
        # Observation config
        self.diversity_enabled = diversity_enabled
        self.nudge_magnitude = float(nudge_magnitude)
        self.nudge_mode_indices = list(nudge_mode_indices) if nudge_mode_indices is not None else ([0] if basis.lower() == 'harmonic' else [3])
        self.num_diversity_pairs = int(num_diversity_pairs)
        self.obs_noise_enabled = bool(obs_noise_enabled)
        self.include_slopes = bool(include_slopes)
        self.obs_delta_t = delta_t if obs_delta_t is None else obs_delta_t
        # RL stability knobs
        self.action_scale = float(action_scale)
        self.dm_clip = float(dm_clip) if dm_clip is not None else None

        self.num_pupil_pixels = pixels * oversizing_factor
        self.pupil_grid_diameter = telescope_diameter * oversizing_factor
        self.pupil_grid = make_pupil_grid(self.num_pupil_pixels, self.pupil_grid_diameter)

        spatial_resolution = wavelength_sci / telescope_diameter
        self.focal_grid = make_focal_grid(q=pixels_per_spacial_res, num_airy=num_airy, spatial_resolution=spatial_resolution)

        VLT_aperture_generator = hcipy.aperture.make_circular_aperture(telescope_diameter)
        self.VLT_aperture = evaluate_supersampled(VLT_aperture_generator, self.pupil_grid, 4)

        self.wf = Wavefront(self.VLT_aperture, wavelength_sci)
        self.wf.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5)

        self.prop = FraunhoferPropagator(self.pupil_grid, self.focal_grid)

        self.unaberrated_PSF = self.prop.forward(self.wf)

        self.camera = NoiselessDetector(self.focal_grid)

        self.basis = basis.lower()
        if self.basis == 'zernike':
            # Number of Zernike modes
            self.dm_modes = make_zernike_basis(num_modes=self.num_modes, D=self.telescope_diameter, grid=self.pupil_grid, starting_mode=1, ansi=False, radial_cutoff=True, use_cache=True)
            self.dm_noise_modes = make_zernike_basis(num_modes=num_noise_modes, D=self.telescope_diameter, grid=self.pupil_grid, starting_mode=1, ansi=False, radial_cutoff=True, use_cache=True)
            # Normalizing each mode with the peak-to-peak value (max - min)
            self.dm_modes = ModeBasis([mode / np.ptp(mode) for mode in self.dm_modes], self.pupil_grid)
            self.dm_noise_modes = ModeBasis([mode / np.ptp(mode) for mode in self.dm_noise_modes], self.pupil_grid)
        elif self.basis == 'orthogonal':
            # Number of orthogonal modes
            zernike_basis = make_zernike_basis(num_modes=self.num_modes, D=self.telescope_diameter, grid=self.pupil_grid, starting_mode=1, ansi=False, radial_cutoff=True, use_cache=True)
            zernike_basis_noise = make_zernike_basis(num_modes=self.num_noise_modes, D=self.telescope_diameter, grid=self.pupil_grid, starting_mode=1, ansi=False, radial_cutoff=True, use_cache=True)

            def orthagonalize(surfaces):
                orthogonal_surfaces = []
                for i in range(surfaces.shape[0]):
                    v = surfaces[i].flatten()
                    for j in range(i):
                        u = orthogonal_surfaces[j]
                        coef = np.dot(v, u) / np.dot(u, u)
                        proj = coef * u
                        v = v - proj
                    orthogonal_surfaces.append(v)

                orthogonal_surfaces = np.array(orthogonal_surfaces)
                return orthogonal_surfaces 
            
            orthogonal_surfaces = orthagonalize(np.array(zernike_basis))
            orthogonal_surfaces_noise = orthagonalize(np.array(zernike_basis_noise))
            self.dm_modes = ModeBasis(Field(orthogonal_surfaces.T, grid=self.pupil_grid))
            self.dm_noise_modes = ModeBasis(Field(orthogonal_surfaces_noise.T, grid=self.pupil_grid))
        else:
            # Number of harmonic modes
            self.dm_modes = make_disk_harmonic_basis(self.pupil_grid, num_modes, telescope_diameter, 'neumann')
            self.dm_noise_modes = make_disk_harmonic_basis(self.pupil_grid, num_noise_modes, telescope_diameter, 'neumann')
            # Normalizing each mode with the peak-to-peak value (max - min)
            self.dm_modes = ModeBasis([mode / np.ptp(mode) for mode in self.dm_modes], self.pupil_grid)
            self.dm_noise_modes = ModeBasis([mode / np.ptp(mode) for mode in self.dm_noise_modes], self.pupil_grid)

        self.deformable_mirror = DeformableMirror(self.dm_modes)
        self.noise_gen_mirror = DeformableMirror(self.dm_noise_modes)

        self.lyot_mask = evaluate_supersampled(circular_aperture(telescope_diameter * lyot_fraction), self.pupil_grid, 4) # keep at point 8 for now, removes noise, test .7
        self.coro = VortexCoronagraph(self.pupil_grid, coronagraph_charge)
        self.lyot_stop = Apodizer(self.lyot_mask)

        self.f_number = 50
        self.num_lenslets = 40 # 40 lenslets along one diameter
        self.sh_diameter = 5e-3 # m

        # Zooms in on the microlens array
        self.magnification = self.sh_diameter / self.telescope_diameter
        self.magnifier = Magnifier(self.magnification)

        self.shwfs = SquareShackHartmannWavefrontSensorOptics(self.pupil_grid.scaled(self.magnification), self.f_number, \
                                                 self.num_lenslets, self.sh_diameter)
        self.shwfse = ShackHartmannWavefrontSensorEstimator(self.shwfs.mla_grid, self.shwfs.micro_lens_array.mla_index)
        self.max_value = 1

        self.slopes_shape = self.get_slopes().shape
        base_h, base_w = self.get_camera_image().shape
        # Determine number of channels for observation image stack
        self.channels = 1
        if self.diversity_enabled:
            # Baseline + (±nudge) per diversity pair
            self.channels = 1 + 2 * max(1, self.num_diversity_pairs)
        self.camera_shape = [self.channels, base_h, base_w]
        self.iteration_counter = num_iterations

        self.max_value = np.max(self.prop(self.wf).intensity)

        # print(self.max_value)

        # Use array low/high to match shapes explicitly. Allow wide ranges for robustness.
        image_low = np.zeros(self.camera_shape, dtype=np.float32)
        image_high = np.full(self.camera_shape, np.inf, dtype=np.float32)  # bright spikes allowed
        if self.include_slopes:
            slopes_low = np.full(self.slopes_shape, -np.inf, dtype=np.float32)
            slopes_high = np.full(self.slopes_shape, np.inf, dtype=np.float32)
        strehl_low = np.array([0.0], dtype=np.float32)
        strehl_high = np.array([np.inf], dtype=np.float32)  # contrast proxy may exceed 1

        # Build default nudge vectors (one per requested mode index), repeated per diversity pair if needed
        self.nudge_vectors = []
        for _ in range(max(1, self.num_diversity_pairs)):
            for idx in self.nudge_mode_indices:
                v = np.zeros(self.num_modes, dtype=np.float64)
                if 0 <= idx < self.num_modes:
                    v[idx] = self.nudge_magnitude
                self.nudge_vectors.append(v)

        obs_spaces = {
            "image": spaces.Box(low=image_low, high=image_high, shape=self.camera_shape, dtype=np.float32),
            "contrast": spaces.Box(low=strehl_low, high=strehl_high, shape=(1,), dtype=np.float32)
        }
        if self.include_slopes:
            obs_spaces["slopes"] = spaces.Box(low=slopes_low, high=slopes_high, shape=self.slopes_shape, dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

        self.action_space = spaces.Box(low=-1e1, high=1e1, shape=(num_modes,), dtype=np.float32)

        # self.reset()

    # -------------------------
    # Modularity helpers
    # -------------------------
    def set_nudge_vectors(self, vectors: list[np.ndarray] | np.ndarray):
        """Set one or multiple nudge vectors used to generate diversity images.

        vectors can be a list of 1D arrays (num_modes,) or a 2D array (k, num_modes).
        """
        if isinstance(vectors, np.ndarray):
            if vectors.ndim == 1:
                vectors = [vectors]
            elif vectors.ndim == 2:
                vectors = [vectors[i] for i in range(vectors.shape[0])]
            else:
                raise ValueError("nudge vectors must be 1D or 2D numpy arrays")
        out = []
        for v in vectors:
            v = np.asarray(v, dtype=np.float64).reshape(-1)
            if v.shape[0] != self.num_modes:
                raise ValueError(f"nudge vector length {v.shape[0]} != num_modes {self.num_modes}")
            out.append(v)
        self.nudge_vectors = out

    def _sanitize_image(self, img: np.ndarray) -> np.ndarray:
        img = np.asarray(img, dtype=np.float32)
        return np.nan_to_num(img, posinf=np.finfo(np.float32).max, neginf=0.0)

    def _capture_image_with_dm(self, dm_act: np.ndarray, delta_t: float, coronagraph_enabled: bool, noise_enabled: bool) -> np.ndarray:
        original = self.deformable_mirror.actuators.copy()
        try:
            self.deformable_mirror.flatten()
            self.deformable_mirror.actuators = np.asarray(dm_act, dtype=np.float64)
            return self._sanitize_image(self.get_camera_image(delta_t=delta_t, coronagraph_enabled=coronagraph_enabled, crop=False, noise_enabled=noise_enabled))
        finally:
            self.deformable_mirror.flatten()
            self.deformable_mirror.actuators = original

    def generate_diversity_images(self, baseline_dm: np.ndarray | None = None, delta_t: float | None = None, noise_enabled: bool | None = None) -> np.ndarray:
        """Return stacked images (C, H, W): baseline plus ± nudges if enabled.

        - If diversity is disabled, returns a single-channel stack [baseline].
        - If enabled, returns [baseline, +v1, -v1, +v2, -v2, ...].
        """
        if baseline_dm is None:
            baseline_dm = self.deformable_mirror.actuators.copy()
        dt = self.obs_delta_t if delta_t is None else delta_t
        nz = self.obs_noise_enabled if noise_enabled is None else noise_enabled

        # Baseline
        baseline_img = self._capture_image_with_dm(baseline_dm, dt, coronagraph_enabled=True, noise_enabled=nz)
        stack = [baseline_img]

        if self.diversity_enabled and self.nudge_vectors:
            for v in self.nudge_vectors:
                plus = self._capture_image_with_dm(baseline_dm + v, dt, True, nz)
                minus = self._capture_image_with_dm(baseline_dm - v, dt, True, nz)
                stack.extend([plus, minus])

        return np.stack(stack, axis=0).astype(np.float32)

    def set_random_noise_dm(self, noise=1e-7):
        # Put actuators at random values, putting a little more power in low-order modes
        noise = np.abs(noise)
        
        self.noise_gen_mirror.flatten()

        action = np.random.randn(self.num_noise_modes)

        self.noise_gen_mirror.actuators = action

        action *= noise * self.wavelength_sci / np.std(self.noise_gen_mirror.surface)

        magnitude = np.linalg.norm(action)

        action /= magnitude
        action *= noise

        if self.basis == 'zernike':
            action *= np.sqrt(5.0, dtype=np.float16) # scaling zernike so that it is similar to harmonic noise levels.
        
        elif self.basis == 'orthogonal':
            action /= np.sqrt(5.0, dtype=np.float16) # scaling orthogonal so that it is similar to harmonic noise levels.


        self.noise_gen_mirror.actuators = action


    def set_random_dm(self, noise=1e-7):
        # Put actuators at random values, putting a little more power in low-order modes
        noise = np.abs(noise)

        self.deformable_mirror.flatten()

        action = np.random.randn(self.num_modes)  / (np.arange(self.num_modes) + 10)

        self.deformable_mirror.actuators = action

        action *= noise * self.wavelength_sci / np.std(self.deformable_mirror.surface)

        magnitude = np.linalg.norm(action)

        action /= magnitude
        action *= noise

        if self.basis == 'zernike':
            action *= np.sqrt(5.0, dtype=np.float16) # scaling zernike so that it is similar to harmonic noise levels.

        elif self.basis == 'orthogonal':
            action /= np.sqrt(5.0, dtype=np.float16) # scaling orthogonal so that it is similar to harmonic noise levels.


        self.deformable_mirror.actuators = action

    def set_dm(self, action):
        # Additive update relative to current actuators (preserve previous state)
        self.deformable_mirror.actuators = np.asarray(self.deformable_mirror.actuators, dtype=np.float64) + np.asarray(action, dtype=np.float64)
        if self.dm_clip is not None:
            self.deformable_mirror.actuators = np.clip(self.deformable_mirror.actuators, -self.dm_clip, self.dm_clip)


    def get_slopes(self):
        wfs_wf = self.shwfs(self.magnifier(self.deformable_mirror(self.wf)))
        # Produces an image as if the camera was exposed to the light for this amount of time.
        self.camera.integrate(wfs_wf, 1)
        image = self.camera.read_out()

        slopes = self.shwfse.estimate([image])
        return slopes


    def get_perfect_adjustment(self):
        return self.deformable_mirror.actuators * -1
    
    def get_camera_image(self, delta_t=1e3, crop=False, crop_width=40, coronagraph_enabled=True, noise_enabled=False):
        def crop_image(img, width=40):
            if len(img.shape) == 1:
                img = img.reshape(int(np.sqrt(img.shape[0])), int(np.sqrt(img.shape[0])))

            center = (img.shape[0] // 2, img.shape[1] // 2)
            half_width = width // 2
            return img[center[0] - half_width: center[0] + half_width, center[1] - half_width: center[1] + half_width]

        # Read out WFS camera

        if coronagraph_enabled:
            propagrated_wf = self.prop(self.lyot_stop(self.coro(self.deformable_mirror(self.noise_gen_mirror(self.wf)))))
        else: 
            propagrated_wf = self.prop(self.lyot_stop(self.deformable_mirror(self.noise_gen_mirror(self.wf))))

        self.camera.integrate(propagrated_wf, delta_t)
        wfs_image = self.camera.read_out()
        if noise_enabled: wfs_image = large_poisson(wfs_image).astype('float')
        wfs_image = wfs_image.reshape(int(np.sqrt(wfs_image.size)), int(np.sqrt(wfs_image.size)))

        return crop_image(wfs_image, width=crop_width) if crop else wfs_image


    def lod_to_pixels(self, lod: float) -> float:
        """Convert a separation in units of λ/D to pixels on the detector image.

        With the focal grid created via make_focal_grid(q=pixels_per_spacial_res, ...),
        the pixel scale is q pixels per λ/D. Hence: pixels = lod * q.
        """
        return float(lod) * float(self.pixels_per_spacial_res)

    def pixels_to_lod(self, pixels: float) -> float:
        """Convert pixels on the detector image to units of λ/D."""
        return float(pixels) / float(self.pixels_per_spacial_res)

    def get_contrast(self, corona_image=None, clear_image=None, delta_t=None, *, inner_lod: float | None = None, outer_lod: float | None = None, half: str = "right"):
        # Generate images if not provided; use noiseless frames for a stable metric
        if corona_image is None:
            if delta_t is not None:
                corona_image = self.get_camera_image(delta_t, coronagraph_enabled=True, crop=False, noise_enabled=False)
            else:
                corona_image = self.get_camera_image(coronagraph_enabled=True, crop=False, noise_enabled=False)

        if clear_image is None:
            if delta_t is not None:
                clear_image = self.get_camera_image(delta_t, coronagraph_enabled=False, crop=False, noise_enabled=False)
            else:
                clear_image = self.get_camera_image(coronagraph_enabled=False, crop=False, noise_enabled=False)

        # Ensure numpy arrays and same shape
        corona_image = np.asarray(corona_image)
        clear_image = np.asarray(clear_image)
        assert corona_image.shape == clear_image.shape, "get_contrast images different shapes."

        img_height, img_width = corona_image.shape

        def create_circular_mask(h, w, center=None, radius=None):
            if center is None:
                center = (int(w / 2), int(h / 2))
            if radius is None:
                radius = min(center[0], center[1], w - center[0], h - center[1])
            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
            return dist_from_center <= radius  # boolean mask

        # D-shaped region (half-annulus): outer minus inner, restricted to chosen half-plane
        # Radii can be provided in λ/D; otherwise fallback to pixel radii 18 and 35
        if inner_lod is not None:
            r_in = int(round(self.lod_to_pixels(inner_lod)))
        else:
            r_in = 3
        if outer_lod is not None:
            r_out = int(round(self.lod_to_pixels(outer_lod)))
        else:
            r_out = 10
        inner_circle = create_circular_mask(img_height, img_width, radius=r_in)
        outer_circle = create_circular_mask(img_height, img_width, radius=r_out)
        half_plane = np.zeros((img_height, img_width), dtype=bool)
        if half.lower() == "left":
            half_plane[:, : img_width // 2] = True
        else:
            half_plane[:, img_width // 2 :] = True  # right half (default)
        annulus = np.logical_and(outer_circle, np.logical_not(inner_circle))
        mask = annulus # np.logical_and(half_plane, annulus)  # boolean mask

        # Contrast definition: mean coronagraph intensity in D-shaped mask over PEAK of aberrated non-coronagraph image
        num = float(np.mean(corona_image[mask]))
        denom = float(np.max(clear_image))  # use global peak, not masked
        denom = max(denom, 1e-20)

        # fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
        # im0 = axs[0].imshow(inner_circle.astype(float), cmap="gray")

        # axs[0].set_title("Mask")
        # im1 = axs[1].imshow(corona_image, cmap="inferno")
        # axs[1].set_title("Coronagraph")
        # im2 = axs[2].imshow(clear_image, cmap="inferno")
        # axs[2].set_title("Clear")
        # for ax in axs:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        # plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        # plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        # plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
        # plt.show()

        # print(f"corona masked: {corona_image[mask]}")

        if (num > denom and False):
            print(f"ERROR IMPOSSIBLE VALUE ENCOUNTERED! Dumping info!")
            # Dump relevant variables and visualize masks/images for debugging
            try:
                def arr_stats(a):
                    a = np.asarray(a)
                    return {
                        "shape": tuple(a.shape),
                        "dtype": str(a.dtype),
                        "min": float(np.nanmin(a)),
                        "max": float(np.nanmax(a)),
                        "mean": float(np.nanmean(a)),
                        "sum": float(np.nansum(a)),
                        "nan_count": int(np.isnan(a).sum()),
                        "inf_count": int(np.isinf(a).sum()),
                    }

                masked_mean = float(np.mean(corona_image[mask]))
                masked_max = float(np.max(corona_image[mask]))
                print("=== Contrast Debug Dump ===")
                print(f"time: {datetime.now().isoformat()}")
                print(f"delta_t={delta_t}, inner_lod={inner_lod}, outer_lod={outer_lod}, half='{half}'")
                print(f"r_in={r_in}, r_out={r_out}, pixels_per_spacial_res={self.pixels_per_spacial_res}")
                print(f"num(mean masked corona)={num}, denom(peak clear)={denom}")
                print(f"masked_mean={masked_mean}, masked_max={masked_max}")
                print(f"mask_coverage={float(mask.mean())}, annulus_coverage={float(annulus.mean())}, half_plane_coverage={float(half_plane.mean())}")
                print(f"telescope_diameter={self.telescope_diameter}, wavelength_sci={self.wavelength_sci}, coronagraph_charge={self.coronagraph_charge}")
                print(f"num_modes={self.num_modes}, dm_norm={float(np.linalg.norm(self.deformable_mirror.actuators))}")
                print("corona_image:", arr_stats(corona_image))
                print("clear_image:", arr_stats(clear_image))

                # Visualize images and masks
                fig, axs = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)

                im0 = axs[0, 0].imshow(corona_image, cmap="inferno")
                axs[0, 0].imshow(mask, cmap="cool", alpha=0.3)
                axs[0, 0].set_title("Coronagraph Image + Mask")
                plt.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

                im1 = axs[0, 1].imshow(clear_image, cmap="inferno")
                axs[0, 1].imshow(mask, cmap="cool", alpha=0.3)
                axs[0, 1].set_title("Clear (No Coro) + Mask")
                plt.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

                im2 = axs[0, 2].imshow(corona_image * mask, cmap="inferno")
                axs[0, 2].set_title("Masked Coronagraph Image")
                plt.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

                axs[1, 0].imshow(inner_circle, cmap="gray")
                axs[1, 0].set_title("Inner Circle")

                axs[1, 1].imshow(outer_circle, cmap="gray")
                axs[1, 1].set_title("Outer Circle")

                axs[1, 2].imshow(annulus, cmap="gray")
                axs[1, 2].imshow(half_plane, cmap="cool", alpha=0.3)
                axs[1, 2].set_title("Annulus + Half-Plane")

                for ax in axs.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])

                plt.show()
            except Exception as ex:
                print(f"[contrast debug] visualization failed: {ex}")
        return num / denom

    def get_strehl_ratio(self):
        # max_aberrated_value = get_camera_image(self, delta_t=1e3, coronagraph_enabled=False, crop=False, noise_enabled=False).max()
        wf_aberrated = self.deformable_mirror(self.wf)
        psf_aberrated = self.prop(wf_aberrated).intensity
        peak_aberrated = np.max(psf_aberrated)

        # print(f"psf_aberrated.shape: {psf_aberrated.shape}")
        # print(f"pupil grid shape: {self.pupil_grid.shape}")
        # print(f"VLT_aperture shape: {self.VLT_aperture.shape}")
        # print(f"wavefront shape: {self.wf.grid.shape}")
        # print(f"aberrated wavefront shape: {wf_aberrated.grid.shape}")
        # print(f"propagated wavefront shape: {self.prop(wf_aberrated).grid.shape}")
        # print(f"focal grid shape: {self.focal_grid.shape}")
        # psf_ideal = self.prop(self.wf).intensity
        # peak_ideal = np.max(psf_ideal)

        strehl = peak_aberrated / self.max_value

        return strehl

    def _get_obs(self):
        # Images stack (C,H,W)
        images = self.generate_diversity_images()
        # Slopes and strehl
        if self.include_slopes:
            slopes = np.asarray(self.get_slopes(), dtype=np.float32)
            slopes = np.nan_to_num(slopes, posinf=0.0, neginf=0.0)
        contrast = np.array([self.get_contrast(delta_t=1e15)], dtype=np.float32)
        contrast = np.nan_to_num(contrast, posinf=np.finfo(np.float32).max, neginf=0.0)

        observation = {
            "image": images,
            "contrast": contrast
        }
        if self.include_slopes:
            observation["slopes"] = slopes
    
        # Robustness: avoid hard crash; warn if out of bounds
        if not self.observation_space.contains(observation):
            # Optionally clip images to observation space high if finite
            img_space = self.observation_space.spaces["image"]
            finite_high = np.isfinite(img_space.high)
            if np.any(finite_high):
                observation["image"] = np.minimum(observation["image"], img_space.high)
        return observation


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the environment to a starting state
        self.deformable_mirror.flatten()
        if options is not None and "noise" in options:
            self.set_random_dm(noise=options["noise"])
        else:
            self.set_random_dm()

        observation = self._get_obs()
        # Reset iteration counter.

        """self.current_state = ... # Define your initial state
        observation = self.current_state # or transform the state into an observation"""

        info = {}
        return observation, info

    def _compute_reward(self):
        # Use high-exposure contrast as a proxy; ensure numerical stability
        contrast = self.get_contrast(delta_t=1e15)        
        # Guard against NaN/Inf and non-positive values
        c = float(np.nan_to_num(contrast, nan=1e-20, posinf=1e-20, neginf=1e-20))
        c = max(c, 1e-20)
        return -np.log10(c)  # higher is better

    def step(self, action):
        # Update the environment state based on the action
        assert action.shape == self.deformable_mirror.actuators.shape

        # Scale down the action to keep DM updates numerically safe
        self.set_dm(action=action * self.action_scale)
        self.iteration_counter -= 1

        reward = self._compute_reward()

        terminated = self.iteration_counter <= 0
        truncated = False

        info = {}

        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Implement visualization if needed
        ...

    def close(self):
        # Implement cleanup if needed
        ...


if __name__ == "__main__":
    e = CoronagraphEnvironment(num_modes=40, num_airy=5, pixels_per_spacial_res=2)
    e.set_random_dm(noise=1e-7)
    diversity_imgs = e.generate_diversity_images()

    # Visualize stacked diversity images (C, H, W) using inferno colormap
    imgs = diversity_imgs
    if imgs.ndim == 2:
        imgs = imgs[None, ...]
    C, H, W = imgs.shape

    ncols = int(np.ceil(np.sqrt(C)))
    nrows = int(np.ceil(C / ncols))

    vmin = float(np.nanmin(imgs))
    vmax = float(np.nanmax(imgs))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    im = None
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i < C:
            im = ax.imshow(imgs[i], cmap='inferno', vmin=vmin, vmax=vmax)
            ax.set_title(f'Channel {i}')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Intensity')

    plt.tight_layout()
    plt.show()

    print(diversity_imgs.shape)

    obs = e._get_obs()

    print(obs)

    exit()


    N = 400
    avgs = np.zeros(shape=(N,))

    for i in range(200, N):
        for _ in range(50):
            e.set_random_dm(noise=i * 1e-7)
            avgs[i] = max(avgs[i], e.get_contrast(delta_t=1e15))

    print(avgs)

    plt.plot(np.arange(N), avgs)
    plt.show()



    exit()

    e.get_contrast(delta_t=1000)

    e.set_random_dm(noise=0.01)

    e.get_contrast()

    e.set_random_dm(noise=0.01)

    e.get_contrast()

    e.set_random_dm(noise=0.1)

    e.get_contrast()

    exit()
   
    values_of_delta_t = [10**(i/2) for i in range(-10, 20, 1)]
    repetitions = 10
    X = []
    y = []
    errors = []


    for delta_t in values_of_delta_t:
        print(f"testing delta_t = {delta_t}")
        entries = []

        for _ in range(repetitions):
            entries.append(e.get_contrast(delta_t=delta_t))
        
        X.append(delta_t)
        y.append(np.mean(entries))
        errors.append(np.std(entries))
    
    plt.plot(X, y)
    plt.fill_between(X, np.array(y) - np.array(errors), np.array(y) + np.array(errors), alpha=0.3)
    plt.xlabel("Noise")
    plt.ylabel("Contrast")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Contrast vs Noise with Error Range")
    plt.savefig("contrast_vs_noise.png", dpi=300)
    plt.show()
    exit()


    values_of_noise = [np.pow(10,i/10) for i in range(-60, 10, 2)]

    repetitions = 10

    X = []
    y = []
    errors = []

    for noise in values_of_noise:
        entries = []
        print(f"simulating {noise} noise")

        for _ in range(repetitions):
            e.set_random_dm(noise=noise)
            entries.append(e.get_contrast(delta_t=1000))
        
        X.append(noise)
        y.append(np.mean(entries))
        errors.append(np.std(entries))
    
    plt.plot(X, y)
    plt.fill_between(X, np.array(y) - np.array(errors), np.array(y) + np.array(errors), alpha=0.3)
    plt.xlabel("Noise")
    plt.ylabel("Contrast")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Contrast vs Noise with Error Range")
    plt.savefig("contrast_vs_noise.png", dpi=300)
    plt.show()

    exit() 



    
    e.set_random_dm(noise=1e-4)

    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(np.zeros((240, 240)), cmap='inferno', animated=True)
    ax.axis('off')
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'Constant noise image sim {current_time}.mp4'
    anim = FFMpegWriter(filename, framerate=20)

    for _ in range(100):
        plt.clf()
        crop_size = 40
        image = (np.clip(e.get_camera_image(delta_t=1), a_min = 10 ** -20, a_max = None))
        # print(f"IMAGE SHAPE: {image.shape}")
        image = image.reshape(240, 240)


        center = image.shape[0] // 2
        half_crop = crop_size // 2
        cropped_image = image[center - half_crop:center + half_crop, center - half_crop:center + half_crop]

        # print(cropped_image.shape)

        plt.imshow(cropped_image, cmap='inferno')
        plt.colorbar()
        anim.add_frame()
    

    anim.close()