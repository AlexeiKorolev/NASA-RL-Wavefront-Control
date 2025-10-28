import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hcipy import *
import hcipy
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


class CoronagraphEnvironment(gym.Env):
    def __init__(self, telescope_diameter = 8., oversizing_factor = 16 / 15, 
                 wavelength_sci = 2.2e-6, num_modes = 500, zero_magnitude_flux = 3.9e10, #3.9e10 photon/s for a mag 0 star
                stellar_magnitude = 5, delta_t = 1e-3, pixels = 240, # sec, so a loop speed of 1kHz.
                num_iterations = 10, coronagraph_charge=4, num_airy=7, pixels_per_spacial_res=4,
                # Diversity / observation configuration
                diversity_enabled: bool = True,
                nudge_magnitude: float = 3e-7,
                nudge_mode_indices: list | None = None,
                num_diversity_pairs: int = 1,
                obs_noise_enabled: bool = False,
                obs_delta_t: float | None = None):
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
        self.wavelength_sci = wavelength_sci
        # Observation config
        self.diversity_enabled = diversity_enabled
        self.nudge_magnitude = float(nudge_magnitude)
        self.nudge_mode_indices = list(nudge_mode_indices) if nudge_mode_indices is not None else [0]
        self.num_diversity_pairs = int(num_diversity_pairs)
        self.obs_noise_enabled = bool(obs_noise_enabled)
        self.obs_delta_t = delta_t if obs_delta_t is None else obs_delta_t

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

        # Number of harmonic modes
        self.dm_modes = make_disk_harmonic_basis(self.pupil_grid, num_modes, telescope_diameter, 'neumann')
        # Normalizing each mode with the peak-to-peak value (max - min)
        self.dm_modes = ModeBasis([mode / np.ptp(mode) for mode in self.dm_modes], self.pupil_grid)

        self.deformable_mirror = DeformableMirror(self.dm_modes)

        self.lyot_mask = evaluate_supersampled(circular_aperture(telescope_diameter * 0.8), self.pupil_grid, 4) # keep at point 8 for now, removes noise, test .7
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

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=image_low, high=image_high, shape=self.camera_shape, dtype=np.float32),
            "slopes": spaces.Box(low=slopes_low, high=slopes_high, shape=self.slopes_shape, dtype=np.float32),
            "strehl": spaces.Box(low=strehl_low, high=strehl_high, shape=(1,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=-1e-3, high=1e-3, shape=(num_modes,), dtype=np.float32)

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

    def set_random_dm(self, noise=1e-7):
        # Put actuators at random values, putting a little more power in low-order modes
        self.deformable_mirror.actuators = np.random.randn(self.num_modes)  / (np.arange(self.num_modes) + 10)

        # Normalize the DM surface so that we get a reasonable surface RMS.
        self.deformable_mirror.actuators *= noise * self.wavelength_sci / np.std(self.deformable_mirror.surface)

        magnitude = np.linalg.norm(self.deformable_mirror.actuators)

        self.deformable_mirror.actuators /= magnitude
        self.deformable_mirror.actuators *= noise


    def set_dm(self, action):
        # Additive update relative to current actuators (preserve previous state)
        self.deformable_mirror.actuators = np.asarray(self.deformable_mirror.actuators, dtype=np.float64) + np.asarray(action, dtype=np.float64)


    def get_slopes(self):
        wfs_wf = self.shwfs(self.magnifier(self.deformable_mirror(self.wf)))
        # Produces an image as if the camera was exposed to the light for this amount of time.
        self.camera.integrate(wfs_wf, 1)
        image = self.camera.read_out()

        slopes = self.shwfse.estimate([image])
        return slopes


    def get_perfect_adjustment(self):
        return self.deformable_mirror.actuators * -1
    

    def get_camera_image(self, delta_t=1e3, crop=False, crop_width=40, coronagraph_enabled=True, noise_enabled=True):
        def crop_image(img, width=40):
            if len(img.shape) == 1:
                img = img.reshape(int(np.sqrt(img.shape[0])), int(np.sqrt(img.shape[0])))

            center = (img.shape[0] // 2, img.shape[1] // 2)
            half_width = width // 2
            return img[center[0] - half_width: center[0] + half_width, center[1] - half_width: center[1] + half_width]

        # Read out WFS camera

        if coronagraph_enabled:
            propagrated_wf = self.prop(self.lyot_stop(self.coro(self.deformable_mirror(self.wf))))
        else: 
            propagrated_wf = self.prop(self.lyot_stop(self.deformable_mirror(self.wf)))

        self.camera.integrate(propagrated_wf, delta_t)
        wfs_image = self.camera.read_out()
        if noise_enabled: wfs_image = large_poisson(wfs_image).astype('float')
        wfs_image = wfs_image.reshape(int(np.sqrt(wfs_image.size)), int(np.sqrt(wfs_image.size)))

        return crop_image(wfs_image, width=crop_width) if crop else wfs_image


    def get_contrast(self, corona_image=None, clear_image=None, delta_t=None):
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

        # D-shaped region (half-annulus): outer minus inner, restricted to right half-plane
        inner_circle = create_circular_mask(img_height, img_width, radius=18)
        outer_circle = create_circular_mask(img_height, img_width, radius=35)
        half_plane = np.zeros((img_height, img_width), dtype=bool)
        half_plane[:, img_width // 2 :] = True  # right half
        annulus = np.logical_and(outer_circle, np.logical_not(inner_circle))
        mask = np.logical_and(half_plane, annulus)  # boolean mask

        # Contrast definition: mean coronagraph intensity in D-shaped mask over PEAK of aberrated non-coronagraph image
        num = float(np.mean(corona_image[mask]))
        denom = float(np.max(clear_image))  # use global peak, not masked
        denom = max(denom, 1e-20)
        return num / denom

    def get_strehl_ratio(self):
        wf_aberrated = self.deformable_mirror(self.wf)
        psf_aberrated = self.prop(wf_aberrated).intensity
        peak_aberrated = np.max(psf_aberrated)

        psf_ideal = self.prop(self.wf).intensity
        peak_ideal = np.max(psf_ideal)

        strehl = peak_aberrated / peak_ideal

        return strehl

    def _get_obs(self):
        # Images stack (C,H,W)
        images = self.generate_diversity_images(baseline_dm=self.deformable_mirror.actuators.copy())
        # Slopes and strehl
        slopes = np.asarray(self.get_slopes(), dtype=np.float32)
        slopes = np.nan_to_num(slopes, posinf=0.0, neginf=0.0)
        strehl = np.array([self.get_contrast(delta_t=1e15)], dtype=np.float32)
        strehl = np.nan_to_num(strehl, posinf=np.finfo(np.float32).max, neginf=0.0)

        observation = {
            "image": images,
            "slopes": slopes,
            "strehl": strehl
        }

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
        return -np.log10(contrast + 1e-20) # Tiny positive value to ensure it's positive.

    def step(self, action):
        # Update the environment state based on the action
        assert action.shape == self.deformable_mirror.actuators.shape

        self.set_dm(action=action)
        self.iteration_counter -= 1

        reward = self._compute_reward()

        terminated = self.iteration_counter <= 0
        truncated = False

        info = {}

        observation = self._get_obs()
        
        return observation, reward, terminated, truncated, info

        ...
        # Calculate the reward
        reward = ...
        # Determine if the episode is terminated or truncated
        terminated = False
        truncated = False
        # Provide any extra information
        info = {}
        observation = self.current_state # or transform the state into an observation
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        # Implement visualization if needed
        ...

    def close(self):
        # Implement cleanup if needed
        ...


if __name__ == "__main__":
    print(f"RUNNING!")
    e = CoronagraphEnvironment(num_modes=40)
    e.set_random_dm(noise=1e-8)
    plt.imshow(e.get_camera_image(delta_t=1e-3, noise_enabled=False), cmap='inferno')
    plt.colorbar()
    # plt.show()
    # plt.savefig("coronagraph_image.png", dpi=300)
    plt.savefig("/scratch/network/ak9088/coronagraph_image.png", dpi=300)

    exit()



    avgs = np.zeros_like(e.deformable_mirror.actuators)

    N = 400
    for _ in range(N):
        e.set_random_dm(noise=1e-7)
        avgs += e.deformable_mirror.actuators
    
    avgs /= N

    print(avgs)

    plt.plot(np.arange(len(e.deformable_mirror.actuators)), avgs)
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