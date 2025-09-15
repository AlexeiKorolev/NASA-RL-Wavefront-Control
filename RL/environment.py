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
                num_iterations = 10, coronagraph_charge=4, num_airy=7, pixels_per_spacial_res=4):
        super().__init__()

        print(f"initializing coronagraph env. might take a minute.")

        self.telescope_diameter = telescope_diameter
        self.oversizing_factor = oversizing_factor
    
        self.num_pupil_pixels = pixels * oversizing_factor
        self.pupil_grid_diameter = telescope_diameter * oversizing_factor
        self.pupil_grid = make_pupil_grid(self.num_pupil_pixels, self.pupil_grid_diameter)

        spatial_resolution = wavelength_sci / telescope_diameter
        self.focal_grid = make_focal_grid(q=pixels_per_spacial_res, num_airy=num_airy, spatial_resolution=spatial_resolution)

        VLT_aperture_generator = hcipy.aperture.make_circular_aperture(telescope_diameter)
        self.VLT_aperture = evaluate_supersampled(VLT_aperture_generator, self.pupil_grid, 4)

        self.wavelength_sci = wavelength_sci

        self.wf = Wavefront(self.VLT_aperture, wavelength_sci)
        self.wf.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5)

        self.prop = FraunhoferPropagator(self.pupil_grid, self.focal_grid)

        self.unaberrated_PSF = self.prop.forward(self.wf)

        self.camera = NoiselessDetector(self.focal_grid)

        # Number of harmonic modes
        self.num_modes = num_modes
        self.dm_modes = make_disk_harmonic_basis(self.pupil_grid, num_modes, telescope_diameter, 'neumann')
        # Normalizing each mode with the peak-to-peak value (max - min)
        self.dm_modes = ModeBasis([mode / np.ptp(mode) for mode in self.dm_modes], self.pupil_grid)

        self.deformable_mirror = DeformableMirror(self.dm_modes)

        self.lyot_mask = evaluate_supersampled(circular_aperture(telescope_diameter * 0.95), self.pupil_grid, 4)
        self.coro = VortexCoronagraph(self.pupil_grid, coronagraph_charge)
        self.lyot_stop = Apodizer(self.lyot_mask)

        self.delta_t = delta_t

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
        self.camera_shape = self.get_camera_image().shape
        self.iteration_counter = num_iterations

        self.max_value = np.max(self.prop(self.wf).intensity)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=1, shape=self.camera_shape, dtype=np.float32),
            "slopes": spaces.Box(low=-1e-3, high=1e-3, shape=self.slopes_shape, dtype=np.float32),
            "strehl": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        self.action_space = spaces.Box(low=-1e-3, high=1e-3, shape=(num_modes,), dtype=np.float32)

    def set_random_dm(self, noise=1e-2):


        # Put actuators at random values, putting a little more power in low-order modes
        self.deformable_mirror.actuators = np.random.randn(self.num_modes)  / (np.arange(self.num_modes) + 10)

        # Normalize the DM surface so that we get a reasonable surface RMS.
        self.deformable_mirror.actuators *= noise * self.wavelength_sci / np.std(self.deformable_mirror.surface)

        magnitude = np.linalg.norm(self.deformable_mirror.actuators)

        self.deformable_mirror.actuators /= magnitude
        self.deformable_mirror.actuators *= noise


    def set_dm(self, action):
        self.deformable_mirror.actuators += action


    def get_slopes(self):
        wfs_wf = self.shwfs(self.magnifier(self.deformable_mirror(self.wf)))
        # Produces an image as if the camera was exposed to the light for this amount of time.
        self.camera.integrate(wfs_wf, 1)
        image = self.camera.read_out()

        slopes = self.shwfse.estimate([image])
        return slopes


    def get_perfect_adjustment(self):
        return self.deformable_mirror.actuators * -1
    

    def get_camera_image(self, delta_t=1e3, crop=False, crop_width=40, coronagraph_enabled=True):
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
        wfs_image = large_poisson(wfs_image).astype('float')
        wfs_image = wfs_image.reshape(int(np.sqrt(wfs_image.size)), int(np.sqrt(wfs_image.size)))

        return crop_image(wfs_image, width=crop_width) if crop else wfs_image


    def get_contrast(self, corona_image=None, clear_image=None, delta_t=None):
        if corona_image == None:
            corona_image = self.get_camera_image(delta_t, coronagraph_enabled=True, crop=False) if delta_t != None else self.get_camera_image(coronagraph_enabled=True)

        if clear_image == None:
            clear_image = self.get_camera_image(delta_t, coronagraph_enabled=False, crop=False) if delta_t != None else self.get_camera_image(coronagraph_enabled=False)
        
        # Area of interest definition.

        corona_image = np.array(corona_image)
        clear_image = np.array(clear_image)

        assert corona_image.shape == clear_image.shape, "get_contrast images different shapes."
        
        img_height, img_width = corona_image.shape

        def create_circular_mask(h, w, center=None, radius=None):
            if center is None:
                center = (int(w/2), int(h/2))
            if radius is None:
                radius = min(center[0], center[1], w-center[0], h-center[1])

            Y, X = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

            mask = dist_from_center <= radius
            return mask

        inner_circle = create_circular_mask(img_height, img_width, radius=18)
        outer_circle = create_circular_mask(img_height, img_width, radius=35)
        right_side = np.zeros((img_height, img_width), dtype=int)
        right_side[:, :img_width // 2] = 1

        mask = np.where(np.logical_and(right_side, np.logical_and(outer_circle, np.logical_not(inner_circle))), 1, 0)
        # plt.imshow(mask)
        # plt.colorbar()
        # plt.show()
        
        # plt.imshow(np.where(mask, corona_image, np.zeros_like(corona_image)))
        # plt.colorbar()
        # plt.show()
        # plt.imshow(corona_image)
        # plt.colorbar()
        # plt.show()

        return np.mean(corona_image[mask]) / np.max(clear_image[mask])

    def get_strehl_ratio(self):
        wf_aberrated = self.deformable_mirror(self.wf)
        psf_aberrated = self.prop(wf_aberrated).intensity
        peak_aberrated = np.max(psf_aberrated)

        psf_ideal = self.prop(self.wf).intensity
        peak_ideal = np.max(psf_ideal)

        strehl = peak_aberrated / peak_ideal

        return strehl

    def _get_obs(self):
        image = self.get_camera_image().astype(np.float32)
        slopes = self.get_slopes().astype(np.float32)
        strehl = np.array([self.get_strehl_ratio()], dtype=np.float32)

        observation = {
            "image": image,
            "slopes": slopes,
            "strehl": strehl
        }

        # print(f"observation: {observation}")
        # print(f"max_value: {self.max_value}")
        # print(f"image min: {np.min(observation['image'])}, max: {np.max(observation['image'])}")
        # print(f"slopes min: {np.min(observation['slopes'])}, max: {np.max(observation['slopes'])}")
        # print(f"strehl min: {np.min(observation['strehl'])}, max: {np.max(observation['strehl'])}")
        

        assert self.observation_space.contains(observation), "Observation doesn't match space"
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
        return -np.log10(self.get_contrast + 1e-20) # Tiny positive value to ensure its positive.

    def step(self, action):
        # Update the environment state based on the action
        assert action.shape == self.deformable_mirror.actuators.shape

        self.set_dm(action=action * 1e-8)
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

    e = CoronagraphEnvironment(num_modes=40)

    e.set_random_dm(noise=0.01)

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