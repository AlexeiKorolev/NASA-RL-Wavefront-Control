import gymnasium as gym
from gymnasium import spaces
import numpy as np
from hcipy import *
import hcipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage


class CoronagraphEnvironment(gym.Env):
    def __init__(self, telescope_diameter = 8., oversizing_factor = 16 / 15, wavelength_wfs = 0.7e-6, 
                 wavelength_sci = 2.2e-6, num_modes = 500, zero_magnitude_flux = 3.9e10, #3.9e10 photon/s for a mag 0 star
                stellar_magnitude = 5, delta_t = 1e-3, pixels = 240, # sec, so a loop speed of 1kHz.
                num_iterations = 10
                ):
        super().__init__()

        print(f"initializing coronagraph env. might take a minute.")

        self.telescope_diameter = telescope_diameter
        self.oversizing_factor = oversizing_factor
    
        self.num_pupil_pixels = pixels * oversizing_factor
        self.pupil_grid_diameter = telescope_diameter * oversizing_factor
        self.pupil_grid = make_pupil_grid(self.num_pupil_pixels, self.pupil_grid_diameter)

        spatial_resolution = wavelength_sci / telescope_diameter
        self.focal_grid = make_focal_grid(q=4, num_airy=30, spatial_resolution=spatial_resolution)

        VLT_aperture_generator = hcipy.aperture.make_circular_aperture(telescope_diameter)
        self.VLT_aperture = evaluate_supersampled(VLT_aperture_generator, self.pupil_grid, 4)

        self.wavelength_wfs = wavelength_wfs
        self.wavelength_sci = wavelength_sci

        self.wf = Wavefront(self.VLT_aperture, wavelength_sci)
        self.wf.total_power = zero_magnitude_flux * 10**(-stellar_magnitude / 2.5)

        spatial_resolution = wavelength_sci / telescope_diameter

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
        charge = 2
        self.coro = VortexCoronagraph(self.pupil_grid, charge)
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
        # Define observation and action spaces
        # self.observation_space = spaces.Box(low=..., high=..., shape=(...), dtype=np.float32)  # Adjust shape based on your state representation
        # self.action_space = spaces.Discrete(...) # Adjust number based on your possible actions

        # Initialize other environment-specific attributes

    def set_random_dm(self, noise=0.3):
        # Put actuators at random values, putting a little more power in low-order modes
        self.deformable_mirror.actuators = np.random.randn(self.num_modes)  / (np.arange(self.num_modes) + 10)

        # Normalize the DM surface so that we get a reasonable surface RMS.
        self.deformable_mirror.actuators *= noise * self.wavelength_sci / np.std(self.deformable_mirror.surface)

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

    def get_camera_image(self):
        # Read out WFS camera
        propagrated_wf = self.prop(self.lyot_stop(self.coro(self.deformable_mirror(self.wf))))
        self.camera.integrate(propagrated_wf, self.delta_t)
        wfs_image = self.camera.read_out()
        wfs_image = large_poisson(wfs_image).astype('float')

        return wfs_image

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

        """self.current_state = ... # Define your initial state
        observation = self.current_state # or transform the state into an observation"""

        info = {}
        return observation, info

    @staticmethod
    def reward_function(strehl):
        return strehl

    def step(self, action):
        # Update the environment state based on the action
        assert action.shape == self.deformable_mirror.actuators.shape

        self.set_dm(action=action)
        self.iteration_counter -= 1

        reward = CoronagraphEnvironment.reward_function(self.get_strehl_ratio())

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
    e = CoronagraphEnvironment(num_modes=4)
    print(e.action_space)