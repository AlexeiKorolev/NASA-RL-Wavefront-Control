- Initial results with image ml model. Not getting consistently below 1e-5 contrast possibly need defocus image as well? Is is possible to do WFC with just the camera image? Slopes still performs better

- Does my contrast match my image? I have a long exposure time.

- Why is contrast increasing and then decreasing? From what I could tell, it's because large values of noise spread the image thin, and create good contrast. This will be a problem with RL for reward function.

- Is there any technical difference between defocusing a lens and adding a defocus Zernike aberration mode?

- Quick check, using really large delta_t for the camera image should be ok?

# Change to vortex to charge 4 or charge 6.

# hcipy: "Emiel Por" <epor@ucsc.edu>

# Define region of interest as either a D-shape, or C-shape (half anulus), while only one dm.

# Resolution limit is 1.22 lambda / D for circular aperture.

# 3 lambda / D is the inner working angle (ranges 1 - 6, usually between 2 - 3, best angle is where planet throughput is 50%) (inner circle within the anulus).

# Outer working angle, 10 lambda / D (maybe 7 to start). (Careful to not set beyond Nyquist limit), for example, if 48 x 48 actuators, 24 l/D is limit, because beyond that point, the wave will get aliased.

# Possibly make everything lower quality (Dm surface, output pixels, etc.) \*but check that it doesn't break the coronagraph, still 10\*\*-10 contrast when there is no error.

# Change contrast to hcipy contrast: https://docs.hcipy.org/dev/api/hcipy.metrics.get_mean_ra

# Defocus is just one way to make diversity images

# One image is NOT sufficient, need to play with the DM.

# Get advice from a professor @ Pton.

# Best way to compute contrast is wrto a planet in high throughput region. And planet shape is the same as the unblocked and unaberrated starshape. But when errors on DM grow, then there are no longer clean images, and spreads out light everywhere. If error on DM exceeds ~1 radian of phase (possibly 0.1 noise), corrupts images, just constrain DM to not exceed that value. Check hcipy for prebuilt functions as well.

# Make optimistic assumptions, just remove all sources of noise.

# Check computational cost of noise.

# Homework: rerun NN with tweaks above, and see if we can get 10\*\*-10 contrast.

# DIVERSITY IMAGE CAN JUST BE WITH A NEW WAVELENGTH OF LIGHT????

# Changes I made:

- fixed everything to be in terms of spacial_resolution.
- reduced the resolution of DM surface
- fixed contrast

* Note: hcipy contrast is behaving weirdly.. really noisy images are still >10 \*\* 10 contrast

# Probe the actuators a couple of directions. Std way of doing it: take a probe and an -probe. ASSUMING electric field didn't change (slow changes over time), I'll get 2 images from which I can estimate the electric field (real and complex parts). So no way a single image can contain all the information. Image is just L2 norm of electric field. Probe structure isn't exactly defined.

# Not entirely clear if current probing is best for RL. Read more: EFC and probing, pairwise probing (tyler groff, AJ Riggs).

# For RL, can just use past images as diversity images (but only for a short period of time, prior to large changes in the electric field).

# If I get down to 50x50 images (small images), there will be an artificial limit for contrast.

# hcipy raw contrast needs to be normalized with energy
