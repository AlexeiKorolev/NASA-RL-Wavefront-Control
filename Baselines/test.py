from hcipy import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# For notebook animations
from matplotlib import animation
from IPython.display import HTML

mpl.rcParams['figure.dpi'] = 100

pupil_grid = make_pupil_grid(256, 1.5)
focal_grid = make_focal_grid(8, 12)
prop = FraunhoferPropagator(pupil_grid, focal_grid)


aperture = evaluate_supersampled(circular_aperture(1), pupil_grid, 4)
lyot_mask = evaluate_supersampled(circular_aperture(0.95), pupil_grid, 4)

plt.subplot(1,2,1)
plt.title('Aperture')
imshow_field(aperture, cmap='gray')
plt.subplot(1,2,2)
plt.title('Lyot stop')
imshow_field(lyot_mask, cmap='gray')
plt.show()

wf = Wavefront(aperture)
img_ref = prop(wf).intensity

imshow_field(np.log10(img_ref / img_ref.max()), vmin=-5, cmap='inferno')
plt.show()



charge = 3
coro = VortexCoronagraph(pupil_grid, charge)
lyot_stop = Apodizer(lyot_mask)

wf = Wavefront(aperture)
lyot_plane = coro(wf)

imshow_field(lyot_plane.intensity, cmap='inferno')
plt.show()
