from hcipy import *
from progressbar import progressbar

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import time
import os

from environment import CoronagraphEnviroment

e = CoronagraphEnviroment()

imshow_field(e.get_camera_image())
plt.show()