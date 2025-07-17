# Deformable Mirror Control with Reinforcement Learning and High-Contrast Imaging

This project explores the intersection of adaptive optics, high-contrast imaging, and machine learning. It provides a simulation framework for controlling a deformable mirror (DM) in a coronagraphic telescope system using reinforcement learning (RL) and neural networks. The goal is to optimize wavefront correction for exoplanet imaging and other high-contrast astronomical applications.

## Features

- **Coronagraphic Environment Simulation**  
  The core environment ([RL/environment.py](RL/environment.py)) models a telescope with a DM, vortex coronagraph, Lyot stop, and Shack-Hartmann wavefront sensor (SH-WFS). It simulates realistic optical propagation, noise, and atmospheric turbulence.

- **Reinforcement Learning Integration**  
  RL agents (e.g., PPO from Stable-Baselines3) can interact with the environment to learn optimal DM control policies. Training scripts and notebooks are provided ([RL/RL_training.ipynb](RL/RL_training.ipynb)).

- **Neural Network-Based Control**  
  Includes pipelines for training neural networks to predict DM settings from wavefront sensor slopes or focal plane images ([RL/testing_image_and_slopes.ipynb](RL/testing_image_and_slopes.ipynb), [RL/testing_slopes_small.ipynb](RL/testing_slopes_small.ipynb)).

- **High-Contrast Imaging Metrics**  
  Computes Strehl ratio, contrast, and other performance metrics. Supports visualization of PSFs, DM surfaces, and wavefront sensor images.

- **Dataset Generation**  
  Tools for generating large datasets of DM settings, slopes, and images for supervised learning.

- **Classical AO Benchmarks**  
  Includes classical AO control and simulation scripts for comparison ([dm_corrector.ipynb](dm_corrector.ipynb), [dm_custom.ipynb](dm_custom.ipynb)).

- **Extensive Visualization**  
  Notebooks and scripts for visualizing optical fields, DM modes, and RL agent performance.

### Prerequisites

- Python 3.8+
- [HCIPy](https://hcipy.org/) (High Contrast Imaging for Python)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- NumPy, Matplotlib, SciPy, tqdm, scikit-learn, PyTorch

Install dependencies:

```sh
pip install hcipy stable-baselines3 torch matplotlib scipy tqdm scikit-learn
```
