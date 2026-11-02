# Working notes

Running research notes and job references from the cluster runs.

## Ideas and findings

1. Optimal normalization.
2. Zernike converges slower and overfits more?
3. Transformers vs fully-connected
4. RL for coronagraphic imaging!
5. Fine tuning on adverserial noise vs. adding different levels of noise to the data vs. gaussian neural nets.
6. Generate basis via Gramm Schmidt https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
7. Try linear / quadradic reward functions for RL.
8. Fit a linear model raw to the image (1D NN) for a baseline
9. Try different action scales.
10. Seperate direction and magnitude from the output of the neural network. Possibly split the training too.

11. Train on different fixed levels of noise and use a neural net ensemble.
12. Investigate different nudge vectors.

distribution graph: https://www.desmos.com/calculator/jpz9ge36zg

Slopes 512 model: /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_192105
Zernike non-random: /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_192016/exp-fc1-zscore-ep10000-bs262144-lr1e-3-s42-h512-256-128-images.slurm
Zernike random: /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_192023
Transfoermer: /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_195013
USE STREHL FOR RL FIRST
Transformer random: /scratch/network/ak9088/NASA-RL-Wavefront-Control/RL/jobs/20251116_195050

Linear model for baseline: 20251127_230433
