# Reproducibility Materials for the AFDM PAPR Reduction Study

This README is intended to accompany the GitHub release of the author-generated implementation materials.

## Contents

- Complete training and evaluation code for the AFDM simulation framework.
- Model configuration definitions for all enabled neural baselines and PolyNet variants.
- Fixed random seed settings used for the main experiments and reviewer-response experiments.
- Simulation parameter tables covering AFDM frame size, modulation, SNR, masking threshold, channel settings, and training schedule.
- Scripts for regenerating BER, MSE, PAPR, threshold sensitivity, modulation robustness, Doppler robustness, full-frame ablation, statistical validation, modeling-assumption sensitivity, and hardware-oriented benchmark tables.
- Plotting scripts for reproducing the supplementary statistical, sensitivity, robustness, and complexity figures.
- Trained model weights used for the reported simulation evaluations, where permitted by the repository size and submission rules.

## Environment

- Python with PyTorch, NumPy, Matplotlib, and standard scientific-computing dependencies.
- GPU benchmark values in the manuscript were measured on an NVIDIA GeForce RTX 3090 using the same trained weights.
- CPU-only execution can reproduce the simulation tables, but GPU latency and training-time values should be regenerated on CUDA hardware.

## Reproducibility Notes

- The main experiments use a fixed random seed of 42 unless otherwise stated.
- Repeated Monte Carlo experiments use explicitly listed independent seeds.
- The statistical-validation tables report run-level variability, aggregate uncertainty intervals, and formal significance tests.
- The modeling-assumption sensitivity experiment is simulation-based and is not a substitute for RF hardware, USRP, or UAV field validation.
