# File Inventory

This repository contains the materials needed to reproduce the reported simulation tables, reviewer-response experiments, local GPU measurements, and generated figures.

## Source code

- `src/train_gis_ace_net.py`: main AFDM simulation, training, evaluation, and GPU benchmark script.

## Reviewer-response and plotting scripts

- `scripts/reviewer_additional_experiments.py`: additional validation experiments requested during review.
- `scripts/reviewer_targeted_experiments.py`: targeted sensitivity, modulation, Doppler, and full-frame ablation experiments.
- `scripts/reviewer2_stat_sensitivity_experiments.py`: repeated-run variability, uncertainty quantification, and significance tests.
- `scripts/plot_reviewer_experiments.py`: plotting script for regenerated reviewer-response figures.

## Model weights

The trained model weights used for the reported simulation evaluations are stored in `saved_models/`.

- `DNN.pth`
- `ResNet.pth`
- `GatedMLP.pth`
- `SE-ResNet.pth`
- `PolyNet.pth`
- `PolyNet-Volterra.pth`

## Configurations

- `configs/simulation_config.json`: AFDM system, masking, training, and evaluation parameters.
- `configs/model_configs.json`: model input dimensions, hidden dimensions, normalization, activation, and weight-file mapping.
- `configs/reviewer_experiment_index.json`: mapping between reviewer-response experiments, scripts, result files, and figures.

## Numerical results

All released numerical tables are stored as CSV files in `results/`.

## Figures

Generated figures are stored in `figures/`. PNG files are provided for quick viewing. TIF versions are included for the figures prepared in publication style where available.

## Reproducibility documentation

- `README.md`: main repository guide.
- `docs/Reproducibility_Materials_README.md`: detailed reproducibility notes.
- `docs/seeds_and_initialization.md`: random seeds and initialization settings.
- `docs/file_inventory.md`: this file.
