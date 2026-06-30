# AFDM PAPR Reduction with PolyNet-Volterra

This repository contains the author-generated materials for the AFDM PAPR reduction study using the PolyNet-Volterra reconstruction network. It is organized as a reproducibility package for the manuscript simulations, reviewer-requested experiments, local GPU measurements, generated figures, and numerical result tables.

## Repository contents

```text
.
|-- src/
|   `-- train_gis_ace_net.py
|-- scripts/
|   |-- reviewer_additional_experiments.py
|   |-- reviewer_targeted_experiments.py
|   |-- reviewer2_stat_sensitivity_experiments.py
|   `-- plot_reviewer_experiments.py
|-- saved_models/
|   |-- DNN.pth
|   |-- ResNet.pth
|   |-- GatedMLP.pth
|   |-- SE-ResNet.pth
|   |-- PolyNet.pth
|   `-- PolyNet-Volterra.pth
|-- configs/
|   |-- simulation_config.json
|   |-- model_configs.json
|   `-- reviewer_experiment_index.json
|-- results/
|   `-- *.csv
|-- figures/
|   `-- *.png and selected *.tif
|-- docs/
|   |-- Reproducibility_Materials_README.md
|   |-- seeds_and_initialization.md
|   `-- file_inventory.md
|-- requirements.txt
`-- README.md
```

## What is included

- Complete source code for the AFDM simulation, training, evaluation, and GPU benchmarking workflow.
- Reviewer-requested experimental scripts for repeated Monte Carlo validation, uncertainty quantification, significance testing, threshold sensitivity, modulation retraining, Doppler robustness, and full-frame input ablation.
- Trained model weights for the main neural baselines and PolyNet-Volterra.
- Model configuration files, simulation parameters, dependency information, fixed random seeds, and initialization notes.
- Numerical result files used to regenerate the reported tables.
- Generated figures and tables used for manuscript revision and reviewer response.
- Reproducibility documentation describing the directory structure, environment requirements, regeneration steps, simulation scope, and local GPU performance benchmark results.

## Environment

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The main code was prepared with PyTorch, NumPy, Matplotlib, and standard scientific-computing packages. CPU execution can reproduce most simulation tables. GPU latency and training-time values are hardware-specific and were measured on an NVIDIA GeForce RTX 3090.

## Main reproduction steps

Train and evaluate the main AFDM reconstruction models:

```bash
python src/train_gis_ace_net.py
```

Regenerate local GPU latency values when CUDA and the trained weights are available:

```bash
python src/train_gis_ace_net.py --benchmark-only
```

Regenerate reviewer-requested experiments:

```bash
python scripts/reviewer_additional_experiments.py
python scripts/reviewer_targeted_experiments.py
python scripts/reviewer2_stat_sensitivity_experiments.py
python scripts/plot_reviewer_experiments.py
```

## Key result files

Statistical validation:

- `results/stat_repeated_runs_detail.csv`
- `results/stat_repeated_runs_summary.csv`
- `results/stat_uncertainty_quantification.csv`
- `results/stat_significance_tests.csv`
- `figures/fig_s9_extended_statistical_validation.png`

Hardware-oriented validation:

- `results/hardware_resource_validation.csv`
- `results/complexity_latency_gpu.csv`
- `results/training_time_gpu.csv`

Sensitivity, robustness, and ablation results:

- `results/threshold_sensitivity.csv`
- `results/modulation_retraining.csv`
- `results/modulation_robustness.csv`
- `results/doppler_robustness.csv`
- `results/full_frame_ablation.csv`

## Configuration and reproducibility notes

- Main random seed: `42`.
- Repeated Monte Carlo experiments use independent seeds recorded in the result tables.
- Model and simulation settings are listed in `configs/`.
- Random seed and initialization details are documented in `docs/seeds_and_initialization.md`.
- The trained weights in `saved_models/` allow evaluation tables to be regenerated without retraining.
- The local GPU benchmark files report RTX 3090 measurements and should be regenerated if hardware-specific latency or training time is needed on another device.

## Data availability wording

The source code, simulation scripts, reviewer-requested experimental scripts, trained model weights, model configurations, dependency information, fixed random seeds, initialization settings, numerical result files, generated figures and tables, and reproducibility documentation are provided in this repository. The accompanying README files describe the directory structure, environment requirements, regeneration steps, simulation scope, and local GPU performance benchmark results.
