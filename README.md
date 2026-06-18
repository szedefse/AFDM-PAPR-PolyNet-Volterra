# AFDM PAPR Reduction with PolyNet-Volterra

This repository contains the author-generated code, trained model weights, numerical results, and figures for the AFDM PAPR reduction study using the PolyNet-Volterra reconstruction network.

The materials are organized to support reproducibility of the manuscript tables, reviewer-response experiments, and supplementary figures.

## Repository Structure

```text
.
├── src/
│   └── train_gis_ace_net.py
├── scripts/
│   ├── reviewer_additional_experiments.py
│   ├── reviewer_targeted_experiments.py
│   ├── reviewer2_stat_sensitivity_experiments.py
│   └── plot_reviewer_experiments.py
├── saved_models/
│   ├── DNN.pth
│   ├── ResNet.pth
│   ├── GatedMLP.pth
│   ├── SE-ResNet.pth
│   ├── PolyNet.pth
│   └── PolyNet-Volterra.pth
├── results/
│   └── *.csv
├── figures/
│   └── *.png
└── docs/
    └── Reproducibility_Materials_README.md
```

## Environment

Recommended Python packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

The simulation and repeated-evaluation scripts can run on CPU. GPU-specific latency and training-time values in `results/complexity_latency_gpu.csv` and `results/training_time_gpu.csv` were measured on an NVIDIA GeForce RTX 3090.

## Main Scripts

Train and evaluate the main models:

```bash
python src/train_gis_ace_net.py
```

Regenerate the RTX 3090 GPU latency table, if CUDA is available and trained weights are present:

```bash
python src/train_gis_ace_net.py --benchmark-only
```

Regenerate additional reviewer experiments:

```bash
python scripts/reviewer_additional_experiments.py
python scripts/reviewer_targeted_experiments.py
python scripts/reviewer2_stat_sensitivity_experiments.py
python scripts/plot_reviewer_experiments.py
```

## Key Result Files

Statistical validation:

- `results/stat_repeated_runs_detail.csv`
- `results/stat_repeated_runs_summary.csv`
- `results/stat_uncertainty_quantification.csv`
- `results/stat_significance_tests.csv`
- `figures/fig_s9_extended_statistical_validation.png`

Modeling-assumption sensitivity:

- `results/modeling_assumption_sensitivity.csv`
- `results/modeling_assumption_sensitivity_detail.csv`
- `figures/fig_s10_modeling_assumption_sensitivity.png`

Hardware-oriented resource validation:

- `results/hardware_resource_validation.csv`
- `results/complexity_latency_gpu.csv`
- `results/training_time_gpu.csv`

Other robustness and ablation results:

- `results/threshold_sensitivity.csv`
- `results/modulation_retraining.csv`
- `results/modulation_robustness.csv`
- `results/doppler_robustness.csv`
- `results/full_frame_ablation.csv`

## Reproducibility Notes

- Main random seed: `42`.
- Repeated Monte Carlo experiments use explicit independent seeds recorded in the result tables.
- The trained weights in `saved_models/` are included so that evaluation tables can be regenerated without retraining.
- CPU-only runs can reproduce simulation tables, but GPU latency and training-time tables should be regenerated on CUDA hardware for hardware-specific claims.
- The modeling-assumption sensitivity experiment is simulation-based and does not replace RF hardware, USRP, or UAV field validation.

## Data Availability Statement Template

The source code, trained model configurations, numerical data, and scripts required to reproduce the reported tables and figures are available in this repository. The repository also includes a reproducibility README describing environment requirements, directory structure, random seeds, and regeneration steps.
