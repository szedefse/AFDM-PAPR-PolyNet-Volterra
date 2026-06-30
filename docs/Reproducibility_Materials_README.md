# Reproducibility Materials for the AFDM PAPR Reduction Study

This document gives a compact guide to the reproducibility materials released with the AFDM PAPR reduction study.

## Included materials

- Source code for the AFDM simulation, model training, evaluation, and GPU benchmarking workflow.
- Reviewer-requested experimental scripts for statistical validation, threshold sensitivity, high-order modulation retraining, Doppler robustness, full-frame ablation, and local hardware-oriented measurements.
- Trained model weights for the main baselines and PolyNet-Volterra.
- Model configuration files and simulation parameter files.
- Fixed random seed and initialization documentation.
- Numerical result files in CSV format.
- Generated figures in PNG format, with selected publication-style TIF files where available.

## Environment requirements

The required Python dependencies are listed in `requirements.txt` at the repository root. A typical setup is:

```bash
pip install -r requirements.txt
```

GPU-specific benchmark values in the manuscript were measured on an NVIDIA GeForce RTX 3090. CPU-only execution can reproduce most simulation tables, but GPU latency and training time should be regenerated on CUDA hardware if hardware-specific measurements are needed.

## Regeneration workflow

1. Install the dependencies.
2. Run `python src/train_gis_ace_net.py` to train and evaluate the main models.
3. Run `python src/train_gis_ace_net.py --benchmark-only` to regenerate GPU latency values when trained weights and CUDA are available.
4. Run the scripts in `scripts/` to regenerate reviewer-requested experiments and plots.
5. Compare the regenerated CSV and figure files with the released files under `results/` and `figures/`.

## Scope

The released materials cover simulation-based AFDM experiments and local GPU measurements. They do not include measured RF waveforms, USRP experiments, UAV field tests, or hardware-in-the-loop validation.

## Reproducibility notes

- The main experiments use the fixed seed `42`.
- Repeated Monte Carlo experiments use explicitly listed independent seeds.
- The reviewer-response statistical tables report repeated-run variability, uncertainty intervals, and formal significance tests.
- The model weights are included so that evaluation can be reproduced without retraining all models.
