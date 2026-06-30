# Random Seeds and Initialization Settings

This document records the deterministic settings used for the released AFDM PAPR reduction experiments.

## Main seed

The main training and evaluation script fixes the global seed as follows.

```python
torch.manual_seed(42)
np.random.seed(42)
```

The seed value `42` is used for the main experiments unless a reviewer-response script explicitly lists independent seeds for repeated validation.

## Repeated validation seeds

Repeated Monte Carlo validation uses independent seeds recorded in the result files under `results/`. In particular, see:

- `results/stat_repeated_runs_detail.csv`
- `results/stat_repeated_runs_summary.csv`
- `results/stat_uncertainty_quantification.csv`
- `results/stat_significance_tests.csv`

## Initialization

Model parameters use the default PyTorch initialization associated with each layer unless a model-specific module defines its own initialization routine.

- Linear layers in the DNN, ResNet, GatedMLP, SE-ResNet, PolyNet, and PolyNet-Volterra models use PyTorch default initialization.
- BatchNorm and LayerNorm layers use PyTorch default affine parameters.
- The SIREN helper model in `src/train_gis_ace_net.py` includes a dedicated sine-layer initialization routine, but it is not one of the six main manuscript baselines stored in `saved_models/`.

## Hardware-specific measurements

The GPU latency and training-time tables were measured on an NVIDIA GeForce RTX 3090. These values can differ across hardware and driver versions. The numerical files are included for transparency, and the benchmarking code is retained so that readers can regenerate hardware-specific measurements on their own machines.
