import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(exist_ok=True)


def read_csv(name):
    with open(EXP_DIR / name, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def as_float(rows, key):
    return np.array([float(r[key]) for r in rows], dtype=float)


def savefig(name):
    path = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_statistical_validation():
    rows = [r for r in read_csv("ber_20db_wilson_ci.csv") if r["Model"] not in {"Ideal_NoMask (理想下限)", "Masked_NoNet (未恢复上限)"}]
    labels = [r["Model"] for r in rows]
    ber = as_float(rows, "BER")
    low = as_float(rows, "Wilson95_low")
    high = as_float(rows, "Wilson95_high")
    x = np.arange(len(labels))
    yerr = np.vstack([ber - low, high - ber])

    plt.figure(figsize=(9.5, 5.8))
    plt.errorbar(x, ber, yerr=yerr, fmt="o", capsize=5, linewidth=2, color="#2f5d8c")
    plt.yscale("log")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("BER at 20 dB (log scale)")
    plt.title("Statistical Validation: BER with Wilson 95% Confidence Intervals")
    plt.grid(True, which="both", linestyle=":", alpha=0.45)
    return savefig("fig_s1_statistical_validation.png")


def plot_threshold_sensitivity():
    rows = read_csv("threshold_sensitivity.csv")
    x = as_float(rows, "Threshold")
    ber = as_float(rows, "BER_mean")
    ber_std = as_float(rows, "BER_std")
    papr = as_float(rows, "PAPR_Reduction_dB_mean")
    papr_std = as_float(rows, "PAPR_Reduction_dB_std")
    mask = 100 * as_float(rows, "Mask_Ratio_mean")

    fig, ax1 = plt.subplots(figsize=(8.8, 5.6))
    ax1.errorbar(x, papr, yerr=papr_std, marker="o", color="#1f77b4", linewidth=2.3, capsize=4, label="PAPR reduction")
    ax1.set_xlabel("Masking threshold")
    ax1.set_ylabel("PAPR reduction (dB)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, linestyle=":", alpha=0.45)

    ax2 = ax1.twinx()
    ax2.errorbar(x, ber, yerr=ber_std, marker="s", color="#c43c39", linewidth=2.3, capsize=4, label="BER")
    ax2.set_yscale("log")
    ax2.set_ylabel("BER at 20 dB (log scale)", color="#c43c39")
    ax2.tick_params(axis="y", labelcolor="#c43c39")

    for xi, mi in zip(x, mask):
        ax1.annotate(f"{mi:.1f}% masked", (xi, papr[list(x).index(xi)]), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)
    plt.title("Masking Threshold Sensitivity")
    return savefig("fig_s2_threshold_sensitivity.png")


def plot_modulation_stress():
    rows = read_csv("modulation_robustness.csv")
    labels = ["QPSK" if int(float(r["Modulation_order"])) == 4 else f"{int(float(r['Modulation_order']))}-QAM" for r in rows]
    ber = as_float(rows, "BER_mean")
    ber_std = as_float(rows, "BER_std")
    papr = as_float(rows, "PAPR_Reduction_dB_mean")
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    axes[0].bar(x, papr, color="#5c8f68", width=0.55)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("PAPR reduction (dB)")
    axes[0].set_title("PAPR compression")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.4)

    axes[1].bar(x, ber, yerr=ber_std, color="#b5534f", width=0.55, capsize=5)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("BER at 20 dB (log scale)")
    axes[1].set_title("Out-of-distribution BER")
    axes[1].grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    fig.suptitle("High-Order Modulation Stress Test with QPSK-Trained Weights", y=1.02)
    return savefig("fig_s3_modulation_stress.png")


def plot_doppler_robustness():
    rows = read_csv("doppler_robustness.csv")
    x = as_float(rows, "Doppler_kmax")
    ber = as_float(rows, "BER_mean")
    ber_std = as_float(rows, "BER_std")
    mse = as_float(rows, "Masked_MSE_mean")
    mse_std = as_float(rows, "Masked_MSE_std")

    fig, ax1 = plt.subplots(figsize=(8.8, 5.5))
    ax1.errorbar(x, ber, yerr=ber_std, marker="o", color="#764b8e", linewidth=2.3, capsize=4)
    ax1.set_yscale("log")
    ax1.set_xlabel("Maximum Doppler index kmax")
    ax1.set_ylabel("BER at 20 dB (log scale)", color="#764b8e")
    ax1.tick_params(axis="y", labelcolor="#764b8e")
    ax1.grid(True, which="both", linestyle=":", alpha=0.45)

    ax2 = ax1.twinx()
    ax2.errorbar(x, mse, yerr=mse_std, marker="s", color="#d6862d", linewidth=2.3, capsize=4)
    ax2.set_ylabel("Masked-region MSE", color="#d6862d")
    ax2.tick_params(axis="y", labelcolor="#d6862d")
    plt.title("Doppler Robustness of PolyNet-Volterra")
    return savefig("fig_s4_doppler_robustness.png")


def plot_complexity_latency():
    rows = read_csv("complexity_latency.csv")
    labels = [r["Model"] for r in rows]
    flops = as_float(rows, "Linear_FLOPs_per_frame") / 1e6
    latency = as_float(rows, "Single_frame_CPU_latency_ms")
    params = as_float(rows, "Parameters") / 1000

    plt.figure(figsize=(8.8, 5.6))
    plt.scatter(flops, latency, s=params * 1.8, color="#337d8d", alpha=0.78, edgecolors="white", linewidths=1.5)
    for label, fx, ly in zip(labels, flops, latency):
        plt.annotate(label, (fx, ly), textcoords="offset points", xytext=(6, 5), fontsize=8)
    plt.xlabel("Linear-layer FLOPs per frame (M)")
    plt.ylabel("Single-frame CPU latency (ms)")
    plt.title("Complexity-Latency Trade-off")
    plt.grid(True, linestyle=":", alpha=0.45)
    return savefig("fig_s5_complexity_latency.png")


def plot_training_time():
    rows = read_csv("training_time_estimate.csv")
    labels = [r["Model"] for r in rows]
    minutes = as_float(rows, "Estimated_minutes_150epochs")
    x = np.arange(len(labels))
    plt.figure(figsize=(9.2, 5.3))
    plt.bar(x, minutes, color="#6874a8", width=0.6)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Estimated CPU training time for 150 epochs (min)")
    plt.title("Estimated Training Time from Short Full-Batch Benchmark")
    plt.grid(True, axis="y", linestyle=":", alpha=0.45)
    return savefig("fig_s6_training_time_estimate.png")


def plot_modulation_retraining():
    rows = read_csv("modulation_retraining.csv")
    labels = [r["Modulation"] for r in rows]
    ber = as_float(rows, "BER_mean")
    ber_std = as_float(rows, "BER_std")
    papr = as_float(rows, "PAPR_Reduction_dB_mean")
    mse = as_float(rows, "Masked_MSE_mean")
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.5))
    axes[0].bar(x, papr, color="#4f8a73", width=0.58)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("PAPR reduction (dB)")
    axes[0].set_title("Peak compression")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.4)

    axes[1].bar(x, ber, yerr=ber_std, color="#bf5a50", width=0.58, capsize=4)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("BER at 20 dB")
    axes[1].set_title("Demodulation accuracy")
    axes[1].grid(True, axis="y", which="both", linestyle=":", alpha=0.4)

    axes[2].bar(x, mse, color="#5977a8", width=0.58)
    axes[2].set_xticks(x, labels)
    axes[2].set_ylabel("Masked-region MSE")
    axes[2].set_title("Reconstruction error")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("PolyNet-Volterra with Modulation-Specific Retraining", y=1.03)
    return savefig("fig_s7_modulation_retraining.png")


def plot_full_frame_ablation():
    rows = read_csv("full_frame_ablation.csv")
    models = []
    for r in rows:
        if r["Model"] not in models:
            models.append(r["Model"])
    full = {r["Model"]: r for r in rows if r["Input_setting"] == "full-frame input"}
    masked = {r["Model"]: r for r in rows if r["Input_setting"] == "masked-only input"}
    x = np.arange(len(models))
    width = 0.38

    full_mse = np.array([float(full[m]["Masked_MSE_mean"]) for m in models])
    masked_mse = np.array([float(masked[m]["Masked_MSE_mean"]) for m in models])
    full_ber = np.array([float(full[m]["BER_mean"]) for m in models])
    masked_ber = np.array([float(masked[m]["BER_mean"]) for m in models])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(x - width / 2, full_mse, width, label="Full-frame", color="#3f7f93")
    axes[0].bar(x + width / 2, masked_mse, width, label="Masked-only", color="#d08a3e")
    axes[0].set_xticks(x, models, rotation=25, ha="right")
    axes[0].set_ylabel("Masked-region MSE")
    axes[0].set_title("Reconstruction error")
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[0].legend(frameon=False)

    axes[1].bar(x - width / 2, full_ber, width, label="Full-frame", color="#3f7f93")
    axes[1].bar(x + width / 2, masked_ber, width, label="Masked-only", color="#d08a3e")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, models, rotation=25, ha="right")
    axes[1].set_ylabel("BER at 20 dB")
    axes[1].set_title("BER impact")
    axes[1].grid(True, axis="y", which="both", linestyle=":", alpha=0.4)
    axes[1].legend(frameon=False)

    fig.suptitle("Full-Frame Input Ablation", y=1.03)
    return savefig("fig_s8_full_frame_ablation.png")


def plot_summary_panel():
    threshold = read_csv("threshold_sensitivity.csv")
    modulation = read_csv("modulation_robustness.csv")
    doppler = read_csv("doppler_robustness.csv")
    complexity = read_csv("complexity_latency.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))

    tx = as_float(threshold, "Threshold")
    axes[0, 0].plot(tx, as_float(threshold, "PAPR_Reduction_dB_mean"), marker="o", color="#1f77b4")
    axes[0, 0].set_title("Threshold vs PAPR")
    axes[0, 0].set_xlabel("Threshold")
    axes[0, 0].set_ylabel("PAPR reduction (dB)")

    mx = np.arange(len(modulation))
    axes[0, 1].bar(mx, as_float(modulation, "BER_mean"), color="#b5534f")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Modulation Stress BER")
    axes[0, 1].set_xticks(mx, ["QPSK", "16-QAM", "64-QAM"])

    dx = as_float(doppler, "Doppler_kmax")
    axes[1, 0].plot(dx, as_float(doppler, "BER_mean"), marker="s", color="#764b8e")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Doppler Robustness")
    axes[1, 0].set_xlabel("kmax")
    axes[1, 0].set_ylabel("BER")

    flops = as_float(complexity, "Linear_FLOPs_per_frame") / 1e6
    lat = as_float(complexity, "Single_frame_CPU_latency_ms")
    axes[1, 1].scatter(flops, lat, color="#337d8d")
    axes[1, 1].set_title("Complexity-Latency")
    axes[1, 1].set_xlabel("FLOPs/frame (M)")
    axes[1, 1].set_ylabel("Latency (ms)")

    for ax in axes.ravel():
        ax.grid(True, linestyle=":", alpha=0.4)
    fig.suptitle("Summary of Additional Reviewer Experiments", fontsize=15, y=1.01)
    return savefig("fig_s0_additional_experiments_summary.png")


def write_summary(paths):
    lines = [
        "# Summary of Additional Reviewer Experiments",
        "",
        "This supplementary summary collects the additional experiments added in response to the reviewers.",
        "",
        "- Statistical validation: BER at 20 dB with Wilson 95% confidence intervals and repeated Monte Carlo standard deviations.",
        "- Threshold sensitivity: the threshold controls the PAPR/BER trade-off; 0.9 is retained as a balanced operating point.",
        "- Modulation stress test: QPSK-trained weights preserve PAPR reduction under 16-QAM/64-QAM but BER degrades, so high-order QAM requires retraining.",
        "- Doppler robustness: increasing kmax from 1 to 4 produces moderate BER/MSE degradation in the tested synthetic channels.",
        "- Complexity: PolyNet-Volterra has the lowest parameter count, lowest linear-layer FLOPs, and lowest measured CPU inference latency among the enabled models.",
        "- Training time: a short full-training-loop CPU benchmark is used to estimate the relative 150-epoch training time for each model.",
        "- Modulation-specific retraining: QPSK, 16-QAM, and 64-QAM were trained and evaluated separately to clarify high-order modulation behavior.",
        "- Full-frame input ablation: masked-only input increases masked-region MSE, supporting the use of full-frame contextual information.",
        "",
        "Generated figures:",
    ]
    for p in paths:
        lines.append(f"- {p.name}")
    (EXP_DIR / "Additional_Experiments_Summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    paths = [
        plot_summary_panel(),
        plot_statistical_validation(),
        plot_threshold_sensitivity(),
        plot_modulation_stress(),
        plot_doppler_robustness(),
        plot_complexity_latency(),
        plot_training_time(),
        plot_modulation_retraining(),
        plot_full_frame_ablation(),
    ]
    write_summary(paths)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
