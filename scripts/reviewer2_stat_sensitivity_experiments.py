import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import reviewer_additional_experiments as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
MODEL_DIR = ROOT / "saved_models"
GPU_COMPLEXITY = ROOT / "results" / "complexity_latency_gpu.csv"
GPU_TRAINING = ROOT / "results" / "training_time_gpu.csv"

FIG_DIR.mkdir(exist_ok=True)

MODEL_NAMES = ["DNN", "ResNet", "GatedMLP", "SE-ResNet", "PolyNet", "PolyNet-Volterra"]
PV = "PolyNet-Volterra"


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def mean_ci(values, confidence=0.95):
    # t-critical for df=9 and 95% CI; close enough for n=10 repeated runs.
    tcrit_by_df = {4: 2.776, 9: 2.262}
    arr = np.array(values, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, mean, mean
    sd = float(arr.std(ddof=1))
    tcrit = tcrit_by_df.get(len(arr) - 1, 1.96)
    half = tcrit * sd / math.sqrt(len(arr))
    return mean, mean - half, mean + half


def wilson_interval(errors, total, z=1.96):
    if total <= 0:
        return 0.0, 0.0, 0.0
    p = errors / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def two_proportion_z_test(e1, n1, e2, n2):
    p_pool = (e1 + e2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (e1 / n1 - e2 / n2) / se
    return z, math.erfc(abs(z) / math.sqrt(2))


def sign_test_p(wins, n):
    if n == 0:
        return 1.0
    k = max(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2**n)
    return min(1.0, 2 * tail)


def p_label(value):
    if value == 0.0 or value < 1e-300:
        return "<1.0e-300"
    return f"{value:.3e}"


def repeated_independent_statistics(repeats=10, frames=10000):
    models = base.load_models(MODEL_NAMES)
    detail_rows = []
    for rep in range(repeats):
        seed = 9100 + rep
        summary = base.evaluate_models(models, frames=frames, batch_size=500, seed=seed)
        for name in MODEL_NAMES:
            item = summary[name]
            detail_rows.append(
                {
                    "Run": rep + 1,
                    "Seed": seed,
                    "Model": name,
                    "Frames": frames,
                    "Symbols": int(item["Symbols"]),
                    "Errors": int(item["Errors"]),
                    "BER": f"{item['BER']:.10e}",
                    "Active_BER": f"{item['Active_BER']:.10e}",
                    "Masked_MSE": f"{item['Masked_MSE']:.10e}",
                    "PAPR_Reduction_dB": f"{item['PAPR_Reduction_dB']:.6f}",
                    "Mask_Ratio": f"{item['Mask_Ratio']:.8f}",
                }
            )
    write_rows(OUT_DIR / "stat_repeated_runs_detail.csv", detail_rows)

    summary_rows = []
    uncertainty_rows = []
    by_model = {name: [r for r in detail_rows if r["Model"] == name] for name in MODEL_NAMES}
    for name in MODEL_NAMES:
        rows = by_model[name]
        ber = [float(r["BER"]) for r in rows]
        mse = [float(r["Masked_MSE"]) for r in rows]
        papr = [float(r["PAPR_Reduction_dB"]) for r in rows]
        ber_mean, ber_std = mean_std(ber)
        mse_mean, mse_std = mean_std(mse)
        papr_mean, papr_std = mean_std(papr)
        errors = sum(int(r["Errors"]) for r in rows)
        symbols = sum(int(r["Symbols"]) for r in rows)
        wilson_ber, wilson_low, wilson_high = wilson_interval(errors, symbols)
        mse_mean_ci, mse_low, mse_high = mean_ci(mse)
        summary_rows.append(
            {
                "Model": name,
                "Independent_runs": repeats,
                "Frames_per_run": frames,
                "BER_mean": f"{ber_mean:.10e}",
                "BER_std": f"{ber_std:.10e}",
                "BER_CV_percent": f"{100 * ber_std / ber_mean:.2f}" if ber_mean else "0.00",
                "Masked_MSE_mean": f"{mse_mean:.10e}",
                "Masked_MSE_std": f"{mse_std:.10e}",
                "PAPR_Reduction_dB_mean": f"{papr_mean:.6f}",
                "PAPR_Reduction_dB_std": f"{papr_std:.6f}",
            }
        )
        uncertainty_rows.append(
            {
                "Model": name,
                "Aggregate_symbols": symbols,
                "Aggregate_errors": errors,
                "Aggregate_BER": f"{wilson_ber:.10e}",
                "Wilson95_low": f"{wilson_low:.10e}",
                "Wilson95_high": f"{wilson_high:.10e}",
                "Wilson95_half_width": f"{(wilson_high - wilson_low) / 2:.10e}",
                "Masked_MSE_mean": f"{mse_mean_ci:.10e}",
                "Masked_MSE_95CI_low": f"{mse_low:.10e}",
                "Masked_MSE_95CI_high": f"{mse_high:.10e}",
            }
        )
    write_rows(OUT_DIR / "stat_repeated_runs_summary.csv", summary_rows)
    write_rows(OUT_DIR / "stat_uncertainty_quantification.csv", uncertainty_rows)

    ref_rows = by_model[PV]
    ref_errors = sum(int(r["Errors"]) for r in ref_rows)
    ref_symbols = sum(int(r["Symbols"]) for r in ref_rows)
    ref_ber_by_run = {int(r["Run"]): float(r["BER"]) for r in ref_rows}
    sig_rows = []
    for name in MODEL_NAMES:
        if name == PV:
            continue
        rows = by_model[name]
        errors = sum(int(r["Errors"]) for r in rows)
        symbols = sum(int(r["Symbols"]) for r in rows)
        z, p = two_proportion_z_test(errors, symbols, ref_errors, ref_symbols)
        diffs = [float(r["BER"]) - ref_ber_by_run[int(r["Run"])] for r in rows]
        wins = sum(1 for d in diffs if d > 0)
        n_non_tie = sum(1 for d in diffs if d != 0)
        sig_rows.append(
            {
                "Comparison": f"{name} vs {PV}",
                "Mean_BER_difference": f"{float(np.mean(diffs)):.10e}",
                "Aggregate_z": f"{z:.4f}",
                "Aggregate_p_value": p_label(p),
                "Runs_with_higher_BER_than_PV": f"{wins}/{n_non_tie}",
                "Paired_sign_test_p_value": p_label(sign_test_p(wins, n_non_tie)),
            }
        )
    write_rows(OUT_DIR / "stat_significance_tests.csv", sig_rows)
    plot_statistical_validation(detail_rows, uncertainty_rows, sig_rows)


def custom_noise(shape_like, noise_std, mode):
    if mode == "AWGN":
        return noise_std * (torch.randn_like(shape_like) + 1j * torch.randn_like(shape_like))
    if mode == "impulsive":
        base_noise = noise_std * (torch.randn_like(shape_like) + 1j * torch.randn_like(shape_like))
        impulse_mask = (torch.rand(shape_like.shape, device=shape_like.device) < 0.02).to(torch.float32)
        impulse_noise = 8 * noise_std * (torch.randn_like(shape_like) + 1j * torch.randn_like(shape_like))
        return base_noise + impulse_mask * impulse_noise
    if mode == "colored":
        white = noise_std * (torch.randn_like(shape_like) + 1j * torch.randn_like(shape_like))
        colored = white.clone()
        rho = 0.70
        scale = math.sqrt(1 - rho * rho)
        for idx in range(1, colored.shape[1]):
            colored[:, idx] = rho * colored[:, idx - 1] + scale * white[:, idx]
        return colored
    raise ValueError(f"Unknown noise mode: {mode}")


def batch_observation_custom(batch_size, scenario):
    x_p, data = base.generate_data(batch_size, 4)
    x_p, data = x_p.to(base.DEVICE), data.to(base.DEVICE)
    s_orig, _ = base.afdm.construct_afdm_frame_basic(x_p, data)
    s_masked, mask_idx = base.afdm.apply_mask(s_orig, 0.9)

    taps = scenario["taps"]
    l_max = scenario["l_max"]
    k_max = scenario["k_max"]
    h_true = base.afdm.gen_channel_matrix(base.N, taps, l_max, k_max, base.afdm.c1).to(base.DEVICE)
    r = torch.matmul(s_masked, h_true.T)

    snr_db = 20
    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1 / snr_linear
    noise_std = math.sqrt(1 / (2 * snr_linear))
    r_noisy = r + custom_noise(r, noise_std, scenario["noise"])

    h_est = h_true
    mismatch = scenario["mismatch"]
    if mismatch > 0:
        scale = torch.mean(torch.abs(h_true)).clamp_min(1e-8)
        perturb = scale * (torch.randn_like(h_true) + 1j * torch.randn_like(h_true)) / math.sqrt(2)
        h_est = h_true + mismatch * perturb

    eye = torch.eye(base.N).cfloat().to(base.DEVICE)
    h_h = torch.conj(h_est.T)
    mmse = torch.matmul(h_h, h_est) + noise_var * eye
    g = torch.matmul(torch.inverse(mmse), h_h)
    s_est = torch.matmul(r_noisy, g.T)

    return {
        "data": data,
        "s_orig": s_orig,
        "s_masked": s_masked,
        "mask_idx": mask_idx,
        "s_est": s_est,
    }


def evaluate_scenario(model, scenario, frames=8000, batch_size=500, seed=12000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    metrics = base.init_metrics()
    papr_orig_values = []
    papr_masked_values = []
    mask_count = 0
    total_count = 0
    batches = math.ceil(frames / batch_size)
    for bi in range(batches):
        current_batch = min(batch_size, frames - bi * batch_size)
        batch = batch_observation_custom(current_batch, scenario)
        papr_orig_values.append(base.papr_db(batch["s_orig"]).detach())
        papr_masked_values.append(base.papr_db(batch["s_masked"]).detach())
        mask_count += int(batch["mask_idx"].sum().item())
        total_count += int(batch["mask_idx"].numel())
        base.update_metrics(metrics, model, batch, 4)
    ber, active_ber, mse = base.finalize_metrics(metrics)
    papr_orig = torch.cat(papr_orig_values).mean().item()
    papr_masked = torch.cat(papr_masked_values).mean().item()
    return {
        "BER": ber,
        "Active_BER": active_ber,
        "Masked_MSE": mse,
        "PAPR_Reduction_dB": papr_orig - papr_masked,
        "Mask_Ratio": mask_count / total_count if total_count else 0,
    }


def modeling_assumption_sensitivity(repeats=5, frames=8000):
    model = base.load_models([PV])[PV]
    scenarios = [
        {"Scenario": "baseline_AWGN", "Stress_type": "baseline condition", "taps": 3, "l_max": 2, "k_max": 1, "noise": "AWGN", "mismatch": 0.0},
        {"Scenario": "higher_Doppler", "Stress_type": "channel", "taps": 3, "l_max": 2, "k_max": 4, "noise": "AWGN", "mismatch": 0.0},
        {"Scenario": "extended_delay", "Stress_type": "channel", "taps": 5, "l_max": 4, "k_max": 1, "noise": "AWGN", "mismatch": 0.0},
        {"Scenario": "joint_delay_Doppler", "Stress_type": "channel", "taps": 5, "l_max": 4, "k_max": 4, "noise": "AWGN", "mismatch": 0.0},
        {"Scenario": "mild_channel_mismatch", "Stress_type": "model mismatch", "taps": 3, "l_max": 2, "k_max": 1, "noise": "AWGN", "mismatch": 0.05},
        {"Scenario": "strong_channel_mismatch", "Stress_type": "model mismatch", "taps": 3, "l_max": 2, "k_max": 1, "noise": "AWGN", "mismatch": 0.10},
        {"Scenario": "impulsive_noise", "Stress_type": "noise", "taps": 3, "l_max": 2, "k_max": 1, "noise": "impulsive", "mismatch": 0.0},
        {"Scenario": "colored_noise", "Stress_type": "noise", "taps": 3, "l_max": 2, "k_max": 1, "noise": "colored", "mismatch": 0.0},
    ]
    rows = []
    detail = []
    for s_idx, scenario in enumerate(scenarios):
        collected = {k: [] for k in ["BER", "Active_BER", "Masked_MSE", "PAPR_Reduction_dB", "Mask_Ratio"]}
        for rep in range(repeats):
            seed = 13000 + 100 * s_idx + rep
            out = evaluate_scenario(model, scenario, frames=frames, seed=seed)
            detail.append(
                {
                    "Scenario": scenario["Scenario"],
                    "Run": rep + 1,
                    "Seed": seed,
                    **{k: f"{v:.10e}" for k, v in out.items()},
                }
            )
            for key in collected:
                collected[key].append(out[key])
        row = {
            "Scenario": scenario["Scenario"],
            "Stress_type": scenario["Stress_type"],
            "Taps": scenario["taps"],
            "l_max": scenario["l_max"],
            "k_max": scenario["k_max"],
            "Noise_model": scenario["noise"],
            "Channel_mismatch_ratio": scenario["mismatch"],
            "Repeats": repeats,
            "Frames_per_repeat": frames,
        }
        for key, values in collected.items():
            m, sd = mean_std(values)
            row[f"{key}_mean"] = f"{m:.10e}" if key != "PAPR_Reduction_dB" else f"{m:.6f}"
            row[f"{key}_std"] = f"{sd:.10e}" if key != "PAPR_Reduction_dB" else f"{sd:.6f}"
        rows.append(row)
    write_rows(OUT_DIR / "modeling_assumption_sensitivity_detail.csv", detail)
    write_rows(OUT_DIR / "modeling_assumption_sensitivity.csv", rows)
    plot_modeling_assumptions(rows)


def hardware_resource_validation():
    complexity = {r["Model"]: r for r in read_rows(GPU_COMPLEXITY)}
    training = {r["Model"]: r for r in read_rows(GPU_TRAINING)}
    rows = []
    for name in MODEL_NAMES:
        c = complexity[name]
        t = training[name]
        params = int(c["Parameters"])
        weight_file = MODEL_DIR / f"{name}.pth"
        rows.append(
            {
                "Model": name,
                "Device": c["Device"],
                "Parameters": params,
                "FP32_parameter_memory_MB": f"{params * 4 / (1024 ** 2):.3f}",
                "Saved_weight_file_MB": f"{weight_file.stat().st_size / (1024 ** 2):.3f}" if weight_file.exists() else "NA",
                "Single_frame_GPU_latency_ms": f"{float(c['Single_frame_GPU_latency_ms']):.4f}",
                "Batch256_GPU_latency_frame_ms": f"{float(c['Batch256_GPU_latency_ms_per_frame']):.4f}",
                "Batch256_throughput_frames_s": f"{float(c['Batch256_throughput_frames_per_s']):.2f}",
                "Training_time_s": f"{float(t['Training_seconds']):.2f}",
            }
        )
    write_rows(OUT_DIR / "hardware_resource_validation.csv", rows)


def plot_statistical_validation(detail_rows, uncertainty_rows, sig_rows):
    labels = MODEL_NAMES
    x = np.arange(len(labels))
    ber_runs = [[float(r["BER"]) for r in detail_rows if r["Model"] == name] for name in labels]
    agg = {r["Model"]: r for r in uncertainty_rows}
    ber = np.array([float(agg[name]["Aggregate_BER"]) for name in labels])
    low = np.array([float(agg[name]["Wilson95_low"]) for name in labels])
    high = np.array([float(agg[name]["Wilson95_high"]) for name in labels])
    pvals = []
    p_labels = []
    for name in labels:
        if name == PV:
            pvals.append(np.nan)
            p_labels.append("ref")
        else:
            row = next(r for r in sig_rows if r["Comparison"].startswith(name + " vs"))
            p_text = row["Aggregate_p_value"]
            pvals.append(1e-300 if p_text.startswith("<") else float(p_text))
            p_labels.append(p_text)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    axes[0].boxplot(ber_runs, labels=labels, showfliers=False)
    for i, runs in enumerate(ber_runs, 1):
        axes[0].scatter(np.full(len(runs), i), runs, s=20, alpha=0.65)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("BER at 20 dB")
    axes[0].set_title("10 independent runs")
    axes[0].tick_params(axis="x", rotation=28)
    axes[0].grid(True, which="both", linestyle=":", alpha=0.4)

    yerr = np.vstack([ber - low, high - ber])
    axes[1].errorbar(x, ber, yerr=yerr, fmt="o", capsize=5, color="#2f5d8c")
    axes[1].set_xticks(x, labels, rotation=28, ha="right")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Aggregate BER with Wilson 95% CI")
    axes[1].set_title("Uncertainty quantification")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.4)

    valid_x = [i for i, p in enumerate(pvals) if not np.isnan(p)]
    neglog = [-math.log10(max(pvals[i], 1e-300)) for i in valid_x]
    axes[2].bar([labels[i] for i in valid_x], neglog, color="#7b6aa8")
    axes[2].set_ylabel("-log10(p)")
    axes[2].set_title("Aggregate BER significance vs PolyNet-Volterra")
    axes[2].tick_params(axis="x", rotation=28)
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Extended Statistical Validation at 20 dB", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_s9_extended_statistical_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_modeling_assumptions(rows):
    labels = [r["Scenario"].replace("_", "\n") for r in rows]
    x = np.arange(len(labels))
    ber = np.array([float(r["BER_mean"]) for r in rows])
    ber_std = np.array([float(r["BER_std"]) for r in rows])
    mse = np.array([float(r["Masked_MSE_mean"]) for r in rows])
    mse_std = np.array([float(r["Masked_MSE_std"]) for r in rows])
    papr = np.array([float(r["PAPR_Reduction_dB_mean"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))
    axes[0].bar(x, ber, yerr=ber_std, capsize=3, color="#3f7f93")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].set_ylabel("BER mean +/- SD")
    axes[0].set_title("BER sensitivity")
    axes[0].grid(True, axis="y", which="both", linestyle=":", alpha=0.4)

    axes[1].bar(x, mse, yerr=mse_std, capsize=3, color="#d08a3e")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].set_ylabel("Masked MSE mean +/- SD")
    axes[1].set_title("Reconstruction sensitivity")
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.4)

    axes[2].bar(x, papr, color="#6c78a8")
    axes[2].set_xticks(x, labels, rotation=25, ha="right")
    axes[2].set_ylabel("PAPR reduction (dB)")
    axes[2].set_title("PAPR reduction stability")
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("Modeling-Assumption Sensitivity of PolyNet-Volterra", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_s10_modeling_assumption_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Writing Reviewer 2 experiments to: {OUT_DIR}")
    repeated_independent_statistics()
    modeling_assumption_sensitivity()
    hardware_resource_validation()
    print("Done.")


if __name__ == "__main__":
    main()
