import csv
import importlib.util
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN_SCRIPT = ROOT / "src" / "train_gis_ace_net.py"
MODEL_DIR = ROOT / "saved_models"
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def load_train_module():
    spec = importlib.util.spec_from_file_location("afdm_train", str(TRAIN_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


afdm = load_train_module()
DEVICE = torch.device("cpu")
N = afdm.N
Q = afdm.Q
D = afdm.D


MODEL_SPECS = {
    "DNN": afdm.DNN_Baseline,
    "ResNet": afdm.ResNet_MLP,
    "GatedMLP": afdm.Gated_MLP,
    "SE-ResNet": afdm.SE_ResNet,
    "PolyNet": afdm.PolyNet,
    "PolyNet-Volterra": afdm.PolyNet_Volterra,
}


def load_models(names):
    models = {}
    for name in names:
        path = MODEL_DIR / f"{name}.pth"
        if not path.exists():
            raise FileNotFoundError(f"Missing model weight: {path}")
        model = MODEL_SPECS[name]().to(DEVICE)
        state = torch.load(str(path), map_location=DEVICE)
        model.load_state_dict(state, strict=True)
        model.eval()
        models[name] = model
    return models


def qam_constellation(order):
    side = int(math.sqrt(order))
    if side * side != order:
        raise ValueError(f"Only square QAM is supported, got M={order}")
    levels = torch.arange(-(side - 1), side, 2, dtype=torch.float32)
    real = levels.repeat_interleave(side)
    imag = levels.repeat(side)
    points = torch.complex(real, imag)
    avg_power = (2.0 / 3.0) * (order - 1)
    return points / math.sqrt(avg_power)


def generate_data(batch_size, order):
    points = qam_constellation(order)
    pilot_idx = torch.randint(0, order, (batch_size, 1))
    data_idx = torch.randint(0, order, (batch_size, D))
    return points[pilot_idx].cfloat(), points[data_idx].cfloat()


def qam_demod(y, order):
    points = qam_constellation(order).to(y.device)
    dist = torch.abs(y.unsqueeze(-1) - points.view(1, 1, -1)) ** 2
    idx = torch.argmin(dist, dim=-1)
    return points[idx]


def papr_db(frame):
    power = torch.abs(frame) ** 2
    papr = power.max(dim=1).values / power.mean(dim=1).clamp_min(1e-12)
    return 10.0 * torch.log10(papr.clamp_min(1e-12))


def batch_observation(batch_size, order, threshold, snr_db, k_max):
    x_p, data = generate_data(batch_size, order)
    x_p, data = x_p.to(DEVICE), data.to(DEVICE)
    s_orig, _ = afdm.construct_afdm_frame_basic(x_p, data)
    s_masked, mask_idx = afdm.apply_mask(s_orig, threshold)

    taps = 3
    l_max = 2
    H = afdm.gen_channel_matrix(N, taps, l_max, k_max, afdm.c1).to(DEVICE)
    r = torch.matmul(s_masked, H.T)

    snr_linear = 10 ** (snr_db / 10)
    noise_var = 1 / snr_linear
    noise_std = math.sqrt(1 / (2 * snr_linear))
    noise = noise_std * (torch.randn_like(r) + 1j * torch.randn_like(r))
    r_noisy = r + noise

    eye = torch.eye(N).cfloat().to(DEVICE)
    H_H = torch.conj(H.T)
    mmse = torch.matmul(H_H, H) + noise_var * eye
    G = torch.matmul(torch.inverse(mmse), H_H)
    s_est = torch.matmul(r_noisy, G.T)

    return {
        "data": data,
        "s_orig": s_orig,
        "s_masked": s_masked,
        "mask_idx": mask_idx,
        "s_est": s_est,
    }


def init_metrics():
    return {
        "err": 0,
        "symbols": 0,
        "active_err": 0,
        "active_symbols": 0,
        "mse_sum": 0.0,
        "masked_points": 0,
    }


def update_metrics(metrics, model, batch, order):
    mask_idx = batch["mask_idx"]
    s_est = batch["s_est"]
    s_out = s_est.clone()
    with torch.no_grad():
        pred = model(s_est)
    if mask_idx.any():
        s_out[mask_idx] = pred[mask_idx]
        diff = pred[mask_idx] - batch["s_orig"][mask_idx]
        metrics["mse_sum"] += (diff.real**2 + diff.imag**2).sum().item()
        metrics["masked_points"] += int(mask_idx.sum().item())

    x_est = torch.matmul(s_out, afdm.TRANSFORM_MATRIX.to(DEVICE).conj())
    data_est = x_est[:, 1 + Q : 1 + Q + D]
    demod = qam_demod(data_est, order)
    errors = demod != batch["data"]
    metrics["err"] += int(errors.sum().item())
    metrics["symbols"] += int(errors.numel())

    active = mask_idx.any(dim=1)
    if active.any():
        active_errors = errors[active]
        metrics["active_err"] += int(active_errors.sum().item())
        metrics["active_symbols"] += int(active_errors.numel())


def finalize_metrics(metrics):
    ber = metrics["err"] / metrics["symbols"] if metrics["symbols"] else 0.0
    active_ber = (
        metrics["active_err"] / metrics["active_symbols"]
        if metrics["active_symbols"]
        else 0.0
    )
    mse = (
        metrics["mse_sum"] / metrics["masked_points"]
        if metrics["masked_points"]
        else 0.0
    )
    return ber, active_ber, mse


def evaluate_models(models, order=4, threshold=0.9, snr_db=20, k_max=1, frames=5000, batch_size=500, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    states = {name: init_metrics() for name in models}
    papr_orig_values = []
    papr_masked_values = []
    mask_count = 0
    total_count = 0

    batches = math.ceil(frames / batch_size)
    for bi in range(batches):
        current_batch = min(batch_size, frames - bi * batch_size)
        batch = batch_observation(current_batch, order, threshold, snr_db, k_max)
        papr_orig_values.append(papr_db(batch["s_orig"]).detach())
        papr_masked_values.append(papr_db(batch["s_masked"]).detach())
        mask_count += int(batch["mask_idx"].sum().item())
        total_count += int(batch["mask_idx"].numel())

        for name, model in models.items():
            update_metrics(states[name], model, batch, order)

    papr_orig = torch.cat(papr_orig_values).mean().item()
    papr_masked = torch.cat(papr_masked_values).mean().item()
    summary = {}
    for name, state in states.items():
        ber, active_ber, mse = finalize_metrics(state)
        summary[name] = {
            "BER": ber,
            "Active_BER": active_ber,
            "Masked_MSE": mse,
            "PAPR_Original_dB": papr_orig,
            "PAPR_Masked_dB": papr_masked,
            "PAPR_Reduction_dB": papr_orig - papr_masked,
            "Mask_Ratio": mask_count / total_count if total_count else 0.0,
            "Symbols": state["symbols"],
            "Errors": state["err"],
        }
    return summary


def mean_std(values):
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def run_repeated_table(models, rows, out_csv):
    fieldnames = list(rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repeated_main_statistics():
    model_names = list(MODEL_SPECS.keys())
    models = load_models(model_names)
    repeats = 5
    frames = 20000
    results = {name: {"BER": [], "Active_BER": [], "Masked_MSE": []} for name in model_names}
    for rep in range(repeats):
        summary = evaluate_models(models, frames=frames, seed=1000 + rep)
        for name in model_names:
            for key in results[name]:
                results[name][key].append(summary[name][key])

    rows = []
    for name in model_names:
        ber_mean, ber_std = mean_std(results[name]["BER"])
        active_mean, active_std = mean_std(results[name]["Active_BER"])
        mse_mean, mse_std = mean_std(results[name]["Masked_MSE"])
        rows.append(
            {
                "Model": name,
                "Repeats": repeats,
                "Frames_per_repeat": frames,
                "BER_mean": f"{ber_mean:.8e}",
                "BER_std": f"{ber_std:.8e}",
                "Active_BER_mean": f"{active_mean:.8e}",
                "Active_BER_std": f"{active_std:.8e}",
                "Masked_MSE_mean": f"{mse_mean:.8e}",
                "Masked_MSE_std": f"{mse_std:.8e}",
            }
        )
    run_repeated_table(models, rows, OUT_DIR / "statistical_ber_20db.csv")


def single_model_scenario_tables():
    model = load_models(["PolyNet-Volterra"])
    repeats = 5
    frames = 20000

    threshold_rows = []
    for threshold in [0.8, 0.9, 1.0, 1.1]:
        ber, active_ber, mse, papr_o, papr_m, papr_r, mask_ratio = [], [], [], [], [], [], []
        for rep in range(repeats):
            summary = evaluate_models(model, threshold=threshold, frames=frames, seed=2000 + rep)
            item = summary["PolyNet-Volterra"]
            ber.append(item["BER"])
            active_ber.append(item["Active_BER"])
            mse.append(item["Masked_MSE"])
            papr_o.append(item["PAPR_Original_dB"])
            papr_m.append(item["PAPR_Masked_dB"])
            papr_r.append(item["PAPR_Reduction_dB"])
            mask_ratio.append(item["Mask_Ratio"])
        threshold_rows.append(compact_row("Threshold", threshold, ber, active_ber, mse, papr_o, papr_m, papr_r, mask_ratio, repeats, frames))
    write_rows(OUT_DIR / "threshold_sensitivity.csv", threshold_rows)

    modulation_rows = []
    for order in [4, 16, 64]:
        ber, active_ber, mse, papr_o, papr_m, papr_r, mask_ratio = [], [], [], [], [], [], []
        for rep in range(repeats):
            summary = evaluate_models(model, order=order, frames=frames, seed=3000 + rep)
            item = summary["PolyNet-Volterra"]
            ber.append(item["BER"])
            active_ber.append(item["Active_BER"])
            mse.append(item["Masked_MSE"])
            papr_o.append(item["PAPR_Original_dB"])
            papr_m.append(item["PAPR_Masked_dB"])
            papr_r.append(item["PAPR_Reduction_dB"])
            mask_ratio.append(item["Mask_Ratio"])
        modulation_rows.append(compact_row("Modulation_order", order, ber, active_ber, mse, papr_o, papr_m, papr_r, mask_ratio, repeats, frames))
    write_rows(OUT_DIR / "modulation_robustness.csv", modulation_rows)

    doppler_rows = []
    for k_max in [1, 2, 3, 4]:
        ber, active_ber, mse, papr_o, papr_m, papr_r, mask_ratio = [], [], [], [], [], [], []
        for rep in range(repeats):
            summary = evaluate_models(model, k_max=k_max, frames=frames, seed=4000 + rep)
            item = summary["PolyNet-Volterra"]
            ber.append(item["BER"])
            active_ber.append(item["Active_BER"])
            mse.append(item["Masked_MSE"])
            papr_o.append(item["PAPR_Original_dB"])
            papr_m.append(item["PAPR_Masked_dB"])
            papr_r.append(item["PAPR_Reduction_dB"])
            mask_ratio.append(item["Mask_Ratio"])
        doppler_rows.append(compact_row("Doppler_kmax", k_max, ber, active_ber, mse, papr_o, papr_m, papr_r, mask_ratio, repeats, frames))
    write_rows(OUT_DIR / "doppler_robustness.csv", doppler_rows)


def compact_row(name_key, name_value, ber, active_ber, mse, papr_o, papr_m, papr_r, mask_ratio, repeats, frames):
    ber_mean, ber_std = mean_std(ber)
    active_mean, active_std = mean_std(active_ber)
    mse_mean, mse_std = mean_std(mse)
    papr_o_mean, papr_o_std = mean_std(papr_o)
    papr_m_mean, papr_m_std = mean_std(papr_m)
    papr_r_mean, papr_r_std = mean_std(papr_r)
    mask_mean, mask_std = mean_std(mask_ratio)
    return {
        name_key: name_value,
        "Repeats": repeats,
        "Frames_per_repeat": frames,
        "BER_mean": f"{ber_mean:.8e}",
        "BER_std": f"{ber_std:.8e}",
        "Active_BER_mean": f"{active_mean:.8e}",
        "Active_BER_std": f"{active_std:.8e}",
        "Masked_MSE_mean": f"{mse_mean:.8e}",
        "Masked_MSE_std": f"{mse_std:.8e}",
        "PAPR_Original_dB_mean": f"{papr_o_mean:.4f}",
        "PAPR_Original_dB_std": f"{papr_o_std:.4f}",
        "PAPR_Masked_dB_mean": f"{papr_m_mean:.4f}",
        "PAPR_Masked_dB_std": f"{papr_m_std:.4f}",
        "PAPR_Reduction_dB_mean": f"{papr_r_mean:.4f}",
        "PAPR_Reduction_dB_std": f"{papr_r_std:.4f}",
        "Mask_Ratio_mean": f"{mask_mean:.8e}",
        "Mask_Ratio_std": f"{mask_std:.8e}",
    }


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def linear_flops(model):
    macs = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            macs += module.in_features * module.out_features
    return 2 * macs


def benchmark_model(model, batch_size=1, repeats=2000):
    x = torch.randn(batch_size, N, dtype=torch.cfloat)
    with torch.no_grad():
        for _ in range(100):
            model(x)
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        elapsed = time.perf_counter() - start
    frames = repeats * batch_size
    return elapsed * 1000.0 / frames, frames / elapsed


def complexity_latency_table():
    rows = []
    models = load_models(list(MODEL_SPECS.keys()))
    for name, model in models.items():
        lat1, thr1 = benchmark_model(model, batch_size=1, repeats=3000)
        lat256, thr256 = benchmark_model(model, batch_size=256, repeats=80)
        rows.append(
            {
                "Model": name,
                "Parameters": sum(p.numel() for p in model.parameters()),
                "Linear_FLOPs_per_frame": linear_flops(model),
                "Single_frame_CPU_latency_ms": f"{lat1:.6f}",
                "Batch256_CPU_latency_ms_per_frame": f"{lat256:.6f}",
                "Batch256_throughput_frames_per_s": f"{thr256:.2f}",
                "Torch": torch.__version__,
                "CPU": platform.processor() or platform.machine(),
                "Threads": torch.get_num_threads(),
            }
        )
    write_rows(OUT_DIR / "complexity_latency.csv", rows)


def main():
    print(f"Writing reviewer experiments to: {OUT_DIR}")
    print(f"Using weights from: {MODEL_DIR}")
    repeated_main_statistics()
    single_model_scenario_tables()
    complexity_latency_table()
    print("Done.")


if __name__ == "__main__":
    main()
