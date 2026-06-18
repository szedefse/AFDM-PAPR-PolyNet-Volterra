import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import reviewer_additional_experiments as base


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "results"
MODEL_DIR = OUT_DIR / "targeted_models"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
N = base.N
Q = base.Q
D = base.D

torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

MODEL_NAMES = ["DNN", "ResNet", "GatedMLP", "SE-ResNet", "PolyNet", "PolyNet-Volterra"]
EPOCHS = 150
BATCHES_PER_EPOCH = 50
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
MASK_THRESHOLD = 0.9
TRAINING_SNR_DB = 20.0


def write_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def input_for_mode(s_est, mask_idx, mode):
    if mode == "full-frame":
        return s_est
    if mode == "masked-only":
        return torch.where(mask_idx, s_est, torch.zeros_like(s_est))
    raise ValueError(f"Unknown input mode: {mode}")


def train_one(model_name, order=4, mode="full-frame", epochs=EPOCHS, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = base.MODEL_SPECS[model_name]().to(DEVICE)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.MSELoss()

    losses = []
    start = time.perf_counter()
    for _ in range(epochs):
        total_loss = 0.0
        used_batches = 0
        for _ in range(BATCHES_PER_EPOCH):
            batch = base.batch_observation(BATCH_SIZE, order, MASK_THRESHOLD, TRAINING_SNR_DB, k_max=1)
            mask_idx = batch["mask_idx"]
            if not mask_idx.any():
                continue

            inp = input_for_mode(batch["s_est"], mask_idx, mode)
            pred = model(inp)
            loss = criterion(
                torch.view_as_real(pred[mask_idx]),
                torch.view_as_real(batch["s_orig"][mask_idx]),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            used_batches += 1
        scheduler.step()
        losses.append(total_loss / max(1, used_batches))

    elapsed = time.perf_counter() - start
    model.eval()
    return model, losses, elapsed


def load_original_model(model_name):
    model = base.MODEL_SPECS[model_name]().to(DEVICE)
    state = torch.load(str(base.MODEL_DIR / f"{model_name}.pth"), map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_or_train(model_name, order, mode):
    if mode == "full-frame" and order == 4:
        return load_original_model(model_name), [], 0.0

    path = MODEL_DIR / f"{model_name}_M{order}_{mode.replace('-', '_')}.pth"
    meta_path = MODEL_DIR / f"{model_name}_M{order}_{mode.replace('-', '_')}_loss.csv"
    if path.exists():
        model = base.MODEL_SPECS[model_name]().to(DEVICE)
        model.load_state_dict(torch.load(str(path), map_location=DEVICE), strict=True)
        model.eval()
        losses = []
        if meta_path.exists():
            with open(meta_path, encoding="utf-8-sig", newline="") as f:
                losses = [float(r["Loss"]) for r in csv.DictReader(f)]
        return model, losses, 0.0

    seed = 10000 + order * 10 + MODEL_NAMES.index(model_name) * 97 + (0 if mode == "full-frame" else 1000)
    model, losses, elapsed = train_one(model_name, order=order, mode=mode, seed=seed)
    torch.save(model.state_dict(), str(path))
    write_rows(meta_path, [{"Epoch": i + 1, "Loss": f"{loss:.8e}"} for i, loss in enumerate(losses)])
    return model, losses, elapsed


def init_metrics():
    return {
        "err": 0,
        "symbols": 0,
        "active_err": 0,
        "active_symbols": 0,
        "mse_sum": 0.0,
        "masked_points": 0,
    }


def update_metrics(metrics, model, batch, order, mode):
    mask_idx = batch["mask_idx"]
    s_est = batch["s_est"]
    inp = input_for_mode(s_est, mask_idx, mode)
    with torch.no_grad():
        pred = model(inp)

    s_out = s_est.clone()
    if mask_idx.any():
        s_out[mask_idx] = pred[mask_idx]
        diff = pred[mask_idx] - batch["s_orig"][mask_idx]
        metrics["mse_sum"] += float((diff.real**2 + diff.imag**2).sum().item())
        metrics["masked_points"] += int(mask_idx.sum().item())

    x_est = torch.matmul(s_out, base.afdm.TRANSFORM_MATRIX.to(DEVICE).conj())
    data_est = x_est[:, 1 + Q : 1 + Q + D]
    demod = base.qam_demod(data_est, order)
    errors = demod != batch["data"]
    metrics["err"] += int(errors.sum().item())
    metrics["symbols"] += int(errors.numel())

    active = mask_idx.any(dim=1)
    if active.any():
        active_errors = errors[active]
        metrics["active_err"] += int(active_errors.sum().item())
        metrics["active_symbols"] += int(active_errors.numel())


def evaluate_one(model, order=4, mode="full-frame", frames=10000, batch_size=500, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    metrics = init_metrics()
    papr_orig_values = []
    papr_masked_values = []
    mask_count = 0
    total_count = 0

    for start in range(0, frames, batch_size):
        current_batch = min(batch_size, frames - start)
        batch = base.batch_observation(current_batch, order, MASK_THRESHOLD, TRAINING_SNR_DB, k_max=1)
        papr_orig_values.append(base.papr_db(batch["s_orig"]).detach())
        papr_masked_values.append(base.papr_db(batch["s_masked"]).detach())
        mask_count += int(batch["mask_idx"].sum().item())
        total_count += int(batch["mask_idx"].numel())
        update_metrics(metrics, model, batch, order, mode)

    ber = metrics["err"] / metrics["symbols"]
    active_ber = metrics["active_err"] / metrics["active_symbols"] if metrics["active_symbols"] else 0.0
    mse = metrics["mse_sum"] / metrics["masked_points"] if metrics["masked_points"] else 0.0
    papr_orig = torch.cat(papr_orig_values).mean().item()
    papr_masked = torch.cat(papr_masked_values).mean().item()
    return {
        "BER": ber,
        "Active_BER": active_ber,
        "Masked_MSE": mse,
        "PAPR_Original_dB": papr_orig,
        "PAPR_Masked_dB": papr_masked,
        "PAPR_Reduction_dB": papr_orig - papr_masked,
        "Mask_Ratio": mask_count / total_count if total_count else 0.0,
        "Symbols": metrics["symbols"],
        "Errors": metrics["err"],
    }


def mean_std(values):
    arr = np.array(values, dtype=float)
    if len(arr) <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def repeated_eval_row(label_fields, model, order, mode, repeats=3, frames=10000, seed0=5000, train_seconds=0.0, final_loss=None):
    collected = {k: [] for k in ["BER", "Active_BER", "Masked_MSE", "PAPR_Reduction_dB", "Mask_Ratio"]}
    for rep in range(repeats):
        out = evaluate_one(model, order=order, mode=mode, frames=frames, seed=seed0 + rep)
        for key in collected:
            collected[key].append(out[key])
    row = dict(label_fields)
    row.update(
        {
            "Repeats": repeats,
            "Frames_per_repeat": frames,
            "BER_mean": f"{mean_std(collected['BER'])[0]:.8e}",
            "BER_std": f"{mean_std(collected['BER'])[1]:.8e}",
            "Active_BER_mean": f"{mean_std(collected['Active_BER'])[0]:.8e}",
            "Active_BER_std": f"{mean_std(collected['Active_BER'])[1]:.8e}",
            "Masked_MSE_mean": f"{mean_std(collected['Masked_MSE'])[0]:.8e}",
            "Masked_MSE_std": f"{mean_std(collected['Masked_MSE'])[1]:.8e}",
            "PAPR_Reduction_dB_mean": f"{mean_std(collected['PAPR_Reduction_dB'])[0]:.4f}",
            "PAPR_Reduction_dB_std": f"{mean_std(collected['PAPR_Reduction_dB'])[1]:.4f}",
            "Mask_Ratio_mean": f"{mean_std(collected['Mask_Ratio'])[0]:.8e}",
            "Mask_Ratio_std": f"{mean_std(collected['Mask_Ratio'])[1]:.8e}",
            "Training_seconds_observed": f"{train_seconds:.2f}",
            "Final_training_loss": "" if final_loss is None else f"{final_loss:.8e}",
        }
    )
    return row


def modulation_retraining_experiment():
    rows = []
    settings = [
        ("QPSK", 4),
        ("16-QAM", 16),
        ("64-QAM", 64),
    ]
    for label, order in settings:
        model, losses, train_seconds = load_or_train("PolyNet-Volterra", order, "full-frame")
        rows.append(
            repeated_eval_row(
                {
                    "Modulation": label,
                    "Training_setting": "modulation-specific training",
                    "Model": "PolyNet-Volterra",
                },
                model,
                order,
                "full-frame",
                repeats=3,
                frames=10000,
                seed0=6000 + order,
                train_seconds=train_seconds,
                final_loss=losses[-1] if losses else None,
            )
        )
    write_rows(OUT_DIR / "modulation_retraining.csv", rows)


def full_frame_input_ablation():
    rows = []
    for model_name in MODEL_NAMES:
        full_model = load_original_model(model_name)
        rows.append(
            repeated_eval_row(
                {"Model": model_name, "Input_setting": "full-frame input"},
                full_model,
                4,
                "full-frame",
                repeats=3,
                frames=10000,
                seed0=7000 + MODEL_NAMES.index(model_name) * 10,
            )
        )

        masked_model, losses, train_seconds = load_or_train(model_name, 4, "masked-only")
        rows.append(
            repeated_eval_row(
                {"Model": model_name, "Input_setting": "masked-only input"},
                masked_model,
                4,
                "masked-only",
                repeats=3,
                frames=10000,
                seed0=8000 + MODEL_NAMES.index(model_name) * 10,
                train_seconds=train_seconds,
                final_loss=losses[-1] if losses else None,
            )
        )
    write_rows(OUT_DIR / "full_frame_ablation.csv", rows)


def main():
    print(f"Targeted experiments writing to {OUT_DIR}")
    modulation_retraining_experiment()
    full_frame_input_ablation()
    print("Done.")


if __name__ == "__main__":
    main()
