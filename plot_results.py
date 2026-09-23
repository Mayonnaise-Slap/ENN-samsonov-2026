import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import equations as eq

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MEASUREMENTS_CSV = RESULTS_DIR / "measurements.csv"
THETA_JSON = RESULTS_DIR / "theta.json"

METRICS = [
    ("latency", "Latency", "s", eq.latency),
    ("memory", "Memory", "bytes", eq.memory),
    ("energy", "Energy", "J", eq.energy),
]

VRAM_CAPACITY_BYTES = 15 * 2**30  # the Colab T4 used for measurement


def load_measurements() -> list[dict]:
    with open(MEASUREMENTS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["S"] = float(row["S"])
        row["B"] = float(row["B"])
        row["is_validation"] = str(row.get("is_validation", "")).strip().lower() in ("true", "1")
        for column in ("latency", "memory", "energy"):
            try:
                row[column] = float(row[column])
            except (TypeError, ValueError):
                row[column] = None  # OOM or blank
    return rows


def load_theta() -> dict:
    return json.loads(THETA_JSON.read_text())


def predict(metric: str, fn, s, b, theta: dict):
    return fn(s, b) if metric == "memory" else fn(s, b, theta[metric])


def _predict_at(metric, fn, x_key, x, fixed, theta):
    return predict(metric, fn, x, fixed, theta) if x_key == "S" else predict(metric, fn, fixed, x, theta)


def plot_vs(rows, theta, x_key: str, fixed_key: str, fixed_values, filename: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (metric, name, unit, fn) in zip(axes, METRICS):
        any_oom = False
        for fixed in fixed_values:
            group = [r for r in rows if r[fixed_key] == fixed]
            if not group:
                continue
            x_all = np.array([r[x_key] for r in group])
            have_value = np.array([r[metric] is not None for r in group])

            x_line = np.geomspace(x_all.min(), x_all.max(), 100)
            y_line = _predict_at(metric, fn, x_key, x_line, fixed, theta)
            (line,) = ax.plot(x_line, y_line, label=f"{fixed_key}={fixed:g}")

            if have_value.any():
                y = np.array([r[metric] for r in group if r[metric] is not None])
                ax.scatter(x_all[have_value], y, color=line.get_color())

            if (~have_value).any():
                any_oom = True
                x_oom = x_all[~have_value]
                y_oom = _predict_at(metric, fn, x_key, x_oom, fixed, theta)
                ax.scatter(x_oom, y_oom, color=line.get_color(), marker="x", s=80, zorder=5)

        if any_oom:
            ax.scatter([], [], marker="x", color="gray", s=80, label="OOM (predicted)")
        if metric == "memory":
            ax.axhline(VRAM_CAPACITY_BYTES, color="gray", linestyle="--", label="15 GB VRAM")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("image size S" if x_key == "S" else "batch size B")
        ax.set_ylabel(f"{name} ({unit})")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def plot_parity(rows, theta):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric, name, unit, fn) in zip(axes, METRICS):
        usable = [r for r in rows if r[metric] is not None]
        n_oom = len(rows) - len(usable)
        s = np.array([r["S"] for r in usable])
        b = np.array([r["B"] for r in usable])
        measured = np.array([r[metric] for r in usable])
        is_validation = np.array([r["is_validation"] for r in usable])
        predicted = predict(metric, fn, s, b, theta)

        for mask, split in ((~is_validation, "calibration"), (is_validation, "validation")):
            if mask.any():
                ax.scatter(measured[mask], predicted[mask], label=split)
        lo, hi = measured.min(), measured.max()
        ax.plot([lo, hi], [lo, hi], "k--")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"measured {name} ({unit})")
        ax.set_ylabel(f"predicted {name} ({unit})")
        ax.set_title(f"{name} ({n_oom} OOM excluded)" if n_oom else name)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "parity.png", bbox_inches="tight")
    plt.close(fig)


def print_error_summary(rows, theta):
    for metric, name, _unit, fn in METRICS:
        usable = [r for r in rows if r[metric] is not None]
        s = np.array([r["S"] for r in usable])
        b = np.array([r["B"] for r in usable])
        measured = np.array([r[metric] for r in usable])
        is_validation = np.array([r["is_validation"] for r in usable])
        predicted = predict(metric, fn, s, b, theta)
        rel_err = np.abs(predicted - measured) / measured
        for mask, split in ((~is_validation, "calibration"), (is_validation, "validation")):
            if mask.any():
                print(f"{name:<10} {split:<12} median |rel err| = {np.median(rel_err[mask]) * 100:6.2f}%  (n={mask.sum()})")

    oom_rows = [r for r in rows if r["memory"] is None]
    if oom_rows:
        print(f"\nOOM configs ({len(oom_rows)}) -- what memory() predicted for them:")
        for r in oom_rows:
            predicted_gib = eq.memory(r["S"], r["B"]) / 2**30
            print(f"  S={r['S']:.0f} B={r['B']:.0f}: predicted memory = {predicted_gib:.2f} GiB")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_measurements()
    theta = load_theta()

    plot_vs(rows, theta, x_key="S", fixed_key="B", fixed_values=[1, 8, 64, 256, 512], filename="vs_image_size.png")
    plot_vs(rows, theta, x_key="B", fixed_key="S", fixed_values=[32, 128, 256, 512, 1024], filename="vs_batch.png")
    plot_parity(rows, theta)
    print_error_summary(rows, theta)
    print(f"wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
