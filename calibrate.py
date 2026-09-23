import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from equations import bytes_moved, flops, latency

RESULTS_DIR = Path(__file__).parent / "results"
MEASUREMENTS_CSV = RESULTS_DIR / "measurements.csv"
THETA_JSON = RESULTS_DIR / "theta.json"


def _load_measurements(path: Path = MEASUREMENTS_CSV) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _is_validation(row: dict) -> bool:
    return str(row.get("is_validation", "")).strip().lower() in ("true", "1")


def _numeric_column(rows: list[dict], column: str):
    s, b, y = [], [], []
    for row in rows:
        value = row.get(column, "")
        try:
            y_val = float(value)
        except (TypeError, ValueError):
            continue
        s.append(float(row["S"]))
        b.append(float(row["B"]))
        y.append(y_val)
    if not y:
        return None
    return np.array(s), np.array(b), np.array(y)


def _fit_log_space(residual_fn, x0: np.ndarray) -> np.ndarray:
    result = least_squares(lambda log_p: residual_fn(np.exp(log_p)), np.log(x0))
    return np.exp(result.x)


def fit_latency_theta(rows: list[dict]) -> dict:
    data = _numeric_column(rows, "latency")
    if data is None:
        raise ValueError("No usable latency measurements to calibrate on")
    s, b, y = data

    f = flops(s, b)
    m = bytes_moved(s, b)

    def residuals(params):
        t_startup, flops_rate, bandwidth = params
        pred = t_startup + np.maximum(f / flops_rate, m / bandwidth)
        return (pred - y) / y

    x0 = np.array([y.min() / 2, f.mean() / y.mean(), m.mean() / y.mean()])
    t_startup, flops_rate, bandwidth = _fit_log_space(residuals, x0)
    return {"t_startup": t_startup, "flops_rate": flops_rate, "bandwidth": bandwidth}


def fit_energy_theta(rows: list[dict], latency_theta: dict) -> dict:
    data = _numeric_column(rows, "energy")
    if data is None:
        raise ValueError("No usable energy measurements to calibrate on")
    s, b, y = data

    f = flops(s, b)
    m = bytes_moved(s, b)
    t = latency(s, b, latency_theta)

    def residuals(params):
        p_idle, e_flop, e_byte = params
        pred = p_idle * t + e_flop * f + e_byte * m
        return (pred - y) / y

    x0 = np.array([y.mean() / t.mean(), y.mean() / (4 * f.mean()), y.mean() / (4 * m.mean())])
    p_idle, e_flop, e_byte = _fit_log_space(residuals, x0)
    return {"p_idle": p_idle, "e_flop": e_flop, "e_byte": e_byte, "latency": latency_theta}


def calibrate(measurements_csv: Path = MEASUREMENTS_CSV) -> dict:
    rows = _load_measurements(measurements_csv)
    calibration_rows = [row for row in rows if not _is_validation(row)]

    latency_theta = fit_latency_theta(calibration_rows)
    energy_theta = fit_energy_theta(calibration_rows, latency_theta)
    return {"latency": latency_theta, "energy": energy_theta}


if __name__ == "__main__":
    theta = calibrate()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(THETA_JSON, "w") as f:
        json.dump(theta, f, indent=2)
