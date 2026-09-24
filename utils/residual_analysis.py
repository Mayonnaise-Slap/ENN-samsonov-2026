import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import equations as eq

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MEASUREMENTS_CSV = RESULTS_DIR / "measurements.csv"


def _load_memory_points():
    s, b, measured = [], [], []
    with open(MEASUREMENTS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                m = float(row["memory"])
            except (TypeError, ValueError):
                continue  # OOM or blank
            s.append(float(row["S"]))
            b.append(float(row["B"]))
            measured.append(m)
    return np.array(s), np.array(b), np.array(measured)


def analyze_memory():
    s, b, measured = _load_memory_points()
    predicted = eq.memory(s, b)

    log_measured = np.log(measured)
    log_predicted = np.log(predicted)
    fit = stats.linregress(log_predicted, log_measured)

    log_err = log_measured - log_predicted
    rmsle = float(np.sqrt(np.mean(log_err ** 2)))

    print(f"log-log: slope={fit.slope:.3f} stderr={fit.stderr:.3f} r={fit.rvalue:.4f}")
    print(f"RMSLE = {rmsle:.4f}: err off by {float(np.exp(rmsle)):.2f} times on avg")
    print(f"log-error: mean={log_err.mean():.4f}  std={log_err.std():.4f}  (n={len(measured)})")


if __name__ == "__main__":
    analyze_memory()
