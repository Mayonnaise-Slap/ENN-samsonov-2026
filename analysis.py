# -*- coding: utf-8 -*-
"""Analysis of the super-convergence reproduction

Runs locally on the result files produced by `samsonovenn2.py` on Colab:
copy `MyDrive/enn-superconvergence/results` into `./results`, then

    uv run analysis.py

Figures go to `./figures`.

# Environment
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)

"""# Loading"""

# %%

def load_runs(root=RESULTS):
    """All finished runs, keyed by run name."""
    runs = {}
    for path in sorted(root.rglob("*.json")):
        run = json.loads(path.read_text())
        if "cfg" in run:
            run["losses"] = np.load(path.with_name(path.stem + ".losses.npy"))
            run["history"] = pd.DataFrame(run["history"])
            runs[run["name"]] = run
    return runs


runs = load_runs()
table = pd.DataFrame(
    [{**run["cfg"], "status": run["status"], **run["final"]} for run in runs.values()],
    index=list(runs),
)
print(f"{len(runs)} runs in {RESULTS}")
if len(table):
    print(table[["exp", "model", "depth", "sched", "bs", "iters", "seed", "status", "test_acc", "best_eval_acc"]])

"""# Experiment 1: LR range test, ResNet-56 (C1, Fig. 2b and 8a)

Paper: test accuracy stays high for LR up to 3 (Fig. 2b), and between LR ≈ 0.2
and 2 the train loss goes up while the test loss goes down (Fig. 8a).

Reproduced if the run doesn't collapse up to LR 3 and the smoothed test accuracy
at LR 3 is at least 90% of its peak.
"""

# %%

for run in (r for r in runs.values() if r["cfg"]["exp"] == "range_test" and r["cfg"]["depth"] == 56):
    hist = run["history"]
    smooth = hist[["test_acc", "test_acc_trainmode", "test_loss"]].rolling(5, center=True, min_periods=1).mean()
    peak = smooth["test_acc"].max()
    lr_at_peak = hist.loc[smooth["test_acc"].idxmax(), "lr"]
    lr_max = hist.loc[smooth["test_acc"] >= 0.9 * peak, "lr"].max()
    acc_at_end = smooth["test_acc"].iloc[-1]
    c1 = run["status"] == "complete" and not run["flags"] and acc_at_end >= 0.9 * peak

    # LR of every iteration, interpolated from the logged evals, for the per-iteration train loss.
    iters = np.arange(1, len(run["losses"]) + 1)
    iter_lr = np.interp(iters, hist["it"], hist["lr"])
    train_loss = pd.Series(run["losses"]).rolling(len(iters) // len(hist), center=True, min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(hist["lr"], hist["test_acc"], color="tab:blue", alpha=0.3, label="eval")
    ax.plot(hist["lr"], smooth["test_acc"], color="tab:blue", label="rolling mean 5")
    ax.plot(hist["lr"], smooth["test_acc_trainmode"], color="tab:orange", label="BN batch statistics, rolling mean 5")
    ax.axhline(0.9 * peak, color="gray", ls=":", label="90% of peak")
    ax.axvline(lr_max, color="gray", ls="--", label=f"lr_max = {lr_max:.2f}")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"ResNet-{run['cfg']['depth']}, CIFAR-10, {run['cfg']['iters']:,} iterations")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.plot(iter_lr, train_loss, label="train loss")
    ax.plot(hist["lr"], smooth["test_loss"], label="test loss")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.set_title("Train and test loss vs LR")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / f"fig2b_range_test_r56_s{run['cfg']['seed']}.png", dpi=150)
    plt.show()

    print(f"\n{run['name']}")
    print(f"  status {run['status']}, flags {run['flags']}, non-finite steps {hist['nonfinite'].sum()}")
    print(f"  peak smoothed acc {peak:.3f} at LR {lr_at_peak:.2f}")
    print(f"  smoothed acc at LR {hist['lr'].iloc[-1]:.2f}: {acc_at_end:.3f} ({acc_at_end / peak:.0%} of peak)")
    print(f"  lr_max: {lr_max:.2f}")
    print(f"  train time {run['final']['t_train'] / 60:.1f} min, eval time {run['final']['t_eval'] / 60:.1f} min")
    print("\n| claim | paper | ours | verdict |")
    print("|---|---|---|---|")
    print(
        f"| C1 | acc stays high up to LR 3 (Fig. 2b) | smoothed acc at LR 3 = {acc_at_end / peak:.0%} "
        f"of peak ({peak:.3f} at LR {lr_at_peak:.2f}) | {'reproduced' if c1 else 'not reproduced'} |"
    )
