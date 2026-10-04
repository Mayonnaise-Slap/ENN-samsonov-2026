import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MAX_SAFE_LR_SLOPE = 1e-3
ESTIMATORS = {
    "lr_estimator": ["weights s1, dense", "weights s1, /10"],
    "lr_estimator_v2": ["weights s1, dense", "grads s1, dense", "weights s1, /10", "weights s10, /10", "grads s10, /10"],
}
PAPER = {
    "c2": {"1cycle": 92.4, "PC-LR long": 91.2},
    "c11": {"inv, 85 epochs": (99.03, 0.04), "1cycle, 12 epochs": (99.25, 0.03)},
}


def load_runs():
    runs = []
    for path in sorted(RESULTS.glob("*/*.json")):
        run = json.loads(path.read_text())
        run["history"] = pd.DataFrame(run["history"])
        run["losses"] = np.load(path.with_name(path.stem + ".losses.npy"))
        run["group"] = run["name"].rsplit("_seed", 1)[0]
        runs.append(run)
    return runs


def select(runs, exp, **match):
    return [r for r in runs if r["cfg"]["exp"] == exp and all(r["cfg"].get(k) == v for k, v in match.items())]


def smooth(values, window=5):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def stack_history(group, column):
    frames = [r["history"].set_index("it")[column].rename(r["cfg"]["seed"]) for r in group]
    table = pd.concat(frames, axis=1)
    return table.index.to_numpy(), table.mean(axis=1).to_numpy(), table.std(axis=1, ddof=1).to_numpy()


def smooth_ramp(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x / (x * x + (1 - x) ** 2)


def one_cycle_lr(total, lr_min, lr_max, cycle_frac=0.9, final_div=100.0, first_scaled=False):
    pct = cycle_frac / 2
    ends = [pct * total - 1, 2 * pct * total - 2, total - 1]
    values = [lr_min, lr_max, lr_min, lr_min / final_div]
    steps = np.arange(total, dtype=float)
    lr = np.empty(total)
    done = np.zeros(total, dtype=bool)
    start = 0.0
    for i, end in enumerate(ends):
        m = ~done & ((steps <= end) | (i == len(ends) - 1))
        lr[m] = values[i] + (values[i + 1] - values[i]) * (steps[m] - start) / (end - start)
        done |= m
        start = end
    slope = (lr_max - lr_min) / (cycle_frac * total / 2)
    if slope > MAX_SAFE_LR_SLOPE:
        warmup = max(100, math.ceil(2 * lr_min / MAX_SAFE_LR_SLOPE))
        ramp = smooth_ramp(steps / warmup)
        ramp[0] = 0.0 if first_scaled else 1.0
        lr = lr * ramp
    return lr


def lr_schedule(run):
    cfg = run["cfg"]
    total = cfg["iters"]
    steps = np.arange(total, dtype=float)
    if cfg["sched"] == "range":
        start = 1e-4
        return cfg["lr_hi"] * (start + (1 - start) * steps / (total - 1))
    if cfg["sched"] in ("const", "const_m0"):
        return np.full(total, cfg["lr0"])
    if cfg["sched"] == "pc":
        milestones = np.array([int(0.5 * total), int(0.75 * total)])
        return cfg["lr0"] * 0.1 ** (steps[:, None] >= milestones[None, :]).sum(axis=1)
    if cfg["sched"] == "inv":
        return cfg["lr0"] * (1 + 1e-4 * steps) ** -0.75
    if cfg["sched"] == "1cycle" and cfg["model"] == "lenet":
        return one_cycle_lr(total, 0.01, cfg["lr_hi"], cycle_frac=10 / 12)
    if cfg["sched"] == "1cycle":
        return one_cycle_lr(total, 0.1, cfg["lr_hi"], first_scaled=cfg["exp"] == "lr_estimator_v2")
    raise ValueError(cfg["sched"])


def check_schedules(runs):
    rows = []
    for run in runs:
        lr = lr_schedule(run)
        h = run["history"]
        logged = h["lr"].to_numpy()
        rebuilt = lr[h["it"].to_numpy() - 1]
        rows.append(dict(run=run["name"], max_rel_err=float(np.max(np.abs(rebuilt - logged) / np.maximum(logged, 1e-12)))))
    table = pd.DataFrame(rows)
    assert (table["max_rel_err"] < 1e-6).all(), table
    return table


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)
    print("saved", FIGURES / name)


def band(ax, x, mean, std, label, **kw):
    line, = ax.plot(x, mean, label=label, **kw)
    if np.isfinite(std).any():
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)


def runs_table(runs):
    rows = []
    for group in pd.unique(pd.Series([r["group"] for r in runs])):
        rs = [r for r in runs if r["group"] == group]
        acc = np.array([r["final"]["test_acc"] for r in rs])
        rows.append(dict(
            group=group, seeds=len(rs), status=",".join(sorted({r["status"] for r in rs})),
            final_test_acc=acc.mean(), final_test_acc_std=acc.std(ddof=1) if len(rs) > 1 else np.nan,
            best_eval_acc=np.mean([r["final"]["best_eval_acc"] for r in rs]),
            mean_last3_eval_acc=np.mean([r["final"]["mean_last3_eval_acc"] for r in rs]),
            n_flags=sum(len(r["flags"]) for r in rs),
            train_min=np.mean([r["final"]["t_train"] for r in rs]) / 60,
            s_per_it=np.mean([r["final"]["t_train"] / r["cfg"]["iters"] for r in rs]),
        ))
    return pd.DataFrame(rows)


def compare(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = dict(mean_a=a.mean(), mean_b=b.mean(), diff=a.mean() - b.mean(), n_a=len(a), n_b=len(b))
    if len(a) > 1 and len(b) > 1:
        out["std_a"], out["std_b"] = a.std(ddof=1), b.std(ddof=1)
        out["pooled_std"] = math.sqrt((out["std_a"] ** 2 + out["std_b"] ** 2) / 2)
        out["welch_t"], out["welch_p"] = stats.ttest_ind(a, b, equal_var=False)
    return out


def range_test_figures(runs):
    group = select(runs, "range_test", model="resnet", depth=56)
    if not group:
        return {}
    _, lr, _ = stack_history(group, "lr")
    _, acc, acc_std = stack_history(group, "test_acc")
    _, acc_tm, acc_tm_std = stack_history(group, "test_acc_trainmode")
    _, train_loss, train_loss_std = stack_history(group, "train_loss")
    _, test_loss, test_loss_std = stack_history(group, "test_loss")
    acc_s, acc_tm_s = smooth(acc), smooth(acc_tm)
    peak = acc_s.max()
    lr_max = lr[acc_s >= 0.9 * peak].max()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lr, acc, alpha=0.3, label="test accuracy (eval mode)")
    band(ax, lr, acc_s, smooth(acc_std), "eval mode, rolling mean 5")
    band(ax, lr, acc_tm_s, smooth(acc_tm_std), "BN batch statistics, rolling mean 5")
    ax.axhline(0.9 * peak, color="gray", ls=":", label="90% of peak")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"LR range test, ResNet-56, CIFAR-10, seeds: {len(group)}")
    ax.legend()
    save(fig, "c1_resnet56_range_test_accuracy.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    band(ax, lr, train_loss, train_loss_std, "train loss (100-iteration window)")
    band(ax, lr, smooth(test_loss), smooth(test_loss_std), "test loss, rolling mean 5")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.set_title("LR range test, ResNet-56: train and test loss")
    ax.legend()
    save(fig, "c1_resnet56_range_test_loss.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lr, 100 * (acc_tm - acc), alpha=0.3, label="per eval")
    ax.plot(lr, smooth(100 * (acc_tm - acc)), label="rolling mean 5")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Accuracy gap, pp")
    ax.set_title("BN batch statistics minus running statistics, test accuracy")
    ax.legend()
    save(fig, "c1_resnet56_bn_gap.png")

    bands = []
    for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.01)):
        m = (lr >= lo) & (lr < hi)
        bands.append(dict(lr_band=f"{lo}-{min(hi, 3.0)}", eval_acc=acc_s[m].mean(), trainmode_acc=acc_tm_s[m].mean(),
                          gap_pp=100 * (acc_tm_s[m] - acc_s[m]).mean(), train_loss=train_loss[m].mean(),
                          test_loss=smooth(test_loss)[m].mean()))
    return dict(
        seeds=len(group), peak_acc=peak, lr_at_peak=lr[acc_s.argmax()], peak_acc_trainmode=acc_tm_s.max(),
        lr_at_peak_trainmode=lr[acc_tm_s.argmax()], acc_at_lr_hi=acc_s[-1], acc_at_lr_hi_trainmode=acc_tm_s[-1],
        lr_max=lr_max, flags=[f for r in group for f in r["flags"]], bands=pd.DataFrame(bands),
    )


def lenet_range_test_figure(runs):
    group = select(runs, "range_test", model="lenet")
    if not group:
        return {}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    diverged = []
    for r in group:
        h = r["history"]
        ax.plot(h["lr"], h["test_acc"], marker="o", label=f"seed {r['cfg']['seed']}")
        if r["status"] == "diverged":
            diverged.append(float(h["lr"].iloc[-1]))
            ax.axvline(h["lr"].iloc[-1], color="gray", ls="--")
    ax.axvspan(0.01, 0.1, color="gray", alpha=0.15, label="1cycle range, paper Table 2")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test accuracy")
    ax.set_title("LR range test 0 -> 4, LeNet, MNIST")
    ax.legend()
    save(fig, "c11_lenet_range_test.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in group:
        lr = lr_schedule(r)
        loss = r["losses"]
        ax.plot(lr[:len(loss)], smooth(loss, 20), label=f"seed {r['cfg']['seed']}")
    ax.set_yscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Train loss, rolling mean 20")
    ax.set_title("LR range test, LeNet: per-iteration train loss")
    ax.legend()
    save(fig, "c11_lenet_range_test_loss.png")
    return dict(seeds=len(group), peak_acc=max(r["history"]["test_acc"].max() for r in group), diverged_at_lr=diverged)


def c2_figures(runs):
    groups = {
        "1cycle 0.1-3, 3k": select(runs, "c2", sched="1cycle"),
        "PC-LR 0.35, 3k": select(runs, "c2", sched="pc", iters=3000),
        "PC-LR 0.35, 24k": select(runs, "c2", sched="pc", iters=24000),
    }
    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return {}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in groups.items():
        it, mean, std = stack_history(group, "test_acc")
        band(ax, it, mean, std, label)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test accuracy (2k subset)")
    ax.set_title("ResNet-56, CIFAR-10: 1cycle vs PC-LR")
    ax.legend()
    save(fig, "c2_test_accuracy.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in groups.items():
        it, mean, std = stack_history(group, "test_acc")
        band(ax, it, mean, std, label)
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test accuracy (2k subset)")
    ax.set_title("ResNet-56, CIFAR-10: 1cycle vs PC-LR, log iteration axis")
    ax.legend()
    save(fig, "c2_test_accuracy_log.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in groups.items():
        ax.plot(np.arange(group[0]["cfg"]["iters"]), lr_schedule(group[0]), label=label)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedules")
    ax.legend()
    save(fig, "c2_lr_schedules.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in groups.items():
        losses = np.vstack([r["losses"] for r in group])
        ax.plot(np.arange(1, losses.shape[1] + 1), smooth(np.nanmean(losses, axis=0), 50), label=label)
    ax.axhline(math.log(10), color="gray", ls=":", label="ln 10")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Train loss, rolling mean 50")
    ax.set_title("Per-iteration train loss")
    ax.legend()
    save(fig, "c2_train_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in groups.items():
        it, train, _ = stack_history(group, "train_acc")
        _, test, _ = stack_history(group, "test_acc")
        ax.plot(it, 100 * (train - test), label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Train window accuracy minus test accuracy, pp")
    ax.set_title("Generalization gap during training")
    ax.legend()
    save(fig, "c2_generalization_gap.png")

    final = {label: [r["final"]["test_acc"] for r in group] for label, group in groups.items()}
    out = dict(final=final, flags={label: [r["flags"] for r in group] for label, group in groups.items()})
    if {"1cycle 0.1-3, 3k", "PC-LR 0.35, 24k"} <= final.keys():
        out["1cycle_3k_vs_pc_24k"] = compare(final["1cycle 0.1-3, 3k"], final["PC-LR 0.35, 24k"])
    if {"1cycle 0.1-3, 3k", "PC-LR 0.35, 3k"} <= final.keys():
        out["1cycle_3k_vs_pc_3k"] = compare(final["1cycle 0.1-3, 3k"], final["PC-LR 0.35, 3k"])
    out["best_eval"] = {label: [r["final"]["best_eval_acc"] for r in group] for label, group in groups.items()}
    return out


def c11_figures(runs):
    groups = {"inv, 85 epochs": select(runs, "c11", sched="inv"), "1cycle, 12 epochs": select(runs, "c11", sched="1cycle")}
    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return {}
    its_per_epoch = next(iter(groups.values()))[0]["cfg"]["eval_every"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, group in groups.items():
        it, mean, std = stack_history(group, "test_acc")
        band(ax, it / its_per_epoch, 100 * (1 - mean), 100 * std, label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test error, %")
    ax.set_title("LeNet, MNIST: test error")
    ax.legend()
    save(fig, "c11_test_error.png")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i, (label, group) in enumerate(groups.items()):
        acc = 100 * np.array([r["final"]["test_acc"] for r in group])
        ax.scatter(np.full(len(acc), i), acc, label=f"{label}, ours")
        paper_mean, paper_std = PAPER["c11"][label]
        ax.errorbar(i + 0.15, paper_mean, yerr=paper_std, fmt="s", color="black", capsize=4,
                    label="paper (mean ± std)" if i == 0 else None)
    ax.set_xticks(range(len(groups)), list(groups))
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylabel("Final test accuracy, %")
    ax.set_title("LeNet, MNIST: final test accuracy per seed")
    ax.legend()
    save(fig, "c11_final_accuracy.png")

    final = {label: [r["final"]["test_acc"] for r in group] for label, group in groups.items()}
    out = dict(final=final)
    if len(final) == 2:
        out["1cycle_12_vs_inv_85"] = compare(final["1cycle, 12 epochs"], final["inv, 85 epochs"])
    return out


def estimator_table(runs):
    rows = []
    for exp, names in ESTIMATORS.items():
        for r in select(runs, exp):
            lr = lr_schedule(r)
            total = r["cfg"]["iters"]
            for name, hook in zip(names, r["hooks"]):
                t = np.array(hook["trace"], dtype=float)
                k, est, ema = t[:, 0].astype(int), t[:, 1], t[:, 2]

                def med(lo, hi, values=ema):
                    m = (k >= lo) & (k < hi)
                    return float(np.median(values[m])) if m.any() else np.nan

                rows.append(dict(
                    exp=exp, sched=r["cfg"]["sched"], seed=r["cfg"]["seed"], variant=name,
                    early=med(0, 300), first_half=med(0, total // 2), mid=med(total // 2, int(0.9 * total)),
                    last_10pct=med(int(0.9 * total), total), est_over_lr=float(np.median(est / np.maximum(lr[k], 1e-12))),
                    spearman_est_lr=float(stats.spearmanr(est, lr[k]).statistic) if np.ptp(lr[k]) > 0 else np.nan,
                ))
    return pd.DataFrame(rows)


def estimator_figures(runs):
    labels = {"pc": "PC-LR 0.1", "1cycle": "1cycle 0.1-3", "const": "constant 0.1, momentum 0.9",
              "const_m0": "constant 0.1, momentum 0"}
    for exp, names in ESTIMATORS.items():
        for r in select(runs, exp):
            lr = lr_schedule(r)
            total = r["cfg"]["iters"]
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
            for name, hook in zip(names, r["hooks"]):
                t = np.array(hook["trace"], dtype=float)
                ax = axes[0] if "dense" in name else axes[1]
                line, = ax.plot(t[:, 0], t[:, 2], label=f"{name}, EMA")
                ax.plot(t[:, 0], t[:, 1], color=line.get_color(), alpha=0.2, lw=0.7)
            axes[0].plot(np.arange(300), lr[:300], color="black", label="LR used")
            axes[1].plot(np.arange(total), lr, color="black", label="LR used")
            axes[0].set_title("every iteration, first 300")
            axes[1].set_title("every 10 iterations, whole run")
            for ax in axes:
                ax.set_yscale("log")
                ax.set_xlabel("Iteration")
                ax.legend(fontsize=8)
            axes[0].set_ylabel("Learning rate")
            fig.suptitle(f"Eq. 8 estimate, ResNet-56, {labels[r['cfg']['sched']]} ({exp}, seed {r['cfg']['seed']})")
            save(fig, f"c9_{exp}_{r['cfg']['sched']}_seed{r['cfg']['seed']}.png")

    table = estimator_table(runs)
    v2 = table[table["exp"] == "lr_estimator_v2"]
    if not v2.empty:
        pivot = v2.groupby(["variant", "sched"])["est_over_lr"].mean().unstack("sched")
        pivot = pivot.reindex(index=ESTIMATORS["lr_estimator_v2"])
        fig, ax = plt.subplots(figsize=(9, 4.5))
        width = 0.8 / len(pivot.columns)
        x = np.arange(len(pivot.index))
        for i, sched in enumerate(pivot.columns):
            ax.bar(x + i * width - 0.4 + width / 2, pivot[sched], width, label=labels[sched])
        ax.set_xticks(x, pivot.index, rotation=15)
        ax.set_yscale("log")
        ax.set_ylabel("median estimate / LR used")
        ax.set_title("Eq. 8 estimate relative to the LR used, by variant and run")
        ax.legend(fontsize=8)
        save(fig, "c9_estimate_over_lr.png")
    return table


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    runs = load_runs()
    print(f"{len(runs)} runs")
    print(check_schedules(runs).to_string(index=False))

    summary = runs_table(runs)
    print(summary.round(4).to_string(index=False))

    c1 = range_test_figures(runs)
    lenet = lenet_range_test_figure(runs)
    c2 = c2_figures(runs)
    c11 = c11_figures(runs)
    c9 = estimator_figures(runs)

    print("\nC1", json.dumps({k: v for k, v in c1.items() if k != "bands"}, indent=1, default=float))
    if "bands" in c1:
        print(c1["bands"].round(4).to_string(index=False))
    print("\nLeNet range test", json.dumps(lenet, indent=1, default=float))
    print("\nC2", json.dumps(c2, indent=1, default=float))
    print("\nC11", json.dumps(c11, indent=1, default=float))
    print("\nC9")
    print(c9.round(4).to_string(index=False))

    FIGURES.mkdir(exist_ok=True)
    summary.to_csv(FIGURES / "runs_summary.csv", index=False)
    c9.to_csv(FIGURES / "c9_estimator_summary.csv", index=False)
    if "bands" in c1:
        c1["bands"].to_csv(FIGURES / "c1_range_test_bands.csv", index=False)


if __name__ == "__main__":
    main()
