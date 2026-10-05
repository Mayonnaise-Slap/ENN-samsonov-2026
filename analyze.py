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

EXP1 = FIGURES / "exp1_range_test"
EXP2 = FIGURES / "exp2_lr_estimator"
EXP2B = FIGURES / "exp2b_lr_estimator_v2"
EXP3 = FIGURES / "exp3_mnist"
EXP4 = FIGURES / "exp4_c2"

MAX_SAFE_LR_SLOPE = 1e-3
ESTIMATORS = {
    "lr_estimator": ["weights s1, dense", "weights s1, /10"],
    "lr_estimator_v2": ["weights s1, dense", "grads s1, dense", "weights s1, /10", "weights s10, /10", "grads s10, /10"],
}
SCHED_LABELS = {"pc": "PC-LR 0.1", "1cycle": "1cycle 0.1-3", "const": "constant 0.1, momentum 0.9",
                "const_m0": "constant 0.1, momentum 0"}
PAPER = {
    "c2": {"1cycle 0.1-3, 3k": (92.4, None), "PC-LR 0.35, 24k": (91.2, None)},
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


def by_group(runs):
    groups = {}
    for r in runs:
        groups.setdefault(r["group"], []).append(r)
    return groups


def seeds(group):
    return [r["cfg"]["seed"] for r in group]


def smooth(values, window=5):
    return pd.Series(values).rolling(window, center=True, min_periods=1).mean().to_numpy()


def describe(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    out = dict(n=len(v), mean=np.nan, std=np.nan, sem=np.nan, ci95_low=np.nan, ci95_high=np.nan, min=np.nan, max=np.nan)
    if len(v) == 0:
        return out
    out.update(mean=v.mean(), min=v.min(), max=v.max())
    if len(v) > 1:
        std = v.std(ddof=1)
        sem = std / math.sqrt(len(v))
        half = stats.t.ppf(0.975, len(v) - 1) * sem
        out.update(std=std, sem=sem, ci95_low=v.mean() - half, ci95_high=v.mean() + half)
    return out


def describe_frame(frame, by, columns):
    rows = []
    for key, sub in frame.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        for column in columns:
            rows.append(dict(zip(by, key)) | dict(metric=column) | describe(sub[column]))
    return pd.DataFrame(rows)


def stack(series_by_seed):
    table = pd.concat(series_by_seed, axis=1).sort_index()
    std = table.std(axis=1, ddof=1) if table.shape[1] > 1 else pd.Series(np.nan, index=table.index)
    return table.index.to_numpy(), table.mean(axis=1).to_numpy(), std.to_numpy()


def stack_history(group, column):
    return stack([r["history"].set_index("it")[column].rename(r["cfg"]["seed"]) for r in group])


def stack_losses(group, window):
    return stack([pd.Series(smooth(r["losses"], window), index=np.arange(1, len(r["losses"]) + 1)).rename(r["cfg"]["seed"])
                  for r in group])


def stack_trace(group, index, column):
    col = {"est": 1, "ema": 2}[column]
    series = []
    for r in group:
        t = np.array(r["hooks"][index]["trace"], dtype=float)
        series.append(pd.Series(t[:, col], index=t[:, 0].astype(int)).rename(r["cfg"]["seed"]))
    return stack(series)


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


def save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / name, dpi=150)
    plt.close(fig)
    print("saved", (directory / name).relative_to(ROOT))


def save_table(table, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    table.to_csv(directory / name, index=False)
    print("saved", (directory / name).relative_to(ROOT))


def band(ax, x, mean, std, label, **kw):
    line, = ax.plot(x, mean, label=label, **kw)
    if np.isfinite(std).any():
        ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)
    return line


def seeds_note(groups):
    counts = sorted({len(g) for g in groups})
    return "seeds: " + "/".join(str(c) for c in counts) + ", mean ± std"


def compare(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = dict(n_a=len(a), n_b=len(b), mean_a=a.mean(), mean_b=b.mean(), diff=a.mean() - b.mean())
    if len(a) > 1 and len(b) > 1:
        out["std_a"], out["std_b"] = a.std(ddof=1), b.std(ddof=1)
        out["pooled_std"] = math.sqrt((out["std_a"] ** 2 + out["std_b"] ** 2) / 2)
        out["diff_over_pooled_std"] = out["diff"] / out["pooled_std"] if out["pooled_std"] > 0 else np.nan
        test = stats.ttest_ind(a, b, equal_var=False)
        out["welch_t"], out["welch_p"] = float(test.statistic), float(test.pvalue)
    return out


def final_table(groups):
    rows = []
    for label, group in groups.items():
        for metric in ("test_acc", "test_loss", "best_eval_acc", "mean_last3_eval_acc", "t_train"):
            rows.append(dict(run=label, metric=metric, seeds=seeds(group)) | describe([r["final"][metric] for r in group]))
    return pd.DataFrame(rows)


def final_accuracy_bars(groups, paper, title, directory, name):
    labels = list(groups)
    x = np.arange(len(labels))
    width = 0.38
    ours = [describe(100 * np.array([r["final"]["test_acc"] for r in groups[label]])) for label in labels]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, [d["mean"] for d in ours], width, yerr=[0 if np.isnan(d["std"]) else d["std"] for d in ours],
           capsize=5, label="ours, mean ± std")
    for i, label in enumerate(labels):
        acc = 100 * np.array([r["final"]["test_acc"] for r in groups[label]])
        ax.scatter(np.full(len(acc), x[i] - width / 2), acc, color="black", s=14, zorder=3,
                   label="ours, seeds" if i == 0 else None)
    paper_x = [x[i] + width / 2 for i, label in enumerate(labels) if label in paper]
    paper_mean = [paper[label][0] for label in labels if label in paper]
    paper_std = [paper[label][1] or 0 for label in labels if label in paper]
    ax.bar(paper_x, paper_mean, width, yerr=paper_std, capsize=5, color="gray", label="paper, mean ± std")
    values = [d["mean"] for d in ours] + paper_mean
    span = max(values) - min(values)
    ax.set_ylim(min(values) - max(0.3, 0.25 * span), max(values) + max(0.15, 0.05 * span))
    ax.set_xticks(x, labels)
    ax.set_ylabel("Final test accuracy, %")
    ax.set_title(title)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8)
    save(fig, directory, name)


def range_readout(run):
    h = run["history"]
    lr = h["lr"].to_numpy()
    acc = smooth(h["test_acc"].to_numpy())
    out = dict(seed=run["cfg"]["seed"], peak_acc=acc.max(), lr_at_peak=lr[acc.argmax()], acc_at_lr_hi=acc[-1],
               acc_at_lr_hi_over_peak=acc[-1] / acc.max(), lr_max=lr[acc >= 0.9 * acc.max()].max(),
               n_flags=len(run["flags"]))
    if "test_acc_trainmode" in h:
        tm = smooth(h["test_acc_trainmode"].to_numpy())
        out.update(peak_acc_trainmode=tm.max(), lr_at_peak_trainmode=lr[tm.argmax()], acc_at_lr_hi_trainmode=tm[-1],
                   gap_pp_mean=100 * (tm - acc).mean())
    return out


def exp1(runs):
    group = select(runs, "range_test", model="resnet", depth=56)
    if not group:
        return
    note = seeds_note([group])
    _, lr, _ = stack_history(group, "lr")
    _, acc, acc_std = stack_history(group, "test_acc")
    _, acc_tm, acc_tm_std = stack_history(group, "test_acc_trainmode")
    _, train_loss, train_loss_std = stack_history(group, "train_loss")
    _, test_loss, test_loss_std = stack_history(group, "test_loss")
    peak = smooth(acc).max()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    band(ax, lr, smooth(acc), smooth(acc_std), "eval mode (running BN statistics)")
    band(ax, lr, smooth(acc_tm), smooth(acc_tm_std), "BN batch statistics")
    ax.axhline(0.9 * peak, color="gray", ls=":", label="90% of peak (eval mode)")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test accuracy, rolling mean 5")
    ax.set_title(f"LR range test, ResNet-56, CIFAR-10 ({note})")
    ax.legend()
    save(fig, EXP1, "range_test_accuracy.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    band(ax, lr, train_loss, train_loss_std, "train loss (100-iteration window)")
    band(ax, lr, smooth(test_loss), smooth(test_loss_std), "test loss, rolling mean 5")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Loss")
    ax.set_title(f"LR range test, ResNet-56: train and test loss ({note})")
    ax.legend()
    save(fig, EXP1, "range_test_loss.png")

    _, gap, gap_std = stack([(100 * (r["history"]["test_acc_trainmode"] - r["history"]["test_acc"]))
                             .set_axis(r["history"]["it"]).rename(r["cfg"]["seed"]) for r in group])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    band(ax, lr, smooth(gap), smooth(gap_std), "batch minus running statistics, rolling mean 5")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test accuracy gap, pp")
    ax.set_title(f"BN statistics gap during the range test ({note})")
    ax.legend()
    save(fig, EXP1, "bn_statistics_gap.png")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in group:
        h = r["history"]
        ax.plot(h["lr"], smooth(h["test_acc"].to_numpy()), lw=1, label=f"seed {r['cfg']['seed']}")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test accuracy, rolling mean 5")
    ax.set_title("LR range test, ResNet-56: individual seeds")
    ax.legend()
    save(fig, EXP1 / "per_seed", "range_test_accuracy_seeds.png")

    readouts = pd.DataFrame([range_readout(r) for r in group])
    save_table(readouts, EXP1, "readouts_per_seed.csv")
    save_table(describe_frame(readouts.assign(run="resnet56"), ["run"], [c for c in readouts if c != "seed"]),
               EXP1, "readouts_summary.csv")

    rows = []
    for r in group:
        h = r["history"]
        lr_r, a, tm = h["lr"].to_numpy(), smooth(h["test_acc"].to_numpy()), smooth(h["test_acc_trainmode"].to_numpy())
        tl, sl = h["train_loss"].to_numpy(), smooth(h["test_loss"].to_numpy())
        for lo, hi in ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.01)):
            m = (lr_r >= lo) & (lr_r < hi)
            rows.append(dict(seed=r["cfg"]["seed"], lr_band=f"{lo}-{min(hi, 3.0)}", eval_acc=a[m].mean(),
                             trainmode_acc=tm[m].mean(), gap_pp=100 * (tm[m] - a[m]).mean(),
                             train_loss=tl[m].mean(), test_loss=sl[m].mean()))
    bands = pd.DataFrame(rows)
    save_table(describe_frame(bands, ["lr_band"], ["eval_acc", "trainmode_acc", "gap_pp", "train_loss", "test_loss"]),
               EXP1, "lr_bands_summary.csv")
    save_table(final_table({"resnet56 range test": group}), EXP1, "final_summary.csv")


def estimator_rows(group, exp):
    rows = []
    for r in group:
        lr = lr_schedule(r)
        total = r["cfg"]["iters"]
        for name, hook in zip(ESTIMATORS[exp], r["hooks"]):
            t = np.array(hook["trace"], dtype=float)
            k, est, ema = t[:, 0].astype(int), t[:, 1], t[:, 2]

            def med(lo, hi):
                m = (k >= lo) & (k < hi)
                return float(np.median(ema[m])) if m.any() else np.nan

            early = k < 300
            rows.append(dict(
                exp=exp, sched=r["cfg"]["sched"], seed=r["cfg"]["seed"], variant=name,
                early_median=med(0, 300), early_peak=float(ema[early].max()) if early.any() else np.nan,
                first_half=med(0, total // 2), second_half=med(total // 2, total),
                mid_50_90=med(total // 2, int(0.9 * total)), last_10pct=med(int(0.9 * total), total),
                est_over_lr=float(np.median(est / np.maximum(lr[k], 1e-12))),
                spearman_est_lr=float(stats.spearmanr(est, lr[k]).statistic) if np.ptp(lr[k]) > 0 else np.nan,
                final_test_acc=r["final"]["test_acc"],
            ))
    return rows


def estimator_figure(group, exp, directory, name, title, yscale="log"):
    lr = lr_schedule(group[0])
    total = group[0]["cfg"]["iters"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=yscale == "log")
    for i, variant in enumerate(ESTIMATORS[exp]):
        ax = axes[0] if "dense" in variant else axes[1]
        k, ema, ema_std = stack_trace(group, i, "ema")
        line = band(ax, k, ema, ema_std, f"{variant}, EMA")
        k_raw, est, _ = stack_trace(group, i, "est")
        ax.plot(k_raw, est, color=line.get_color(), alpha=0.2, lw=0.7)
    axes[0].plot(np.arange(300), lr[:300], color="black", label="LR used")
    axes[1].plot(np.arange(total), lr, color="black", label="LR used")
    axes[0].set_title("every iteration, first 300")
    axes[1].set_title("every 10 iterations, whole run")
    for ax in axes:
        ax.set_yscale(yscale)
        ax.set_xlabel("Iteration")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Learning rate")
    fig.suptitle(title)
    save(fig, directory, name)


def estimator_experiment(runs, exp, directory):
    groups = by_group(select(runs, exp))
    if not groups:
        return pd.DataFrame()
    rows = []
    for group in groups.values():
        sched = group[0]["cfg"]["sched"]
        estimator_figure(group, exp, directory, f"estimate_{sched}.png",
                         f"Eq. 8 estimate, ResNet-56, {SCHED_LABELS[sched]} ({seeds_note([group])})")
        estimator_figure(group, exp, directory, f"estimate_{sched}_linear.png",
                         f"Eq. 8 estimate, ResNet-56, {SCHED_LABELS[sched]} ({seeds_note([group])}), linear scale",
                         yscale="linear")
        for r in group:
            estimator_figure([r], exp, directory / "per_seed", f"estimate_{sched}_seed{r['cfg']['seed']}.png",
                             f"Eq. 8 estimate, ResNet-56, {SCHED_LABELS[sched]}, seed {r['cfg']['seed']}")
        rows += estimator_rows(group, exp)
    table = pd.DataFrame(rows)
    save_table(table, directory, "estimator_per_seed.csv")
    metrics = ["early_median", "early_peak", "first_half", "second_half", "mid_50_90", "last_10pct", "est_over_lr",
               "spearman_est_lr", "final_test_acc"]
    summary = describe_frame(table, ["sched", "variant"], metrics)
    save_table(summary, directory, "estimator_summary.csv")
    save_table(final_table({SCHED_LABELS[g[0]["cfg"]["sched"]]: g for g in groups.values()}), directory, "final_summary.csv")
    return table


def estimate_over_lr_figure(table, directory):
    v2 = table[table["exp"] == "lr_estimator_v2"]
    if v2.empty:
        return
    grouped = v2.groupby(["variant", "sched"])["est_over_lr"]
    mean = grouped.mean().unstack("sched").reindex(index=ESTIMATORS["lr_estimator_v2"])
    std = grouped.std(ddof=1).unstack("sched").reindex(index=ESTIMATORS["lr_estimator_v2"])
    count = grouped.count().unstack("sched").reindex(index=ESTIMATORS["lr_estimator_v2"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(mean.columns)
    x = np.arange(len(mean.index))
    for i, sched in enumerate(mean.columns):
        ax.bar(x + i * width - 0.4 + width / 2, mean[sched], width, yerr=std[sched].fillna(0), capsize=3,
               label=f"{SCHED_LABELS[sched]} (n = {int(count[sched].max())})")
    ax.set_xticks(x, mean.index, rotation=15)
    # ax.set_yscale("log")
    ax.set_ylabel("median estimate / LR used, mean ± std over seeds")
    ax.set_title("Eq. 8 estimate relative to the LR used, by variant and run")
    ax.legend(fontsize=8)
    save(fig, directory, "estimate_over_lr.png")


def exp3(runs):
    rt = select(runs, "range_test", model="lenet")
    if rt:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _, lr, _ = stack([r["history"].set_index("it")["lr"].rename(r["cfg"]["seed"]) for r in rt])
        _, acc, acc_std = stack_history(rt, "test_acc")
        band(ax, lr, acc, acc_std, f"test accuracy ({seeds_note([rt])})", marker="o")
        diverged = [float(r["history"]["lr"].iloc[-1]) for r in rt if r["status"] == "diverged"]
        for i, d in enumerate(diverged):
            ax.axvline(d, color="gray", ls="--", label="divergence" if i == 0 else None)
        ax.axvspan(0.01, 0.1, color="gray", alpha=0.15, label="1cycle range, paper Table 2")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Test accuracy")
        ax.set_title("LR range test 0 -> 4, LeNet, MNIST")
        ax.legend()
        save(fig, EXP3, "lenet_range_test.png")

        fig, ax = plt.subplots(figsize=(7, 4.5))
        it, loss, loss_std = stack_losses(rt, 20)
        lr_it = lr_schedule(rt[0])
        band(ax, lr_it[it - 1], loss, loss_std, f"train loss, rolling mean 20 ({seeds_note([rt])})")
        ax.set_yscale("log")
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Train loss")
        ax.set_title("LR range test, LeNet: per-iteration train loss")
        ax.legend()
        save(fig, EXP3, "lenet_range_test_loss.png")
        save_table(pd.DataFrame([describe(diverged) | dict(metric="diverged_at_lr", seeds=seeds(rt))]),
                   EXP3, "lenet_range_test_summary.csv")

    groups = {"inv, 85 epochs": select(runs, "c11", sched="inv"), "1cycle, 12 epochs": select(runs, "c11", sched="1cycle")}
    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return
    note = seeds_note(groups.values())
    its_per_epoch = next(iter(groups.values()))[0]["cfg"]["eval_every"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, group in groups.items():
        it, mean, std = stack_history(group, "test_acc")
        band(ax, it / its_per_epoch, 100 * (1 - mean), 100 * std, label)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test error, %")
    ax.set_title(f"LeNet, MNIST: test error ({note})")
    ax.legend()
    save(fig, EXP3, "c11_test_error.png")

    for ylim, name in ((None, "c11_accuracy.png"), ((98.0, 100.0), "c11_accuracy_zoom.png")):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (label, group) in enumerate(groups.items()):
            it, train, train_std = stack_history(group, "train_acc")
            _, test, test_std = stack_history(group, "test_acc")
            band(ax, it / its_per_epoch, 100 * train, 100 * train_std, f"{label}, train (epoch window)", color=f"C{i}")
            band(ax, it / its_per_epoch, 100 * test, 100 * test_std, f"{label}, test", color=f"C{i}", ls="--")
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy, %")
        ax.set_title(f"LeNet, MNIST: train and test accuracy ({note})" + (", zoomed" if ylim else ""))
        ax.legend(fontsize=8)
        save(fig, EXP3, name)

    final_accuracy_bars(groups, PAPER["c11"], f"LeNet, MNIST: final test accuracy ({note})", EXP3, "c11_final_accuracy.png")
    save_table(final_table(groups), EXP3, "c11_final_summary.csv")
    if len(groups) == 2:
        final = {label: [r["final"]["test_acc"] for r in group] for label, group in groups.items()}
        save_table(pd.DataFrame([dict(a="1cycle, 12 epochs", b="inv, 85 epochs")
                                 | compare(final["1cycle, 12 epochs"], final["inv, 85 epochs"])]),
                   EXP3, "c11_comparison.csv")


def exp4(runs):
    groups = {
        "1cycle 0.1-3, 3k": select(runs, "c2", sched="1cycle"),
        "PC-LR 0.35, 3k": select(runs, "c2", sched="pc", iters=3000),
        "PC-LR 0.35, 24k": select(runs, "c2", sched="pc", iters=24000),
    }
    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        return
    note = seeds_note(groups.values())

    for xscale, name in (("linear", "test_accuracy.png"), ("log", "test_accuracy_log.png")):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, group in groups.items():
            it, mean, std = stack_history(group, "test_acc")
            band(ax, it, mean, std, label)
        ax.set_xscale(xscale)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Test accuracy (2k subset)")
        ax.set_title(f"ResNet-56, CIFAR-10: 1cycle vs PC-LR ({note})")
        ax.legend()
        save(fig, EXP4, name)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in groups.items():
        ax.plot(np.arange(group[0]["cfg"]["iters"]), lr_schedule(group[0]), label=label)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedules")
    ax.legend()
    save(fig, EXP4, "lr_schedules.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in groups.items():
        it, mean, std = stack_losses(group, 50)
        band(ax, it, mean, std, label)
    ax.axhline(math.log(10), color="gray", ls=":", label="ln 10")
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Train loss, rolling mean 50")
    ax.set_title(f"Per-iteration train loss ({note})")
    ax.legend()
    save(fig, EXP4, "train_loss.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, group in groups.items():
        it, gap, gap_std = stack([(100 * (r["history"]["train_acc"] - r["history"]["test_acc"]))
                                  .set_axis(r["history"]["it"]).rename(r["cfg"]["seed"]) for r in group])
        band(ax, it, gap, gap_std, label)
    ax.set_xscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Train window accuracy minus test accuracy, pp")
    ax.set_title(f"Generalization gap during training ({note})")
    ax.legend()
    save(fig, EXP4, "generalization_gap.png")

    final_accuracy_bars(groups, PAPER["c2"], f"ResNet-56, CIFAR-10: final test accuracy ({note})", EXP4,
                        "final_accuracy.png")
    save_table(final_table(groups), EXP4, "final_summary.csv")
    final = {label: [r["final"]["test_acc"] for r in group] for label, group in groups.items()}
    pairs = [("1cycle 0.1-3, 3k", "PC-LR 0.35, 24k"), ("1cycle 0.1-3, 3k", "PC-LR 0.35, 3k")]
    rows = [dict(a=a, b=b) | compare(final[a], final[b]) for a, b in pairs if a in final and b in final]
    if rows:
        save_table(pd.DataFrame(rows), EXP4, "comparison.csv")
    save_table(pd.DataFrame([dict(run=label, seed=r["cfg"]["seed"], flags=" ".join(r["flags"]))
                             for label, group in groups.items() for r in group]), EXP4, "flags.csv")


def runs_table(runs):
    rows = []
    for group_name, group in by_group(runs).items():
        acc = describe([r["final"]["test_acc"] for r in group])
        rows.append(dict(
            group=group_name, seeds=" ".join(str(s) for s in seeds(group)), n=len(group),
            status=",".join(sorted({r["status"] for r in group})),
            final_test_acc_mean=acc["mean"], final_test_acc_std=acc["std"], final_test_acc_sem=acc["sem"],
            final_test_acc_ci95_low=acc["ci95_low"], final_test_acc_ci95_high=acc["ci95_high"],
            n_flags=sum(len(r["flags"]) for r in group),
            train_min=np.mean([r["final"]["t_train"] for r in group]) / 60,
            s_per_it=np.mean([r["final"]["t_train"] / r["cfg"]["iters"] for r in group]),
        ))
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    runs = load_runs()
    print(f"{len(runs)} runs")
    print(check_schedules(runs).to_string(index=False))
    summary = runs_table(runs)
    print(summary.round(4).to_string(index=False))
    save_table(summary, FIGURES, "runs_summary.csv")

    exp1(runs)
    table2 = estimator_experiment(runs, "lr_estimator", EXP2)
    table2b = estimator_experiment(runs, "lr_estimator_v2", EXP2B)
    estimate_over_lr_figure(table2b, EXP2B)
    exp3(runs)
    exp4(runs)

    for directory in (EXP1, EXP2, EXP2B, EXP3, EXP4):
        for path in sorted(directory.glob("*summary.csv")) + sorted(directory.glob("*comparison.csv")):
            print(f"\n{path.relative_to(ROOT)}")
            print(pd.read_csv(path).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
