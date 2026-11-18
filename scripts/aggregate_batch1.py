"""Aggregate Batch 1 online runs, compute Wilcoxon tests, reference deltas, and plots."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.compare_reference import compare
from experiments.run_reproduction import write_json
from src.evaluation.statistics import paired_dataset_test

RAW_ROOT = ROOT / "results" / "raw" / "batch1"
OUT_ROOT = ROOT / "results" / "batch1"
FIG_ROOT = ROOT / "figures"


def _collect_runs(raw_root):
    """Load all completed run directories under results/raw/batch1."""
    rows = []
    for run_file in sorted(raw_root.rglob("run.json")):
        run_dir = run_file.parent
        state = json.loads(run_file.read_text(encoding="utf-8"))
        if state.get("status") not in ("completed_local", "completed_online"):
            continue
        rel = run_dir.relative_to(raw_root).parts
        # Expected: <suite>/<dataset>/<learner>/<variant>/seed-<n>
        if len(rel) != 5:
            continue
        suite, dataset, learner, variant, _ = rel
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        method_key = next(k for k in metrics if k != "baseline")
        rows.append({
            "suite": suite,
            "dataset": dataset,
            "learner": learner,
            "variant": variant,
            "seed": int(cfg["seed"]),
            "method_key": method_key,
            "status": state["status"],
            "wandb_url": state.get("wandb_url") or "",
            "path": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
            "baseline": metrics["baseline"],
            "method": metrics[method_key],
            "noise_rate": float(cfg.get("noise_rate", 0.0)),
        })
    return rows


def _summarize_groups(rows):
    """Group runs by (suite, dataset, learner, variant) and compute mean/min/max."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["suite"], row["dataset"], row["learner"], row["variant"])].append(row)
    summary = []
    for (suite, ds, learner, variant), items in sorted(grouped.items()):
        items = sorted(items, key=lambda r: r["seed"])
        entry = {
            "suite": suite,
            "dataset": ds,
            "learner": learner,
            "variant": variant,
            "method_key": items[0]["method_key"],
            "n_trials": len(items),
            "five_trials_complete": len(items) == 5,
            "all_public_urls": all(bool(r["wandb_url"]) for r in items),
            "wandb_urls": [r["wandb_url"] for r in items],
            "baseline": {},
            "method": {},
        }
        for side in ("baseline", "method"):
            for metric in ("final_accuracy", "final_sparsity", "rop", "train_seconds"):
                vals = np.array([r[side][metric] for r in items], dtype=float)
                entry[side][metric] = {
                    "mean": float(vals.mean()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                }
        summary.append(entry)
    return summary


def _write_reproduced_csv(summary, out_path):
    """Write normalized CSV matching compare_reference.py schema for main runs."""
    fields = ["dataset", "learner", "model", "seed", "config_id", "source", "final_accuracy", "rop", "final_sparsity"]
    rows = []
    seen_base = set()
    for item in summary:
        if item["suite"] != "main":
            continue
        ds, learner, variant = item["dataset"], item["learner"], item["variant"]
        urls = [u for u in item["wandb_urls"] if u]
        src = urls[0] if urls else f"local:{ds}/{learner}/{variant}"
        if (ds, learner) not in seen_base and variant == "k64_standard_simple_vz0":
            seen_base.add((ds, learner))
            b = item["baseline"]
            rows.append({
                "dataset": ds,
                "learner": learner,
                "model": "baseline",
                "seed": "mean_5trials",
                "config_id": "baseline",
                "source": src,
                "final_accuracy": b["final_accuracy"]["mean"],
                "rop": b["rop"]["mean"],
                "final_sparsity": b["final_sparsity"]["mean"],
            })
        m = item["method"]
        rows.append({
            "dataset": ds,
            "learner": learner,
            "model": "wrs",
            "seed": "mean_5trials",
            "config_id": variant,
            "source": src,
            "final_accuracy": m["final_accuracy"]["mean"],
            "rop": m["rop"]["mean"],
            "final_sparsity": m["final_sparsity"]["mean"],
        })
    with out_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_suite_stats(results, by_key, suite, learner, variant, datasets):
    """Compute paired Wilcoxon stats for ROP and accuracy on one (suite, learner, variant)."""
    if len(datasets) < 2:
        return
    base_rop = {d: by_key[(suite, d, learner, variant)]["baseline"]["rop"]["mean"] for d in datasets}
    cand_rop = {d: by_key[(suite, d, learner, variant)]["method"]["rop"]["mean"] for d in datasets}
    results[f"{suite}_{learner}_{variant}_rop"] = paired_dataset_test(base_rop, cand_rop, alternative="two-sided")
    base_acc = {d: -by_key[(suite, d, learner, variant)]["baseline"]["final_accuracy"]["mean"] for d in datasets}
    cand_acc = {d: -by_key[(suite, d, learner, variant)]["method"]["final_accuracy"]["mean"] for d in datasets}
    acc_stat = paired_dataset_test(base_acc, cand_acc, alternative="two-sided")
    acc_stat["note"] = "negated accuracy so candidate_wins_lower_is_better counts higher accuracy"
    results[f"{suite}_{learner}_{variant}_acc"] = acc_stat


def _compute_wilcoxon(summary):
    """Compute paired Wilcoxon signed-rank tests on main, ablations, and deviation_band."""
    results = {}
    by_key = {(s["suite"], s["dataset"], s["learner"], s["variant"]): s for s in summary}
    main_variants = [
        "k64_standard_simple_vz0",
        "k64_standard_weighted_vz0",
        "k64_exponential_simple_vz0",
        "k64_exponential_weighted_vz0",
        "k64_standard_simple_vz1",
        "k64_standard_weighted_vz1",
    ]
    ablation_variants = ["topk_k64_simple", "topk_k64_weighted", "moving_k64", "ema_g0.9"]
    for learner in ("pac", "fsol"):
        for variant in main_variants:
            datasets = sorted({
                s["dataset"] for s in summary
                if s["suite"] == "main" and s["learner"] == learner and s["variant"] == variant
            })
            _paired_suite_stats(results, by_key, "main", learner, variant, datasets)
        for variant in ablation_variants:
            datasets = sorted({
                s["dataset"] for s in summary
                if s["suite"] == "ablations" and s["learner"] == learner and s["variant"] == variant
            })
            _paired_suite_stats(results, by_key, "ablations", learner, variant, datasets)

    # Deviation band (top_acc - 0.025): 7 changed datasets from deviation_band + news20 (unchanged C_err=100.0) from main.
    band_items = {}
    for s in summary:
        if s["learner"] == "pac" and s["variant"] == "k64_standard_simple_vz0":
            if s["suite"] == "deviation_band":
                band_items[s["dataset"]] = s
            elif s["suite"] == "main" and s["dataset"] == "news20" and s["dataset"] not in band_items:
                band_items[s["dataset"]] = s
    if len(band_items) >= 2:
        ds_names = sorted(band_items)
        base_rop = {d: band_items[d]["baseline"]["rop"]["mean"] for d in ds_names}
        cand_rop = {d: band_items[d]["method"]["rop"]["mean"] for d in ds_names}
        results["deviation_band_pac_k64_standard_simple_vz0_rop"] = paired_dataset_test(base_rop, cand_rop, alternative="two-sided")
        base_acc = {d: -band_items[d]["baseline"]["final_accuracy"]["mean"] for d in ds_names}
        cand_acc = {d: -band_items[d]["method"]["final_accuracy"]["mean"] for d in ds_names}
        acc_stat = paired_dataset_test(base_acc, cand_acc, alternative="two-sided")
        acc_stat["note"] = "negated accuracy so candidate_wins_lower_is_better counts higher accuracy"
        results["deviation_band_pac_k64_standard_simple_vz0_acc"] = acc_stat
    return results


def _compute_runtime_table(summary):
    """Compute moving_k64 and ema_g0.9 train_seconds relative to WRS K=64 (Table 5)."""
    by_key = {(s["suite"], s["dataset"], s["learner"], s["variant"]): s for s in summary}
    datasets = sorted({s["dataset"] for s in summary if s["suite"] == "main"})
    rows = []
    for ds in datasets:
        row = {"dataset": ds}
        for learner in ("pac", "fsol"):
            wrs = by_key.get(("main", ds, learner, "k64_standard_simple_vz0"))
            moving = by_key.get(("ablations", ds, learner, "moving_k64"))
            ema = by_key.get(("ablations", ds, learner, "ema_g0.9"))
            if not wrs:
                continue
            wrs_t = wrs["method"]["train_seconds"]["mean"]
            row[f"{learner}_wrs_s"] = wrs_t
            if moving and wrs_t > 0:
                row[f"{learner}_moving_s"] = moving["method"]["train_seconds"]["mean"]
                row[f"{learner}_moving_ratio"] = moving["method"]["train_seconds"]["mean"] / wrs_t
            if ema and wrs_t > 0:
                row[f"{learner}_ema_s"] = ema["method"]["train_seconds"]["mean"]
                row[f"{learner}_ema_ratio"] = ema["method"]["train_seconds"]["mean"] / wrs_t
        rows.append(row)
    return rows


def _save_fig(fig, stem):
    """Save figure as both PNG (160 dpi) and clean SVG."""
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_ROOT / f"{stem}.png", dpi=160)
    svg = FIG_ROOT / f"{stem}.svg"
    fig.savefig(svg)
    svg.write_text("\n".join(l.rstrip() for l in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def _plot_main_rop(summary):
    """Plot base vs K=64 Standard-Simple WRS ROP (mean + min/max error bars) across datasets."""
    items = [s for s in summary if s["suite"] == "main" and s["variant"] == "k64_standard_simple_vz0"]
    if not items:
        return
    datasets = sorted({s["dataset"] for s in items})
    by_dl = {(s["dataset"], s["learner"]): s for s in items}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    for ax, learner in zip(axes, ("pac", "fsol")):
        ds_list = [d for d in datasets if (d, learner) in by_dl]
        x = np.arange(len(ds_list))
        for offset, side, label, color in ((-0.16, "baseline", f"{learner.upper()} Baseline", "#7A869A"),
                                           (0.16, "method", f"{learner.upper()}-WRS (K=64)", "#1B6CA8")):
            means = np.array([by_dl[(d, learner)][side]["rop"]["mean"] for d in ds_list])
            mins = np.array([by_dl[(d, learner)][side]["rop"]["min"] for d in ds_list])
            maxs = np.array([by_dl[(d, learner)][side]["rop"]["max"] for d in ds_list])
            ax.errorbar(x + offset, means, yerr=[means - mins, maxs - means], fmt="o", capsize=4, label=label, color=color)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(ds_list, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("ROP (lower is more stable)")
        ax.set_title(f"{learner.upper()} vs {learner.upper()}-WRS (K=64, Standard/Simple)")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle("Reproduced ROP Stability Across Datasets (Mean and Min/Max over 5 Trials)")
    _save_fig(fig, "batch1_rop_k64")


def _plot_k_sweep(summary):
    """Plot ROP across K in {1, 4, 16, 64} (Standard/Simple, VZ=False) on available datasets."""
    ks = [1, 4, 16, 64]
    items = [s for s in summary if s["suite"] == "main" and s["variant"] in {f"k{k}_standard_simple_vz0" for k in ks}]
    if not items:
        return
    datasets = sorted({s["dataset"] for s in items})[:4]
    by_key = {(s["dataset"], s["learner"], s["variant"]): s for s in items}
    fig, axes = plt.subplots(2, len(datasets), figsize=(3.6 * len(datasets), 6.2), layout="constrained", squeeze=False)
    for r, learner in enumerate(("pac", "fsol")):
        for c, ds in enumerate(datasets):
            ax = axes[r][c]
            ref = by_key.get((ds, learner, "k64_standard_simple_vz0"))
            if ref:
                b = ref["baseline"]["rop"]
                ax.axhspan(b["min"], b["max"], color="#B0BEC5", alpha=0.35, label="Base min-max")
                ax.axhline(b["mean"], color="#455A64", linewidth=1.4, label="Base mean")
            means, mins, maxs, xs = [], [], [], []
            for idx, k in enumerate(ks):
                it = by_key.get((ds, learner, f"k{k}_standard_simple_vz0"))
                if it:
                    m = it["method"]["rop"]
                    xs.append(idx); means.append(m["mean"]); mins.append(m["min"]); maxs.append(m["max"])
            if xs:
                means, mins, maxs = np.array(means), np.array(mins), np.array(maxs)
                ax.errorbar(xs, means, yerr=[means - mins, maxs - means], fmt="o-", capsize=4, color="#1B6CA8", label="WRS")
            ax.set_xticks(range(len(ks)))
            ax.set_xticklabels([f"K={k}" for k in ks])
            ax.set_title(f"{ds} ({learner.upper()})")
            if c == 0:
                ax.set_ylabel("ROP")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
    fig.suptitle("ROP vs Reservoir Size K (Standard Weighting, Simple Average)")
    _save_fig(fig, "batch1_k_sweep")


def _plot_noise(summary):
    """Plot clean test accuracy and ROP across train-label noise rates."""
    items = [s for s in summary if s["suite"] == "noise"]
    if not items:
        return
    datasets = sorted({s["dataset"] for s in items})
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10.5, 3.4 * len(datasets)), layout="constrained", squeeze=False)
    for r, ds in enumerate(datasets):
        for c, learner in enumerate(("pac", "fsol")):
            ax = axes[r][c]
            sub = sorted([s for s in items if s["dataset"] == ds and s["learner"] == learner], key=lambda s: float(s["variant"].split("-")[1]))
            if not sub:
                continue
            rates = [float(s["variant"].split("-")[1]) * 100 for s in sub]
            for side, label, color in (("baseline", "Base", "#7A869A"), ("method", "WRS (K=64)", "#1B6CA8")):
                means = np.array([s[side]["final_accuracy"]["mean"] for s in sub])
                mins = np.array([s[side]["final_accuracy"]["min"] for s in sub])
                maxs = np.array([s[side]["final_accuracy"]["max"] for s in sub])
                ax.errorbar(rates, means, yerr=[means - mins, maxs - means], fmt="o-", capsize=4, label=label, color=color)
            ax.set_title(f"{ds} — {learner.upper()} (Clean Test Accuracy)")
            ax.set_xlabel("Train label noise rate (%)")
            ax.set_ylabel("Final test accuracy")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.25)
    fig.suptitle("Additional Analysis: Train-Only Label Noise Robustness")
    _save_fig(fig, "batch1_noise_analysis")


def _plot_ablations(summary):
    """Plot ROP across WRS vs Top-K, Moving Average, and EMA (K=64 / gamma=0.9)."""
    by_key = {(s["suite"], s["dataset"], s["learner"], s["variant"]): s for s in summary}
    datasets = sorted({s["dataset"] for s in summary if s["suite"] == "ablations"})
    if not datasets:
        return
    methods = [
        ("main", "k64_standard_simple_vz0", "WRS (K=64)", "#1B6CA8"),
        ("ablations", "topk_k64_simple", "Top-K (Simple)", "#2E8B57"),
        ("ablations", "moving_k64", "Moving Avg (K=64)", "#D97706"),
        ("ablations", "ema_g0.9", "EMA (gamma=0.9)", "#DC2626"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), layout="constrained")
    for ax, learner in zip(axes, ("pac", "fsol")):
        x = np.arange(len(datasets))
        offsets = np.linspace(-0.27, 0.27, len(methods))
        for offset, (suite, var, label, color) in zip(offsets, methods):
            vals = [by_key[(suite, d, learner, var)]["method"]["rop"]["mean"] if (suite, d, learner, var) in by_key else np.nan for d in datasets]
            ax.plot(x + offset, vals, "o", label=label, color=color, markersize=6)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Mean ROP (lower is better)")
        ax.set_title(f"{learner.upper()}: WRS vs Ensembling Ablations")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Ablation Comparison Across 8 Batch 1 Datasets (5-Trial Mean ROP)")
    _save_fig(fig, "batch1_ablations")


def _plot_deviation_band(summary):
    """Plot PAC vs PAC-WRS ROP under ceil(2.5%) vs author's top_acc - 0.025 tuning rule."""
    by_main = {s["dataset"]: s for s in summary if s["suite"] == "main" and s["learner"] == "pac" and s["variant"] == "k64_standard_simple_vz0"}
    by_band = {s["dataset"]: s for s in summary if s["suite"] == "deviation_band" and s["learner"] == "pac" and s["variant"] == "k64_standard_simple_vz0"}
    if "news20" in by_main and "news20" not in by_band:
        by_band["news20"] = by_main["news20"]
    datasets = sorted(set(by_main) & set(by_band))
    if not datasets:
        return
    x = np.arange(len(datasets))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), layout="constrained")
    for ax, title, src_map in (
        (axes[0], "ceil(2.5% * 7) = Top-1 Config (Low Oscillation)", by_main),
        (axes[1], "Author Rule: acc >= max_acc - 0.025 (Adverse Oscillation)", by_band),
    ):
        for offset, side, label, color in ((-0.16, "baseline", "PAC Baseline", "#7A869A"),
                                           (0.16, "method", "PAC-WRS (K=64)", "#1B6CA8")):
            means = np.array([src_map[d][side]["rop"]["mean"] for d in datasets])
            mins = np.array([src_map[d][side]["rop"]["min"] for d in datasets])
            maxs = np.array([src_map[d][side]["rop"]["max"] for d in datasets])
            ax.errorbar(x + offset, means, yerr=[means - mins, maxs - means], fmt="o", capsize=4, label=label, color=color)
        ax.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("ROP (lower is more stable)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Tuning-Policy Deviation Analysis on PAC (Mean and Min/Max over 5 Trials)")
    _save_fig(fig, "batch1_tuning_deviation")


def main():
    """Aggregate runs, write JSON/CSV artifacts, run reference comparison, and export plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = _collect_runs(args.raw_root)
    if not rows:
        print("No completed runs found under", args.raw_root)
        return 1

    index_fields = [
        "suite", "dataset", "learner", "variant", "seed", "status", "wandb_url", "path",
        "base_acc", "base_rop", "base_sparsity", "base_train_s",
        "method_acc", "method_rop", "method_sparsity", "method_train_s",
    ]
    with (args.out_root / "run-index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=index_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "suite": r["suite"], "dataset": r["dataset"], "learner": r["learner"], "variant": r["variant"],
                "seed": r["seed"], "status": r["status"], "wandb_url": r["wandb_url"], "path": r["path"],
                "base_acc": r["baseline"]["final_accuracy"], "base_rop": r["baseline"]["rop"],
                "base_sparsity": r["baseline"]["final_sparsity"], "base_train_s": r["baseline"]["train_seconds"],
                "method_acc": r["method"]["final_accuracy"], "method_rop": r["method"]["rop"],
                "method_sparsity": r["method"]["final_sparsity"], "method_train_s": r["method"]["train_seconds"],
            })

    summary = _summarize_groups(rows)
    write_json(args.out_root / "summary.json", summary)
    _write_reproduced_csv(summary, args.out_root / "reproduced_means.csv")
    write_json(args.out_root / "wilcoxon_stats.json", _compute_wilcoxon(summary))
    write_json(args.out_root / "runtime_table5.json", _compute_runtime_table(summary))

    ref_csv = ROOT / "results" / "reference" / "author_batch1_means.csv"
    comp_out = args.out_root / "reference_comparison.csv"
    if ref_csv.is_file():
        if comp_out.exists():
            comp_out.unlink()
        compare(ref_csv, args.out_root / "reproduced_means.csv", comp_out)

    _plot_main_rop(summary)
    _plot_k_sweep(summary)
    _plot_noise(summary)
    _plot_ablations(summary)
    _plot_deviation_band(summary)
    print(f"Aggregated {len(rows)} runs across {len(summary)} groups into {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
