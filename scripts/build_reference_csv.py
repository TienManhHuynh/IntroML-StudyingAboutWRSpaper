"""Normalize author WRS reference logs for Batch 1 datasets into compare_reference.py schema."""
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF_LOGS = ROOT / "tmp" / "author-ref" / "WRS" / "logs"
OUT_DIR = ROOT / "results" / "reference"
SOURCE = "github:FutureComputing4AI/Weighted-Reservoir-Sampling-Augmented-Training@main"

DATASET_MAP = {
    "dexter_binary_sparse": "dexter",
    "dorothea_binary_sparse": "dorothea",
    "pcmac_binary_sparse": "pcmac",
    "newsgroups_binary_sparse": "newsgroups-binary-cs",
    "news20_binary_sparse": "news20",
    "w8a_binary_sparse": "w8a",
    "sst2_binary_sparse": "sst2",
    "real-sim_binary_sparse": "real-sim",
}
SCHEME_MAP = {"dense": "standard", "exp-dense": "exponential"}
VARIANTS = [
    ("SA", "simple", "vz0"),
    ("SA_VZ", "simple", "vz1"),
    ("WA", "weighted", "vz0"),
    ("WA_VZ", "weighted", "vz1"),
]


def main():
    """Write author_batch1_trials.csv and author_batch1_means.csv from author logs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_rows = []
    grouped = defaultdict(list)

    with (REF_LOGS / "baseline_logs.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            ds = DATASET_MAP.get(row["dataset"])
            if not ds:
                continue
            learner = row["model"].lower()
            entry = {
                "dataset": ds,
                "learner": learner,
                "model": "baseline",
                "seed": row["seed"],
                "config_id": "baseline",
                "source": SOURCE,
                "final_accuracy": float(row["fin_test_acc_inst"]),
                "rop": float(row["L1_inst"]),
                "final_sparsity": float(row["fin_sparsity_inst"]),
            }
            trial_rows.append(entry)
            grouped[(ds, learner, "baseline", "baseline")].append(entry)

    with (REF_LOGS / "master.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            ds = DATASET_MAP.get(row["dataset"])
            if not ds:
                continue
            learner = row["model"].lower()
            ws = SCHEME_MAP[row["weight_scheme"]]
            k_val = int(row["K"])
            for suffix, avg, vz in VARIANTS:
                cfg_id = f"k{k_val}_{ws}_{avg}_{vz}"
                entry = {
                    "dataset": ds,
                    "learner": learner,
                    "model": "wrs",
                    "seed": row["seed"],
                    "config_id": cfg_id,
                    "source": SOURCE,
                    "final_accuracy": float(row[f"fin_test_acc_{suffix}"]),
                    "rop": float(row[f"L1_{suffix}"]),
                    "final_sparsity": float(row[f"fin_sparsity_{suffix}"]),
                }
                trial_rows.append(entry)
                grouped[(ds, learner, "wrs", cfg_id)].append(entry)

    fields = ["dataset", "learner", "model", "seed", "config_id", "source", "final_accuracy", "rop", "final_sparsity"]
    with (OUT_DIR / "author_batch1_trials.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trial_rows)

    mean_rows = []
    for (ds, learner, model, cfg_id), items in sorted(grouped.items()):
        mean_rows.append({
            "dataset": ds,
            "learner": learner,
            "model": model,
            "seed": "mean_5trials",
            "config_id": cfg_id,
            "source": SOURCE,
            "final_accuracy": float(np.mean([x["final_accuracy"] for x in items])),
            "rop": float(np.mean([x["rop"] for x in items])),
            "final_sparsity": float(np.mean([x["final_sparsity"] for x in items])),
        })
    with (OUT_DIR / "author_batch1_means.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(mean_rows)
    print(f"Wrote {len(trial_rows)} trial rows and {len(mean_rows)} mean rows to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
