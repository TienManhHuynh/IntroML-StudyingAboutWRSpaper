"""Create an auditable job PLAN, without loading datasets or training models."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_jobs(config):
    """Return planned main/ablation jobs from a config mapping; no experiments run.

    Parameters: config: the Full scope configuration dictionary.
    Returns: list of unique job dictionaries with execution_ready=False.
    """
    if config["scope"] != "full":
        raise ValueError("This framework requires Full scope.")
    jobs = []

    def add(group, dataset, learner, seed, method, **variant):
        job = {"group": group, "dataset": dataset, "learner": learner,
               "seed": seed, "method": method, **variant,
               "train_fraction": config["train_fraction"], "passes": config["passes"],
               "checkpoint_target": config["checkpoint_target"],
               "include_initial_and_final_checkpoints": config["include_initial_and_final_checkpoints"],
               "selected_hyperparameters": None, "status": "planned", "execution_ready": False,
               "blocked_by": ["implementation_pending", "hosted_source_unverified", "hyperparameter_selection_pending"]}
        key = [group.lower(), dataset, learner, method, str(seed)]
        key.extend("{}-{}".format(k, str(v).lower()) for k, v in sorted(variant.items()))
        job["job_id"] = "__".join(key)
        jobs.append(job)

    for dataset, learner, seed in product(config["dataset_ids"], config["learners"], config["seeds"]):
        add("MAIN", dataset, learner, seed, "baseline")
        wrs = config["wrs"]
        for k, weighting, averaging, zeroing in product(wrs["reservoir_sizes"], wrs["weighting"], wrs["averaging"], wrs["voting_zeroing"]):
            add("MAIN", dataset, learner, seed, "wrs", K=k, weighting=weighting,
                averaging=averaging, voting_zeroing=zeroing)
        for averaging in config["topk"]["averaging"]:
            add("TOPK", dataset, learner, seed, "topk", K=config["topk"]["reservoir_size"], averaging=averaging)
        ablations = config["averaging_baselines"]
        add("AVERAGING", dataset, learner, seed, "moving_average", window=ablations["moving_window"])
        add("AVERAGING", dataset, learner, seed, "exponential_average", gamma=ablations["ema_gamma"])
    return jobs


def main():
    """Parse CLI arguments and print/write planned jobs; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSONL plan output; refuses overwrite")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/full_scope.json").read_text(encoding="utf-8"))
    jobs = build_jobs(config)
    counts = Counter(job["method"] for job in jobs)
    print("FULL SCOPE - PLAN ONLY; no dataset download, training or W&B logging")
    print(json.dumps({"datasets": len(config["dataset_ids"]), "trials": len(config["seeds"]),
                      "planned_jobs": len(jobs), "by_method": dict(counts),
                      "not_in_this_manifest": config["not_in_generated_manifest"]}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                for job in jobs:
                    stream.write(json.dumps(job, sort_keys=True) + "\n")
        except FileExistsError:
            parser.error("Output already exists; choose another path to preserve the previous plan.")
        print("Plan written to {}".format(args.output.resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
