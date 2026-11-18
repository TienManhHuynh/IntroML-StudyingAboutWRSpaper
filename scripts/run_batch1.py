"""Resumable batch orchestrator for Batch 1 WRS reproduction, ablations, and label noise."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from experiments.run_ablations import run_ablation
from experiments.run_reproduction import run_experiment
from src.data.loader import load_dataset

BATCH1_DATASETS = [
    "dexter",
    "dorothea",
    "pcmac",
    "w8a",
    "newsgroups-binary-cs",
    "sst2",
    "real-sim",
    "news20",
]
SEEDS = [42, 43, 44, 45, 46]
NOISE_DATASETS = ["dexter", "pcmac", "w8a", "sst2"]
NOISE_RATES = [0.0, 0.05, 0.1, 0.2, 0.3]


def _load_selected(dataset, learner):
    """Read tuned hyperparameters from results/tuning/<dataset>/<learner>/selection.json."""
    path = ROOT / "results" / "tuning" / dataset / learner / "selection.json"
    return json.loads(path.read_text(encoding="utf-8"))["selected"]["parameters"]


def _is_done(run_dir, wandb_mode):
    """Return True if run_dir already completed at the requested tracking level."""
    rfile = run_dir / "run.json"
    if not rfile.is_file():
        return False
    state = json.loads(rfile.read_text(encoding="utf-8"))
    if wandb_mode == "online":
        return state.get("status") == "completed_online" and bool(state.get("wandb_url"))
    return state.get("status") in ("completed_local", "completed_online")


def _build_main_jobs(datasets, learners):
    """Build headline Table 2/6/7 + Figures 1-4 WRS jobs (K=64 variants + K=1,4,16 sweep)."""
    variants = [
        (64, "standard", "simple", False),
        (64, "standard", "weighted", False),
        (64, "exponential", "simple", False),
        (64, "exponential", "weighted", False),
        (64, "standard", "simple", True),
        (64, "standard", "weighted", True),
        (16, "standard", "simple", False),
        (4, "standard", "simple", False),
        (1, "standard", "simple", False),
    ]
    jobs = []
    for ds in datasets:
        for learner in learners:
            params = _load_selected(ds, learner)
            for K, ws, avg, vz in variants:
                vz_tag = "vz1" if vz else "vz0"
                tag = f"k{K}_{ws}_{avg}_{vz_tag}"
                for seed in SEEDS:
                    jobs.append({
                        "kind": "wrs",
                        "dataset": ds,
                        "learner": learner,
                        "seed": seed,
                        "params": params,
                        "K": K,
                        "weighting": ws,
                        "averaging": avg,
                        "zeroing": vz,
                        "noise_rate": 0.0,
                        "out": str(ROOT / "results" / "raw" / "batch1" / "main" / ds / learner / tag / f"seed-{seed}"),
                    })
    return jobs


def _build_ablation_jobs(datasets, learners):
    """Build Top-K, Moving Average, and EMA jobs matching paper Section 4.2 & Appendix D."""
    specs = [
        ("topk", 64, "simple", 0.9, "topk_k64_simple"),
        ("topk", 64, "weighted", 0.9, "topk_k64_weighted"),
        ("moving_average", 64, "simple", 0.9, "moving_k64"),
        ("exponential_average", 64, "simple", 0.9, "ema_g0.9"),
    ]
    jobs = []
    for ds in datasets:
        for learner in learners:
            params = _load_selected(ds, learner)
            for method, K, avg, gamma, tag in specs:
                for seed in SEEDS:
                    jobs.append({
                        "kind": "ablation",
                        "dataset": ds,
                        "learner": learner,
                        "seed": seed,
                        "params": params,
                        "method": method,
                        "K": K,
                        "averaging": avg,
                        "gamma": gamma,
                        "out": str(ROOT / "results" / "raw" / "batch1" / "ablations" / ds / learner / tag / f"seed-{seed}"),
                    })
    return jobs


def _build_noise_jobs(datasets, learners):
    """Build paired train-label noise sweep jobs (K=64 standard/simple) on clean test sets."""
    jobs = []
    for ds in datasets:
        for learner in learners:
            params = _load_selected(ds, learner)
            for rate in NOISE_RATES:
                rtag = "noise-" + format(rate, ".8g")
                for seed in SEEDS:
                    jobs.append({
                        "kind": "wrs",
                        "dataset": ds,
                        "learner": learner,
                        "seed": seed,
                        "params": params,
                        "K": 64,
                        "weighting": "standard",
                        "averaging": "simple",
                        "zeroing": False,
                        "noise_rate": rate,
                        "out": str(ROOT / "results" / "raw" / "batch1" / "noise" / ds / learner / rtag / f"seed-{seed}"),
                    })
    return jobs


DEVIATION_BAND_C_ERR = {
    "dexter": 1.0,
    "dorothea": 0.01,
    "pcmac": 1000.0,
    "newsgroups-binary-cs": 1000.0,
    "w8a": 1000.0,
    "sst2": 100.0,
    "real-sim": 1000.0,
}


def _build_deviation_band_jobs(datasets):
    """Build PAC vs PAC-WRS (K=64 standard/simple) jobs under author's top_acc - 0.025 band."""
    jobs = []
    for ds in datasets:
        if ds not in DEVIATION_BAND_C_ERR:
            continue
        params = {"C_err": DEVIATION_BAND_C_ERR[ds]}
        for seed in SEEDS:
            jobs.append({
                "kind": "wrs",
                "dataset": ds,
                "learner": "pac",
                "seed": seed,
                "params": params,
                "K": 64,
                "weighting": "standard",
                "averaging": "simple",
                "zeroing": False,
                "noise_rate": 0.0,
                "experiment": "deviation-tuning-band",
                "out": str(ROOT / "results" / "raw" / "batch1" / "deviation_band" / ds / "pac" / "k64_standard_simple_vz0" / f"seed-{seed}"),
            })
    return jobs


def _run_seed_group(task):
    """Run all jobs sharing one (dataset, seed) sequentially using one cached bundle."""
    dataset, seed, jobs, wandb_mode, wandb_project, wandb_entity = task
    bundle = load_dataset(dataset, seed=seed, source_file=str(ROOT / "configs/data_sources.json"))
    completed = 0
    for job in jobs:
        out_dir = Path(job["out"])
        if _is_done(out_dir, wandb_mode):
            completed += 1
            continue
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        params = job["params"]
        common = {
            "dataset": dataset,
            "learner": job["learner"],
            "seed": seed,
            "C_err": params.get("C_err", 1.0),
            "eta": params.get("eta", 1.0),
            "lambda_": params.get("lambda_", 0.01),
            "checkpoints": 200,
            "sources": str(ROOT / "configs/data_sources.json"),
            "wandb_mode": wandb_mode,
            "wandb_project": wandb_project,
            "wandb_entity": wandb_entity,
        }
        if job["kind"] == "ablation":
            cfg = dict(
                common,
                method=job["method"],
                K=job["K"],
                averaging=job["averaging"],
                gamma=job["gamma"],
                experiment="ablation",
            )
            run_ablation(cfg, out_dir, bundle=bundle)
        else:
            exp_name = job.get("experiment") or ("label-noise" if job["noise_rate"] > 0 else "reproduction")
            cfg = dict(
                common,
                include_wrs=True,
                wat_factory="src.wat.wat:WAT",
                K=job["K"],
                weighting=job["weighting"],
                averaging=job["averaging"],
                zeroing=job["zeroing"],
                noise_rate=job["noise_rate"],
                noise_seed=seed + 100000,
                experiment=exp_name,
            )
            run_experiment(cfg, out_dir, bundle=bundle)
        completed += 1
    return (dataset, seed, completed)


def main():
    """Execute requested Batch 1 suites with cached bundles and resumable output dirs."""
    load_dotenv(ROOT / ".env", override=False)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["main", "ablations", "noise", "deviation_band", "all"], required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--learners", nargs="+", choices=["pac", "fsol"], default=["pac", "fsol"])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="online")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "wrs-reproduction-group8"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    args = p.parse_args()

    datasets = args.datasets or BATCH1_DATASETS
    jobs = []
    if args.suite in ("main", "all"):
        jobs.extend(_build_main_jobs(datasets, args.learners))
    if args.suite in ("ablations", "all"):
        jobs.extend(_build_ablation_jobs(datasets, args.learners))
    if args.suite in ("noise", "all"):
        noise_ds = [d for d in datasets if d in NOISE_DATASETS] if args.datasets is None else datasets
        jobs.extend(_build_noise_jobs(noise_ds, args.learners))
    if args.suite in ("deviation_band", "all"):
        jobs.extend(_build_deviation_band_jobs(datasets))

    pending = [j for j in jobs if not _is_done(Path(j["out"]), args.wandb_mode)]
    print(f"Total jobs: {len(jobs)} | Already done: {len(jobs) - len(pending)} | Pending: {len(pending)}")
    if not pending:
        return 0

    grouped = {}
    for job in pending:
        grouped.setdefault((job["dataset"], job["seed"]), []).append(job)
    tasks = [
        (ds, seed, ds_jobs, args.wandb_mode, args.wandb_project, args.wandb_entity)
        for (ds, seed), ds_jobs in sorted(grouped.items())
    ]
    if args.jobs <= 1:
        results = [_run_seed_group(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as pool:
            results = list(pool.map(_run_seed_group, tasks))
    for ds, seed, count in results:
        print(f"Completed {ds} seed={seed}: {count} jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
