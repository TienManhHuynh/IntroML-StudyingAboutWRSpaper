"""Ablation experiment runner for Top-K, Moving Average, and Exponential Moving Average."""
import argparse
from collections import deque
import csv
import heapq
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_reproduction import write_json
from src.data.loader import load_dataset, split_data, validate_features
from src.evaluation.metrics import accuracy, checkpoint_steps, relative_oracle_performance, sparsity
from src.evaluation.tracking import track_run
from src.models.fsol import FSOL
from src.models.pac import PAC


class _LinearWrapperMixin:
    """Shared linear prediction methods backed by get_weights()."""

    def decision_function(self, X):
        """Return 1D scores for a dense or sparse batch X."""
        return np.asarray(X @ self.get_weights()).reshape(-1)

    def predict(self, X):
        """Return {-1, +1} labels with zero scores mapped to +1."""
        return np.where(self.decision_function(X) >= 0, 1, -1)


class TopKEnsemble(_LinearWrapperMixin):
    """Deterministic top-K survival ensemble (paper Section 4.2, Tables 3-4).

    Retains up to K outgoing snapshots with the largest passive survival counts.
    Ties keep the incumbent candidate to match WeightedReservoir tie policy.
    """

    def __init__(self, base_learner, K=64, averaging="simple"):
        if K < 1:
            raise ValueError("K must be a positive integer")
        if averaging not in ("simple", "weighted"):
            raise ValueError("averaging must be 'simple' or 'weighted'")
        self.base_learner = base_learner
        self.K = int(K)
        self.averaging = averaging
        self.survival = 0
        self._serial = 0
        self._heap = []

    def update(self, x, y):
        """Process one sample, offering the pre-update snapshot on positive loss."""
        loss = float(self.base_learner.loss(x, y))
        if loss > 0:
            self._serial += 1
            entry = (self.survival, self._serial, self.base_learner.get_weights().copy())
            if len(self._heap) < self.K:
                heapq.heappush(self._heap, entry)
            elif self.survival > self._heap[0][0]:
                heapq.heapreplace(self._heap, entry)
            self.base_learner.update(x, y)
            self.survival = 0
        else:
            self.survival += 1
        return self

    def get_weights(self):
        """Return detached ensemble weights, or active weights before first update."""
        if not self._heap:
            return self.base_learner.get_weights()
        if self.averaging == "simple":
            weights = np.zeros_like(self._heap[0][2])
            for _, _, vec in self._heap:
                weights += vec
            return weights / len(self._heap)
        survivals = np.array([item[0] for item in self._heap], dtype=float)
        total = survivals.sum()
        coeffs = survivals / total if total > 0 else np.full(len(self._heap), 1.0 / len(self._heap))
        weights = np.zeros_like(self._heap[0][2])
        for coeff, (_, _, vec) in zip(coeffs, self._heap):
            weights += coeff * vec
        return weights


class MovingAverageEnsemble(_LinearWrapperMixin):
    """Sliding-window moving average over the most recent K weight vectors (Appendix D)."""

    def __init__(self, base_learner, K=64):
        if K < 1:
            raise ValueError("K must be a positive integer")
        self.base_learner = base_learner
        self.K = int(K)
        self._window = deque()
        self._sum = None

    def update(self, x, y):
        """Update base learner and append its new weight vector to the K-window."""
        self.base_learner.update(x, y)
        w_new = self.base_learner.get_weights()
        if self._sum is None:
            self._sum = np.zeros_like(w_new)
        if len(self._window) == self.K:
            self._sum -= self._window.popleft()
        self._window.append(w_new)
        self._sum += w_new
        return self

    def get_weights(self):
        """Return detached mean of the current window (or active weights if empty)."""
        if not self._window:
            return self.base_learner.get_weights()
        return self._sum / len(self._window)


class ExponentialAverageEnsemble(_LinearWrapperMixin):
    """Exponential moving average w_bar_t = gamma * w_t + (1 - gamma) * w_bar_{t-1} (Appendix D)."""

    def __init__(self, base_learner, gamma=0.9):
        if not 0 < gamma <= 1:
            raise ValueError("gamma must lie in (0, 1]")
        self.base_learner = base_learner
        self.gamma = float(gamma)
        self._ema = None

    def update(self, x, y):
        """Update base learner and advance exponential moving average."""
        self.base_learner.update(x, y)
        w_t = self.base_learner.get_weights()
        if self._ema is None:
            self._ema = np.zeros_like(w_t)
        self._ema = self.gamma * w_t + (1.0 - self.gamma) * self._ema
        return self

    def get_weights(self):
        """Return detached EMA vector (or active weights before step 1)."""
        if self._ema is None:
            return self.base_learner.get_weights()
        return self._ema.copy()


def build_ablation_model(config, base_learner):
    """Construct the requested ablation wrapper from config."""
    method = config["method"]
    if method == "topk":
        return TopKEnsemble(base_learner, K=config.get("K", 64), averaging=config.get("averaging", "simple"))
    if method == "moving_average":
        return MovingAverageEnsemble(base_learner, K=config.get("K", 64))
    if method == "exponential_average":
        return ExponentialAverageEnsemble(base_learner, gamma=config.get("gamma", 0.9))
    raise ValueError(f"Unsupported ablation method: {method}")


def run_ablation(config, output, bundle=None):
    """Run one paired baseline vs ablation trial, save evidence and track it."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "wandb_url": None, "reportable": False}
    write_json(out / "run.json", status)
    try:
        if bundle is None:
            if config["dataset"] == "synthetic":
                from sklearn.datasets import make_classification
                X, y = make_classification(
                    n_samples=config.get("samples", 300), n_features=20, n_informative=8, random_state=config["seed"]
                )
                bundle = split_data(X, 2 * y - 1, seed=config["seed"], metadata={"dataset": "synthetic", "development_subset": True})
            else:
                bundle = load_dataset(
                    config["dataset"],
                    seed=config["seed"],
                    source_file=config.get("sources", str(ROOT / "configs/data_sources.json")),
                    max_samples=config.get("max_samples"),
                )
        validate_features(bundle.X_train, bundle.y_train)
        validate_features(bundle.X_test, bundle.y_test)
        config = dict(config, data=bundle.metadata)
        write_json(out / "config.json", config)
        np.savez_compressed(out / "split-indices.npz", train=bundle.train_indices, test=bundle.test_indices)
        try:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        versions = {}
        for package in ("numpy", "scipy", "scikit-learn", "wandb", "huggingface-hub", "pyarrow"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        write_json(out / "environment.json", {"git_commit": commit, "dirty": dirty, "python": platform.python_version(), "platform": platform.platform(), "packages": versions})

        def base():
            return (
                PAC(bundle.X_train.shape[1], C_err=config["C_err"])
                if config["learner"] == "pac"
                else FSOL(bundle.X_train.shape[1], eta=config["eta"], lambda_=config["lambda_"])
            )

        method = config["method"]
        models = {"baseline": base(), method: build_ablation_model(config, base())}
        elapsed = {name: 0.0 for name in models}
        checkpoints, evaluation_seconds = [], 0.0
        steps = set(checkpoint_steps(len(bundle.y_train), config.get("checkpoints", 200)).tolist())
        for step in range(len(bundle.y_train) + 1):
            if step in steps:
                start = time.perf_counter()
                for name, model in models.items():
                    w = model.get_weights()
                    pred = np.where(np.asarray(bundle.X_test @ w).reshape(-1) >= 0, 1, -1)
                    checkpoints.append({"step": step, "model": name, "accuracy": accuracy(bundle.y_test, pred), "sparsity": sparsity(w)})
                evaluation_seconds += time.perf_counter() - start
            if step < len(bundle.y_train):
                x = bundle.X_train[step]
                y_val = int(bundle.y_train[step])
                for name, model in models.items():
                    start = time.perf_counter()
                    model.update(x, y_val)
                    elapsed[name] += time.perf_counter() - start

        base_curve = [r["accuracy"] for r in checkpoints if r["model"] == "baseline"]
        metrics = {}
        for name in models:
            rows = [r for r in checkpoints if r["model"] == name]
            metrics[name] = {
                "final_accuracy": rows[-1]["accuracy"],
                "final_sparsity": rows[-1]["sparsity"],
                "rop": relative_oracle_performance(base_curve, [r["accuracy"] for r in rows]),
                "train_seconds": elapsed[name],
                "evaluation_seconds_total": evaluation_seconds,
            }
        with (out / "checkpoints.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "model", "accuracy", "sparsity"])
            writer.writeheader()
            writer.writerows(checkpoints)
        write_json(out / "metrics.json", metrics)
        url = track_run(
            config, checkpoints, metrics, out,
            mode=config.get("wandb_mode", "disabled"),
            project=config.get("wandb_project"),
            entity=config.get("wandb_entity"),
        )
        development = bool(bundle.metadata.get("development_subset")) or config["dataset"] == "synthetic"
        status.update(
            status="completed_online" if url else "completed_local",
            wandb_url=url,
            reportable=False,
            development_run=development,
            note="Ablation run recorded; verify protocol and public W&B before reporting.",
        )
        write_json(out / "run.json", status)
        return status
    except BaseException as exc:
        status.update(status="failed", error_type=type(exc).__name__)
        write_json(out / "run.json", status)
        raise


def parser():
    """Return CLI argument parser for Top-K, Moving Average, and EMA ablations."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--learner", choices=["pac", "fsol"], default="pac")
    p.add_argument("--method", choices=["topk", "moving_average", "exponential_average"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--C-err", dest="C_err", type=float, default=1.0)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--averaging", choices=["simple", "weighted"], default="simple")
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--checkpoints", type=int, default=200)
    p.add_argument("--sources", default=str(ROOT / "configs/data_sources.json"))
    p.add_argument("--wandb-mode", choices=["disabled", "offline", "online"], default="disabled")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "wrs-reproduction-group8"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    p.add_argument("--output", type=Path, required=True)
    return p


def main():
    """Run one ablation trial from CLI arguments."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    args = vars(parser().parse_args())
    out = args.pop("output")
    args["experiment"] = "ablation-development" if args["dataset"] == "synthetic" or args["max_samples"] else "ablation"
    print(json.dumps(run_ablation(args, out), indent=2))


if __name__ == "__main__":
    main()
