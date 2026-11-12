"""Validate source provenance, local Markdown links and Full scope plan integrity."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

from plan_experiments import build_jobs

ROOT = Path(__file__).resolve().parents[1]


def check():
    """Validate repository files and planned coverage; return a list of errors.

    Parameters: none; repository root is derived from this script location.
    Returns: list[str]; empty means scaffold checks passed, not reproduction.
    """
    errors = []
    config = json.loads((ROOT / "configs/full_scope.json").read_text(encoding="utf-8"))
    registry = json.loads((ROOT / "configs/datasets.json").read_text(encoding="utf-8"))["datasets"]
    expected = {"avazu-app", "avazu-site", "criteo", "dexter", "dorothea", "kdd2010-algebra", "mnist8-4-9", "news20", "newsgroups-binary-cs", "pcmac", "rcv1", "real-sim", "sst2", "url", "w8a", "webspam"}
    if config["scope"] != "full" or set(config["dataset_ids"]) != expected or len(config["dataset_ids"]) != 16:
        errors.append("Full scope must cover the exact 16 Table 1 dataset IDs.")
    if {d["id"] for d in registry} != expected or len(registry) != 16:
        errors.append("Dataset registry is incomplete or duplicated.")
    for dataset in registry:
        if dataset["paper_train_samples"] + dataset["paper_test_samples"] != dataset["paper_samples"]:
            errors.append("Paper sample counts inconsistent: " + dataset["id"])
    if len(config["seeds"]) != 5 or len(set(config["seeds"])) != 5:
        errors.append("Five distinct trial seeds required.")
    jobs = build_jobs(config)
    counts = Counter(j["method"] for j in jobs)
    if counts != Counter({"baseline": 160, "wrs": 5120, "topk": 320, "moving_average": 160, "exponential_average": 160}):
        errors.append("Full scope job grid differs from documented 5,920 jobs.")
    if len({j["job_id"] for j in jobs}) != len(jobs):
        errors.append("Duplicate job IDs.")
    if any(j["execution_ready"] or j["status"] != "planned" for j in jobs):
        errors.append("A generated plan must not claim actual execution.")
    sources = json.loads((ROOT / "docs/requirements/source-manifest.json").read_text(encoding="utf-8"))["files"]
    for source in sources:
        path = ROOT / source["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            errors.append("Source snapshot differs: " + source["path"])
    for folder in ["papers", "docs/requirements", "src/models", "src/wat", "src/data", "src/evaluation", "experiments", "tests", "results", "figures", "slides"]:
        if not (ROOT / folder).is_dir():
            errors.append("Missing directory: " + folder)
    markdown = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    for folder in ["docs", "papers", "src", "experiments", "tests", "results", "figures", "slides"]:
        markdown.extend((ROOT / folder).rglob("*.md"))
    for path in markdown:
        for match in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            link = match.group(1).split("#", 1)[0]
            if not link or re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link):
                continue
            target = path.parent / unquote(link)
            if not target.exists():
                errors.append("Broken local link in {}: {}".format(path.relative_to(ROOT), link))
    with (ROOT / "results/coverage.csv").open(encoding="utf-8", newline="") as stream:
        coverage = list(csv.DictReader(stream))
    for group in ["MAIN", "TOPK", "AVERAGING", "RUNTIME", "REFERENCE", "NOISE"]:
        group_rows = [row for row in coverage if row["group"] == group]
        if {row["dataset"] for row in group_rows} != expected or len(group_rows) != 16:
            errors.append("Incomplete or duplicated coverage: " + group)
    for row in coverage:
        if row["status"] == "completed" and (not row["wandb_url"].startswith("https://wandb.ai/") or not row["artifact"]):
            errors.append("Completed coverage lacks W&B URL/artifact: " + row["group"] + "/" + row["dataset"])
    return errors


if __name__ == "__main__":
    try:
        issues = check()
    except (KeyError, ValueError, OSError) as exc:
        print("Scaffold check failed: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)
    if issues:
        print("\n".join(issues), file=sys.stderr)
        raise SystemExit(1)
    print("PASS: 16 datasets, 5,920 planned jobs, unique IDs, source checksums, local links and coverage.")
    print("This validates the scaffold only; algorithms, runs, public W&B and PPTX are pending.")
