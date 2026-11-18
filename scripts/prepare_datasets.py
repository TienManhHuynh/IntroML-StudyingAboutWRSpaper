"""Prepare and verify Table 1 sparse datasets into canonical sparse_npz archives.

Downloads raw assets from their original public hosts (UCI, scikit-feature, scikit-learn,
LIBSVM) into data/raw/ and writes compressed CSR sparse_npz files into data/prepared/
ready for uploading to a Hugging Face dataset repository and pinning in data_sources.json.
"""
import argparse
import bz2
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile

import numpy as np
from scipy import io as spio
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups_vectorized, load_svmlight_file

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def _download(url, dest):
    """Download url to dest if absent, falling back to system curl on Windows SSL chain gaps."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except urllib.error.URLError:
        subprocess.run(["curl.exe", "-fsSL", url, "-o", str(tmp)], check=True)
    tmp.replace(dest)
    return dest


def _parse_uci_sparse(path, n_features, binary_indices=False):
    """Parse UCI dense-index sparse text files (1-based feature indices)."""
    rows, cols, vals = [], [], []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for r, line in enumerate(lines):
        for tok in line.split():
            if binary_indices:
                cols.append(int(tok) - 1)
                vals.append(1.0)
            else:
                idx, val = tok.split(":")
                cols.append(int(idx) - 1)
                vals.append(float(val))
            rows.append(r)
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(lines), n_features), dtype=np.float64)
    X.sum_duplicates()
    X.eliminate_zeros()
    return X


def _filter_nonempty(X, y):
    """Drop zero-nnz rows if present, matching the paper's sparse preprocessing."""
    mask = X.getnnz(axis=1) > 0
    if mask.all():
        return X, y
    return X[mask].tocsr(), y[mask]


def _save_sparse_npz(dataset_id, X, y):
    """Save CSR matrix and {-1,+1} labels as allow_pickle=False sparse_npz and return stats."""
    X = X.tocsr()
    X.sum_duplicates()
    X.eliminate_zeros()
    y = np.asarray(y, dtype=np.int8)
    if not set(np.unique(y)).issubset({-1, 1}):
        raise ValueError(f"{dataset_id}: expected binary labels in {{-1, +1}}, got {np.unique(y)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{dataset_id}.npz"
    np.savez_compressed(
        out,
        data=X.data.astype(np.float64, copy=False),
        indices=X.indices,
        indptr=X.indptr,
        shape=np.asarray(X.shape, dtype=np.int64),
        y=y,
    )
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    sparsity = 1.0 - X.nnz / (X.shape[0] * X.shape[1])
    return {
        "dataset": dataset_id,
        "path": str(out.relative_to(ROOT)),
        "size_mb": round(out.stat().st_size / (1024 * 1024), 3),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "nnz": int(X.nnz),
        "sparsity": round(sparsity, 6),
        "sha256": sha,
    }


def _find_file(root, filename):
    """Find a file recursively inside an extracted archive directory."""
    matches = list(Path(root).rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return matches[0]


def prepare_dexter():
    """Prepare UCI Dexter (train + valid = 600 labeled samples, 20000 features)."""
    zpath = _download("https://archive.ics.uci.edu/static/public/168/dexter.zip", RAW_DIR / "dexter.zip")
    extract_dir = RAW_DIR / "dexter"
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extract_dir)
    X_tr = _parse_uci_sparse(_find_file(extract_dir, "dexter_train.data"), 20000)
    y_tr = np.loadtxt(_find_file(extract_dir, "dexter_train.labels"), dtype=np.int8)
    X_va = _parse_uci_sparse(_find_file(extract_dir, "dexter_valid.data"), 20000)
    y_va = np.loadtxt(_find_file(extract_dir, "dexter_valid.labels"), dtype=np.int8)
    return _save_sparse_npz("dexter", sparse.vstack([X_tr, X_va], format="csr"), np.concatenate([y_tr, y_va]))


def prepare_dorothea():
    """Prepare UCI Dorothea (train + valid = 1150 labeled samples, 100000 binary features)."""
    zpath = _download("https://archive.ics.uci.edu/static/public/169/dorothea.zip", RAW_DIR / "dorothea.zip")
    extract_dir = RAW_DIR / "dorothea"
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(extract_dir)
    X_tr = _parse_uci_sparse(_find_file(extract_dir, "dorothea_train.data"), 100000, binary_indices=True)
    y_tr = np.loadtxt(_find_file(extract_dir, "dorothea_train.labels"), dtype=np.int8)
    X_va = _parse_uci_sparse(_find_file(extract_dir, "dorothea_valid.data"), 100000, binary_indices=True)
    y_va = np.loadtxt(_find_file(extract_dir, "dorothea_valid.labels"), dtype=np.int8)
    return _save_sparse_npz("dorothea", sparse.vstack([X_tr, X_va], format="csr"), np.concatenate([y_tr, y_va]))


def prepare_pcmac():
    """Prepare scikit-feature PCMAC (1943 samples, 3289 features)."""
    mpath = _download("https://jundongl.github.io/scikit-feature/files/datasets/PCMAC.mat", RAW_DIR / "PCMAC.mat")
    mat = spio.loadmat(mpath)
    X = sparse.csr_matrix(mat["X"], dtype=np.float64)
    raw_y = np.asarray(mat["Y"]).reshape(-1)
    y = np.where(raw_y == np.max(raw_y), 1, -1).astype(np.int8)
    return _save_sparse_npz("pcmac", X, y)


def prepare_newsgroups_binary_cs():
    """Prepare Newsgroups (Binary, CS) matching sklearn vectorized + nonzero rows."""
    ds = fetch_20newsgroups_vectorized(subset="all", remove=("headers", "footers", "quotes"))
    cs_indices = {i for i, name in enumerate(ds.target_names) if name.startswith(("comp.", "sci."))}
    y = np.asarray([1 if int(t) in cs_indices else -1 for t in ds.target], dtype=np.int8)
    X, y = _filter_nonempty(ds.data, y)
    return _save_sparse_npz("newsgroups-binary-cs", X, y)


def _load_libsvm_bz2(url, filename, n_features):
    """Download a LIBSVM .bz2 file (or plain file) and parse with load_svmlight_file."""
    path = _download(url, RAW_DIR / filename)
    if path.suffix == ".bz2":
        with bz2.open(path, "rb") as stream:
            return load_svmlight_file(stream, n_features=n_features)
    return load_svmlight_file(str(path), n_features=n_features)


def prepare_news20():
    """Prepare LIBSVM news20.binary (D=1355191)."""
    X, y = _load_libsvm_bz2(
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/news20.binary.bz2",
        "news20.binary.bz2",
        1355191,
    )
    y = np.where(y > 0, 1, -1).astype(np.int8)
    X, y = _filter_nonempty(X, y)
    return _save_sparse_npz("news20", X, y)


def prepare_w8a():
    """Prepare LIBSVM w8a + w8a.t (D=300)."""
    X_tr, y_tr = _load_libsvm_bz2("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a", "w8a", 300)
    X_te, y_te = _load_libsvm_bz2("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t", "w8a.t", 300)
    X = sparse.vstack([X_tr, X_te], format="csr")
    y = np.where(np.concatenate([y_tr, y_te]) > 0, 1, -1).astype(np.int8)
    X, y = _filter_nonempty(X, y)
    return _save_sparse_npz("w8a", X, y)


def prepare_real_sim():
    """Prepare LIBSVM real-sim (D=20958)."""
    X, y = _load_libsvm_bz2(
        "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",
        "real-sim.bz2",
        20958,
    )
    y = np.where(y > 0, 1, -1).astype(np.int8)
    X, y = _filter_nonempty(X, y)
    return _save_sparse_npz("real-sim", X, y)


BUILDERS = {
    "dexter": prepare_dexter,
    "dorothea": prepare_dorothea,
    "pcmac": prepare_pcmac,
    "newsgroups-binary-cs": prepare_newsgroups_binary_cs,
    "news20": prepare_news20,
    "w8a": prepare_w8a,
    "real-sim": prepare_real_sim,
}


def main():
    """Prepare requested datasets and compare against Table 1 metadata in configs/datasets.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(BUILDERS), choices=list(BUILDERS))
    args = parser.parse_args()
    table1 = {d["id"]: d for d in json.loads((ROOT / "configs/datasets.json").read_text(encoding="utf-8"))["datasets"]}
    rows = []
    for name in args.datasets:
        stats = BUILDERS[name]()
        ref = table1[name]
        stats["paper_samples"] = ref["paper_samples"]
        stats["paper_dimensions"] = ref["paper_dimensions"]
        stats["paper_sparsity"] = ref["paper_data_sparsity"]
        rows.append(stats)
        print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
