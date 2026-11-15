"""Reproducible sparse dataset loading, HF provenance and train-only text fitting."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split


@dataclass
class DatasetBundle:
    """Train/test matrices, binary labels, split indices and source metadata."""
    X_train: object
    X_test: object
    y_train: np.ndarray
    y_test: np.ndarray
    metadata: dict
    train_indices: np.ndarray
    test_indices: np.ndarray


def validate_features(X, y):
    """Validate finite numeric features and aligned {-1,+1} labels, returning CSR/dense."""
    matrix = X.tocsr(copy=True) if sparse.issparse(X) else np.asarray(X, dtype=float)
    labels = np.asarray(y)
    if matrix.ndim != 2 or not matrix.shape[1] or labels.shape != (matrix.shape[0],) or len(labels) < 1:
        raise ValueError("Expected nonempty rows, positive feature count and one label per row")
    values = matrix.data if sparse.issparse(matrix) else matrix
    if not np.isfinite(values).all() or not np.isin(labels, [-1, 1]).all():
        raise ValueError("Rejecting NaN/Inf features or labels outside {-1,+1}")
    if sparse.issparse(matrix):
        matrix.sum_duplicates()
        if not np.isfinite(matrix.data).all():
            raise ValueError("Sparse duplicate summation produced non-finite values")
        matrix.eliminate_zeros()
    return matrix, labels.astype(np.int8)


def split_data(X, y, seed=42, train_fraction=.7, metadata=None):
    """Return DatasetBundle using random holdout, no stratification, with split checksum."""
    matrix, labels = validate_features(X, y)
    if len(labels) < 4:
        raise ValueError('At least four samples are required for a split')
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    train, test = train_test_split(np.arange(len(labels)), train_size=train_fraction, random_state=seed)
    meta = dict(metadata or {})
    meta.update({"n_samples": len(labels), "n_features": matrix.shape[1], "n_train": len(train), "n_test": len(test),
                 "split_seed": seed, "train_fraction": train_fraction, "stratified": False,
                 "split_sha256": hashlib.sha256(train.astype('<i8').tobytes() + test.astype('<i8').tobytes()).hexdigest()})
    return DatasetBundle(matrix[train], matrix[test], labels[train], labels[test], meta, train, test)


def training_label_noise(y_train, rate, seed):
    """Return copied labels and mask; flip each train label with probability rate."""
    if not 0 <= rate <= 1 or not np.isin(y_train, [-1, 1]).all():
        raise ValueError("Noise rate must be [0,1] and labels binary")
    mask = np.random.default_rng(seed).random(len(y_train)) < rate
    noisy = np.array(y_train, copy=True)
    noisy[mask] *= -1
    return noisy, mask


def load_dataset(dataset_id, seed=42, train_fraction=.7, source_file='configs/data_sources.json', max_samples=None):
    """Download a pinned HF asset and return split features/labels with SHA256 provenance.

    source_file entries define repo_id, revision, filename, format and binary label mapping.
    Supported formats: svmlight, numeric npz (X,y), text parquet (text/label columns).
    max_samples is explicitly a development subset; None loads the complete source asset.
    """
    from huggingface_hub import hf_hub_download
    specs = json.loads(Path(source_file).read_text(encoding='utf-8'))
    if dataset_id not in specs:
        raise ValueError(f"No verified hosted source for {dataset_id}; register a pinned HF source first")
    spec = specs[dataset_id]
    revision = spec['revision']
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError("Dataset revision must be a full HF commit SHA")
    cache = Path(__file__).resolve().parents[2] / 'data/cache/huggingface'
    path = Path(hf_hub_download(repo_id=spec['repo_id'], filename=spec['filename'], repo_type='dataset', revision=revision, cache_dir=cache))
    sha = _sha256(path)
    if spec.get('sha256') and sha != spec['sha256']:
        raise ValueError("Dataset checksum mismatch")
    metadata = {'dataset': dataset_id, 'provider': 'huggingface', 'repo_id': spec['repo_id'], 'revision': revision,
                'filename': spec['filename'], 'sha256': sha, 'source_notes': spec.get('notes'),
                'development_subset': max_samples is not None}
    text = None
    if spec['format'] == 'svmlight':
        from sklearn.datasets import load_svmlight_file
        X, y = load_svmlight_file(str(path), n_features=spec.get('n_features'))
    elif spec['format'] == 'npz':
        with np.load(path, allow_pickle=False) as data:
            X, y = data['X'], data['y']
    elif spec['format'] == 'text_parquet':
        import pyarrow.parquet as pq
        data = pq.read_table(path, columns=[spec['text_column'], spec['label_column']]).to_pydict()
        text, y = np.asarray(data[spec['text_column']], dtype=str), np.asarray(data[spec['label_column']])
        X = None
    else:
        raise ValueError("Unsupported source format")
    mapping = spec['label_mapping']
    if not np.isfinite(y).all() or not np.equal(y, np.floor(y)).all():
        raise ValueError('Source labels must be finite integral class IDs')
    try:
        y = np.asarray([mapping[str(int(label))] for label in y], dtype=np.int8)
    except (KeyError, ValueError) as exc:
        raise ValueError("Unmapped source label") from exc
    if max_samples is not None:
        if max_samples < 4 or max_samples > len(y):
            raise ValueError("max_samples must be between 4 and source row count")
        chosen = np.sort(np.random.default_rng(0).choice(len(y), max_samples, replace=False))
        metadata['subset_selection_seed'] = 0
        metadata['subset_indices_sha256'] = hashlib.sha256(chosen.astype('<i8').tobytes()).hexdigest()
        y = y[chosen]
        if text is not None:
            text = text[chosen]
        else:
            X = X[chosen]
    if text is None:
        return split_data(X, y, seed, train_fraction, metadata)
    from sklearn.feature_extraction.text import CountVectorizer
    train, test = train_test_split(np.arange(len(y)), train_size=train_fraction, random_state=seed)
    vectorizer = CountVectorizer(**spec.get('vectorizer', {}))
    X_train = vectorizer.fit_transform(text[train])
    X_test = vectorizer.transform(text[test])
    validate_features(X_train, y[train]); validate_features(X_test, y[test])
    metadata.update({'n_samples': len(y), 'n_features': X_train.shape[1], 'n_train': len(train), 'n_test': len(test),
                     'split_seed': seed, 'train_fraction': train_fraction, 'stratified': False,
                     'split_sha256': hashlib.sha256(train.astype('<i8').tobytes()+test.astype('<i8').tobytes()).hexdigest(),
                     'preprocessing': 'CountVectorizer fitted on train only', 'vocabulary_sha256': hashlib.sha256(json.dumps({k:int(v) for k,v in vectorizer.vocabulary_.items()}, sort_keys=True).encode()).hexdigest()})
    return DatasetBundle(X_train, X_test, y[train], y[test], metadata, train, test)


def _sha256(path):
    """Hash a file in bounded memory, returning its SHA256 hex digest."""
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
