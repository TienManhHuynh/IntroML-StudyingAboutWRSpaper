"""Metrics for paired online-learning trajectories (paper Section 4 / Appendix B)."""
import numpy as np


def accuracy(y_true, y_pred):
    """Return correct-label fraction for equally sized, nonempty 1D label arrays."""
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or truth.shape != pred.shape or not truth.size:
        raise ValueError("Accuracy requires nonempty, equally sized 1D arrays")
    if not np.isin(truth, [-1, 1]).all() or not np.isin(pred, [-1, 1]).all():
        raise ValueError("Labels must be -1 or +1")
    return float(np.mean(truth == pred))


def sparsity(weights, zero_tolerance=0.0):
    """Return fraction of zero coordinates in finite 1D model weights; default exact zero."""
    w = np.asarray(weights)
    if w.ndim != 1 or not w.size or not np.isfinite(w).all() or not np.isfinite(zero_tolerance) or zero_tolerance < 0:
        raise ValueError("Expected a nonempty finite vector and nonnegative tolerance")
    return float(np.mean(np.abs(w) <= zero_tolerance))


def relative_oracle_performance(base_accuracy, method_accuracy):
    """Return mean(maximum.accumulate(base)-method) on one shared logging grid.

    Parameters: baseline and method checkpoint accuracy arrays in [0,1].
    Returns: float ROP; negative values are valid. This is the Appendix B.3 proxy.
    """
    base, method = np.asarray(base_accuracy, dtype=float), np.asarray(method_accuracy, dtype=float)
    if base.ndim != 1 or not base.size or base.shape != method.shape:
        raise ValueError("ROP requires aligned nonempty checkpoint arrays")
    if not all(np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all() for x in (base, method)):
        raise ValueError("Accuracy fractions must be finite and in [0,1]")
    return float(np.mean(np.maximum.accumulate(base) - method))


def checkpoint_steps(n_train, target=200):
    """Return distinct initial/final and near-even integer checkpoint steps for a pass."""
    if n_train <= 0 or target <= 0:
        raise ValueError("Training size and checkpoint target must be positive")
    return np.unique(np.linspace(0, n_train, min(n_train, target) + 1, dtype=int))


def deviation(paper_value, reproduced_value, near_zero=1e-12):
    """Return signed absolute delta and relative percent; relative is None near zero."""
    if not np.isfinite([paper_value, reproduced_value]).all():
        raise ValueError("Cannot compare non-finite values")
    delta = float(reproduced_value - paper_value)
    return {"delta": delta, "relative_percent": None if abs(paper_value) <= near_zero else 100 * delta / abs(paper_value)}
