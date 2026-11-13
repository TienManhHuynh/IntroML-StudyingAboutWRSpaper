"""Common linear binary learner operations; inputs are not normalized or biased."""
import numpy as np
from scipy import sparse


class LinearLearner:
    """Initialize n_features zero weights. Zero prediction scores map to +1."""

    def __init__(self, n_features):
        if not isinstance(n_features, (int, np.integer)) or n_features <= 0:
            raise ValueError("n_features must be a positive integer")
        self._w = np.zeros(n_features, dtype=float)

    def _row(self, x):
        """Return (indices or full slice, values) for one dense or sparse sample x."""
        if sparse.issparse(x):
            row = x.tocsr()
            if row.shape != (1, self._w.size):
                raise ValueError("expected one sparse row with n_features columns")
            if not row.has_canonical_format:
                row = row.copy()
                row.sum_duplicates()
            return row.indices, row.data.astype(float, copy=False)
        row = np.asarray(x, dtype=float)
        if row.shape not in ((self._w.size,), (1, self._w.size)):
            raise ValueError("expected one sample with n_features values")
        return slice(None), row.reshape(-1)

    def _example(self, x, y):
        """Return coordinates, values and pre-update hinge loss for sample x, label y."""
        if y not in (-1, 1):
            raise ValueError("y must be -1 or +1")
        idx, v = self._row(x)
        return idx, v, max(0.0, 1.0 - y * float(self._w[idx] @ v))

    def loss(self, x, y):
        """Return hinge loss for sample x and label y without changing weights."""
        return self._example(x, y)[2]

    def get_weights(self):
        """Return an independent snapshot of the current prediction weights."""
        return self._w.copy()

    def decision_function(self, X):
        """Return a 1D array of scores for dense or sparse batch X."""
        matrix = X if sparse.issparse(X) else np.atleast_2d(np.asarray(X, dtype=float))
        if matrix.ndim != 2 or matrix.shape[1] != self._w.size:
            raise ValueError("expected a batch with n_features columns")
        return np.asarray(matrix @ self._w).reshape(-1)

    def predict(self, X):
        """Return labels -1 or +1 for batch X; ties map to +1."""
        return np.where(self.decision_function(X) >= 0, 1, -1)
