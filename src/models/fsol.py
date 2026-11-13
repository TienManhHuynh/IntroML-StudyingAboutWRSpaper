"""First-order sparse learner described in Section 2 of Wu et al. (2024)."""
import numpy as np
from ._linear import LinearLearner


class FSOL(LinearLearner):
    """Create a learner with n_features, learning rate eta and penalty lambda_."""

    def __init__(self, n_features, *, eta, lambda_):
        super().__init__(n_features)
        if not np.isfinite(eta) or eta <= 0 or not np.isfinite(lambda_) or lambda_ < 0:
            raise ValueError("eta must be positive and lambda_ nonnegative, both finite")
        self.eta = float(eta)
        self._threshold = self.eta * lambda_
        self._theta = np.zeros_like(self._w)

    def update(self, x, y):
        """Learn from sample x, label y; return True iff hinge loss was positive."""
        idx, v, loss = self._example(x, y)
        if loss == 0:
            return False
        self._theta[idx] += self.eta * y * v
        # Constant threshold: untouched coordinates retain their previous weights.
        theta = self._theta[idx]
        self._w[idx] = np.sign(theta) * np.maximum(np.abs(theta) - self._threshold, 0)
        return True
