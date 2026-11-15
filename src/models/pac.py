"""PA-II from the update described in Section 2 of Wu et al. (2024)."""
import numpy as np
from ._linear import LinearLearner


class PAC(LinearLearner):
    """Create a zero-initialized PA-II learner with n_features and penalty C_err."""

    def __init__(self, n_features, *, C_err):
        super().__init__(n_features)
        if not np.isfinite(C_err) or C_err <= 0:
            raise ValueError("C_err must be finite and positive")
        self.C_err = float(C_err)

    def update(self, x, y):
        """Learn from sample x, label y; return True iff the aggressive branch ran."""
        idx, v, loss = self._example(x, y)
        if loss == 0:
            return False
        tau = loss / (float(v @ v) + 0.5 / self.C_err)
        self._w[idx] += tau * y * v
        return True
