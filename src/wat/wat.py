"""WRS-Augmented Training wrapper for TV1's loss/update/get_weights API."""
from typing import Protocol

import numpy as np

from .reservoir import WeightedReservoir


class BaseLearner(Protocol):
    """TV1 integration surface; loss must not mutate learner state.

    get_weights returns a dense 1D vector. update accepts the original sparse
    or dense sample and owns all base-specific state (e.g. FSOL theta).
    """

    def loss(self, x, y) -> float:
        """Return nonnegative margin loss before updating."""
        ...

    def update(self, x, y):
        """Apply one base-method update."""
        ...

    def get_weights(self) -> np.ndarray:
        """Return current prediction vector."""
        ...


class WAT:
    """Wrap a passive-aggressive learner without feeding back the ensemble.

    Parameters: base_learner implementing BaseLearner, K, weighting,
    averaging ('simple'/'weighted'), voting_zeroing and optional seed.
    The active terminal candidate is not flushed (Algorithm 1).
    """

    def __init__(self, base_learner: BaseLearner, K=64, weighting="standard",
                 averaging="simple", voting_zeroing=False, seed=None):
        if averaging not in ("simple", "weighted"):
            raise ValueError("averaging must be 'simple' or 'weighted'")
        if not isinstance(voting_zeroing, (bool, np.bool_)):
            raise ValueError("voting_zeroing must be boolean")
        self.base_learner = base_learner
        self.reservoir = WeightedReservoir(K, weighting, seed)
        self.averaging = averaging
        self.voting_zeroing = bool(voting_zeroing)
        self.survival = 0

    def update(self, x, y):
        """Process one labeled sample; return self for chaining.

        Passive means loss == 0, not merely a correct predicted sign.
        Capture outgoing weights before a potentially in-place base update.
        No update return value is required from TV1.
        """
        if y not in (-1, 1):
            raise ValueError("y must be -1 or +1")
        loss = float(self.base_learner.loss(x, y))
        if not np.isfinite(loss) or loss < 0:
            raise ValueError("base learner loss must be finite and nonnegative")
        if loss > 0:
            self.reservoir.offer(self.base_learner.get_weights(), self.survival)
            self.base_learner.update(x, y)
            self.survival = 0
        else:
            self.survival += 1
        return self

    def get_weights(self):
        """Return detached ensemble; use active weights while reservoir is empty.

        Does not flush the terminal candidate or consume random numbers.
        """
        if not len(self.reservoir):
            return np.array(self.base_learner.get_weights(), dtype=float, copy=True)
        return self.reservoir.aggregate(self.averaging, self.voting_zeroing)

    def decision_function(self, X):
        """Return 1D scores for a dense/sparse batch or a single dense vector."""
        return np.asarray(X @ self.get_weights()).reshape(-1)

    def predict(self, X):
        """Return labels -1/+1; score == 0 maps to +1 (documented convention)."""
        return np.where(self.decision_function(X) >= 0, 1, -1)
