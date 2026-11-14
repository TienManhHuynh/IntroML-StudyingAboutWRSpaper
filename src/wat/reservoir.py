"""Bounded weighted sampling of model snapshots (paper Algorithm 1).

Use an exponential race: log(-log(u)) - log(b + epsilon), smaller wins.
This orders candidates exactly like descending u**(1/(b+epsilon)), while
avoiding overflow for b=exp(s) and priorities rounding to one.
"""
from dataclasses import dataclass
import heapq
import math
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class Candidate:
    """Detached model snapshot, survival, log averaging weight and race key."""

    weights: np.ndarray
    survival: int
    log_weight: float
    log_race: float


class WeightedReservoir:
    """Keep at most K snapshots, sampled without replacement by survival weight.

    Parameters: capacity (positive integer), weighting ('standard' or
    'exponential'), seed for a private NumPy RNG. Storage is O(KD).
    """

    def __init__(self, capacity, weighting="standard", seed=None):
        if isinstance(capacity, bool) or not isinstance(capacity, Integral) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if weighting not in ("standard", "exponential"):
            raise ValueError("weighting must be 'standard' or 'exponential'")
        self.capacity = int(capacity)
        self.weighting = weighting
        self._rng = np.random.default_rng(seed)
        self._heap = []
        self._serial = 0
        self._shape = None

    def __len__(self):
        return len(self._heap)

    @property
    def candidates(self):
        """Return detached entries in insertion order; callers cannot mutate storage."""
        return tuple(
            Candidate(c.weights.copy(), c.survival, c.log_weight, c.log_race)
            for _, _, c in sorted(self._heap, key=lambda entry: entry[1])
        )

    def offer(self, weights, survival):
        """Consider a 1D finite dense vector before its learner is updated.

        Returns True if retained. Snapshot copy occurs only on acceptance.
        Survival zero has paper's epsilon sampling mass, but zero averaging
        weight under standard weighting. Equal keys keep existing entries.
        """
        if isinstance(survival, bool) or not isinstance(survival, Integral) or survival < 0:
            raise ValueError("survival must be a nonnegative integer")
        vector = np.asarray(weights, dtype=float)
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("weights must be a nonempty finite 1D vector")
        if self._shape is not None and vector.shape != self._shape:
            raise ValueError("all snapshots must have the same shape")
        log_weight = (math.log(survival) if survival else -math.inf)
        if self.weighting == "exponential":
            log_weight = float(survival)
        log_mass = float(np.logaddexp(log_weight, math.log(1e-8)))
        # 1-random() is in (0,1]; exclude the single endpoint u=1.
        u = 1.0
        while u == 1.0:
            u = 1.0 - self._rng.random()
        log_race = math.log(-math.log(u)) - log_mass
        self._serial += 1
        if len(self) == self.capacity and log_race >= -self._heap[0][0]:
            return False
        candidate = Candidate(vector.copy(), int(survival), log_weight, log_race)
        # Negate race to make heap root the worst (largest race) candidate.
        entry = (-log_race, self._serial, candidate)
        if len(self) < self.capacity:
            heapq.heappush(self._heap, entry)
        else:
            heapq.heapreplace(self._heap, entry)
        self._shape = vector.shape
        return True

    def aggregate(self, averaging="simple", voting_zeroing=False):
        """Return a fresh ensemble vector; empty reservoir raises ValueError.

        Weighted averaging uses sampling weights without epsilon. If all are
        zero, fall back to simple averaging. VZ uses a strict zero majority.
        """
        if averaging not in ("simple", "weighted"):
            raise ValueError("averaging must be 'simple' or 'weighted'")
        if not self._heap:
            raise ValueError("cannot aggregate an empty reservoir")
        entries = [entry[2] for entry in self._heap]
        logs = np.array([c.log_weight for c in entries])
        coefficients = np.ones(len(entries))
        if averaging == "weighted" and np.isfinite(logs).any():
            coefficients = np.exp(logs - logs.max())
        coefficients /= coefficients.sum()
        # Do not stack K vectors into an additional dense K-by-D array.
        result = np.zeros_like(entries[0].weights)
        zeros = np.zeros(result.shape, dtype=np.int64) if voting_zeroing else None
        for coefficient, candidate in zip(coefficients, entries):
            result += coefficient * candidate.weights
            if zeros is not None:
                zeros += candidate.weights == 0
        if zeros is not None:
            result[zeros > len(entries) / 2] = 0
        return result
