"""Small exact cases and sampling-distribution checks, independent of TV1 code."""
import math

import numpy as np
import pytest
from scipy import sparse

from src.wat import WAT, WeightedReservoir


class FixedRandom:
    def __init__(self, uniforms):
        self.values = iter(1 - np.array(uniforms, dtype=float))

    def random(self):
        return next(self.values)


def test_reservoir_matches_direct_priority_reference():
    survivals = [0, 1, 5, 2, 9, 3, 4]
    uniforms = [.2, .7, .4, .9, .3, .6, .8]
    reservoir = WeightedReservoir(3)
    reservoir._rng = FixedRandom(uniforms)
    for index, survival in enumerate(survivals):
        reservoir.offer([index], survival)
        assert len(reservoir) <= 3
    expected = sorted(range(len(survivals)), key=lambda i:
                      uniforms[i] ** (1 / (survivals[i] + 1e-8)), reverse=True)[:3]
    assert {int(c.weights[0]) for c in reservoir.candidates} == set(expected)


def test_snapshot_and_public_access_do_not_alias():
    reservoir = WeightedReservoir(2)
    vector = np.array([1., 2.])
    reservoir.offer(vector, 1)
    vector[:] = 99
    reservoir.candidates[0].weights[:] = -99
    reservoir.aggregate()[:] = 50
    np.testing.assert_array_equal(reservoir.candidates[0].weights, [1, 2])


def test_exact_tie_keeps_existing_candidate():
    reservoir = WeightedReservoir(1)
    reservoir._rng = FixedRandom([.5, .5])
    assert reservoir.offer([1], 2)
    assert not reservoir.offer([2], 2)
    assert reservoir.candidates[0].weights[0] == 1


@pytest.mark.parametrize("weighting", ["standard", "exponential"])
def test_seed_reproducibility_and_bound(weighting):
    left = WeightedReservoir(4, weighting, seed=42)
    right = WeightedReservoir(4, weighting, seed=42)
    for i in range(100):
        assert left.offer([i, -i], i) == right.offer([i, -i], i)
        assert len(left) <= 4
    for a, b in zip(left.candidates, right.candidates):
        assert a.log_race == b.log_race
        np.testing.assert_array_equal(a.weights, b.weights)


@pytest.mark.parametrize("weighting,expected", [
    ("standard", 3 / 4), ("exponential", math.exp(3) / (math.exp(1) + math.exp(3)))])
def test_k1_selection_frequency_matches_weight_ratio(weighting, expected):
    # Independent trials; a generous fixed tolerance avoids a fragile exact count.
    wins = 0
    for seed in range(3000):
        reservoir = WeightedReservoir(1, weighting, seed)
        reservoir.offer([1], 1)
        reservoir.offer([3], 3)
        wins += reservoir.candidates[0].weights[0] == 3
    assert abs(wins / 3000 - expected) < .035


def test_averages_use_actual_size_and_original_weights():
    reservoir = WeightedReservoir(8)
    reservoir.offer([2, 0], 1)
    reservoir.offer([0, 8], 3)
    np.testing.assert_allclose(reservoir.aggregate("simple"), [1, 4])
    np.testing.assert_allclose(reservoir.aggregate("weighted"), [.5, 6])


def test_exponential_extreme_survival_is_finite_and_correct():
    reservoir = WeightedReservoir(3, "exponential", seed=1)
    reservoir.offer([2, 4], 10000)
    reservoir.offer([6, 8], 10001)
    a = 1 / (1 + math.e)
    np.testing.assert_allclose(reservoir.aggregate("weighted"),
                               a * np.array([2, 4]) + (1-a) * np.array([6, 8]))
    assert all(np.isfinite(c.log_race) for c in reservoir.candidates)
    # Capacity one must still order huge weights without exp overflow.
    one = WeightedReservoir(1, "exponential")
    one._rng = FixedRandom([.5, .5])
    one.offer([1], 10000)
    assert one.offer([2], 10001)


def test_zero_weights_fallback_and_strict_zero_majority():
    reservoir = WeightedReservoir(4)
    for vector in ([0, 0, 4], [0, 0, 4], [0, 2, 4], [8, 2, 4]):
        reservoir.offer(vector, 0)
    np.testing.assert_allclose(reservoir.aggregate("weighted"), [2, 1, 4])
    # First coordinate: 3/4 zero -> zero; second: 2/4 -> keep.
    np.testing.assert_allclose(reservoir.aggregate("weighted", True), [0, 1, 4])


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        WeightedReservoir(capacity)


def test_invalid_reservoir_inputs():
    reservoir = WeightedReservoir(2)
    with pytest.raises(ValueError):
        reservoir.aggregate()
    for vector, survival in [([], 1), ([np.nan], 1), ([[1]], 1), ([1], -1), ([1], .5)]:
        with pytest.raises(ValueError):
            reservoir.offer(vector, survival)
    reservoir.offer([1], 0)
    with pytest.raises(ValueError):
        reservoir.offer([1, 2], 1)
    with pytest.raises(ValueError):
        WeightedReservoir(2, "unknown")


class InPlacePA:
    """Minimal PA-II test learner, deliberately returning aliased weights."""

    def __init__(self):
        self.w = np.zeros(2)
        self.updates = 0

    def loss(self, x, y):
        return max(0., 1 - y * np.dot(self.w, x))

    def update(self, x, y):
        self.w += self.loss(x, y) / (np.dot(x, x) + .5) * y * np.asarray(x)
        self.updates += 1

    def get_weights(self):
        return self.w


def test_wat_margin_survival_and_preupdate_snapshot():
    base = InPlacePA()
    base.w[:] = [2, 0]
    wat = WAT(base, K=8)
    wat.update([1, 0], 1).update([1, 1], 1)
    assert wat.survival == 2
    assert base.updates == 0
    # Correct positive prediction but margin .5: must be aggressive.
    wat.update([.25, 0], 1)
    assert base.updates == 1
    assert wat.survival == 0
    assert wat.reservoir.candidates[0].survival == 2
    np.testing.assert_array_equal(wat.get_weights(), [2, 0])
    assert base.w[0] > 2


@pytest.mark.parametrize("averaging", ["simple", "weighted"])
def test_wat_ensemble_zeroing_and_independent_active_state(averaging):
    base = InPlacePA()
    wat = WAT(base, K=4, averaging=averaging, voting_zeroing=True)
    for vector, survival in [([0, 1], 1), ([0, 2], 2), ([9, 3], 3)]:
        wat.reservoir.offer(vector, survival)
    expected_second = 2 if averaging == "simple" else 14 / 6
    np.testing.assert_allclose(wat.get_weights(), [0, expected_second])
    np.testing.assert_array_equal(base.w, [0, 0])


def test_inference_empty_terminal_and_sparse_batch():
    base = InPlacePA()
    base.w[:] = [2, -1]
    wat = WAT(base, seed=12)
    wat.update([1, 0], 1)
    detached = wat.get_weights()
    detached[:] = 100
    X = np.array([[1, 1], [0, 1], [0, 0]])
    for batch in (X, sparse.csr_matrix(X)):
        np.testing.assert_array_equal(wat.decision_function(batch), [1, -1, 0])
        np.testing.assert_array_equal(wat.predict(batch), [1, -1, 1])
    assert wat.survival == 1
    assert len(wat.reservoir) == 0  # No terminal flush or inference insertion.
    np.testing.assert_array_equal(base.w, [2, -1])


def test_checkpoint_reads_do_not_change_future_sampling():
    a, b = WAT(InPlacePA(), K=2, seed=9), WAT(InPlacePA(), K=2, seed=9)
    for i in range(20):
        x, y = [1, i % 2], (-1 if i % 3 else 1)
        a.update(x, y)
        a.get_weights()
        a.predict(np.eye(2))
        b.update(x, y)
    np.testing.assert_array_equal(a.get_weights(), b.get_weights())
    assert [c.log_race for c in a.reservoir.candidates] == [c.log_race for c in b.reservoir.candidates]


def test_invalid_wat_options_and_loss():
    with pytest.raises(ValueError):
        WAT(InPlacePA(), averaging="unknown")
    with pytest.raises(ValueError):
        WAT(InPlacePA(), voting_zeroing="false")
    wat = WAT(InPlacePA())
    with pytest.raises(ValueError):
        wat.update([1, 0], 0)
    for invalid in (float("nan"), float("inf"), -1):
        wat.base_learner.loss = lambda x, y, invalid=invalid: invalid
        with pytest.raises(ValueError):
            wat.update([1, 0], 1)
    assert len(wat.reservoir) == 0
