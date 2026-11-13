import numpy as np
import unittest
from scipy.sparse import csr_matrix
from src.models.fsol import FSOL


class TestLearner(unittest.TestCase):
    def test_threshold_passive_and_accumulated_theta(self):
        for row in (np.array, csr_matrix):
            with self.subTest(row=row):
                model = FSOL(2, eta=1, lambda_=0.5)
                x = row([[2., -0.25]])
                assert model.update(x, 1)
                saved = model.get_weights()
                np.testing.assert_allclose(saved, [1.5, 0])
                assert not model.update(x, 1)
                # Hidden theta[1] remains -0.25, although its prediction weight is zero.
                assert model.update(row([[0., 0.5]]), -1)
                np.testing.assert_allclose(model.get_weights(), [1.5, -0.25])
                np.testing.assert_allclose(saved, [1.5, 0])
                np.testing.assert_allclose(model.decision_function(row([[1., 0.], [0., 1.]])), [1.5, -0.25])


    def test_threshold_scales_with_learning_rate(self):
        model = FSOL(2, eta=0.5, lambda_=1)
        model.update([1, -2], 1)
        np.testing.assert_allclose(model.get_weights(), [0, -0.5])
