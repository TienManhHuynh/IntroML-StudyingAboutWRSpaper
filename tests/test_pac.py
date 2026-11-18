import numpy as np
import unittest
from scipy.sparse import csr_matrix
from src.models.pac import PAC


class TestLearner(unittest.TestCase):
    def test_pa_ii_margin_and_snapshot(self):
        for row in (np.array, csr_matrix):
            with self.subTest(row=row):
                model = PAC(2, C_err=0.5)
                x = row([[1., 0.]])
                before = model.get_weights()
                assert model.loss(x, 1) == 1
                assert model.update(x, 1)
                np.testing.assert_allclose(model.get_weights(), [0.5, 0])
                assert model.predict(x).tolist() == [1]
                assert model.loss(x, 1) == 0.5  # Correct label is insufficient.
                assert model.update(x, 1)
                np.testing.assert_allclose(model.get_weights(), [0.75, 0])
                assert not model.update(row([[2., 0.]]), 1)
                np.testing.assert_allclose(model.get_weights(), [0.75, 0])
                np.testing.assert_array_equal(before, [0, 0])


    def test_negative_label_duplicate_sparse_and_batch(self):
        model = PAC(2, C_err=0.5)
        x = csr_matrix(([1., 1.], [0, 0], [0, 2]), shape=(1, 2))
        model.update(x, -1)
        np.testing.assert_allclose(model.get_weights(), [-0.4, 0])
        np.testing.assert_array_equal(model.predict([[1, 0], [0, 1]]), [-1, 1])
