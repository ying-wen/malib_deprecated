# Created by yingwen at 2019-03-10

import numpy as np
import tensorflow as tf
from malib.distributions import SquashBijector


class SquashBijectorTest(tf.test.TestCase):
    def setUp(self):
        self.squash_bijector = SquashBijector()

    def test_forward(self):
        x = np.zeros((1,))
        expected = np.tanh(x)
        y = self.squash_bijector.forward(x)
        self.assertEqual(expected, y.numpy())

    def test_inverse(self):
        x = np.zeros((1,))
        expected = np.arctan(x)
        y = self.squash_bijector.inverse(x)
        self.assertEqual(expected, y.numpy())

    def forward_log_det_jacobian(self):
        x = np.zeros((1,))
        expected = 2. * (np.log(2.) - x - np.log(1.0 + np.exp(x)))
        y = self.squash_bijector.inverse(x)
        self.assertEqual(expected, y.numpy())