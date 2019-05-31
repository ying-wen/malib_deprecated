import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import gym

from malib.value_functions import MLPValueFunction
from malib.core import Serializable


class ValueFunctionTest(tf.test.TestCase):
    def setUp(self):
        self.env = gym.envs.make('MountainCarContinuous-v0')
        self.hidden_layer_sizes = (128, 128)
        self.Q = MLPValueFunction(
            input_shapes=(self.env.observation_space.shape, self.env.action_space.shape),
            output_shape=(1,),
            hidden_layer_sizes=self.hidden_layer_sizes,
            name='Q'
        )
        self.V = MLPValueFunction(
            input_shapes=(self.env.observation_space.shape,),
            output_shape=(1,),
            hidden_layer_sizes=self.hidden_layer_sizes,
            name='V'
        )

    def test_multi_output(self):
        Q5 = MLPValueFunction(
            input_shapes=(self.env.observation_space.shape, self.env.action_space.shape),
            output_shape=(5,),
            hidden_layer_sizes=self.hidden_layer_sizes,
            name='Q5'
        )
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        action1_np = self.env.action_space.sample()
        action2_np = self.env.action_space.sample()
        observations_np = np.stack((observation1_np, observation2_np)).astype(np.float32)

        actions_np = np.stack((action1_np, action2_np))

        conditions = [observations_np, actions_np]

        q_values_np = Q5.get_values_np(conditions)
        q_values = Q5.get_values(conditions)

        self.assertEqual(q_values_np.shape, (2, 5))
        self.assertEqual(q_values.shape, (2, 5))

    def test_values(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        action1_np = self.env.action_space.sample()
        action2_np = self.env.action_space.sample()
        observations_np = np.stack((observation1_np, observation2_np)).astype(np.float32)

        actions_np = np.stack((action1_np, action2_np))

        conditions = [observations_np, actions_np]

        q_values_np = self.Q.get_values_np(conditions)
        q_values = self.Q.get_values(conditions)

        v_values_np = self.V.get_values_np([observations_np])
        v_values = self.V.get_values([observations_np])

        self.assertEqual(q_values_np.shape, (2, 1))
        self.assertEqual(q_values.shape, (2, 1))
        self.assertEqual(v_values_np.shape, (2, 1))
        self.assertEqual(v_values.shape, (2, 1))

    def test_trainable_variables(self):
        self.assertEqual(
            len(self.Q.trainable_variables),
            2 * (len(self.hidden_layer_sizes) + 1))
        self.assertEqual(
            len(self.V.trainable_variables),
            2 * (len(self.hidden_layer_sizes) + 1))

    def test_clone_target(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack(
            (observation1_np, observation2_np)
        ).astype(np.float32)

        weights = self.V.get_weights()
        values_np = self.V.get_values_np([observations_np])

        target_name = '{}_{}'.format('target', self.V._name)
        target_V = Serializable.clone(self.V, name=target_name)

        weights_2 = target_V.get_weights()

        self.assertEqual(target_V._name, target_name)
        self.assertIsNot(weights, weights_2)
        for weight, weight_2 in zip(weights, weights_2):
            np.testing.assert_array_equal(weight.shape, weight_2.shape)
        np.testing.assert_equal(
            values_np.shape, target_V.get_values_np([observations_np]).shape)

    def test_serialize_deserialize(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack(
            (observation1_np, observation2_np)
        ).astype(np.float32)

        weights = self.V.get_weights()
        values_np = self.V.get_values_np([observations_np])

        serialized = pickle.dumps(self.V)
        deserialized = pickle.loads(serialized)

        weights_2 = deserialized.get_weights()

        for weight, weight_2 in zip(weights, weights_2):
            np.testing.assert_array_equal(weight, weight_2)
        np.testing.assert_equal(
            values_np.shape, deserialized.get_values_np([observations_np]).shape)


if __name__ == '__main__':
    tf.test.main()