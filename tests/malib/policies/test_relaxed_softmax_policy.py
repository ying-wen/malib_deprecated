import pickle
from collections import OrderedDict

import numpy as np
import tensorflow as tf

import gym

from malib.policies import RelaxedSoftmaxMLPPolicy
from malib.core import Serializable


class GaussianPolicyTest(tf.test.TestCase):
    def setUp(self):
        self.env = gym.envs.make('MountainCarContinuous-v0')
        self.hidden_layer_sizes = (128, 128)
        self.policy = RelaxedSoftmaxMLPPolicy(
            input_shapes=(self.env.observation_space.shape, ),
            output_shape=self.env.action_space.shape,
            hidden_layer_sizes=self.hidden_layer_sizes,
            name='Policy'
        )
        self.cond_polcy = RelaxedSoftmaxMLPPolicy(
            input_shapes=(
                self.env.observation_space.shape,
                self.env.action_space.shape),
            output_shape=self.env.action_space.shape,
            hidden_layer_sizes=self.hidden_layer_sizes,
            name='CondPolicy')

    def test_actions_and_log_pis_symbolic(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack((observation1_np, observation2_np))

        observations_tf = tf.constant(observations_np, dtype=tf.float32)

        actions = self.policy.get_actions([observations_tf])
        log_pis = self.policy.log_pis([observations_tf], actions)

        self.assertEqual(actions.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis.shape, (2, 1))

        actions_np = self.evaluate(actions)
        log_pis_np = self.evaluate(log_pis)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis_np.shape, (2, 1))

    def test_cond_policy(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        action1_np = self.env.action_space.sample()
        action2_np = self.env.action_space.sample()
        observations_np = np.stack(
            (observation1_np, observation2_np)).astype(
            np.float32)
        actions_np = np.stack((action1_np, action2_np))
        conditions = [observations_np, actions_np]

        actions_np = self.cond_polcy.get_actions_np(conditions)
        actions = self.cond_polcy.get_actions(conditions)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))
        self.assertEqual(actions.shape, (2, *self.env.action_space.shape))

    def test_actions_and_log_pis_numeric(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack((observation1_np, observation2_np))

        actions_np = self.policy.get_actions_np([observations_np])
        log_pis_np = self.policy.log_pis_np([observations_np], actions_np)

        self.assertEqual(actions_np.shape, (2, *self.env.action_space.shape))
        self.assertEqual(log_pis_np.shape, (2, 1))

    def test_env_step_with_actions(self):
        observation1_np = self.env.reset()
        action = self.policy.get_actions_np(observation1_np[None])[0, ...]
        self.env.step(action)

    def test_env_step_with_action(self):
        observation1_np = self.env.reset()
        action = self.policy.get_action_np(observation1_np)
        self.env.step(action)
        self.assertEqual(action.shape, self.env.action_space.shape)

    def test_trainable_variables(self):
        self.assertEqual(
            len(self.policy.trainable_variables),
            2 * (len(self.hidden_layer_sizes) + 1))

    def test_get_diagnostics(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack((observation1_np, observation2_np))

        diagnostics = self.policy.get_diagnostics([observations_np])

        self.assertTrue(isinstance(diagnostics, OrderedDict))
        # self.assertEqual(
        #     tuple(diagnostics.keys()),
        #     ('shifts-mean',
        #      'shifts-std',
        #      'log_scale_diags-mean',
        #      'log_scale_diags-std',
        #      '-log-pis-mean',
        #      '-log-pis-std',
        #      'raw-actions-mean',
        #      'raw-actions-std',
        #      'actions-mean',
        #      'actions-std'))

        for value in diagnostics.values():
            self.assertTrue(np.isscalar(value))

    def test_clone_target(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack(
            (observation1_np, observation2_np)
        ).astype(np.float32)

        weights = self.policy.get_weights()
        actions_np = self.policy.get_actions_np([observations_np])
        log_pis_np = self.policy.log_pis_np([observations_np], actions_np)

        target_name = '{}_{}'.format('target', self.policy._name)
        target_policy = Serializable.clone(self.policy, name=target_name)

        weights_2 = target_policy.get_weights()
        log_pis_np_2 = target_policy.log_pis_np([observations_np], actions_np)

        self.assertEqual(target_policy._name, target_name)
        self.assertIsNot(weights, weights_2)
        for weight, weight_2 in zip(weights, weights_2):
            np.testing.assert_array_equal(weight.shape, weight_2.shape)
        np.testing.assert_array_equal(log_pis_np.shape, log_pis_np_2.shape)
        np.testing.assert_equal(
            actions_np.shape,
            self.policy.get_actions_np(
                [observations_np]).shape)

    def test_serialize_deserialize(self):
        observation1_np = self.env.reset()
        observation2_np = self.env.step(self.env.action_space.sample())[0]
        observations_np = np.stack(
            (observation1_np, observation2_np)
        ).astype(np.float32)

        weights = self.policy.get_weights()
        actions_np = self.policy.get_actions_np([observations_np])
        log_pis_np = self.policy.log_pis_np([observations_np], actions_np)

        serialized = pickle.dumps(self.policy)
        deserialized = pickle.loads(serialized)

        weights_2 = deserialized.get_weights()
        log_pis_np_2 = deserialized.log_pis_np([observations_np], actions_np)

        for weight, weight_2 in zip(weights, weights_2):
            np.testing.assert_array_equal(weight, weight_2)
        np.testing.assert_array_equal(log_pis_np, log_pis_np_2)
        np.testing.assert_equal(
            actions_np.shape,
            deserialized.get_actions_np(
                [observations_np]).shape)




if __name__ == '__main__':
    tf.test.main()
