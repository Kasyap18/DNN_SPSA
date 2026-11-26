"""Focused numerical and integration checks. Run: python -m unittest discover -s tests"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

from neural_network import DenseTanhNetwork, encode_targets, mean_squared_error, loss_and_gradient
from spsa_optimizer import spsa_minimize
from training import get_parameter_blocks, train_spsa, train_gradient_optimizer
from evaluation import save_checkpoint, predict_checkpoint


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.network = DenseTanhNetwork([2, 3, 2], slope=0.5)
        self.parameters = self.network.initialize(4)
        self.inputs = np.array([[0.2, -0.5], [0.8, 0.1], [-0.2, 0.9]])
        self.targets = encode_targets([0, 1, 0], 2)

    def test_gradient_matches_central_differences(self):
        _, gradient = loss_and_gradient(self.network, self.parameters, self.inputs, self.targets)
        numerical = np.zeros_like(gradient)
        for index in range(len(gradient)):
            shift = np.zeros_like(gradient)
            shift[index] = 1e-6
            numerical[index] = (
                mean_squared_error(self.network, self.parameters + shift, self.inputs, self.targets)
                - mean_squared_error(self.network, self.parameters - shift, self.inputs, self.targets)
            ) / 2e-6
        np.testing.assert_allclose(gradient, numerical, rtol=1e-5, atol=1e-8)

    def test_spsa_calls_and_convergence(self):
        calls = []
        def sphere(parameters):
            calls.append(parameters.copy())
            return np.sum(parameters**2)
        initial = np.array([3.0, -2.0])
        parameters, history = spsa_minimize(sphere, initial, np.random.default_rng(4),
            steps=200, learning_rate=0.2, alpha=0.602, return_history=True)
        self.assertLess(np.sum(parameters**2), 0.01 * np.sum(initial**2))
        self.assertEqual(len(calls), 601)
        self.assertEqual(history['objective_evaluations'][-1], 400)
        self.assertEqual(history['monitoring_evaluations'][-1], 201)
        np.testing.assert_array_equal(initial, [3, -2])

    def test_block_coverage_and_order(self):
        for order in ('holistic', 'bottom_up', 'top_down', 'random'):
            for split in (False, True):
                blocks = get_parameter_blocks(self.network, order, split, np.random.default_rng(3))
                indices = np.concatenate([np.arange(self.network.n_parameters)[block] for block in blocks])
                np.testing.assert_array_equal(np.sort(indices), np.arange(self.network.n_parameters))
        blocks = get_parameter_blocks(self.network, 'top_down', False)
        self.assertEqual(blocks[0].start, self.network.layers[-1]['weights'].start)

    def test_spsa_freezes_inactive_parameters_and_never_backpropagates(self):
        parameters_seen = []
        from neural_network import mean_squared_error as actual_loss
        def capture(network, parameters, inputs, targets):
            parameters_seen.append(parameters.copy())
            return actual_loss(network, parameters, inputs, targets)
        with patch('training.mean_squared_error', side_effect=capture), patch(
            'training.loss_and_gradient', side_effect=AssertionError('SPSA used backpropagation')
        ):
            _, history = train_spsa(self.network, self.parameters, self.inputs, self.targets,
                epochs=1, batch_size=3, inner_steps=1, verbose=False)
        # First two calls are validation and epoch-zero monitoring, then +/- pairs.
        calls = parameters_seen[2:-1]
        for index, block in enumerate(get_parameter_blocks(self.network)):
            plus, minus = calls[2 * index:2 * index + 2]
            inactive = np.ones(len(plus), dtype=bool)
            inactive[block] = False
            np.testing.assert_array_equal(plus[inactive], minus[inactive])
        self.assertEqual(history['objective_evaluations'][-1], 8)

    def test_inner_steps_must_be_a_positive_integer(self):
        for value in (0, -1, 1.5, 1.0, True, False, np.bool_(True), '2', None):
            with self.subTest(inner_steps=value):
                with patch('training.spsa_minimize') as optimizer:
                    with self.assertRaisesRegex(ValueError, 'inner_steps must be a positive integer'):
                        train_spsa(self.network, self.parameters, self.inputs, self.targets,
                                   epochs=1, inner_steps=value, verbose=False)
                    optimizer.assert_not_called()
        for value in (1, np.int64(2)):
            with self.subTest(inner_steps=value):
                _, history = train_spsa(
                    self.network, self.parameters, self.inputs, self.targets,
                    epochs=1, batch_size=3, inner_steps=value, verbose=False,
                )
                self.assertEqual(history['objective_evaluations'][-1], 8 * value)

    def test_baseline_updates_and_shared_batch_order(self):
        seen = []
        def fixed_gradient(network, parameters, inputs, targets):
            seen.append(inputs.copy())
            return 0.0, np.ones_like(parameters)
        with patch('training.loss_and_gradient', side_effect=fixed_gradient):
            sgd, _ = train_gradient_optimizer(self.network, self.parameters, self.inputs,
                self.targets, epochs=2, batch_size=2, learning_rate=0.01, verbose=False)
            adam, _ = train_gradient_optimizer(self.network, self.parameters, self.inputs,
                self.targets, optimizer='adam', epochs=2, batch_size=2, learning_rate=0.01, verbose=False)
        np.testing.assert_allclose(sgd, self.parameters - 0.04)
        np.testing.assert_allclose(adam, self.parameters - 0.04 / (1 + 1e-8))
        for first, second in zip(seen[:4], seen[4:]):
            np.testing.assert_array_equal(first, second)

    def test_checkpoint_roundtrip_including_scaling(self):
        mean, scale = np.array([1., 2.]), np.array([2., 3.])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.npz'
            save_checkpoint(path, self.network, self.parameters, {}, mean=mean, scale=scale)
            predicted = predict_checkpoint(path, self.inputs * scale + mean)
        np.testing.assert_array_equal(predicted, self.network.predict(self.parameters, self.inputs))

    def test_invalid_inputs_and_binary_prediction(self):
        for sizes in ([2.5, 2], [0, 2], [True, 2]):
            with self.assertRaises(ValueError):
                DenseTanhNetwork(sizes)
        for slope in (float('nan'), float('inf'), 0):
            with self.assertRaises(ValueError):
                DenseTanhNetwork([2, 2], slope)
        with self.assertRaises(ValueError):
            encode_targets([-1, 2], 2)
        with self.assertRaises(ValueError):
            spsa_minimize(lambda p: float('nan'), [1.], np.random.default_rng())
        network = DenseTanhNetwork([1, 1])
        np.testing.assert_array_equal(network.predict(np.array([1., 0.]), [[-2.], [2.]]), [0, 1])


if __name__ == '__main__':
    unittest.main()
