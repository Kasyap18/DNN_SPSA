"""Data preparation and training loops for SPSA, SGD, and Adam.

Read get_parameter_blocks, then train_spsa, then train_gradient_optimizer.
SPSA uses forward loss evaluations; only the baselines use backpropagation.
"""

from time import perf_counter
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural_network import encode_targets, mean_squared_error, loss_and_gradient
from spsa_optimizer import spsa_minimize


def prepare_moons_data(seed=42, n_samples=600, noise=0.2):
    """Return a 60/20/20 stratified split, scaled using training data only."""
    inputs, labels = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    train, remaining = train_test_split(np.arange(n_samples), test_size=0.4,
                                       random_state=seed, stratify=labels)
    validation, test = train_test_split(remaining, test_size=0.5,
                                      random_state=seed + 1, stratify=labels[remaining])
    scaler = StandardScaler().fit(inputs[train])
    data = {"mean": scaler.mean_, "scale": scaler.scale_}
    for name, indices in (("train", train), ("validation", validation), ("test", test)):
        data[f"X_{name}"] = scaler.transform(inputs[indices])
        data[f"y_{name}"] = labels[indices]
        data[f"Y_{name}"] = encode_targets(labels[indices], 2)
        data[f"{name}_indices"] = indices
    return data


def get_parameter_blocks(network, order="bottom_up", split_weights_biases=True, rng=None):
    """Choose slices to optimize, leaving all other coordinates fixed."""
    if order == "holistic":
        return [slice(0, network.n_parameters)]
    indices = list(range(len(network.layers)))
    if order == "top_down":
        indices.reverse()
    elif order == "random":
        if rng is None:
            raise ValueError("Pass an RNG when using random layer order.")
        rng.shuffle(indices)
    elif order != "bottom_up":
        raise ValueError("order must be holistic, bottom_up, top_down, or random.")
    blocks = []
    for index in indices:
        layer = network.layers[index]
        if split_weights_biases:
            blocks.extend([layer["weights"], layer["biases"]])
        else:
            blocks.append(slice(layer["weights"].start, layer["biases"].stop))
    return blocks


def _validate_training(network, parameters, inputs, targets, epochs, batch_size, validation_data):
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    inputs, targets = np.asarray(inputs, dtype=float), np.asarray(targets, dtype=float)
    if inputs.ndim != 2 or not np.isfinite(inputs).all() or not np.isfinite(targets).all():
        raise ValueError("Training data must be finite; inputs must be a 2D batch.")
    mean_squared_error(network, parameters, inputs, targets)
    if validation_data is not None:
        mean_squared_error(network, parameters, *validation_data)
    return inputs, targets


def _new_history():
    return {key: [] for key in ("epoch", "training_loss", "validation_loss",
                               "objective_evaluations", "gradient_evaluations",
                               "monitoring_evaluations", "elapsed_seconds")}


def _record(history, network, parameters, inputs, targets, validation_data,
            epoch, objective_calls, gradient_calls, started):
    history["epoch"].append(epoch)
    history["training_loss"].append(mean_squared_error(network, parameters, inputs, targets))
    validation_loss = None
    if validation_data is not None:
        validation_loss = mean_squared_error(network, parameters, *validation_data)
    history["validation_loss"].append(validation_loss)
    history["objective_evaluations"].append(objective_calls)
    history["gradient_evaluations"].append(gradient_calls)
    history["monitoring_evaluations"].append(len(history["epoch"]) * (1 + (validation_data is not None)))
    history["elapsed_seconds"].append(perf_counter() - started)


def train_spsa(
    network, initial_parameters, inputs, targets, *, epochs=12, batch_size=128,
    inner_steps=50, learning_rate=0.001, perturbation=0.1, order="bottom_up",
    split_weights_biases=True, seed=1000, shuffle_seed=None, validation_data=None,
    alpha=1.0, gamma=1.0 / 6.0, verbose=True,
):
    """Train with forward-only SPSA updates on each minibatch and block.

    The a_t/c_t schedules restart at t=1 for each block optimization, as in
    the original notebook. The default shared RNG preserves its update sequence.
    Supply shuffle_seed for comparisons: minibatch randomness then becomes
    independent of perturbations and identical to the baseline batch order.
    History includes epoch zero; monitoring calls are separate from updates.
    """
    if (
        isinstance(inner_steps, (bool, np.bool_))
        or not isinstance(inner_steps, (int, np.integer))
        or inner_steps < 1
    ):
        raise ValueError("inner_steps must be a positive integer.")
    parameters = np.array(initial_parameters, dtype=float, copy=True)
    inputs, targets = _validate_training(network, parameters, inputs, targets,
                                        epochs, batch_size, validation_data)
    rng = np.random.default_rng(seed)
    shuffle_rng = rng if shuffle_seed is None else np.random.default_rng(shuffle_seed)
    get_parameter_blocks(network, order, split_weights_biases, np.random.default_rng(seed))
    history = _new_history()
    calls = 0
    started = perf_counter()
    _record(history, network, parameters, inputs, targets, validation_data, 0, 0, 0, started)
    for epoch in range(1, epochs + 1):
        permutation = shuffle_rng.permutation(len(inputs))
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start:start + batch_size]
            batch_x, batch_y = inputs[indices], targets[indices]
            for block in get_parameter_blocks(network, order, split_weights_biases, rng):
                frozen_parameters = parameters.copy()

                def block_objective(candidate_block):
                    nonlocal calls
                    candidate = frozen_parameters.copy()
                    candidate[block] = candidate_block
                    calls += 1
                    return mean_squared_error(network, candidate, batch_x, batch_y)

                parameters[block] = spsa_minimize(
                    block_objective, parameters[block], rng, steps=inner_steps,
                    learning_rate=learning_rate, perturbation=perturbation,
                    alpha=alpha, gamma=gamma,
                )
        _record(history, network, parameters, inputs, targets, validation_data,
                epoch, calls, 0, started)
        if verbose:
            print(f"Epoch {epoch:02d}/{epochs} - loss: {history['training_loss'][-1]:.4f}")
    return parameters, history


def train_gradient_optimizer(
    network, initial_parameters, inputs, targets, *, optimizer="sgd", epochs=12,
    batch_size=128, learning_rate=0.01, shuffle_seed=1000, validation_data=None,
    beta1=0.9, beta2=0.999, epsilon=1e-8, verbose=True,
):
    """Train using ordinary minibatch SGD or bias-corrected Adam.

    Adam's moments and step counter persist across ALL batches and epochs.
    Each gradient evaluation includes forward propagation and backpropagation;
    it is not equivalent in cost to one SPSA objective evaluation.
    """
    if optimizer not in ("sgd", "adam"):
        raise ValueError("optimizer must be sgd or adam.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if not (0 <= beta1 < 1 and 0 <= beta2 < 1 and np.isfinite(epsilon) and epsilon > 0):
        raise ValueError("Adam requires beta1/beta2 in [0,1) and positive finite epsilon.")
    parameters = np.array(initial_parameters, dtype=float, copy=True)
    inputs, targets = _validate_training(network, parameters, inputs, targets,
                                        epochs, batch_size, validation_data)
    rng = np.random.default_rng(shuffle_seed)
    first_moment, second_moment = np.zeros_like(parameters), np.zeros_like(parameters)
    history = _new_history()
    step = 0
    started = perf_counter()
    _record(history, network, parameters, inputs, targets, validation_data, 0, 0, 0, started)
    for epoch in range(1, epochs + 1):
        permutation = rng.permutation(len(inputs))
        for start in range(0, len(inputs), batch_size):
            indices = permutation[start:start + batch_size]
            _, gradient = loss_and_gradient(network, parameters, inputs[indices], targets[indices])
            step += 1
            if optimizer == "sgd":
                parameters -= learning_rate * gradient
            else:
                first_moment = beta1 * first_moment + (1 - beta1) * gradient
                second_moment = beta2 * second_moment + (1 - beta2) * gradient**2
                corrected_first = first_moment / (1 - beta1**step)
                corrected_second = second_moment / (1 - beta2**step)
                parameters -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
            if not np.isfinite(parameters).all():
                raise FloatingPointError("Parameters became non-finite during baseline training.")
        _record(history, network, parameters, inputs, targets, validation_data,
                epoch, 0, step, started)
        if verbose:
            print(f"{optimizer.upper()} epoch {epoch:02d}/{epochs} - loss: {history['training_loss'][-1]:.4f}")
    return parameters, history
