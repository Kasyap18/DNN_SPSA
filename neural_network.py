"""Neural-network model used by the SPSA experiments.

The network is fully connected and uses tanh activations. All trainable
weights and biases are stored in one flat NumPy vector so SPSA can perturb
either the complete parameter vector or selected layer-wise slices.
"""

import numpy as np


class DenseTanhNetwork:
    """A fully connected neural network with tanh activations.

    Parameters
    ----------
    layer_sizes : sequence of int
        Number of units in each layer. For MNIST, use (784, 196, 10).
    slope : float, default=1.0
        Multiplier applied before each tanh activation.

    Notes
    -----
    Parameters are stored in a single one-dimensional vector. Each layer
    records slices into that vector for its weights and biases.
    """

    def __init__(self, layer_sizes, slope=1.0):
        if len(layer_sizes) < 2:
            raise ValueError("A network must contain an input and output layer.")
        if any(
            isinstance(size, (bool, np.bool_))
            or not isinstance(size, (int, np.integer))
            or size <= 0
            for size in layer_sizes
        ):
            raise ValueError("All layer sizes must be positive integers.")
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("slope must be finite and positive.")

        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.slope = float(slope)
        self.layers = []

        parameter_offset = 0

        for input_size, output_size in zip(
            self.layer_sizes[:-1], self.layer_sizes[1:]
        ):
            weight_count = input_size * output_size
            weight_slice = slice(
                parameter_offset, parameter_offset + weight_count
            )
            parameter_offset += weight_count

            bias_slice = slice(
                parameter_offset, parameter_offset + output_size
            )
            parameter_offset += output_size

            self.layers.append(
                {
                    "weights": weight_slice,
                    "biases": bias_slice,
                    "shape": (input_size, output_size),
                }
            )

        self.n_parameters = parameter_offset

    def initialize(self, seed=0):
        """Create an initialized flat parameter vector.

        A seeded NumPy random generator makes initialization reproducible.
        Biases start at zero.
        """
        rng = np.random.default_rng(seed)
        parameters = np.zeros(self.n_parameters, dtype=float)

        for layer in self.layers:
            input_size, output_size = layer["shape"]
            limit = np.sqrt(6.0 / (input_size + output_size)) / self.slope

            parameters[layer["weights"]] = rng.uniform(
                -limit,
                limit,
                size=layer["weights"].stop - layer["weights"].start,
            )

        return parameters

    def forward(self, parameters, inputs):
        """Run forward propagation and return the output scores."""
        parameters = np.asarray(parameters, dtype=float)
        activations = np.asarray(inputs, dtype=float)

        if parameters.ndim != 1 or parameters.size != self.n_parameters:
            raise ValueError(
                f"parameters must be a flat vector of length {self.n_parameters}."
            )

        if activations.ndim == 1:
            activations = activations.reshape(1, -1)

        if activations.ndim != 2:
            raise ValueError("inputs must be a 1D sample or a 2D batch.")
        if activations.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Each input must contain {self.layer_sizes[0]} features."
            )

        for layer in self.layers:
            weights = parameters[layer["weights"]].reshape(layer["shape"])
            biases = parameters[layer["biases"]]
            weighted_sum = activations @ weights + biases
            activations = np.tanh(self.slope * weighted_sum)

        return activations

    def predict(self, parameters, inputs):
        """Return the class index with the largest output score."""
        output_scores = self.forward(parameters, inputs)
        # A single tanh output represents binary classes by its sign.
        if output_scores.shape[1] == 1:
            return (output_scores[:, 0] >= 0).astype(int)
        return np.argmax(output_scores, axis=1)


def encode_targets(labels, number_of_classes=10):
    """Convert integer class labels to -1/+1 targets for tanh outputs.

    Use number_of_classes=1 for one-output binary classification (labels 0/1).
    Otherwise return one column per class, matching the existing MNIST code.
    """
    labels = np.asarray(labels)
    if not isinstance(number_of_classes, (int, np.integer)) or number_of_classes < 1:
        raise ValueError("number_of_classes must be a positive integer.")
    limit = 2 if number_of_classes == 1 else number_of_classes
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be a one-dimensional integer array.")
    if np.any((labels < 0) | (labels >= limit)):
        raise ValueError("A label is outside the configured class range.")
    if number_of_classes == 1:
        return 2.0 * labels[:, None] - 1.0
    return 2.0 * np.eye(number_of_classes)[labels] - 1.0


def mean_squared_error(network, parameters, inputs, targets):
    """Mean over examples of the SUM of squared output residuals.

    We do not divide by output width. This preserves the notebook's loss scale.
    The paper writes a sum over examples; our mean removes batch-size scaling.
    """
    predictions = network.forward(parameters, inputs)
    targets = np.asarray(targets, dtype=float)
    if targets.shape != predictions.shape or len(targets) == 0:
        raise ValueError("targets must match the non-empty network output shape.")
    loss = float(np.mean(np.sum((predictions - targets) ** 2, axis=1)))
    if not np.isfinite(loss):
        raise FloatingPointError("Loss is non-finite; check data and parameters.")
    return loss


def loss_and_gradient(network, parameters, inputs, targets):
    """Backpropagation for the SGD/Adam baselines ONLY.

    Return the same loss as mean_squared_error and its analytical gradient
    in the network's flat parameter layout. SPSA never calls this function.
    """
    loss = mean_squared_error(network, parameters, inputs, targets)
    parameters = np.asarray(parameters, dtype=float)
    activations = [np.asarray(inputs, dtype=float).reshape(-1, network.layer_sizes[0])]
    for layer in network.layers:
        weights = parameters[layer["weights"]].reshape(layer["shape"])
        biases = parameters[layer["biases"]]
        activations.append(np.tanh(network.slope * (activations[-1] @ weights + biases)))

    gradient = np.zeros_like(parameters)
    # Derivative of the mean squared residual loss with respect to output.
    output_derivative = 2.0 * (activations[-1] - targets) / len(activations[0])
    for index in reversed(range(len(network.layers))):
        layer = network.layers[index]
        # Chain rule: tanh(s*z) has derivative s*(1-tanh(s*z)**2).
        delta = output_derivative * network.slope * (1.0 - activations[index + 1] ** 2)
        gradient[layer["weights"]] = (activations[index].T @ delta).ravel()
        gradient[layer["biases"]] = delta.sum(axis=0)
        weights = parameters[layer["weights"]].reshape(layer["shape"])
        output_derivative = delta @ weights.T
    return loss, gradient
