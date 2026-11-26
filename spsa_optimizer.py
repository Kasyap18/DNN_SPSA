"""Two-evaluation SPSA, following Algorithm 1 of Wulff et al. (2018).

This is the scheduled algorithm used in our notebook, not the adaptive
SimpleArt optimizer. No derivatives or backpropagation are used here.
"""

import numpy as np


def spsa_minimize(
    objective, initial_parameters, rng, steps=50, learning_rate=0.001,
    perturbation=0.1, alpha=1.0, gamma=1.0 / 6.0, *, return_history=False,
):
    """Minimize a scalar objective and return a new parameter vector.

    Each step uses a_t = learning_rate / t**alpha and
    c_t = perturbation / t**gamma, starting at t=1 on EACH call.
    The caller supplies a NumPy Generator to control reproducibility.

    With return_history=True, also return losses and parameters at step zero
    and after each update. These extra monitoring calls are counted separately
    from the two objective evaluations used by each optimizer step.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a non-negative integer.")
    for name, value in (("learning_rate", learning_rate), ("perturbation", perturbation)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in (("alpha", alpha), ("gamma", gamma)):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative.")
    parameters = np.array(initial_parameters, dtype=float, copy=True)
    if parameters.ndim != 1 or not parameters.size or not np.isfinite(parameters).all():
        raise ValueError("initial_parameters must be a non-empty, finite flat vector.")

    def evaluate(candidate):
        value = np.asarray(objective(candidate))
        if value.ndim != 0 or not np.isfinite(value):
            raise ValueError("objective must return one finite scalar.")
        return float(value)

    history = {"loss": [], "parameters": [], "objective_evaluations": [],
               "monitoring_evaluations": []}

    def record(step):
        history["loss"].append(evaluate(parameters))
        history["parameters"].append(parameters.copy())
        history["objective_evaluations"].append(2 * step)
        history["monitoring_evaluations"].append(step + 1)

    if return_history:
        record(0)
    for step in range(1, steps + 1):
        a_t = learning_rate / step**alpha
        c_t = perturbation / step**gamma
        # Perturb every coordinate at once, independently by -1 or +1.
        delta = rng.choice([-1.0, 1.0], size=parameters.shape)
        loss_plus = evaluate(parameters + c_t * delta)
        loss_minus = evaluate(parameters - c_t * delta)
        # For +/-1 perturbations, dividing by delta equals multiplying by it.
        gradient_estimate = ((loss_plus - loss_minus) / (2.0 * c_t)) * delta
        parameters -= a_t * gradient_estimate
        if not np.isfinite(parameters).all():
            raise FloatingPointError("SPSA diverged; inspect the learning rate and loss scale.")
        if return_history:
            record(step)
    return (parameters, history) if return_history else parameters
