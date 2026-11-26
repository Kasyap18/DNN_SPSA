"""Classification metrics, learning curves, and portable NumPy checkpoints."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from neural_network import DenseTanhNetwork


def evaluate_classifier(network, parameters, inputs, labels, title="Confusion matrix", *, show=True):
    """Print held-out metrics, optionally display a matrix, and return predictions."""
    predictions = network.predict(parameters, inputs)
    print(f"Accuracy: {accuracy_score(labels, predictions):.3%}")
    classes = np.arange(max(2, network.layer_sizes[-1]))
    print(classification_report(labels, predictions, labels=classes, zero_division=0))
    if show:
        matrix = confusion_matrix(labels, predictions, labels=classes)
        ConfusionMatrixDisplay(matrix, display_labels=classes).plot(cmap="Blues", values_format="d")
        plt.title(title)
        plt.tight_layout()
        plt.show()
    return predictions


def plot_loss_curves(histories, path=None):
    """Plot training and validation loss, including the untrained epoch-zero point."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        axes[0].plot(history["epoch"], history["training_loss"], label=name)
        if any(value is not None for value in history["validation_loss"]):
            axes[1].plot(history["epoch"], history["validation_loss"], label=name)
    for axis, title in zip(axes, ("Training loss", "Validation loss")):
        axis.set(title=title, xlabel="Epoch", ylabel="Mean summed squared error")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    figure.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=160, bbox_inches="tight")
    return figure


def save_checkpoint(path, network, parameters, metadata, *, mean=None, scale=None):
    """Save architecture, parameters, metadata, and optional feature scaling."""
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (network.n_parameters,) or not np.isfinite(parameters).all():
        raise ValueError("Invalid checkpoint parameters.")
    width = network.layer_sizes[0]
    mean = np.zeros(width) if mean is None else np.asarray(mean, dtype=float)
    scale = np.ones(width) if scale is None else np.asarray(scale, dtype=float)
    if mean.shape != (width,) or scale.shape != (width,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("mean/scale must match input width; scale must be positive and finite.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, parameters=parameters, layer_sizes=network.layer_sizes,
                        slope=network.slope, metadata=json.dumps(metadata), mean=mean, scale=scale)


def load_checkpoint(path):
    """Load without pickle; also support older checkpoints without preprocessing."""
    with np.load(path, allow_pickle=False) as saved:
        sizes = tuple(saved["layer_sizes"].tolist())
        return {"parameters": saved["parameters"].copy(), "layer_sizes": sizes,
                "slope": float(saved["slope"]), "metadata": json.loads(str(saved["metadata"])),
                "mean": saved["mean"].copy() if "mean" in saved else np.zeros(sizes[0]),
                "scale": saved["scale"].copy() if "scale" in saved else np.ones(sizes[0])}


def predict_checkpoint(path, raw_inputs):
    """Rebuild the model, apply saved scaling, and predict from raw features."""
    saved = load_checkpoint(path)
    network = DenseTanhNetwork(saved["layer_sizes"], saved["slope"])
    inputs = (np.asarray(raw_inputs) - saved["mean"]) / saved["scale"]
    return network.predict(saved["parameters"], inputs)


def save_experiment(path, record):
    """Write JSON containing configuration, histories, metrics, and seeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
