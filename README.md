# SPSA neural-network optimization

Train a neural network using **Simultaneous Perturbation Stochastic Approximation (SPSA)** without backpropagation, and compare it with SGD and Adam. A sphere-function example first demonstrates convergence toward a known minimum.

## 1. Run the project

Use Python 3.11 or newer (validated locally with Python 3.13). From this folder, create an environment if needed and install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For a fresh checkout, first run `git clone https://github.com/Kasyap18/DNN_SPSA.git` and `cd DNN_SPSA`. Open that folder in VS Code, then follow the environment setup above.

The MNIST cache is not committed. Before running the MNIST section, run this once in a notebook cell from the project folder:

```python
from pathlib import Path
from urllib.request import urlretrieve

dataset_path = Path("data/mnist.npz")
dataset_path.parent.mkdir(parents=True, exist_ok=True)
if not dataset_path.exists():
    urlretrieve(
        "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        dataset_path,
    )
print("MNIST cache ready:", dataset_path)
```

Moons experiments can run without this download. To collaborate, create a branch with `git switch -c your-change`, commit your changes, push the branch, and open a pull request. Direct pushes require collaborator access; otherwise fork the repository and submit a pull request from the fork.

Select `.venv` as the notebook kernel in VS Code. Open and run these notebooks in order:

1. [01_SPSA_Implementation_and_Training.ipynb](01_SPSA_Implementation_and_Training.ipynb): theory, sphere minimization, small-network SPSA training, evaluation, and checkpoint reload.
2. [02_Optimizer_Comparison.ipynb](02_Optimizer_Comparison.ipynb): three-seed SPSA/SGD/Adam experiment, loss curves, measured results, and interpretation.

Each notebook can also run independently from a fresh kernel. The Moons experiments use generated data and require no download. The first notebook also contains active MNIST cells using the cached dataset, with 1,600 training, 400 validation, and 1,000 test images. Running all its cells starts the MNIST training run, which may take several minutes.

## 2. Project structure

| File or directory | Responsibility |
| --- | --- |
| [neural_network.py](neural_network.py) | Parameter layout, initialization, forward pass, predictions, target encoding, loss, and baseline gradients |
| [spsa_optimizer.py](spsa_optimizer.py) | Two-evaluation SPSA updates with decaying learning rate and perturbation |
| [training.py](training.py) | Data preparation, parameter blocks, SPSA training, SGD, and Adam |
| [evaluation.py](evaluation.py) | Classification reports, confusion matrices, loss plots, checkpoints, and JSON records |
| [requirements.txt](requirements.txt) | Dependencies |
| [tests/test_project.py](tests/test_project.py) | Numerical correctness and integration checks |
| [outputs/figures/](outputs/figures/) | New convergence and comparison plots |
| [outputs/checkpoints/](outputs/checkpoints/) | New trained models with architecture and preprocessing |
| [outputs/experiments/](outputs/experiments/) | New split data, settings, metrics, and histories |
| [outputs/phase4/](outputs/phase4/) and [outputs/phase5/](outputs/phase5/) | Preserved historical MNIST experiments |
| [SimpleArt/spsa](https://github.com/SimpleArt/spsa) | Upstream reference; the local patched checkout is excluded from Git |
| `data/` | Locally downloaded MNIST cache, excluded from Git |
| `outputs/cleanup_backup.zip` | Local backup of earlier work, excluded from Git |

## 3. Follow the code

### Network and loss

`DenseTanhNetwork` stores every weight and bias in one flat vector. Each layer records slices into that vector and the weight matrix shape. `initialize` creates seeded Xavier-style uniform weights and zero biases. `forward` applies `tanh(slope * (inputs @ weights + biases))` through each layer.

`predict` chooses the largest output score. For a one-output binary network it thresholds the tanh score at zero. Scores are not probabilities. `encode_targets` maps labels to -1/+1 targets. `mean_squared_error` averages the **sum** of squared output residuals over examples; it does not average across output units.

### SPSA update

At step t, sample independent perturbations delta_i from {-1, +1}, then:

```text
 a_t = learning_rate / t**alpha
 c_t = perturbation / t**gamma
 loss_plus  = loss(parameters + c_t * delta)
 loss_minus = loss(parameters - c_t * delta)
 estimated_gradient = (loss_plus - loss_minus) / (2*c_t) * delta
 parameters = parameters - a_t * estimated_gradient
```

The estimate requires two loss evaluations regardless of parameter count. No derivative is calculated. `spsa_minimize` returns a new vector without changing the caller's initial vector. Optional history includes step zero and costs additional monitoring evaluations.

### Holistic and layer-wise training

`get_parameter_blocks` supports `holistic`, `bottom_up`, `top_down`, and `random`. With `split_weights_biases=True`, each layer's weights are optimized before its biases. Holistic mode always optimizes the entire vector. Random order is sampled for each minibatch.

`train_spsa` visits each minibatch, then each block. It freezes all other parameters while optimizing the active block. Both perturbed losses use the same minibatch. As in the original notebook, the decay schedule restarts at t=1 for every block optimization.

The default shared RNG preserves the original update sequence. For comparisons, provide `shuffle_seed`: batch shuffling then uses an independent RNG, ensuring all optimizers receive the same batches despite SPSA's extra random draws.

### Conventional baselines

`loss_and_gradient` explicitly applies backpropagation and is called only by `train_gradient_optimizer`. SGD subtracts the learning rate times the gradient. Adam maintains first and second gradient moments, applies bias correction, and scales the update. Adam's state persists across batches and epochs.

### Evaluation and checkpoints

Histories contain epoch-zero loss, training/validation loss, elapsed seconds, optimizer evaluation counts, and monitoring counts. Validation data never updates weights. Test accuracy is measured after training.

`save_checkpoint` stores parameters, architecture, activation slope, experiment metadata, and feature mean/scale. `predict_checkpoint` reconstructs the network and applies the saved scaling to raw inputs. Supply the same preprocessing used for training when saving a checkpoint; identity scaling is the default.

## 4. Comparison protocol

### Keep the experiments separate

| Experiment | Dataset and network | Results |
| --- | --- | --- |
| Historical MNIST (archive only) | Digit images; 784 -> 196 -> 10; earlier experimental settings | [Phase 5 records](outputs/phase5/); excluded from the current notebook presentation |
| Current MNIST SPSA | Digit images; 784 -> 196 -> 10; one training seed | [Current run record](outputs/experiments/mnist_spsa_current_run.json); 51.9% test accuracy; SPSA only |
| Current Moons | Generated two-class data; 2 -> 8 -> 2; three training seeds | [Comparison notebook](02_Optimizer_Comparison.ipynb) and [Moons records](outputs/experiments/moons_optimizer_comparison.json) |

Present these as separate experiments. Do not combine their accuracy, loss, or runtime values into one optimizer ranking: their datasets, architectures, and training settings differ. The protocol below applies only to Moons.

The runnable comparison uses 600 moons examples, a stratified 360/120/120 split, training-only standardization, and a 2 -> 8 -> 2 network. It runs 40 epochs with batch size 64 for three initialization seeds. Within a trial, all methods share initial weights and minibatch order.

SPSA uses bottom-up separate weight/bias blocks, five inner steps, a=0.03 and c=0.1. SGD uses learning rate 0.03; Adam uses 0.01 with beta1=0.9, beta2=0.999 and epsilon=1e-8. These are illustrative settings, not exhaustive tuning.

Per trial, SPSA performs `40 * 6 * 4 * 5 * 2 = 9,600` update-objective evaluations. SGD and Adam each perform `40 * 6 = 240` gradient evaluations. Each method additionally records 82 monitoring losses, plus initial data-validation loss checks. These call types have different costs. Elapsed time includes monitoring but excludes initial input validation, final test evaluation, and checkpoint writing.

Plots show mean loss and one standard deviation across training seeds. The data split remains fixed. Equal epochs are not equal compute, and three trials do not establish a universal optimizer ranking. All results are calculated from the executed experiments and saved in [moons_optimizer_comparison.json](outputs/experiments/moons_optimizer_comparison.json).

## 5. Research relationship

The project implements the scheduled SPSA estimator and alternating layer-wise update idea from Algorithms 1 and 2 of Wulff, Schuecker and Bauckhage (2018). It preserves the earlier notebook's MNIST architecture and training defaults.

Differences from an exact paper reproduction:

- The default moons network is a small demonstration, not the paper's exact toy architecture.
- The paper writes a summed loss over examples; this code uses a mean for batch-size-independent scaling.
- The earlier notebook's MNIST targets are -1/+1; the paper's MNIST section describes 0/1 one-hot targets.
- A 784 -> 196 -> 10 dense network has 155,830 parameters including biases, not the count stated in the paper.
- The Moons comparison and current MNIST SPSA run are separate experiments; historical MNIST comparisons remain archived only.

[SimpleArt/spsa](https://github.com/SimpleArt/spsa) is a reference implementation with automatic tuning and adaptive momentum. The new experiments call our scheduled implementation, not the SimpleArt optimizer. Its local source and the earlier integration experiments are retained.

## 6. Verification

The test suite is included in [tests/test_project.py](tests/test_project.py). Include the [tests/](tests/) directory when sharing the project. Run this command from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Checks cover finite-difference validation of backpropagation, sphere convergence, SPSA evaluation counts, parameter-block coverage, frozen inactive parameters, absence of backpropagation in SPSA, SGD/Adam updates, checkpoint predictions, and invalid input handling, including positive-integer validation of `inner_steps`.

## References

- Wulff, Schuecker and Bauckhage, [*SPSA for Layer-Wise Training of Deep Networks*](https://doi.org/10.1007/978-3-030-01424-7_55) (2018). The paper PDF is not distributed in this repository.
- [SimpleArt/spsa](https://github.com/SimpleArt/spsa). Earlier integration experiments used commit `55ff58868f9b5e515dff4b55c8d0671f0f124f3d`.
