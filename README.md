# Score-Repellent Monte Carlo

This repository contains the code for Score-Repellent Monte Carlo: Toward Efficient Non-Markovian Sampler with Constant Memory in General State Spaces, including continuous-state experiments, discrete MNIST/GWG experiments, and CIFAR-10 EBM  mode-coverage notebooks.

## Contribution Note

This repository consolidates the simulation code used in the SRMC paper.

- Initial implementations of the continuous-state modules were provided by Jie Hu.
- Initial code for the discrete MNIST/GWG experiments was provided by Jinyoung Choi.
- Initial code for the CIFAR-10 EBM mode-coverage experiments was provided by Geeho Kim.
- Continued development of the initial continuous-state experiments, repository
  integration, cleanup, documentation, and public release setup are maintained
  by Lingyun Chen.
- The repository is released under the supervision of Do Young Eun and the SRMC coauthors.

Please see `CONTRIBUTIONS.md` for a more detailed breakdown of experimental
and repository contributions.

## Repository Layout

```text
.
|-- main.py                         # JSON-config continuous experiment runner
|-- samplers.py                     # Continuous SRMC and baseline samplers
|-- potentials.py                   # Continuous target distributions
|-- utils.py                        # MSE and ESS utilities
|-- CONTRIBUTIONS.md                # Contribution and code-provenance notes
|-- experiments/
|   |-- figure1_metastable_demo.py
|   |-- figure2_continuous_alpha_sweep.py
|   |-- production_ablation.py
|   |-- discrete_gwg_mnist/         # Discrete MNIST/GWG SRMC experiments
|   `-- cifar10_ebm_mode_coverage/  # CIFAR-10 EBM mode-coverage notebooks
|-- results/
|   `-- figure2_continuous_alpha_sweep/
`-- docs/
    `-- release_scope.md
```

## Setup

The root environment is for the continuous-state experiments only:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

On macOS/Linux, use:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Smoke Test

```bash
python - <<'PY'
import numpy as np
from potentials import CorrelatedGaussian
from samplers import MALA, ScoreTiltedMCMC

target = CorrelatedGaussian(dim=2, rho=0.5)
x0 = np.zeros(2)

for sampler in [
    MALA(target, step_size=0.1, rng=np.random.default_rng(1)),
    ScoreTiltedMCMC(target, step_size=0.1, alpha=0.5, theta_step=1.0, rng=np.random.default_rng(2)),
]:
    samples, diag = sampler.run(x0, n_steps=100, burn_in=20)
    print(type(sampler).__name__, samples.shape, round(diag.acceptance_rate, 3))
PY
```

## Reproducing the Continuous Figure 2 Sweep

The full script runs 30 replicates per condition and may take time:

```bash
python experiments/figure2_continuous_alpha_sweep.py
```

Precomputed plots and summary tables are included under:

```text
results/figure2_continuous_alpha_sweep/
```

## Reproducing the Discrete MNIST / GWG Experiments

The MNIST/GWG code is self-contained under:

```text
experiments/discrete_gwg_mnist/
```

It keeps its own `samplers.py` and `utils.py` because these are discrete-state
GWG utilities, not duplicates of the continuous-state root modules.

From that directory, install the optional MNIST/GWG dependencies:

```bash
cd experiments/discrete_gwg_mnist
python -m pip install -r requirements.txt
```

The static binarized MNIST files and `mnist_classifier.pt` are included. To
train or provide a GWG MNIST EBM checkpoint, use `pcd_ebm_ema.py`; the mixing
script expects a checkpoint path such as `model_gwg_mnist/best_ckpt.pt`.

Example SRMC mode-mixing run:

```bash
python mixing.py \
    --ckpt_path model_gwg_mnist/best_ckpt.pt \
    --save_dir output_srmc_mnist_mode_mixing \
    --start_from real \
    --sampler sr \
    --alpha 0.00001 \
    --ema \
    --total_steps 10000 \
    --eval_every 100 \
    --burn_in 0 \
    --batch_size 20
```

Then evaluate generated image diversity:

```bash
python eval_metrics.py --image_dir output_srmc_mnist_mode_mixing
```

## Reproducing the CIFAR-10 EBM Mode-Coverage Experiments

The CIFAR-10 EBM release code is staged under:

```text
experiments/cifar10_ebm_mode_coverage/
```

This part uses its own TensorFlow 1.x environment and is intentionally separate
from the continuous-state Python package. From that directory:

```bash
cd experiments/cifar10_ebm_mode_coverage
conda create -n srmc-ebm python=3.7 -y
conda activate srmc-ebm
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks top-to-bottom:

```text
exp1_gmm_mode_coverage.ipynb
exp2_cifar10_single_chain.ipynb
exp3_cifar10_multi_chain.ipynb
```

For experiments 2 and 3, the pretrained CIFAR-10 EBM checkpoint is not bundled.
Follow `experiments/cifar10_ebm_mode_coverage/README.md` for the expected
checkpoint location under `sandbox_cachedir/`.

## Sampler Names

The JSON runner accepts the following public sampler names:

```text
MALA
SR-MALA
HMC
SR-HMC
ULD
SR-ULD
UnadjustedLangevin
UnadjustedScoreTilted
ScoreTilted
ScoreTiltedHMC
```

`SR-MALA`, `SR-HMC`, and `SR-ULD` all support fixed alpha, linear alpha warmup,
and rational adaptive alpha through:

```json
{
  "alpha": 1.0,
  "alpha_adaptive": true,
  "alpha_C": 1000.0,
  "alpha_warmup_steps": 0
}
```

When comparing multiple settings of the same sampler in one JSON config, use
`label` to keep the output entries distinct:

```json
{"name": "SR-MALA", "label": "SR-MALA adaptive alpha=1", "step_size": 0.1, "alpha": 1.0, "theta_step": 1.0, "alpha_adaptive": true}
```

## License

This code is released under the MIT License. See `LICENSE`.
