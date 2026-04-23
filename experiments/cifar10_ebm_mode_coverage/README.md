# SRMC — Score-Repellent MCMC for Mode Coverage

Public release of the three mode-coverage experiments that accompany the
Score-Repellent MCMC (SR-MCMC) paper. Each experiment is packaged as a single,
self-contained Jupyter notebook and runs SR-MCMC only (no Langevin-baseline
comparison in this release). All randomness (Python, NumPy, PyTorch CPU/GPU,
TensorFlow, cuDNN) is seeded so that repeated runs produce the same results.

## Repository layout

```
SRMC_ebm_public/
├── README.md                             # this file
├── requirements.txt                      # pinned dependencies
├── models.py                             # TF1 EBM architectures (ResNet32Large, ...)
├── utils.py                              # TF1 utilities (weights, spectral norm, restore)
├── exp1_gmm_mode_coverage.ipynb          # Experiment 1 (PyTorch only)
├── exp2_cifar10_single_chain.ipynb       # Experiment 2 (TF1 + PyTorch)
└── exp3_cifar10_multi_chain.ipynb        # Experiment 3 (TF1 + PyTorch)
```

Outputs (figures, `.npy` predictions, `metrics.json`) are written under
`mode_coverage_results/` next to the notebook you ran.

## The three experiments

| # | Notebook | Target distribution | What it measures |
|---|----------|---------------------|------------------|
| 1 | `exp1_gmm_mode_coverage.ipynb` | Gaussian Mixture of 1,000 CIFAR-10 images (100 per class), `σ=4` | Cumulative unique modes discovered by 50 parallel chains starting from mode 0 |
| 2 | `exp2_cifar10_single_chain.ipynb` | Pretrained unconditional CIFAR-10 EBM (`cifar10_large_model_uncond`, iter 121,200) | Class distribution of 2,000 samples produced by one SR-LD chain |
| 3 | `exp3_cifar10_multi_chain.ipynb` | Same pretrained EBM | Class distribution of `num_chains` final samples (default 1,000 chains × 450 steps) |

Experiments 2 and 3 use the pretrained CIFAR-10 ResNet20 from
`chenyaofo/pytorch-cifar-models` as the mode-labelling classifier; it is
downloaded automatically by `torch.hub` on first use.

## Environment

The CIFAR-10 EBM was trained with TensorFlow 1.12, so experiments 2 and 3
require a TF-1.12 environment. Experiment 1 is PyTorch-only.

```bash
# Option A — conda (recommended)
conda create -n srmc python=3.7 -y
conda activate srmc
pip install -r requirements.txt

# Option B — use the original OpenAI EBM env file (if you kept it)
conda env create -f ebm.yml
conda activate ebm
```

Key pinned versions:

- `tensorflow==1.12.0`, `tensorflow-gpu==1.12.0`
- `torch`, `torchvision`   (any version compatible with your CUDA)
- `baselines==0.1.5`   (required by EBM graph utilities)
- `numpy==1.15.1`, `scipy==0.19.1`, `matplotlib`, `tqdm`, `absl-py`

GPU: an NVIDIA GPU with a CUDA runtime matching TF 1.12 (typically CUDA 9.0)
is recommended for experiments 2 and 3.

## Data

- **CIFAR-10** is auto-downloaded by `torchvision` into `./data/` on first run.
  No manual step.
- **Pretrained CIFAR-10 EBM checkpoint** is **not included** in this repo.
  Obtain it from the original OpenAI EBM release
  (<https://sites.google.com/view/igebm>) and place it at:

  ```
  sandbox_cachedir/
    cachedir/
      cifar10_large_model_uncond/
        model_121200.data-00000-of-00001
        model_121200.index
        model_121200.meta
  ```

  Create the path relative to the notebook's working directory (i.e. the
  repository root if you launch Jupyter from there), or edit `FLAGS.logdir`
  / `FLAGS.exp` in the first code cell of exp2 / exp3.

## How to run

From the repository root:

```bash
jupyter notebook
```

Then open and run, top-to-bottom:

1. `exp1_gmm_mode_coverage.ipynb`
2. `exp2_cifar10_single_chain.ipynb`
3. `exp3_cifar10_multi_chain.ipynb`

Each notebook is self-contained — you can run them in any order. Experiments 2
and 3 each spin up their own TensorFlow session; restart the kernel between
them (or run them in separate Jupyter instances) to avoid flag / graph
collisions.

## Acknowledgements

The TF1 EBM architecture (`models.py`) and utilities (`utils.py`) are derived
from the original OpenAI EBM release
(<https://github.com/openai/ebm_code_release>).
