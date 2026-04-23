# Contributions

This repository contains the consolidated simulation code for the SRMC paper.
It combines continuous-state SRMC experiment code with the discrete MNIST/GWG
and CIFAR-10 EBM experiment code used in the revised-paper experimental
pipeline.

## Code Contributions

- Jinyoung Choi: provided the initial code corresponding to the discrete
  MNIST/GWG experiments, staged in this repository under
  `experiments/discrete_gwg_mnist/`.
- Geeho Kim: provided the initial code corresponding to the CIFAR-10 EBM
  mode-coverage experiments, staged in this repository under
  `experiments/cifar10_ebm_mode_coverage/`.
- Jie Hu: provided the initial implementations of the continuous-state
  `main.py`, `samplers.py`, and `potentials.py` modules, and coordinated
  repository planning and public release discussion.
- Lingyun Chen: continued development of the initial continuous-state code,
  integrated the contributed code into a unified public repository, reorganized
  the directory structure, cleaned release artifacts, prepared documentation,
  and maintained the public release setup.
- Do Young Eun: supervised the repository release and public dissemination of
  the code base.

## Experiment-source Mapping

| Repository path | Experiment block | Initial code source |
|---|---|---|
| `main.py`, `samplers.py`, `potentials.py` | Continuous-state SRMC simulations and baseline comparisons | Initial implementations by Jie Hu; continued development and release integration by Lingyun Chen |
| `utils.py` | Continuous-state MSE, ESS, and summary utilities | Integrated SRMC continuous experiment code |
| `experiments/figure1_metastable_demo.py` | Figure 1 illustrative metastability experiment | Integrated SRMC continuous experiment code |
| `experiments/figure2_continuous_alpha_sweep.py` | Continuous Figure 2 alpha-sweep experiments | Integrated SRMC continuous experiment code |
| `experiments/production_ablation.py` | Earlier production-ablation experiment script | Integrated SRMC continuous experiment code |
| `experiments/discrete_gwg_mnist/` | Discrete MNIST/GWG SRMC experiments | Jinyoung Choi |
| `experiments/cifar10_ebm_mode_coverage/` | CIFAR-10 EBM mode-coverage experiments | Geeho Kim |
