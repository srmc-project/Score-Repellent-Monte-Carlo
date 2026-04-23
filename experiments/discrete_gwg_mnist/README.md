# Score-Tilted MCMC (SRMC) for Discrete Distributions

This codebase is heavily based on the official implementation of ["Oops I Took A Gradient: Scalable Sampling for Discrete Distributions"](https://github.com/wgrathwohl/GWG_release) (Grathwohl et al., ICML 2021). We sincerely thank the authors for their foundational work and for providing the excellent Gibbs-With-Gradients (GWG) codebase.

This repository extends the original GWG framework by introducing the **Score-Tilted MCMC (SRMC)** sampler. Our implementation incorporates a continuous gradient proxy (**Shifted Gradient**) to efficiently handle high-dimensional discrete spaces while effectively preserving the essential properties of the exact discrete score.

## How to Run Score-Tilted MCMC (SRMC)

We introduce a new sampler option, `--sampler sr`. You can control the repulsion strength and the behavior of the score evaluation via the provided hyperparameters.

### Mode Mixing 
This is to explicitly evaluate how well the MCMC chains mix and transition between different modes.

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
        --n_subsample 20

### Evaluating Sample Diversity (Entropy, KL, Vendi)
We quantitatively evaluate the diversity and mode-mixing behavior of the generated images. We use a pre-trained MNIST classifier to calculate the marginal distribution of the generated samples.

    python eval_metrics.py \
        --image_dir /path/to/generated/images
