"""Main driver for running MCMC experiments.

This script reads a JSON configuration file describing an experiment,
runs multiple replicates of various samplers on a specified target
distribution, computes diagnostic statistics, and writes the results
to disk.  It is intended to be run from the command line:

    python main.py --config config.json --output results.json

The configuration file must specify the target distribution, the
samplers to evaluate, the number of steps and burn–in, and the number
of replicates.  Optional fields include a ``seed`` (for
reproducibility) and a ``true_mean`` vector used to compute mean
squared error.

Example config::

    {
      "experiment_name": "gaussian_mixture_2d",
      "target": {"name": "GaussianMixture", "params": {"dim": 2, "separation": 6.0}},
      "samplers": [
        {"name": "MALA", "step_size": 0.1},
        {"name": "Underdamped", "step_size": 0.01, "friction": 1.0},
        {"name": "ScoreTilted", "step_size": 0.05, "alpha": 0.1, "theta_step": 1.0}
      ],
      "n_steps": 5000,
      "burn_in": 500,
      "num_replicates": 5,
      "seed": 42,
      "true_mean": [0.0, 0.0]
    }

Available target names are ``CorrelatedGaussian``, ``GaussianMixture``,
``GaussianMixtureThreeMode``, ``GaussianMixtureTriangle``, ``SyntheticLogistic``.
Sampler names correspond to classes defined in ``samplers.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from potentials import (
    CorrelatedGaussian,
    GaussianMixture,
    SyntheticLogistic,
    GaussianMixtureThreeMode,
    GaussianMixtureTriangle,
)
from samplers import (
    MALA,
    ScoreTiltedMCMC,
    UnderdampedLangevin,
    # MetropolisUnderdampedLangevin,
    # UnderdampedScoreTiltedMCMC,
    SamplerDiagnostics,
    UnadjustedScoreTiltedULD,
    HMC,
    ScoreTiltedHMC,
    UnadjustedLangevin,
)
from utils import effective_sample_size, mean_squared_error, summary_statistics


# Map string names to target classes
TARGET_MAP = {
    "CorrelatedGaussian": CorrelatedGaussian,
    "GaussianMixture": GaussianMixture,
    "GaussianMixtureThreeMode": GaussianMixtureThreeMode,
    "SyntheticLogistic": SyntheticLogistic,
    "GaussianMixtureTriangle": GaussianMixtureTriangle,
}

# Map string names to sampler constructors
SAMPLER_MAP = {
    "MALA": MALA,
    "Underdamped": UnderdampedLangevin,
    "ULD": UnadjustedLangevin,
    "ScoreTilted": ScoreTiltedMCMC,
    "SRMALA": ScoreTiltedMCMC,
    "SR-MALA": ScoreTiltedMCMC,
    # [DEPRECATED] MetropolisUnderdamped
    # "MetropolisUnderdamped": MetropolisUnderdampedLangevin,
    # [DEPRECATED] UnderdampedScoreTilted
    # "UnderdampedScoreTilted": UnderdampedScoreTiltedMCMC,
    
    # new additions
    "HMC": HMC,
    "ScoreTiltedHMC": ScoreTiltedHMC,
    "SRHMC": ScoreTiltedHMC,
    "SR-HMC": ScoreTiltedHMC,
    "UnadjustedLangevin": UnadjustedLangevin,
    "UnadjustedScoreTilted": UnadjustedScoreTiltedULD,
    "SRULD": UnadjustedScoreTiltedULD,
    "SR-ULD": UnadjustedScoreTiltedULD,
}


def run_experiment(config: Dict[str, Any], output_path: Path) -> None:
    """Run an experiment as specified in a configuration dictionary.

    Parameters
    ----------
    config : dict
        Parsed JSON configuration.
    output_path : Path
        Path to write the results JSON file.
    """
    exp_name = config.get("experiment_name", "experiment")
    target_cfg = config["target"]
    target_name = target_cfg["name"]
    target_params = target_cfg.get("params", {})
    sampler_cfgs = config["samplers"]
    n_steps = config["n_steps"]
    burn_in = config.get("burn_in", 0)
    num_reps = config.get("num_replicates", 1)
    seed = config.get("seed", None)
    true_mean = np.array(config.get("true_mean", []), dtype=float) if "true_mean" in config else None

    # Instantiate target distribution
    if target_name not in TARGET_MAP:
        raise ValueError(f"Unknown target distribution {target_name}")
    target_cls = TARGET_MAP[target_name]

    # Preallocate result storage
    results: Dict[str, Any] = {
        "experiment_name": exp_name,
        "target": target_name,
        "target_params": target_params,
        "sampler_results": {},
        "config": config,
    }
     # --- FIX 1: Deterministic Seed & Init Handling ---
    # Master RNG for generating seeds and initial positions
    base_rng = np.random.default_rng(seed)
    
    # 1. Generate specific seeds for each replicate (to be used by samplers)
    replicate_seeds = base_rng.integers(0, 2**32 - 1, size=num_reps)
    # 2. Pre-generate initial positions (x0) for each replicate
    # This ensures EVERY sampler starts at the EXACT same positions.
    replicate_x0s = []
    # We need to instantiate a temp target to get dimension if not explicitly in params
    # (A bit hacky but robust for variable dim targets)
    temp_params = target_params.copy()
    if "cov_matrices" in temp_params and temp_params["cov_matrices"] is not None:
        temp_params["cov_matrices"] = [np.array(c) for c in temp_params["cov_matrices"]]
    temp_target = target_cls(**temp_params)
    dim = getattr(temp_target, "dim", target_params.get("dim", 2)) # Default 2 if unknown
    
    for _ in range(num_reps):
        replicate_x0s.append(base_rng.normal(size=dim))
    # -------------------------------------------------
    # For each sampler
    for sampler_cfg in sampler_cfgs:
        sampler_name = sampler_cfg["name"]
        result_key = sampler_cfg.get("label", sampler_name)
        if sampler_name not in SAMPLER_MAP:
            raise ValueError(f"Unknown sampler {sampler_name}")
        sampler_cls = SAMPLER_MAP[sampler_name]
        # collect metrics across replicates
        replicate_stats: List[Dict[str, Any]] = []
        for rep in range(num_reps):
            init_params = target_params.copy()
            if "cov_matrices" in init_params and init_params["cov_matrices"] is not None:
                init_params["cov_matrices"] = [np.array(c) for c in init_params["cov_matrices"]]
            # instantiate new target for each replicate (some targets generate data on init)
            target = target_cls(**init_params)
             # Use pre-generated x0
            x0 = replicate_x0s[rep].copy()
            
            # Use pre-generated seed
            rng = np.random.default_rng(replicate_seeds[rep])
            
            kwargs = {k: v for k, v in sampler_cfg.items() if k not in {"name"}}
            # Factory logic (simplified for clarity, can be optimized)
            if sampler_name in ["Underdamped", "MetropolisUnderdamped", "UnadjustedLangevin", "ULD"]:
                sampler = sampler_cls(
                    target,
                    step_size=kwargs["step_size"],
                    friction=kwargs.get("friction", 1.0),
                    rng=rng,
                )
            elif sampler_name in ["ScoreTilted", "SRMALA", "SR-MALA"]:
                sampler = sampler_cls(
                    target,
                    step_size=kwargs["step_size"],
                    alpha=kwargs["alpha"],
                    theta_step=kwargs.get("theta_step", 1.0),
                    use_shifted_gradient=kwargs.get("use_shifted_gradient", True),
                    alpha_warmup_steps=kwargs.get("alpha_warmup_steps", 0),
                    alpha_adaptive=kwargs.get("alpha_adaptive", False),
                    alpha_C=kwargs.get("alpha_C", 100.0),
                    rng=rng,
                )
            elif sampler_name in ["UnderdampedScoreTilted", "UnadjustedScoreTilted", "SRULD", "SR-ULD"]:
                sampler = sampler_cls(
                    target,
                    step_size=kwargs["step_size"],
                    alpha=kwargs["alpha"],
                    theta_step=kwargs.get("theta_step", 1.0),
                    friction=kwargs.get("friction", 1.0),
                    use_shifted_gradient=kwargs.get("use_shifted_gradient", True),
                    alpha_warmup_steps=kwargs.get("alpha_warmup_steps", 0),
                    alpha_adaptive=kwargs.get("alpha_adaptive", False),
                    alpha_C=kwargs.get("alpha_C", 100.0),
                    rng=rng,
                )
            elif sampler_name == "HMC":
                sampler = sampler_cls(
                    target,
                    step_size=kwargs["step_size"],
                    n_leapfrog=kwargs.get("n_leapfrog", 5),
                    rng=rng,
                )
            elif sampler_name in ["ScoreTiltedHMC", "SRHMC", "SR-HMC"]:
                sampler = sampler_cls(
                    target,
                    step_size=kwargs["step_size"],
                    alpha=kwargs["alpha"],
                    theta_step=kwargs.get("theta_step", 1.0),
                    n_leapfrog=kwargs.get("n_leapfrog", 5),
                    use_shifted_gradient=kwargs.get("use_shifted_gradient", True),
                    alpha_warmup_steps=kwargs.get("alpha_warmup_steps", 0),
                    alpha_adaptive=kwargs.get("alpha_adaptive", False),
                    alpha_C=kwargs.get("alpha_C", 100.0),
                    rng=rng,
                )
            else:
                sampler = sampler_cls(target, step_size=kwargs["step_size"], rng=rng)
            
            samples, diagnostics = sampler.run(x0, n_steps=n_steps, burn_in=burn_in)
            # compute summary statistics
            stats = summary_statistics(samples)
            ess = stats["ess"]
            mse = None
            if true_mean is not None and len(true_mean) == dim:
                mse = mean_squared_error(samples, true_mean)
            replicate_stats.append(
                {
                    "samples": None,  # omit raw samples to reduce output size
                    "ess": ess.tolist(),
                    "diagnostics": diagnostics.as_dict(),
                    "mse": mse,
                }
            )
        # aggregate statistics across replicates
        ess_array = np.array([r["ess"] for r in replicate_stats])
        mean_ess = np.mean(ess_array, axis=0)
        std_ess = np.std(ess_array, axis=0)
        acc_rates = [r["diagnostics"]["acceptance_rate"] for r in replicate_stats]
        mean_acc = float(np.mean(acc_rates))
        std_acc = float(np.std(acc_rates))
        runtimes = [r["diagnostics"]["runtime"] for r in replicate_stats]
        mean_runtime = float(np.mean(runtimes))
        mse_vals = [r["mse"] for r in replicate_stats if r["mse"] is not None]
        mean_mse = float(np.mean(mse_vals)) if mse_vals else None
        results["sampler_results"][result_key] = {
            "sampler_name": sampler_name,
            "mean_ess": mean_ess.tolist(),
            "std_ess": std_ess.tolist(),
            "mean_acceptance_rate": mean_acc,
            "std_acceptance_rate": std_acc,
            "mean_runtime": mean_runtime,
            "mean_mse": mean_mse,
            # omit replicate details to reduce output size
            # "replicates": replicate_stats, 
        }
    # Write results to output file
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run adaptive MCMC experiments from a config file.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON configuration file.")
    parser.add_argument("--output", type=str, required=True, help="Path to JSON file to write results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    output_path = Path(args.output)
    with config_path.open("r") as f:
        config = json.load(f)
    run_experiment(config, output_path)


if __name__ == "__main__":
    main()
