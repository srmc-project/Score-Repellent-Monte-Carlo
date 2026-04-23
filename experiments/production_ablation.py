"""
Stage 5: Production Runs for Ablation Study

Ablation structure (3 groups x 3 configurations each):
- Overdamped Group: MALA, SR-MALA (Shifted x Fixed/Adaptive)
- Underdamped Group: ULD, SR-ULD (Shifted x Fixed/Adaptive)
- HMC Group: HMC, SR-HMC (Shifted x Fixed/Adaptive)

Settings:
- num_replicates = 30
- n_steps = 100000
- burn_in = 10000
- seed = 2026

For HMC x-axis: use total leapfrog steps (iteration × n_leapfrog)
"""

import numpy as np
import json
import pickle
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from potentials import (
    CorrelatedGaussian,
    GaussianMixture,
    GaussianMixtureTriangle,
    SyntheticLogistic,
)
from samplers import (
    ScoreTiltedMCMC,
    UnadjustedScoreTiltedULD,
    ScoreTiltedHMC,
    HMC,
    MALA,
    UnadjustedLangevin,
)
from utils import effective_sample_size, mean_squared_error

# =============================================================================
# Configuration
# =============================================================================

SEED = 2026
N_STEPS = 100000
BURN_IN = 10000
NUM_REPLICATES = 30
N_WORKERS = 12

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "rerun_2026-01-28"
RAW_DIR = OUTPUT_DIR / "raw_runs"
PLOT_DIR = OUTPUT_DIR / "final_plots"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Recording MSE at these intervals for convergence curves
RECORD_INTERVAL = 500  # Record MSE every 500 steps


# =============================================================================
# Target Configurations (optimal params from Stage 2-4)
# =============================================================================

TARGET_CONFIGS = {
    "CorrelatedGaussian": {
        "cls_name": "CorrelatedGaussian",
        "params": {"dim": 10, "rho": 0.9},
        "true_mean": [0.0] * 10,
        "dim": 10,
        # Optimal params from Stage 3-4
        "mala": {"step_size": 0.1},
        "uld": {"step_size": 0.1, "friction": 0.5},
        "hmc": {"step_size": 0.2, "n_leapfrog": 20},
        "sr_mala": {"step_size": 0.1, "alpha": 5.0, "theta_step": 1.0, "alpha_C": 1000},
        "sr_uld": {"step_size": 0.1, "alpha": 1.5, "theta_step": 0.5, "friction": 0.5, "alpha_C": 1000},
        "sr_hmc": {"step_size": 0.2, "alpha": 10.0, "theta_step": 0.3, "n_leapfrog": 20, "alpha_C": 1000},
    },
    "GaussianMixture": {
        "cls_name": "GaussianMixture",
        "params": {
            "dim": 2,
            "separation": 4.0,
            "weights": [0.8, 0.2],
            "cov_matrices": [[[0.0324, 0], [0, 0.0324]], [[1, 0], [0, 1]]],
        },
        "true_mean": [0.8 * (-2.0) + 0.2 * 2.0, 0.0],
        "dim": 2,
        "mala": {"step_size": 0.1},
        "uld": {"step_size": 0.1, "friction": 0.5},
        "hmc": {"step_size": 0.1, "n_leapfrog": 5},
        "sr_mala": {"step_size": 0.1, "alpha": 0.5, "theta_step": 0.3, "alpha_C": 1000},
        "sr_uld": {"step_size": 0.1, "alpha": 0.1, "theta_step": 0.5, "friction": 0.5, "alpha_C": 1000},
        "sr_hmc": {"step_size": 0.1, "alpha": 0.1, "theta_step": 1.0, "n_leapfrog": 5, "alpha_C": 1000},
    },
    "GaussianMixtureTriangle": {
        "cls_name": "GaussianMixtureTriangle",
        "params": {"dim": 2, "separation": 5.0},
        "true_mean": [0.0, 0.0],
        "dim": 2,
        "mala": {"step_size": 0.03},
        "uld": {"step_size": 0.1, "friction": 0.5},
        "hmc": {"step_size": 0.15, "n_leapfrog": 5},
        "sr_mala": {"step_size": 0.03, "alpha": 1.0, "theta_step": 0.3, "alpha_C": 1000},
        "sr_uld": {"step_size": 0.1, "alpha": 2.5, "theta_step": 1.0, "friction": 0.1, "alpha_C": 1000},
        "sr_hmc": {"step_size": 0.15, "alpha": 0.1, "theta_step": 0.5, "n_leapfrog": 5, "alpha_C": 1000},
    },
    "SyntheticLogistic": {
        "cls_name": "SyntheticLogistic",
        "params": {"n_samples": 100, "dim": 10, "seed": 42},
        "true_mean": [0.0] * 10,
        "dim": 10,
        "mala": {"step_size": 0.05},
        "uld": {"step_size": 0.1, "friction": 0.5},
        "hmc": {"step_size": 0.02, "n_leapfrog": 30},
        "sr_mala": {"step_size": 0.05, "alpha": 0.1, "theta_step": 0.5, "alpha_C": 1000},
        "sr_uld": {"step_size": 0.1, "alpha": 0.1, "theta_step": 0.3, "friction": 0.1, "alpha_C": 1000},
        "sr_hmc": {"step_size": 0.02, "alpha": 0.1, "theta_step": 0.3, "n_leapfrog": 30, "alpha_C": 1000},
    },
}


def get_target_cls(cls_name: str):
    """Get target class by name."""
    return {
        "CorrelatedGaussian": CorrelatedGaussian,
        "GaussianMixture": GaussianMixture,
        "GaussianMixtureTriangle": GaussianMixtureTriangle,
        "SyntheticLogistic": SyntheticLogistic,
    }[cls_name]


# =============================================================================
# Sampler Configuration Classes
# =============================================================================

@dataclass
class SamplerConfig:
    """Configuration for a sampler run."""
    group: str  # "overdamped", "underdamped", "hmc"
    name: str   # e.g., "MALA", "SR-MALA_shifted_fixed"
    is_baseline: bool
    use_shifted: bool = False
    alpha_adaptive: bool = False
    n_leapfrog: int = 1  # For x-axis normalization


def get_sampler_configs() -> List[SamplerConfig]:
    """Return all sampler configurations for ablation study."""
    configs = []
    
    # === Overdamped Group (MALA-based) ===
    configs.append(SamplerConfig(
        group="overdamped", name="MALA", is_baseline=True, n_leapfrog=1))
    configs.append(SamplerConfig(
        group="overdamped", name="SR-MALA_shifted_fixed", is_baseline=False,
        use_shifted=True, alpha_adaptive=False, n_leapfrog=1))
    configs.append(SamplerConfig(
        group="overdamped", name="SR-MALA_shifted_adaptive", is_baseline=False,
        use_shifted=True, alpha_adaptive=True, n_leapfrog=1))
    
    # === Underdamped Group (ULD-based) ===
    configs.append(SamplerConfig(
        group="underdamped", name="ULD", is_baseline=True, n_leapfrog=1))
    configs.append(SamplerConfig(
        group="underdamped", name="SR-ULD_shifted_fixed", is_baseline=False,
        use_shifted=True, alpha_adaptive=False, n_leapfrog=1))
    configs.append(SamplerConfig(
        group="underdamped", name="SR-ULD_shifted_adaptive", is_baseline=False,
        use_shifted=True, alpha_adaptive=True, n_leapfrog=1))
    
    # === HMC Group ===
    # n_leapfrog will be set per-target
    configs.append(SamplerConfig(
        group="hmc", name="HMC", is_baseline=True, n_leapfrog=1))
    configs.append(SamplerConfig(
        group="hmc", name="SR-HMC_shifted_fixed", is_baseline=False,
        use_shifted=True, alpha_adaptive=False, n_leapfrog=1))
    configs.append(SamplerConfig(
        group="hmc", name="SR-HMC_shifted_adaptive", is_baseline=False,
        use_shifted=True, alpha_adaptive=True, n_leapfrog=1))
    
    return configs


# =============================================================================
# Single Run Function (for parallel execution)
# =============================================================================

def run_single_replicate(args: Tuple) -> Dict[str, Any]:
    """
    Run a single (target, sampler_config, seed) configuration.
    Returns convergence data and final metrics.
    """
    target_name, sampler_config_dict, seed = args
    
    # Reconstruct SamplerConfig from dict
    sc = SamplerConfig(**sampler_config_dict)
    
    config = TARGET_CONFIGS[target_name]
    target_cls = get_target_cls(config["cls_name"])
    target = target_cls(**config["params"])
    true_mean = np.array(config["true_mean"])
    dim = config["dim"]
    
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=dim)
    
    # Set n_leapfrog for HMC group
    if sc.group == "hmc":
        n_leapfrog = config["hmc"]["n_leapfrog"]
    else:
        n_leapfrog = 1
    
    try:
        start_time = time.perf_counter()
        
        # Create sampler based on config
        if sc.group == "overdamped":
            if sc.is_baseline:
                # MALA baseline
                sampler = MALA(
                    target,
                    step_size=config["mala"]["step_size"],
                    rng=rng,
                )
            else:
                # SR-MALA
                params = config["sr_mala"]
                sampler = ScoreTiltedMCMC(
                    target,
                    step_size=params["step_size"],
                    alpha=params["alpha"],
                    theta_step=params["theta_step"],
                    use_shifted_gradient=sc.use_shifted,
                    alpha_adaptive=sc.alpha_adaptive,
                    alpha_C=params["alpha_C"] if sc.alpha_adaptive else 100,
                    rng=rng,
                )
        
        elif sc.group == "underdamped":
            if sc.is_baseline:
                # ULD baseline (use ULD group params)
                params = config["uld"]
                sampler = UnadjustedLangevin(
                    target,
                    step_size=params["step_size"],
                    friction=params.get("friction", 1.0),
                    rng=rng,
                )
            else:
                # SR-ULD
                params = config["sr_uld"]
                sampler = UnadjustedScoreTiltedULD(
                    target,
                    step_size=params["step_size"],
                    alpha=params["alpha"],
                    theta_step=params["theta_step"],
                    friction=params.get("friction", 1.0),
                    use_shifted_gradient=sc.use_shifted,
                    alpha_adaptive=sc.alpha_adaptive,
                    alpha_C=params["alpha_C"] if sc.alpha_adaptive else 100,
                    rng=rng,
                )
        
        elif sc.group == "hmc":
            if sc.is_baseline:
                # HMC baseline
                sampler = HMC(
                    target,
                    step_size=config["hmc"]["step_size"],
                    n_leapfrog=config["hmc"]["n_leapfrog"],
                    rng=rng,
                )
            else:
                # SR-HMC
                params = config["sr_hmc"]
                sampler = ScoreTiltedHMC(
                    target,
                    step_size=params["step_size"],
                    alpha=params["alpha"],
                    theta_step=params["theta_step"],
                    n_leapfrog=params["n_leapfrog"],
                    use_shifted_gradient=sc.use_shifted,
                    alpha_adaptive=sc.alpha_adaptive,
                    alpha_C=params["alpha_C"] if sc.alpha_adaptive else 100,
                    rng=rng,
                )
        
        # Run sampling with convergence tracking
        samples, diag = sampler.run(x0, n_steps=N_STEPS, burn_in=BURN_IN)
        
        cpu_time = time.perf_counter() - start_time
        
        if np.any(~np.isfinite(samples)):
            return {"failed": True, "sampler": sc.name, "target": target_name}
        
        # Compute convergence curve (cumulative MSE at intervals)
        n_samples = len(samples)
        record_points = list(range(RECORD_INTERVAL, n_samples + 1, RECORD_INTERVAL))
        mse_curve = []
        for i in record_points:
            cumulative_mean = np.mean(samples[:i], axis=0)
            mse = np.sum((cumulative_mean - true_mean) ** 2)
            mse_curve.append(float(mse))
        
        # Final metrics
        ess = effective_sample_size(samples)
        final_mse = mean_squared_error(samples, true_mean)
        acc_rate = diag.acceptance_rate
        
        return {
            "failed": False,
            "target": target_name,
            "sampler": sc.name,
            "group": sc.group,
            "n_leapfrog": n_leapfrog,
            "mse_curve": mse_curve,
            "record_points": record_points,
            "ess_min": float(np.min(ess)),
            "ess_mean": float(np.mean(ess)),
            "final_mse": float(final_mse),
            "acceptance_rate": float(acc_rate),
            "cpu_time": float(cpu_time),
            "n_samples": n_samples,
        }
        
    except Exception as e:
        return {"failed": True, "sampler": sc.name, "target": target_name, "error": str(e)}


# =============================================================================
# Main Production Run
# =============================================================================

def run_production_for_target(target_name: str) -> Dict[str, Any]:
    """Run all sampler configurations for a single target."""
    
    print(f"\n{'='*60}")
    print(f"Target: {target_name}")
    print(f"{'='*60}")
    
    config = TARGET_CONFIGS[target_name]
    sampler_configs = get_sampler_configs()
    
    # Update n_leapfrog for HMC group
    hmc_n_leapfrog = config["hmc"]["n_leapfrog"]
    for sc in sampler_configs:
        if sc.group == "hmc":
            sc.n_leapfrog = hmc_n_leapfrog
    
    # Generate seeds
    base_rng = np.random.default_rng(SEED)
    rep_seeds = base_rng.integers(0, 2**32 - 1, size=NUM_REPLICATES).tolist()
    
    # Prepare all tasks
    tasks = []
    for sc in sampler_configs:
        sc_dict = {
            "group": sc.group,
            "name": sc.name,
            "is_baseline": sc.is_baseline,
            "use_shifted": sc.use_shifted,
            "alpha_adaptive": sc.alpha_adaptive,
            "n_leapfrog": sc.n_leapfrog,
        }
        for seed in rep_seeds:
            tasks.append((target_name, sc_dict, seed))
    
    total_tasks = len(tasks)
    print(f"  Total tasks: {total_tasks} ({len(sampler_configs)} samplers × {NUM_REPLICATES} replicates)")
    
    # Run in parallel
    results_by_sampler = {sc.name: [] for sc in sampler_configs}
    completed = 0
    
    import sys
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(run_single_replicate, task): task for task in tasks}
        
        for future in as_completed(futures):
            result = future.result()
            if not result.get("failed", True):
                results_by_sampler[result["sampler"]].append(result)
            completed += 1
            
            if completed % 50 == 0 or completed == total_tasks:
                print(f"  Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.0f}%)")
                sys.stdout.flush()
    
    # Aggregate results
    summary = {
        "target": target_name,
        "n_steps": N_STEPS,
        "burn_in": BURN_IN,
        "num_replicates": NUM_REPLICATES,
        "hmc_n_leapfrog": hmc_n_leapfrog,
        "samplers": {},
    }
    
    for sampler_name, results in results_by_sampler.items():
        if not results:
            print(f"  WARNING: {sampler_name} has no successful runs!")
            continue
        
        # Aggregate metrics
        n_success = len(results)
        ess_mins = [r["ess_min"] for r in results]
        final_mses = [r["final_mse"] for r in results]
        cpu_times = [r["cpu_time"] for r in results]
        acc_rates = [r["acceptance_rate"] for r in results]
        
        # Aggregate MSE curves (all should have same length)
        curve_length = len(results[0]["mse_curve"])
        mse_curves = np.array([r["mse_curve"] for r in results])
        
        summary["samplers"][sampler_name] = {
            "group": results[0]["group"],
            "n_leapfrog": results[0]["n_leapfrog"],
            "n_success": n_success,
            "ess_min_mean": float(np.mean(ess_mins)),
            "ess_min_std": float(np.std(ess_mins)),
            "final_mse_mean": float(np.mean(final_mses)),
            "final_mse_std": float(np.std(final_mses)),
            "final_mse_median": float(np.median(final_mses)),
            "final_mse_p10": float(np.percentile(final_mses, 10)),
            "final_mse_p90": float(np.percentile(final_mses, 90)),
            "cpu_time_mean": float(np.mean(cpu_times)),
            "cpu_time_std": float(np.std(cpu_times)),
            "acceptance_rate_mean": float(np.mean(acc_rates)),
            "record_points": results[0]["record_points"],
            "mse_curve_median": np.median(mse_curves, axis=0).tolist(),
            "mse_curve_p10": np.percentile(mse_curves, 10, axis=0).tolist(),
            "mse_curve_p90": np.percentile(mse_curves, 90, axis=0).tolist(),
            "mse_curve_mean": np.mean(mse_curves, axis=0).tolist(),
        }
        
        # Print summary
        print(f"  {sampler_name}: ESS={np.mean(ess_mins):.1f}, MSE={np.mean(final_mses):.6f}, "
              f"CPU={np.mean(cpu_times):.1f}s, n={n_success}")
    
    return summary


def main():
    print("="*60)
    print("STAGE 5: PRODUCTION RUNS (ABLATION STUDY)")
    print("="*60)
    print(f"Settings: n_steps={N_STEPS}, burn_in={BURN_IN}, num_replicates={NUM_REPLICATES}")
    print(f"Workers: {N_WORKERS}")
    print()
    print("Ablation Structure:")
    print("  Overdamped: MALA, SR-MALA (Shifted x Fixed/Adaptive)")
    print("  Underdamped: ULD, SR-ULD (Shifted x Fixed/Adaptive)")
    print("  HMC: HMC, SR-HMC (Shifted x Fixed/Adaptive)")
    
    all_results = {}
    
    for target_name in TARGET_CONFIGS.keys():
        summary = run_production_for_target(target_name)
        all_results[target_name] = summary
        
        # Save per-target results
        result_file = RAW_DIR / f"production_{target_name}_summary.json"
        with open(result_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {result_file}")
    
    # Save all results
    all_file = RAW_DIR / "production_all_summary.json"
    with open(all_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results: {all_file}")
    
    # Print final summary table
    print("\n" + "="*60)
    print("FINAL SUMMARY (Final MSE, lower is better)")
    print("="*60)
    
    for target_name, res in all_results.items():
        print(f"\n{target_name}:")
        print("-" * 50)
        
        for group in ["overdamped", "underdamped", "hmc"]:
            print(f"  [{group.upper()}]")
            for sampler_name, data in res["samplers"].items():
                if data["group"] == group:
                    marker = "★" if data.get("n_leapfrog", 1) > 1 else ""
                    print(f"    {sampler_name:30s}: MSE={data['final_mse_mean']:.6f} "
                          f"(±{data['final_mse_std']:.6f}), ESS={data['ess_min_mean']:.1f}")


if __name__ == "__main__":
    main()
