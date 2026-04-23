from __future__ import annotations

import json
import multiprocessing
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from potentials import CorrelatedGaussian, SyntheticLogistic
from samplers import HMC, MALA, ScoreTiltedHMC, ScoreTiltedMCMC


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR / "figure2_continuous_alpha_screening_0_0p5_1_2_3_5_c1000_target_init_20260327"
PLOT_PDF = OUTPUT_ROOT / "figure2_continuous_alpha_screening_0_0p5_1_2_3_5_c1000_target_init_20260327.pdf"
PLOT_PNG = OUTPUT_ROOT / "figure2_continuous_alpha_screening_0_0p5_1_2_3_5_c1000_target_init_20260327.png"
CLOCK_PLOT_PDF = OUTPUT_ROOT / "figure2_continuous_alpha_screening_0_0p5_1_2_3_5_c1000_target_init_cpu_clock_20260327.pdf"
CLOCK_PLOT_PNG = OUTPUT_ROOT / "figure2_continuous_alpha_screening_0_0p5_1_2_3_5_c1000_target_init_cpu_clock_20260327.png"
SUMMARY_JSON = OUTPUT_ROOT / "summary_figure2_continuous_alpha_screening_0_0p5_1_2_3_5_c1000_target_init_20260327.json"
REPORT_MD = OUTPUT_ROOT / "report_figure2_continuous_alpha_screening_0_0p5_1_2_3_5_c1000_target_init_20260327.md"
INIT_CACHE_JSON = OUTPUT_ROOT / "initial_points_target_init_20260327.json"

NUM_REPLICATES = 30
TOP_LEVEL_SEED = 20260327
BURN_IN_FRAC = 0.3
ALPHA_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]
FIXED_C = 1000.0
MALA_RECORD_INTERVAL = 100
HMC_RECORD_INTERVAL_LEAPFROG = 100
INIT_SEEDS = {
    "correlated_gaussian": 2026032701,
    "synthetic_logistic_reference": 2026032702,
}
SYNTHETIC_LOGISTIC_REFERENCE_HMC = {
    "step_size": 0.025,
    "n_leapfrog": 10,
    "n_steps": 35_000,
    "burn_in": 10_000,
    "thin": 200,
}

TRUE_MEAN_LOGISTIC = [
    -0.6212977967644445,
    -0.6799150597543924,
    0.7575568874980533,
    0.5150891994605535,
    -1.6048458013509426,
    1.1549478950607461,
    -0.06097425566741736,
    -1.0928740175979428,
    0.921151244865071,
    0.5188626731914046,
]

BLOCKS: Dict[str, Dict[str, Any]] = {
    "hmc_correlated_gaussian": {
        "algorithm": "HMC",
        "target_name": "CorrelatedGaussian",
        "target_cls": CorrelatedGaussian,
        "target_params": {"dim": 20, "rho": 0.9},
        "true_mean": [0.0] * 20,
        "dim": 20,
        "fixed_c": FIXED_C,
        "sampler_params": {"step_size": 0.2, "n_leapfrog": 20, "theta_step": 1.0},
        "matched_total_steps": 10_000,
    },
    "hmc_synthetic_logistic": {
        "algorithm": "HMC",
        "target_name": "SyntheticLogistic",
        "target_cls": SyntheticLogistic,
        "target_params": {"n_samples": 200, "dim": 10, "seed": 42},
        "true_mean": TRUE_MEAN_LOGISTIC,
        "dim": 10,
        "fixed_c": FIXED_C,
        "sampler_params": {"step_size": 0.025, "n_leapfrog": 10, "theta_step": 0.03},
        "matched_total_steps": 10_000,
    },
    "mala_correlated_gaussian": {
        "algorithm": "MALA",
        "target_name": "CorrelatedGaussian",
        "target_cls": CorrelatedGaussian,
        "target_params": {"dim": 20, "rho": 0.9},
        "true_mean": [0.0] * 20,
        "dim": 20,
        "fixed_c": FIXED_C,
        "sampler_params": {"step_size": 0.2, "theta_step": 1.0},
        "matched_total_steps": 10_000,
    },
    "mala_synthetic_logistic": {
        "algorithm": "MALA",
        "target_name": "SyntheticLogistic",
        "target_cls": SyntheticLogistic,
        "target_params": {"n_samples": 200, "dim": 10, "seed": 42},
        "true_mean": TRUE_MEAN_LOGISTIC,
        "dim": 10,
        "fixed_c": FIXED_C,
        "sampler_params": {"step_size": 0.01, "theta_step": 0.3},
        "matched_total_steps": 10_000,
    },
}

COLORS = {
    0.5: "#1E88E5",
    1.0: "#FB8C00",
    2.0: "#8E24AA",
    3.0: "#43A047",
    5.0: "#6D4C41",
}
BASELINE_COLOR = "#D81B60"


def _failure_reason(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _target_key(block_cfg: Dict[str, Any]) -> str:
    return block_cfg["target_name"].lower()


def _replicate_setup() -> List[int]:
    rng = np.random.default_rng(TOP_LEVEL_SEED)
    return [int(s) for s in rng.integers(0, 2**32 - 1, size=NUM_REPLICATES).tolist()]


def _make_sampler(block_cfg: Dict[str, Any], schedule: str, alpha_target: float, rng: np.random.Generator) -> Any:
    params = block_cfg["sampler_params"]
    target = block_cfg["target_cls"](**block_cfg["target_params"])

    if block_cfg["algorithm"] == "MALA":
        if schedule == "baseline":
            sampler = MALA(target, step_size=params["step_size"], rng=rng)
        else:
            sampler = ScoreTiltedMCMC(
                target,
                step_size=params["step_size"],
                alpha=alpha_target,
                theta_step=params["theta_step"],
                use_shifted_gradient=True,
                alpha_warmup_steps=0,
                alpha_adaptive=schedule == "adaptive",
                alpha_C=block_cfg["fixed_c"],
                rng=rng,
            )
        return target, sampler

    if schedule == "baseline":
        sampler = HMC(
            target,
            step_size=params["step_size"],
            n_leapfrog=params["n_leapfrog"],
            rng=rng,
        )
    else:
        sampler = ScoreTiltedHMC(
            target,
            step_size=params["step_size"],
            alpha=alpha_target,
            theta_step=params["theta_step"],
            n_leapfrog=params["n_leapfrog"],
            use_shifted_gradient=True,
            alpha_warmup_steps=0,
            alpha_adaptive=schedule == "adaptive",
            alpha_C=block_cfg["fixed_c"],
            rng=rng,
        )
    return target, sampler


def _build_initial_points() -> Dict[str, Dict[str, Any]]:
    if INIT_CACHE_JSON.exists():
        return json.loads(INIT_CACHE_JSON.read_text(encoding="utf-8"))

    correlated_cfg = BLOCKS["hmc_correlated_gaussian"]
    correlated_target = correlated_cfg["target_cls"](**correlated_cfg["target_params"])
    correlated_rng = np.random.default_rng(INIT_SEEDS["correlated_gaussian"])
    correlated_x0 = correlated_rng.multivariate_normal(
        mean=np.zeros(correlated_cfg["dim"], dtype=float),
        cov=correlated_target.cov,
        size=NUM_REPLICATES,
    )

    logistic_cfg = BLOCKS["hmc_synthetic_logistic"]
    logistic_target = logistic_cfg["target_cls"](**logistic_cfg["target_params"])
    logistic_rng = np.random.default_rng(INIT_SEEDS["synthetic_logistic_reference"])
    logistic_sampler = HMC(
        logistic_target,
        step_size=SYNTHETIC_LOGISTIC_REFERENCE_HMC["step_size"],
        n_leapfrog=SYNTHETIC_LOGISTIC_REFERENCE_HMC["n_leapfrog"],
        rng=logistic_rng,
    )
    logistic_x0_ref = logistic_rng.normal(size=logistic_cfg["dim"])
    logistic_samples, logistic_diag = logistic_sampler.run(
        logistic_x0_ref,
        n_steps=SYNTHETIC_LOGISTIC_REFERENCE_HMC["n_steps"],
        burn_in=SYNTHETIC_LOGISTIC_REFERENCE_HMC["burn_in"],
    )
    logistic_x0 = logistic_samples[:: SYNTHETIC_LOGISTIC_REFERENCE_HMC["thin"]][:NUM_REPLICATES]
    if logistic_x0.shape[0] < NUM_REPLICATES:
        raise RuntimeError("Reference HMC chain did not produce enough SyntheticLogistic initial draws.")

    payload = {
        "correlatedgaussian": {
            "policy": "exact_target_sample",
            "exception_documented": False,
            "init_seed": INIT_SEEDS["correlated_gaussian"],
            "shared_across_blocks": ["hmc_correlated_gaussian", "mala_correlated_gaussian"],
            "x0_values": correlated_x0.tolist(),
        },
        "syntheticlogistic": {
            "policy": "proxy_target_sample_via_reference_hmc",
            "exception_documented": True,
            "exception_reason": "SyntheticLogistic does not implement exact target sampling in potentials.py",
            "init_seed": INIT_SEEDS["synthetic_logistic_reference"],
            "shared_across_blocks": ["hmc_synthetic_logistic", "mala_synthetic_logistic"],
            "reference_sampler": {
                "algorithm": "HMC",
                "step_size": SYNTHETIC_LOGISTIC_REFERENCE_HMC["step_size"],
                "n_leapfrog": SYNTHETIC_LOGISTIC_REFERENCE_HMC["n_leapfrog"],
                "n_steps": SYNTHETIC_LOGISTIC_REFERENCE_HMC["n_steps"],
                "burn_in": SYNTHETIC_LOGISTIC_REFERENCE_HMC["burn_in"],
                "thin": SYNTHETIC_LOGISTIC_REFERENCE_HMC["thin"],
                "runtime": float(logistic_diag.runtime),
                "acceptance_rate": float(logistic_diag.acceptance_rate),
            },
            "x0_values": logistic_x0.tolist(),
        },
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    INIT_CACHE_JSON.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _curve_from_samples(samples: np.ndarray, true_mean: np.ndarray, record_points: np.ndarray) -> Tuple[List[float], List[List[float]]]:
    cumsum = np.cumsum(samples, axis=0)
    running_means = cumsum[record_points - 1] / record_points[:, None]
    curve = np.mean((running_means - true_mean) ** 2, axis=1)
    return curve.astype(float).tolist(), running_means.astype(float).tolist()


def _run_single(task: Tuple[str, str, float, int, List[float]]) -> Dict[str, Any]:
    block_name, schedule, alpha_target, seed, x0_values = task
    block_cfg = BLOCKS[block_name]
    true_mean = np.asarray(block_cfg["true_mean"], dtype=float)
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0_values, dtype=float)
    _, sampler = _make_sampler(block_cfg, schedule, alpha_target, rng)

    try:
        if block_cfg["algorithm"] == "HMC":
            n_leapfrog = block_cfg["sampler_params"]["n_leapfrog"]
            n_steps = block_cfg["matched_total_steps"] // n_leapfrog
            burn_in = int(round(BURN_IN_FRAC * n_steps))
            record_interval = max(1, HMC_RECORD_INTERVAL_LEAPFROG // n_leapfrog)
            samples, diagnostics = sampler.run(x0, n_steps=n_steps, burn_in=burn_in, record_interval=record_interval)
            if diagnostics.recorded_step_indices and diagnostics.recorded_elapsed_times:
                record_points = np.asarray(diagnostics.recorded_step_indices, dtype=int) - burn_in
                cpu_time_points = np.asarray(diagnostics.recorded_elapsed_times, dtype=float)
            else:
                record_points = np.asarray(list(range(record_interval, samples.shape[0] + 1, record_interval)), dtype=int)
                if record_points.size == 0 or record_points[-1] != samples.shape[0]:
                    record_points = np.append(record_points, samples.shape[0])
                cpu_time_points = float(diagnostics.runtime) * (record_points / samples.shape[0])
            curve, estimator_curve = _curve_from_samples(samples, true_mean, record_points)
            global_x = ((burn_in + record_points) * n_leapfrog).astype(int).tolist()
        else:
            n_steps = block_cfg["matched_total_steps"]
            burn_in = int(round(BURN_IN_FRAC * n_steps))
            record_interval = MALA_RECORD_INTERVAL
            samples, diagnostics = sampler.run(x0, n_steps=n_steps, burn_in=burn_in, record_interval=record_interval)
            if diagnostics.recorded_step_indices and diagnostics.recorded_elapsed_times:
                record_points = np.asarray(diagnostics.recorded_step_indices, dtype=int) - burn_in
                cpu_time_points = np.asarray(diagnostics.recorded_elapsed_times, dtype=float)
            else:
                record_points = np.asarray(list(range(record_interval, samples.shape[0] + 1, record_interval)), dtype=int)
                if record_points.size == 0 or record_points[-1] != samples.shape[0]:
                    record_points = np.append(record_points, samples.shape[0])
                cpu_time_points = float(diagnostics.runtime) * (record_points / samples.shape[0])
            curve, estimator_curve = _curve_from_samples(samples, true_mean, record_points)
            global_x = (burn_in + record_points).astype(int).tolist()
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "block_name": block_name,
            "schedule": schedule,
            "alpha_target": alpha_target,
            "error": _failure_reason(exc),
        }

    if samples.size == 0 or np.any(~np.isfinite(samples)):
        return {
            "ok": False,
            "block_name": block_name,
            "schedule": schedule,
            "alpha_target": alpha_target,
            "error": "ValueError: empty_or_non_finite_samples",
        }

    final_mse = float(curve[-1])
    if not np.isfinite(final_mse):
        return {
            "ok": False,
            "block_name": block_name,
            "schedule": schedule,
            "alpha_target": alpha_target,
            "error": "ValueError: non_finite_final_mse",
        }

    return {
        "ok": True,
        "block_name": block_name,
        "schedule": schedule,
        "alpha_target": alpha_target,
        "curve_x": global_x,
        "cpu_time_x": cpu_time_points.astype(float).tolist(),
        "curve_y": curve,
        "estimator_curve": estimator_curve,
        "final_mse": final_mse,
        "runtime": float(diagnostics.runtime),
        "acceptance_rate": float(diagnostics.acceptance_rate),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok_rows = [row for row in rows if row["ok"]]
    fail_rows = [row for row in rows if not row["ok"]]
    payload: Dict[str, Any] = {
        "num_total": len(rows),
        "num_success": len(ok_rows),
        "num_fail": len(fail_rows),
        "failure_reasons": dict(Counter(row["error"] for row in fail_rows)),
    }
    if not ok_rows:
        return payload

    curves = np.asarray([row["curve_y"] for row in ok_rows], dtype=float)
    cpu_time = np.asarray([row["cpu_time_x"] for row in ok_rows], dtype=float)
    estimator_curves = np.asarray([row["estimator_curve"] for row in ok_rows], dtype=float)
    final_mse = np.asarray([row["final_mse"] for row in ok_rows], dtype=float)
    runtime = np.asarray([row["runtime"] for row in ok_rows], dtype=float)
    acceptance = np.asarray([row["acceptance_rate"] for row in ok_rows], dtype=float)
    n = len(ok_rows)

    payload.update(
        {
            "curve_x": ok_rows[0]["curve_x"],
            "cpu_time_points_mean": np.mean(cpu_time, axis=0).tolist(),
            "cpu_time_points_std": np.std(cpu_time, axis=0).tolist(),
            "curve_mean": np.mean(curves, axis=0).tolist(),
            "curve_std": np.std(curves, axis=0).tolist(),
            "curve_ci95": (1.96 * np.std(curves, axis=0) / np.sqrt(n)).tolist(),
            "running_estimator_mean": np.mean(estimator_curves, axis=0).tolist(),
            "running_estimator_std": np.std(estimator_curves, axis=0).tolist(),
            "final_mse_mean": float(np.mean(final_mse)),
            "final_mse_std": float(np.std(final_mse)),
            "mean_runtime": float(np.mean(runtime)),
            "std_runtime": float(np.std(runtime)),
            "mean_acceptance_rate": float(np.mean(acceptance)),
            "std_acceptance_rate": float(np.std(acceptance)),
            "mse_definition": "mean_squared_per_dimension",
            "mse_burn_in_frac": BURN_IN_FRAC,
        }
    )
    return payload


def _stable_log_limits(block: Dict[str, Any]) -> Tuple[float, float]:
    stable_values: List[float] = []
    final_values: List[float] = []

    for schedule in ["baseline", "fixed", "adaptive"]:
        items = [("0.0", block["results"]["baseline"]["0.0"])] if schedule == "baseline" else block["results"][schedule].items()
        for _, rec in items:
            if rec.get("num_success", 0) == 0:
                continue
            final_mse = float(rec["final_mse_mean"])
            if np.isfinite(final_mse) and final_mse > 0:
                final_values.append(final_mse)

    if not final_values:
        return 1e-12, 1.0

    best_final = min(final_values)
    stable_threshold = max(best_final * 1e6, best_final * 10.0)

    for schedule in ["baseline", "fixed", "adaptive"]:
        items = [("0.0", block["results"]["baseline"]["0.0"])] if schedule == "baseline" else block["results"][schedule].items()
        for _, rec in items:
            if rec.get("num_success", 0) == 0:
                continue
            final_mse = float(rec["final_mse_mean"])
            if not np.isfinite(final_mse) or final_mse <= 0 or final_mse > stable_threshold:
                continue
            stable_values.extend(float(y) for y in rec["curve_mean"] if np.isfinite(y) and float(y) > 0)

    if not stable_values:
        stable_values = [v for v in final_values if np.isfinite(v) and v > 0]

    y_min = min(stable_values)
    y_max = max(stable_values)
    y_min = 10 ** np.floor(np.log10(y_min) - 0.15)
    y_max = 10 ** np.ceil(np.log10(y_max) + 0.15)
    if y_max <= y_min:
        y_max = y_min * 10.0
    return float(y_min), float(y_max)


def _plot_series(
    ax: Any,
    x: np.ndarray,
    y: np.ndarray,
    ci95: np.ndarray,
    *,
    y_min: float,
    y_max: float,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
    errorevery: int,
) -> bool:
    lower_bound = y_min * 1.001
    upper_bound = y_max / 1.001
    clipped_y = np.clip(y, lower_bound, upper_bound)
    lower_err = np.minimum(ci95, np.maximum(clipped_y - lower_bound, 0.0))
    upper_err = np.minimum(ci95, np.maximum(upper_bound - clipped_y, 0.0))
    ax.errorbar(
        x,
        clipped_y,
        yerr=np.vstack([lower_err, upper_err]),
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        errorevery=errorevery,
        capsize=3,
        elinewidth=1.0,
        label=label,
    )
    return bool(np.any((y > upper_bound) | (y < lower_bound)))


def _cpu_clock_x(block: Dict[str, Any], rec: Dict[str, Any]) -> np.ndarray:
    cpu_points = rec.get("cpu_time_points_mean")
    if cpu_points:
        return np.asarray(cpu_points, dtype=float)
    curve_x = np.asarray(rec["curve_x"], dtype=float)
    total_steps = float(block["matched_total_steps"])
    mean_runtime = float(rec["mean_runtime"])
    return mean_runtime * (curve_x / total_steps)


def _run_block(
    block_name: str,
    workers: int,
    replicate_seeds: List[int],
    initial_points: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    block_cfg = BLOCKS[block_name]
    x0_values = initial_points[_target_key(block_cfg)]["x0_values"]
    tasks = [("baseline", 0.0)]
    tasks.extend(("fixed", alpha) for alpha in ALPHA_VALUES)
    tasks.extend(("adaptive", alpha) for alpha in ALPHA_VALUES)

    grouped: Dict[str, Dict[float, List[Dict[str, Any]]]] = {
        "baseline": {0.0: []},
        "fixed": {alpha: [] for alpha in ALPHA_VALUES},
        "adaptive": {alpha: [] for alpha in ALPHA_VALUES},
    }

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_run_single, (block_name, schedule, alpha, replicate_seeds[idx], x0_values[idx]))
            for schedule, alpha in tasks
            for idx in range(NUM_REPLICATES)
        ]
        for future in as_completed(futures):
            row = future.result()
            grouped[row["schedule"]][row["alpha_target"]].append(row)

    return {
        "block_name": block_name,
        "algorithm": block_cfg["algorithm"],
        "target_name": block_cfg["target_name"],
        "fixed_c": block_cfg["fixed_c"],
        "matched_total_steps": block_cfg["matched_total_steps"],
        "num_replicates": NUM_REPLICATES,
        "burn_in_frac": BURN_IN_FRAC,
        "replicate_seeds": replicate_seeds,
        "initialization_policy": initial_points[_target_key(block_cfg)]["policy"],
        "results": {
            "baseline": {"0.0": _aggregate(grouped["baseline"][0.0])},
            "fixed": {str(alpha): _aggregate(grouped["fixed"][alpha]) for alpha in ALPHA_VALUES},
            "adaptive": {str(alpha): _aggregate(grouped["adaptive"][alpha]) for alpha in ALPHA_VALUES},
        },
    }


def _plot(summary: Dict[str, Any], output_pdf: Path, output_png: Path, x_axis: str = "steps") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    order = [
        "hmc_correlated_gaussian",
        "hmc_synthetic_logistic",
        "mala_correlated_gaussian",
        "mala_synthetic_logistic",
    ]

    for ax, block_name in zip(axes.flatten(), order):
        block = summary["blocks"][block_name]
        algo = block["algorithm"]
        target = block["target_name"]
        y_min, y_max = _stable_log_limits(block)
        offscale_labels: List[str] = []

        baseline = block["results"]["baseline"]["0.0"]
        x = _cpu_clock_x(block, baseline) if x_axis == "cpu_clock" else np.asarray(baseline["curve_x"], dtype=float)
        y = np.asarray(baseline["curve_mean"], dtype=float)
        ci95 = np.asarray(baseline["curve_ci95"], dtype=float)
        if _plot_series(
            ax,
            x,
            y,
            ci95,
            y_min=y_min,
            y_max=y_max,
            color=BASELINE_COLOR,
            linestyle="-",
            linewidth=2.6,
            label="baseline (= fixed/adaptive 0)",
            errorevery=8,
        ):
            offscale_labels.append("baseline")

        for schedule, linestyle in [("fixed", "-"), ("adaptive", "--")]:
            for alpha in ALPHA_VALUES:
                rec = block["results"][schedule][str(alpha)]
                if rec["num_success"] == 0:
                    continue
                x = _cpu_clock_x(block, rec) if x_axis == "cpu_clock" else np.asarray(rec["curve_x"], dtype=float)
                y = np.asarray(rec["curve_mean"], dtype=float)
                ci95 = np.asarray(rec["curve_ci95"], dtype=float)
                label = f"{schedule} alpha={'*' if schedule == 'adaptive' else ''}{alpha:g}"
                clipped = _plot_series(
                    ax,
                    x=x,
                    y=y,
                    ci95=ci95,
                    y_min=y_min,
                    y_max=y_max,
                    color=COLORS[alpha],
                    linestyle=linestyle,
                    linewidth=2.0,
                    label=label,
                    errorevery=8,
                )
                if clipped:
                    offscale_labels.append(label)

        ax.set_yscale("log")
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{algo} + {target} (screening, C=1000)", fontsize=11)
        ax.set_ylabel("Running MSE")
        ax.set_xlabel("CPU Clock Time (s)" if x_axis == "cpu_clock" else ("Global Leapfrog Steps" if algo == "HMC" else "Global Step Index"))
        if offscale_labels:
            ax.text(
                0.02,
                0.03,
                "Off-scale clipped: " + ", ".join(offscale_labels),
                transform=ax.transAxes,
                fontsize=7,
                verticalalignment="bottom",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
            )
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(output_pdf, dpi=170)
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


def _write_report(summary: Dict[str, Any]) -> None:
    lines = [
        "# Figure2 Continuous Alpha Screening 0/0.5/1/2/3/5 (Target-Init, 2026-03-27)",
        "",
        "## Protocol",
        "",
        "- targets and algorithms: `HMC/MALA x CorrelatedGaussian/SyntheticLogistic`",
        "- compared conditions:",
        "  - baseline representing `fixed/adaptive alpha=0`",
        "  - fixed `alpha in {0.5, 1, 2, 3, 5}`",
        "  - adaptive `alpha_target in {0.5, 1, 2, 3, 5}`",
        f"- unified adaptive schedule constant: `C={FIXED_C:.0f}`",
        "- initialization follows the current repository protocol:",
        "  - exact target draws for `CorrelatedGaussian`",
        "  - proxy target draws via long reference HMC for `SyntheticLogistic`",
        f"- matched total steps: `{BLOCKS['mala_correlated_gaussian']['matched_total_steps']}`",
        f"- num_replicates: `{NUM_REPLICATES}`",
        f"- burn-in: `mse_burn_in_frac={BURN_IN_FRAC}`",
        "- MSE uses the canonical per-dimension definition on post-burn samples only",
        "",
        "## Final MSE Table",
        "",
        "| block | baseline | fixed 0.5 | adaptive 0.5 | fixed 1 | adaptive 1 | fixed 2 | adaptive 2 | fixed 3 | adaptive 3 | fixed 5 | adaptive 5 | best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for block_name in [
        "hmc_correlated_gaussian",
        "hmc_synthetic_logistic",
        "mala_correlated_gaussian",
        "mala_synthetic_logistic",
    ]:
        block = summary["blocks"][block_name]
        candidates = {"baseline": block["results"]["baseline"]["0.0"]["final_mse_mean"]}
        for alpha in ALPHA_VALUES:
            candidates[f"fixed alpha={alpha:g}"] = block["results"]["fixed"][str(alpha)]["final_mse_mean"]
            candidates[f"adaptive alpha*={alpha:g}"] = block["results"]["adaptive"][str(alpha)]["final_mse_mean"]
        best_name = min(candidates, key=candidates.get)
        lines.append(
            f"| {block_name} | {block['results']['baseline']['0.0']['final_mse_mean']:.6g} | "
            f"{block['results']['fixed']['0.5']['final_mse_mean']:.6g} | {block['results']['adaptive']['0.5']['final_mse_mean']:.6g} | "
            f"{block['results']['fixed']['1.0']['final_mse_mean']:.6g} | {block['results']['adaptive']['1.0']['final_mse_mean']:.6g} | "
            f"{block['results']['fixed']['2.0']['final_mse_mean']:.6g} | {block['results']['adaptive']['2.0']['final_mse_mean']:.6g} | "
            f"{block['results']['fixed']['3.0']['final_mse_mean']:.6g} | {block['results']['adaptive']['3.0']['final_mse_mean']:.6g} | "
            f"{block['results']['fixed']['5.0']['final_mse_mean']:.6g} | {block['results']['adaptive']['5.0']['final_mse_mean']:.6g} | "
            f"{best_name} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- step plot: `{PLOT_PDF}` and `{PLOT_PNG}`",
            f"- cpu-clock plot: `{CLOCK_PLOT_PDF}` and `{CLOCK_PLOT_PNG}`",
            f"- machine-readable summary: `{SUMMARY_JSON}`",
        ]
    )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    workers = max(1, multiprocessing.cpu_count() // 2)
    replicate_seeds = _replicate_setup()
    initial_points = _build_initial_points()
    summary = {
        "protocol_label": "figure2_continuous_alpha_screening_0_0p5_1_2_3_5_c1000_target_init",
        "top_level_seed": TOP_LEVEL_SEED,
        "mse_definition": "mean_squared_per_dimension",
        "mse_burn_in_frac": BURN_IN_FRAC,
        "num_replicates": NUM_REPLICATES,
        "alpha_values": [0.0] + ALPHA_VALUES,
        "fixed_c": FIXED_C,
        "replicate_seeds": replicate_seeds,
        "initialization": initial_points,
        "blocks": {},
    }

    for block_name in BLOCKS:
        summary["blocks"][block_name] = _run_block(block_name, workers, replicate_seeds, initial_points)

    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _plot(summary, PLOT_PDF, PLOT_PNG, x_axis="steps")
    _plot(summary, CLOCK_PLOT_PDF, CLOCK_PLOT_PNG, x_axis="cpu_clock")
    _write_report(summary)


if __name__ == "__main__":
    main()
