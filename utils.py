"""Utility functions for analysing MCMC output.

This module provides functions to compute diagnostics such as
effective sample size (ESS), integrated autocorrelation time (IACT),
and mean squared error of Monte Carlo estimates.
"""

from __future__ import annotations

from typing import Dict
import numpy as np

def ess_geyer_ips(x: np.ndarray) -> float:
    """
    x: shape (N,) 1D chain
    Returns ESS with Geyer's initial positive sequence.
    """
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if N < 3:
        return float(N)

    x = x - x.mean()
    var = np.dot(x, x) / N
    if var <= 0:
        return 1.0

    # autocovariance via FFT (fast + stable)
    f = np.fft.rfft(x, n=2*N)
    acov = np.fft.irfft(f * np.conjugate(f))[:N] / N
    rho = acov / acov[0]

    # Geyer IPS: sum pairs until they become negative
    t = 1
    s = 0.0
    while t + 1 < N:
        pair = rho[t] + rho[t+1]
        if pair <= 0:
            break
        s += pair
        t += 2

    tau = 1.0 + 2.0 * s  # IACT
    ess = N / tau
    # clamp
    return float(min(N, max(1.0, ess)))

def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """
    samples: shape (N, D)
    returns ESS per-dimension shape (D,)
    """
    samples = np.asarray(samples)
    return np.array([ess_geyer_ips(samples[:, d]) for d in range(samples.shape[1])])



def mean_squared_error(samples: np.ndarray, true_mean: np.ndarray) -> float:
    """Compute the mean squared error of the sample mean estimate.
    Averages error across all dimensions.
    """
    est_mean = np.mean(samples, axis=0)
    diff = est_mean - true_mean
    return float(np.mean(diff ** 2))


def cumulative_mean_squared_error(samples: np.ndarray, true_mean: np.ndarray) -> np.ndarray:
    """
    Computes cumulative MSE curve: || (1/t * Sum_{i=1}^t x_i) - true_mean ||^2
    Averages error across all dimensions at each time step.
    
    samples: (N, D)
    true_mean: (D,)
    returns: (N,)
    """
    N, D = samples.shape
    cum_sum = np.cumsum(samples, axis=0)
    counts = np.arange(1, N + 1)[:, None]
    cum_mean = cum_sum / counts # (N, D)
    
    diff = cum_mean - true_mean
    # Use mean(axis=1) to be consistent with mean_squared_error
    return np.mean(diff**2, axis=1) 


def summary_statistics(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute summary statistics for the chain.

    Returns a dictionary with the mean, variance, and ESS per dimension.

    Parameters
    ----------
    samples : np.ndarray
        Chain samples of shape ``(n_samples, dim)``.

    Returns
    -------
    dict
        Dictionary containing arrays for mean, variance, and ESS.
    """
    # Ensure samples are numpy array
    samples = np.asarray(samples)
    
    mean = np.mean(samples, axis=0)
    var = np.var(samples, axis=0)
    
    # Calculate ESS
    ess = effective_sample_size(samples)
    
    return {"mean": mean, "var": var, "ess": ess}
