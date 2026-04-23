"""Potential functions for adaptive MCMC experiments.

Optimized version (Batch-Ready):
1. Supports inputs x of shape (D,) OR (N, D).
2. Fully vectorized using einsum with ellipsis (...) for batch handling.
3. Pre-converts lists to numpy arrays to avoid overhead.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logsumexp

class BasePotential:
    """Abstract base class for potential functions."""

    def log_prob(self, x: np.ndarray) -> float | np.ndarray:
        raise NotImplementedError

    def grad(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Default Finite Difference HVP."""
        eps = 1e-5
        grad_x = self.grad(x)
        grad_x_eps = self.grad(x + eps * v)
        return (grad_x_eps - grad_x) / eps

    def hvp_with_grad(self, x: np.ndarray, v: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
        """Return Hessian-vector product, reusing grad(x) if possible."""
        if getattr(type(self), 'hvp') is BasePotential.hvp:
            # Subclass uses default FD; reuse grad_x
            eps = 1e-5
            grad_x_eps = self.grad(x + eps * v)
            return (grad_x_eps - grad_x) / eps
        else:
            # Subclass has analytic HVP
            return self.hvp(x, v)


class CorrelatedGaussian(BasePotential):
    """Multivariate Gaussian with pairwise correlation."""

    def __init__(self, dim: int, rho: float = 0.9) -> None:
        self.dim = dim
        self.rho = rho
        eye = np.eye(dim)
        off_diag = rho * (np.ones((dim, dim)) - eye)
        self.cov = eye + off_diag
        self.inv_cov = np.linalg.inv(self.cov)
        _, logdet = np.linalg.slogdet(self.cov)
        self.log_norm_const = -0.5 * (logdet + dim * np.log(2 * np.pi))

    def log_prob(self, x: np.ndarray) -> float | np.ndarray:
        x = np.asarray(x)
        # x: (..., D) -> (...,)
        # einsum handles batch dimensions automatically
        quad = -0.5 * np.einsum("...i,ij,...j->...", x, self.inv_cov, x)
        return self.log_norm_const + quad

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        # result: (..., D)
        return -np.einsum("ij,...j->...i", self.inv_cov, x)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        return -np.einsum("ij,...j->...i", self.inv_cov, v)


class GaussianMixture(BasePotential):
    """Two-component Gaussian mixture distribution with optional weights/covariances."""

    def __init__(
        self,
        dim: int = 2,
        separation: float = 5.0,
        weights: list[float] | None = None,
        cov_matrices: list[np.ndarray] | None = None,
    ) -> None:
        self.dim = dim
        self.mean1 = np.zeros(dim)
        self.mean1[0] = -separation / 2
        self.mean2 = np.zeros(dim)
        self.mean2[0] = separation / 2
        self.means = np.stack((self.mean1, self.mean2), axis=0)

        if weights is None:
            self.weights = np.array([0.5, 0.5], dtype=float)
        else:
            self.weights = np.asarray(weights, dtype=float)
            self.weights = self.weights / self.weights.sum()
        self.log_weights = np.log(self.weights)

        if cov_matrices is None:
            covs = [np.eye(dim), np.eye(dim)]
        else:
            covs = [np.asarray(c, dtype=float) for c in cov_matrices]
        self.covs = np.asarray(covs, dtype=float)
        self.inv_covs = np.asarray([np.linalg.inv(c) for c in self.covs])
        _, logdets = np.linalg.slogdet(self.covs)
        self.log_norm_consts = -0.5 * (dim * np.log(2 * np.pi) + logdets)

    def log_prob(self, x: np.ndarray) -> float | np.ndarray:
        x = np.asarray(x)
        diff = x[..., None, :] - self.means
        quad = -0.5 * np.einsum("kij,...kj,...ki->...k", self.inv_covs, diff, diff)
        log_ps = self.log_norm_consts + quad + self.log_weights
        m = np.max(log_ps, axis=-1, keepdims=True)
        return (m + np.log(np.sum(np.exp(log_ps - m), axis=-1, keepdims=True))).squeeze(-1)

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        diff = x[..., None, :] - self.means
        quad = -0.5 * np.einsum("kij,...kj,...ki->...k", self.inv_covs, diff, diff)
        log_ps = self.log_norm_consts + quad + self.log_weights
        m = np.max(log_ps, axis=-1, keepdims=True)
        exps = np.exp(log_ps - m)
        responsibilities = exps / np.sum(exps, axis=-1, keepdims=True)
        grad_components = -np.einsum("kij,...kj->...ki", self.inv_covs, diff)
        return np.sum(responsibilities[..., None] * grad_components, axis=-2)

    def sample(self, rng: np.random.Generator, n_samples: int = 1) -> np.ndarray:
        component_ids = rng.choice(len(self.weights), size=n_samples, p=self.weights)
        samples = np.empty((n_samples, self.dim), dtype=float)
        for k in range(len(self.weights)):
            mask = component_ids == k
            if np.any(mask):
                samples[mask] = rng.multivariate_normal(
                    mean=self.means[k],
                    cov=self.covs[k],
                    size=int(np.sum(mask)),
                )
        if n_samples == 1:
            return samples[0]
        return samples


class SyntheticLogistic(BasePotential):
    """Synthetic Bayesian logistic regression model."""

    def __init__(self, n_samples: int = 100, dim: int = 10, seed: int | None = None) -> None:
        rng = np.random.default_rng(seed)
        self.dim = dim
        self.X = rng.normal(size=(n_samples, dim))
        true_beta = rng.normal(size=dim)
        logits = self.X @ true_beta
        probs = 1 / (1 + np.exp(-logits))
        self.y = rng.binomial(1, probs)
        self.prior_cov_inv = np.eye(dim)

    def log_prob(self, beta: np.ndarray) -> float:
        z = self.X @ beta
        log_lik = np.sum(self.y * z - np.logaddexp(0.0, z))
        log_prior = -0.5 * beta.T @ self.prior_cov_inv @ beta
        return float(log_lik + log_prior)

    def grad(self, beta: np.ndarray) -> np.ndarray:
        z = self.X @ beta
        p = expit(z)
        grad_lik = self.X.T @ (self.y - p)
        grad_prior = -self.prior_cov_inv @ beta
        return grad_lik + grad_prior

    def hvp(self, beta: np.ndarray, v: np.ndarray) -> np.ndarray:
        z = self.X @ beta
        p = expit(z)
        W = p * (1 - p)
        Xv = self.X @ v
        WXv = W * Xv
        hess_v = -self.X.T @ WXv - v
        return hess_v


class GaussianMixtureThreeMode(BasePotential):
    """Three-component Gaussian mixture distribution (Equal weights)."""

    def __init__(self, dim: int = 2, separation: float = 5.0) -> None:
        self.dim = dim
        self.means = np.zeros((3, dim))
        self.means[0, 0] = -separation
        self.means[1, 0] = 0.0
        self.means[2, 0] = separation
        self.log_norm_const = -0.5 * (dim * np.log(2 * np.pi))
        self.log_weight = -np.log(3.0)

    def log_prob(self, x: np.ndarray) -> float | np.ndarray:
        x = np.asarray(x)
        diff = x[..., None, :] - self.means # (..., 3, D)
        quad = -0.5 * np.einsum("...ki,...ki->...k", diff, diff)
        log_ps = self.log_norm_const + quad + self.log_weight
        
        m = np.max(log_ps, axis=-1, keepdims=True)
        return (m + np.log(np.sum(np.exp(log_ps - m), axis=-1, keepdims=True))).squeeze(-1)

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        diff = x[..., None, :] - self.means
        quad = -0.5 * np.einsum("...ki,...ki->...k", diff, diff)
        log_ps = self.log_norm_const + quad + self.log_weight

        m = np.max(log_ps, axis=-1, keepdims=True)
        exps = np.exp(log_ps - m)
        responsibilities = exps / np.sum(exps, axis=-1, keepdims=True)

        return np.sum(responsibilities[..., None] * (-diff), axis=-2)


class GaussianMixtureTriangle(BasePotential):
    """
    Three-component Gaussian mixture arranged in a Triangle.
    Supports full covariance matrices. Batch-ready.
    """

    def __init__(
        self, 
        dim: int = 2, 
        separation: float = 5.0, 
        cov_matrices: list[np.ndarray] | None = None
    ) -> None:
        assert dim >= 2
        self.dim = dim
        self.separation = separation
        
        self.means = np.zeros((3, dim))
        self.means[0, 0] = -separation
        self.means[1, 0] = separation
        self.means[2, 1] = separation 
        
        if cov_matrices is None:
            covs = [np.eye(dim) for _ in range(3)]
        else:
            covs = [np.array(c) for c in cov_matrices]
        
        self.inv_covs = np.array([np.linalg.inv(c) for c in covs]) # Shape (3, D, D)
        _, logdets = np.linalg.slogdet(covs)
        self.log_norm_consts = np.array(-0.5 * (dim * np.log(2 * np.pi) + logdets))
        self.log_weight = -np.log(3.0)

    def log_prob(self, x: np.ndarray) -> float | np.ndarray:
        x = np.asarray(x)
        # Support both (D,) and (N, D)
        # x: (..., D) -> (..., 1, D)
        # means: (3, D) -> implicitly (1, 3, D) for broadcasting
        # diff: (..., 3, D)
        diff = x[..., None, :] - self.means
        
        # quad: (..., 3)
        # inv_covs: (3, D, D) matches 'kij'
        # diff: (..., 3, D) matches '...kj' and '...ki'
        # output: (..., 3) matches '...k'
        quad = -0.5 * np.einsum("kij,...kj,...ki->...k", self.inv_covs, diff, diff)
        
        log_ps = self.log_norm_consts + quad + self.log_weight
        
        # sum over components (last axis)
        m = np.max(log_ps, axis=-1, keepdims=True)
        result = m + np.log(np.sum(np.exp(log_ps - m), axis=-1, keepdims=True))
        return result.squeeze(-1)

    def grad(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        diff = x[..., None, :] - self.means # (..., 3, D)

        quad = -0.5 * np.einsum("kij,...kj,...ki->...k", self.inv_covs, diff, diff)
        log_ps = self.log_norm_consts + quad + self.log_weight

        m = np.max(log_ps, axis=-1, keepdims=True)
        exps = np.exp(log_ps - m)
        responsibilities = exps / np.sum(exps, axis=-1, keepdims=True) # (..., 3)

        # Component gradients: -Sigma^-1 (x - mu)
        # Shape: (..., 3, D)
        grad_components = -np.einsum("kij,...kj->...ki", self.inv_covs, diff)
        
        # Weighted sum over components (axis -2)
        # responsibilities: (..., 3) -> (..., 3, 1)
        return np.sum(responsibilities[..., None] * grad_components, axis=-2)

    def hvp(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        # Fallback to Finite Difference
        eps = 1e-5
        grad_x = self.grad(x)
        grad_x_eps = self.grad(x + eps * v)
        return (grad_x_eps - grad_x) / eps
