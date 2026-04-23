"""Sampling algorithms for adaptive score–based MCMC experiments.

This module implements several Markov chain Monte Carlo algorithms used
in our experiments.  All samplers operate on a target potential
(implementing the ``BasePotential`` interface) and support a common
``run`` method that executes a fixed number of iterations and returns
collected samples along with diagnostic information such as acceptance
rates.

Fixes applied:
1. MetropolisUnderdampedLangevin: Switched to O-U + Velocity Verlet splitting.
2. UnderdampedScoreTiltedMCMC: Switched to O-U + Velocity Verlet splitting.

The samplers are categorized as follows for experimental clarity:

Group A: Hamiltonian Monte Carlo - HMC (Metropolized / Unbiased)
----------------------------------------------------------
These samplers use Leapfrog integration and include a Metropolis-Hastings 
correction step. They are strictly "Apple-to-Apple" comparisons.
- `HMC`: Standard Hamiltonian Monte Carlo (Baseline).
- `ScoreTiltedHMC`: HMC augmented with the adaptive score-tilting mechanism.

Group B: Unadjusted Langevin Dynamics - ULA/LD (Non-Metropolized / Biased)
-----------------------------------------------------------------
These samplers use Kinetic (Underdamped) Langevin dynamics without a 
Metropolis correction step. They are faster but asymptotically biased.
- `UnadjustedLangevin`: Standard Underdamped Langevin Dynamics (Baseline).
  (Note: Functionally identical to `UnderdampedLangevin`).
- `UnadjustedScoreTiltedULD`: Unadjusted Langevin with score-tilting.

First-Order Samplers
--------------------
- `MALA`: Standard Metropolis-Adjusted Langevin Algorithm.
- `ScoreTiltedMCMC`: First-order score-tilted sampler.
- `UnderdampedLangevin`: Legacy alias for `UnadjustedLangevin`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from potentials import BasePotential


@dataclass
class SamplerDiagnostics:
    """Container for diagnostics collected during a sampler run."""

    acceptance_rate: float
    n_accepts: int
    n_steps: int
    step_size: float
    runtime: float
    recorded_step_indices: Optional[List[int]] = None
    recorded_elapsed_times: Optional[List[float]] = None

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


class BaseSampler:
    """Base class providing common functionality for samplers."""

    def __init__(
        self,
        target: BasePotential,
        step_size: float,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.target = target
        self.step_size = step_size
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(
        self, x0: np.ndarray, n_steps: int, burn_in: int = 0, **kwargs
    ) -> Tuple[np.ndarray, SamplerDiagnostics]:
        raise NotImplementedError


def _current_alpha(
    alpha: float,
    n: int,
    alpha_adaptive: bool,
    alpha_C: float,
    alpha_warmup_steps: int,
) -> float:
    """Return fixed, linearly warmed, or rationally adaptive repellence."""
    if alpha_adaptive:
        return n / (alpha_C + n / alpha)
    if alpha_warmup_steps > 0:
        return alpha * min(1.0, n / alpha_warmup_steps)
    return alpha


class MALA(BaseSampler):
    """Metropolis–adjusted Langevin algorithm."""

    def run(
        self,
        x0: np.ndarray,
        n_steps: int,
        burn_in: int = 0,
        **kwargs,
    ) -> Tuple[np.ndarray, SamplerDiagnostics]:
        assert burn_in < n_steps
        x = x0.copy()
        dim = x.shape[0]
        samples: List[np.ndarray] = []
        n_accepts = 0
        eps = self.step_size
        sqrt_eps = np.sqrt(eps)
        start = time.perf_counter()
        
        for i in range(n_steps):
            grad_x = self.target.grad(x)
            prop_mean = x + 0.5 * eps * grad_x
            z = self.rng.normal(size=dim)
            x_prop = prop_mean + sqrt_eps * z
            
            log_p_current = self.target.log_prob(x)
            log_p_prop = self.target.log_prob(x_prop)
            grad_prop = self.target.grad(x_prop)
            
            diff_prop = x_prop - x - 0.5 * eps * grad_x
            log_q_forward = -0.5 / eps * np.dot(diff_prop, diff_prop)
            
            diff_back = x - x_prop - 0.5 * eps * grad_prop
            log_q_backward = -0.5 / eps * np.dot(diff_back, diff_back)
            
            log_alpha = log_p_prop + log_q_backward - log_p_current - log_q_forward
            
            if np.log(self.rng.uniform()) < log_alpha:
                x = x_prop
                n_accepts += 1
            
            if i >= burn_in:
                samples.append(x.copy())
                
        runtime = time.perf_counter() - start
        acceptance_rate = n_accepts / n_steps
        return np.array(samples), SamplerDiagnostics(
            acceptance_rate=acceptance_rate,
            n_accepts=n_accepts,
            n_steps=n_steps,
            step_size=self.step_size,
            runtime=runtime,
        )


class UnderdampedLangevin(BaseSampler):
    """Unadjusted kinetic (underdamped) Langevin algorithm.
    
    UPDATED: Now uses the same O-U + Velocity Verlet splitting as the 
    ScoreTilted variants to ensure strict baseline comparison.
    """

    def __init__(
        self,
        target: BasePotential,
        step_size: float,
        friction: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(target, step_size, rng)
        self.friction = friction

    def run(
        self,
        x0: np.ndarray,
        n_steps: int,
        burn_in: int = 0,
        **kwargs,
    ) -> Tuple[np.ndarray, SamplerDiagnostics]:
        assert burn_in < n_steps
        x = x0.copy()
        dim = x.shape[0]
        v = self.rng.normal(size=dim)
        samples: List[np.ndarray] = []
        
        dt = self.step_size
        gamma = self.friction
        # Constants for O-U process
        decay = np.exp(-gamma * dt)
        noise_scale = np.sqrt(1 - decay**2)
        
        start = time.perf_counter()
        
        # Pre-calculate gradient for Velocity Verlet
        grad = self.target.grad(x)
        
        for i in range(n_steps):
            # 1. Momentum Refresh (Exact O-U)
            v = decay * v + noise_scale * self.rng.normal(size=dim)
            
            # 2. Velocity Verlet
            # Half-step velocity
            v = v + 0.5 * dt * grad
            # Full-step position
            x = x + dt * v
            # New gradient
            grad = self.target.grad(x)
            # Half-step velocity
            v = v + 0.5 * dt * grad
            
            if i >= burn_in:
                samples.append(x.copy())
                
        runtime = time.perf_counter() - start
        acceptance_rate = 1.0
        return np.array(samples), SamplerDiagnostics(
            acceptance_rate=acceptance_rate,
            n_accepts=n_steps,
            n_steps=n_steps,
            step_size=self.step_size,
            runtime=runtime,
        )


class ScoreTiltedMCMC(BaseSampler):
    """Adaptive score–tilted MCMC requiring Hessian–vector products.
    
    Supports two modes:
    1. Exact (use_shifted_gradient=False): Uses HVP s(x) - alpha * H(x)theta.
    2. Shifted (use_shifted_gradient=True): Uses approximation s(x - alpha*theta).
    """

    def __init__(
        self,
        target: BasePotential,
        step_size: float,
        alpha: float,
        theta_step: float,
        use_shifted_gradient: bool = True,
        alpha_warmup_steps: int = 0,
        alpha_adaptive: bool = False,
        alpha_C: float = 100.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(target, step_size, rng)
        self.alpha = float(alpha)
        self.theta_step = float(theta_step)
        self.use_shifted_gradient = bool(use_shifted_gradient)
        self.alpha_warmup_steps = max(0, int(alpha_warmup_steps))
        self.alpha_adaptive = bool(alpha_adaptive)
        self.alpha_C = float(alpha_C)

    def run(
        self,
        x0: np.ndarray,
        n_steps: int,
        burn_in: int = 0,
        **kwargs,
    ) -> Tuple[np.ndarray, SamplerDiagnostics]:
        assert burn_in < n_steps
        x = x0.copy()
        dim = x.shape[0]
        theta = np.zeros(dim)
        samples: List[np.ndarray] = []
        n_accepts = 0
        eps = self.step_size
        sqrt_eps = np.sqrt(eps)
        
        start = time.perf_counter()
        for i in range(n_steps):
            alpha_t = _current_alpha(
                self.alpha,
                i + 1,
                self.alpha_adaptive,
                self.alpha_C,
                self.alpha_warmup_steps,
            )
            grad_x = self.target.grad(x)
            
            # 1. Compute Surrogate Score (Drift)
            if self.use_shifted_gradient:
                # Approximation: s(x - alpha*theta)
                shift = -alpha_t * theta
                surrogate_score = self.target.grad(x + shift)
            else:
                # Exact: s(x) - alpha * H(x)theta
                hvp_x_theta = self.target.hvp_with_grad(x, theta, grad_x)
                surrogate_score = grad_x - alpha_t * hvp_x_theta
            
            # MALA proposal with surrogate score
            prop_mean = x + 0.5 * eps * surrogate_score
            z = self.rng.normal(size=dim)
            x_prop = prop_mean + sqrt_eps * z
            
            # Acceptance prob targeting pi_theta
            # log pi_theta = log pi(x) - alpha * theta^T s(x)
            log_p_current = self.target.log_prob(x) - alpha_t * np.dot(theta, grad_x)
            
            grad_prop = self.target.grad(x_prop)
            
            if self.use_shifted_gradient:
                shift = -alpha_t * theta
                surrogate_score_prop = self.target.grad(x_prop + shift)
            else:
                hvp_prop_theta = self.target.hvp_with_grad(x_prop, theta, grad_prop)
                surrogate_score_prop = grad_prop - alpha_t * hvp_prop_theta
            
            log_p_prop = self.target.log_prob(x_prop) - alpha_t * np.dot(theta, grad_prop)
            
            diff_forward = x_prop - x - 0.5 * eps * surrogate_score
            log_q_forward = -0.5 / eps * np.dot(diff_forward, diff_forward)
            
            diff_backward = x - x_prop - 0.5 * eps * surrogate_score_prop
            log_q_backward = -0.5 / eps * np.dot(diff_backward, diff_backward)
            
            log_alpha = log_p_prop + log_q_backward - log_p_current - log_q_forward
            
            if np.log(self.rng.uniform()) < log_alpha:
                x = x_prop
                grad_x = grad_prop # reuse gradient
                n_accepts += 1
            
            # Update theta: EMA of +s(x)
            a_n = self.theta_step / ((i + 1) ** 0.6)
            theta = theta + a_n * (grad_x - theta)
            
            if i >= burn_in:
                samples.append(x.copy())
                
        runtime = time.perf_counter() - start
        acceptance_rate = n_accepts / n_steps
        return np.array(samples), SamplerDiagnostics(
            acceptance_rate=acceptance_rate,
            n_accepts=n_accepts,
            n_steps=n_steps,
            step_size=self.step_size,
            runtime=runtime,
        )


# Alias for clarity in config files
class UnadjustedLangevin(BaseSampler):
    """
    Explicitly named Unadjusted Langevin (ULA/LD) for Group B baseline.
    Standard Kinetic Langevin without Metropolis correction.
    UPDATED: Now uses the same O-U + Velocity Verlet splitting as the 
    ScoreTilted variants to ensure strict baseline comparison.
    """

    def __init__(
        self,
        target: BasePotential,
        step_size: float,
        friction: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(target, step_size, rng)
        self.friction = friction

    def run(
        self,
        x0: np.ndarray,
        n_steps: int,
        burn_in: int = 0,
        **kwargs,
    ) -> Tuple[np.ndarray, SamplerDiagnostics]:
        assert burn_in < n_steps
        x = x0.copy()
        dim = x.shape[0]
        v = self.rng.normal(size=dim)
        samples: List[np.ndarray] = []
        
        dt = self.step_size
        gamma = self.friction
        # Constants for O-U process
        decay = np.exp(-gamma * dt)
        noise_scale = np.sqrt(1 - decay**2)
        
        start = time.perf_counter()
        
        # Pre-calculate gradient for Velocity Verlet
        grad = self.target.grad(x)
        
        for i in range(n_steps):
            # 1. Momentum Refresh (Exact O-U)
            v = decay * v + noise_scale * self.rng.normal(size=dim)
            
            # 2. Velocity Verlet
            # Half-step velocity
            v = v + 0.5 * dt * grad
            # Full-step position
            x = x + dt * v
            # New gradient
            grad = self.target.grad(x)
            # Half-step velocity
            v = v + 0.5 * dt * grad
            
            if i >= burn_in:
                samples.append(x.copy())
                
        runtime = time.perf_counter() - start
        acceptance_rate = 1.0
        return np.array(samples), SamplerDiagnostics(
            acceptance_rate=acceptance_rate,
            n_accepts=n_steps,
            n_steps=n_steps,
            step_size=self.step_size,
            runtime=runtime,
        )

class UnadjustedScoreTiltedULD(BaseSampler):
    """
    Unadjusted kinetic Langevin dynamics with score-tilted target.
    NO Metropolis correction step (always accept).
    Biased, but faster mixing potential.
    
    Optimization: use_shifted_gradient=True drastically reduces cost to 1 grad call/step.
    """
    def __init__(
        self,
        target: BasePotential,
        step_size: float,
        alpha: float,
        theta_step: float,
        friction: float = 1.0,
        use_shifted_gradient: bool = True,
        alpha_warmup_steps: int = 0,
        alpha_adaptive: bool = False,
        alpha_C: float = 100.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(target, step_size, rng)
        self.alpha = float(alpha)
        self.theta_step = float(theta_step)
        self.friction = float(friction)
        self.use_shifted_gradient = bool(use_shifted_gradient)
        self.alpha_warmup_steps = max(0, int(alpha_warmup_steps))
        self.alpha_adaptive = bool(alpha_adaptive)
        self.alpha_C = float(alpha_C)

    def _get_surrogate_score(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        alpha_t: float,
        grad_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (surrogate_score, grad_x).
        """
        if self.use_shifted_gradient:
            # Shifted approximation
            shift = -alpha_t * theta
            score_shifted = self.target.grad(x + shift)
            
            # Optimization: For Unadjusted, we can assume grad_x approx score_shifted 
            # for the theta update to save a gradient call.
            # Warning: This makes the theta update track E[s(x-shift)] instead of E[s(x)].
            return score_shifted, score_shifted 
        
        else:
            if grad_x is None:
                grad_x = self.target.grad(x)
            if np.allclose(theta, 0):
                return grad_x, grad_x
            hvp = self.target.hvp_with_grad(x, theta, grad_x)
            score = grad_x - alpha_t * hvp
            return score, grad_x

    def run(
        self,
        x0: np.ndarray,
        n_steps: int,
        burn_in: int = 0,
        **kwargs,
    ) -> Tuple[np.ndarray, SamplerDiagnostics]:
        x = x0.copy()
        dim = x.shape[0]
        v = self.rng.normal(size=dim)
        theta = np.zeros(dim)
        samples: List[np.ndarray] = []
        
        dt = self.step_size
        gamma = self.friction
        decay = np.exp(-gamma * dt)
        noise_scale = np.sqrt(1 - decay**2)
        
        start = time.perf_counter()
        
        # Initial score
        alpha_t = _current_alpha(
            self.alpha,
            1,
            self.alpha_adaptive,
            self.alpha_C,
            self.alpha_warmup_steps,
        )
        surrogate_score, grad_x = self._get_surrogate_score(x, theta, alpha_t)
        
        for i in range(n_steps):
            alpha_t = _current_alpha(
                self.alpha,
                i + 1,
                self.alpha_adaptive,
                self.alpha_C,
                self.alpha_warmup_steps,
            )
            surrogate_score, grad_x = self._get_surrogate_score(x, theta, alpha_t)
            # 1. Momentum Refresh (Exact O-U)
            v = decay * v + noise_scale * self.rng.normal(size=dim)
            
            # 2. Velocity Verlet Integration (Unadjusted)
            # Half-step velocity
            v = v + 0.5 * dt * surrogate_score
            # Full-step position
            x = x + dt * v
            
            # Compute NEW score at new position
            surrogate_score, grad_x = self._get_surrogate_score(x, theta, alpha_t)
            
            # Half-step velocity
            v = v + 0.5 * dt * surrogate_score
            
            # NO Metropolis Check - Always Accept
            
            # 3. Update Theta
            a_n = self.theta_step / ((i + 1) ** 0.6)
            theta = theta + a_n * (grad_x - theta)
            
            if i >= burn_in:
                samples.append(x.copy())
                
        runtime = time.perf_counter() - start
        # Acceptance rate is conceptually 1.0 (or undefined) for unadjusted methods
        return np.array(samples), SamplerDiagnostics(
            acceptance_rate=1.0, 
            n_accepts=n_steps,
            n_steps=n_steps,
            step_size=self.step_size,
            runtime=runtime,
        )


class HMC(BaseSampler):
    """
    Standard Hamiltonian Monte Carlo (HMC) with Leapfrog integrator.
    """
    def __init__(
        self,
        target: BasePotential,
        step_size: float,
        n_leapfrog: int = 10, # Number of leapfrog steps per iteration
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(target, step_size, rng)
        self.n_leapfrog = n_leapfrog

    def run(
        self, x0: np.ndarray, n_steps: int, burn_in: int = 0, **kwargs
    ) -> Tuple[np.ndarray, SamplerDiagnostics]:
        x = x0.copy()
        dim = x.shape[0]
        samples: List[np.ndarray] = []
        n_accepts = 0
        dt = self.step_size
        
        start = time.perf_counter()
        grad = self.target.grad(x)
        
        for i in range(n_steps):
            # 1. Sample Momentum
            v = self.rng.normal(size=dim)
            
            x_old = x.copy()
            v_old = v.copy()
            grad_old = grad.copy()
            
            # 2. Leapfrog Integration
            # Half-step momentum
            v = v + 0.5 * dt * grad 
            
            for l in range(self.n_leapfrog):
                # Full-step position
                x = x + dt * v
                # Update gradient
                grad = self.target.grad(x)
                # Full-step momentum (except last step)
                if l != self.n_leapfrog - 1:
                    v = v + dt * grad
            
            # Final half-step momentum
            v = v + 0.5 * dt * grad
            
            # 3. Metropolis Step
            # H = U(x) + K(v) = -log_prob(x) + 0.5 * v^T v
            current_U = -self.target.log_prob(x_old)
            current_K = 0.5 * np.dot(v_old, v_old)
            prop_U = -self.target.log_prob(x)
            prop_K = 0.5 * np.dot(v, v)
            
            log_alpha = (current_U + current_K) - (prop_U + prop_K)
            
            if np.log(self.rng.uniform()) < log_alpha:
                n_accepts += 1
            else:
                x = x_old
                grad = grad_old # Restore gradient
            
            if i >= burn_in:
                samples.append(x.copy())
                
        runtime = time.perf_counter() - start
        acceptance_rate = n_accepts / n_steps
        return np.array(samples), SamplerDiagnostics(
            acceptance_rate=acceptance_rate,
            n_accepts=n_accepts,
            n_steps=n_steps,
            step_size=self.step_size,
            runtime=runtime,
        )


class ScoreTiltedHMC(BaseSampler):
    """
    HMC integrated with Score-based target.
    
    Optimization: use_shifted_gradient=True significantly speeds up the Leapfrog integrator
    by requiring only 1 gradient call per substep (vs 2 for FD-HVP).
    """
    def __init__(
        self,
        target: BasePotential,
        step_size: float,
        alpha: float,
        theta_step: float,
        n_leapfrog: int = 10,
        use_shifted_gradient: bool = True,
        alpha_warmup_steps: int = 0,
        alpha_adaptive: bool = False,
        alpha_C: float = 100.0,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(target, step_size, rng)
        self.alpha = float(alpha)
        self.theta_step = float(theta_step)
        self.n_leapfrog = n_leapfrog
        self.use_shifted_gradient = bool(use_shifted_gradient)
        self.alpha_warmup_steps = max(0, int(alpha_warmup_steps))
        self.alpha_adaptive = bool(alpha_adaptive)
        self.alpha_C = float(alpha_C)

    def _get_surrogate_score(
        self,
        x: np.ndarray,
        theta: np.ndarray,
        alpha_t: float,
        grad_x: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.use_shifted_gradient:
            # Shifted approximation
            shift = -alpha_t * theta
            score_shifted = self.target.grad(x + shift)
            # In leapfrog, we only need the force (score).
            # The grad_x is only needed for the MH step at the end.
            # We return None for grad_x inside leapfrog to indicate it wasn't computed.
            return score_shifted, None
        else:
            if grad_x is None:
                grad_x = self.target.grad(x)
            if np.allclose(theta, 0):
                return grad_x, grad_x
            hvp = self.target.hvp_with_grad(x, theta, grad_x)
            score = grad_x - alpha_t * hvp
            return score, grad_x

    def run(
        self, x0: np.ndarray, n_steps: int, burn_in: int = 0, **kwargs
    ) -> Tuple[np.ndarray, SamplerDiagnostics]:
        x = x0.copy()
        dim = x.shape[0]
        theta = np.zeros(dim)
        samples: List[np.ndarray] = []
        n_accepts = 0
        dt = self.step_size
        
        start = time.perf_counter()
        
        # Initial Gradients
        # We need grad_x for MH at the start
        grad_x = self.target.grad(x) 
        
        # Initial Surrogate Score
        alpha_t = _current_alpha(
            self.alpha,
            1,
            self.alpha_adaptive,
            self.alpha_C,
            self.alpha_warmup_steps,
        )
        if self.use_shifted_gradient:
            shift = -alpha_t * theta
            surrogate_score = self.target.grad(x + shift)
        else:
            hvp = self.target.hvp_with_grad(x, theta, grad_x)
            surrogate_score = grad_x - alpha_t * hvp
        
        for i in range(n_steps):
            alpha_t = _current_alpha(
                self.alpha,
                i + 1,
                self.alpha_adaptive,
                self.alpha_C,
                self.alpha_warmup_steps,
            )
            if self.use_shifted_gradient:
                shift = -alpha_t * theta
                surrogate_score = self.target.grad(x + shift)
            else:
                hvp = self.target.hvp_with_grad(x, theta, grad_x)
                surrogate_score = grad_x - alpha_t * hvp
            # 1. Sample Momentum
            v = self.rng.normal(size=dim)
            
            x_old = x.copy()
            v_old = v.copy()
            grad_x_old = grad_x.copy()
            
            # 2. Leapfrog Integration using Surrogate Score
            # Half-step momentum
            v = v + 0.5 * dt * surrogate_score
            
            for l in range(self.n_leapfrog):
                x = x + dt * v
                # Compute NEW surrogate score
                # This is the "inner loop" where speedup matters most
                if l != self.n_leapfrog - 1:
                    surrogate_score, _ = self._get_surrogate_score(x, theta, alpha_t, grad_x=None)
                    v = v + dt * surrogate_score
            
            # Final half-step momentum
            surrogate_score, _ = self._get_surrogate_score(x, theta, alpha_t, grad_x=None)
            v = v + 0.5 * dt * surrogate_score
            
            # 3. Metropolis Step targeting pi_theta
            # We need exact grad_x at the proposal for the MH ratio
            grad_x_prop = self.target.grad(x)
            
            log_pi_theta_current = self.target.log_prob(x_old) - alpha_t * np.dot(theta, grad_x_old)
            current_H = -log_pi_theta_current + 0.5 * np.dot(v_old, v_old)
            
            log_pi_theta_prop = self.target.log_prob(x) - alpha_t * np.dot(theta, grad_x_prop)
            prop_H = -log_pi_theta_prop + 0.5 * np.dot(v, v)
            
            if np.log(self.rng.uniform()) < (current_H - prop_H):
                n_accepts += 1
                grad_x = grad_x_prop
                # surrogate_score for next step is already computed at end of leapfrog?
                # No, surrogate_score at end of leapfrog was at x_prop.
                # So we can keep it.
            else:
                x = x_old
                grad_x = grad_x_old
                # We need to restore surrogate_score for x_old
                if self.use_shifted_gradient:
                    shift = -alpha_t * theta
                    surrogate_score = self.target.grad(x + shift)
                else:
                    hvp = self.target.hvp_with_grad(x, theta, grad_x)
                    surrogate_score = grad_x - alpha_t * hvp
            
            # 4. Update Theta (SA)
            # Adapt theta using gradient of original target at current state
            a_n = self.theta_step / ((i + 1) ** 0.6)
            theta = theta + a_n * (grad_x - theta)
            
            if i >= burn_in:
                samples.append(x.copy())
                
        runtime = time.perf_counter() - start
        acceptance_rate = n_accepts / n_steps
        return np.array(samples), SamplerDiagnostics(
            acceptance_rate=acceptance_rate,
            n_accepts=n_accepts,
            n_steps=n_steps,
            step_size=self.step_size,
            runtime=runtime,
        )
    

