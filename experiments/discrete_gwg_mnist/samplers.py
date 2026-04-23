import torch
import torch.nn as nn
import torch.distributions as dists
import utils
import numpy as np

# ScoreTiltedMCMC for Binary Data
class ScoreTiltedMCMC(nn.Module):
    """
    ScoreTiltedMCMC for Binary Data.  
    Implements the Adaptive Score-Tilted MCMC algorithm adapted for discrete spaces.
    
    Modes:
    1. Exact (use_shifted_gradient=False): Uses Autograd to compute HVP (Hessian-Vector Product).
       Drift ~ s(x) - alpha * H(x) * theta
    2. Shifted (use_shifted_gradient=True): Uses Gradient at shifted continuous point.
       Drift ~ s(x - alpha * theta)
    """
    def __init__(self, dim, n_steps=10, step_size=1.0, alpha=1.0, theta_step=0.1, use_shifted_gradient=True, temp=2.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self.step_size = step_size
        self.alpha = alpha
        self.theta_step = theta_step # Learning rate for theta (gamma_0)
        self.use_shifted_gradient = use_shifted_gradient
        self.temp = temp
        
        self._ar = 0.
        self.register_buffer('theta', torch.zeros(1, dim)) 
        self.register_buffer('n_steps_count', torch.tensor(0, dtype=torch.long))

    def _manual_bernoulli_log_prob(self, x, probs):
        eps = 1e-7
        probs = probs.clamp(eps, 1. - eps)
        return (x * torch.log(probs) + (1. - x) * torch.log(1. - probs)).sum(-1)

    def _manual_ebm_forward(self, model, x):
        if hasattr(model, 'mean') and model.mean is not None:
            bd = self._manual_bernoulli_log_prob(x, model.mean)
        else:
            bd = 0.
        logp = model.net(x).squeeze()
        return logp + bd

    def _unwrap_and_compute_score(self, model_func, x):
        # Case 1: Raw EBM
        if hasattr(model_func, 'net'):
            return self._manual_ebm_forward(model_func, x)
        
        # Case 2: AIS Lambda
        if hasattr(model_func, '__closure__') and model_func.__closure__ is not None:
            ais_model = None
            beta = None
            for cell in model_func.__closure__:
                obj = cell.cell_contents
                if hasattr(obj, 'model') and hasattr(obj, 'init_dist'):
                    ais_model = obj
                elif isinstance(obj, (float, np.floating, int)) or (torch.is_tensor(obj) and obj.numel() == 1):
                    beta = obj

            if ais_model is not None and beta is not None:
                logpx = self._manual_ebm_forward(ais_model.model, x)
                if hasattr(ais_model.init_dist, 'probs'):
                    logpi = self._manual_bernoulli_log_prob(x, ais_model.init_dist.probs)
                elif hasattr(ais_model.init_dist, 'logits'):
                    probs = torch.sigmoid(ais_model.init_dist.logits)
                    logpi = self._manual_bernoulli_log_prob(x, probs)
                else:
                    logpi = ais_model.init_dist.log_prob(x).sum(-1)
                return logpx * beta + logpi * (1. - beta)

        # Case 3: Fallback
        return model_func(x).sum()

    def _get_continuous_gradients(self, x, theta, model, grad_x=None):
        """
        Computes the continuous 'Drift' vector and the Score.
        Returns:
            surrogate_drift: Vector D s.t. GWG_logits ~ -(2x-1) * D
            grad_x: Standard score s(x)
        """
        if not x.requires_grad:
            x_in = x.detach().float().requires_grad_(True)
        else:
            x_in = x

        # 1. Compute Standard Score s(x) 
        if grad_x is None or not grad_x.requires_grad:
            # Need graph for HVP, or just value for Shifted
            create_graph = (not self.use_shifted_gradient) 
            out = self._unwrap_and_compute_score(model, x_in)
            grad_x = torch.autograd.grad(out.sum(), x_in, create_graph=create_graph)[0]
        
        # 2. Compute Surrogate Drift
        if self.use_shifted_gradient:
            # Mode A: Shifted Gradient (Approximation)
            # Drift = s(x - alpha * theta)
            shift = -self.alpha * theta
            x_shifted = x_in + shift
            
            # Manual forward to handle continuous x_shifted
            out_shifted = self._unwrap_and_compute_score(model, x_shifted)
            surrogate_drift = torch.autograd.grad(out_shifted.sum(), x_shifted, create_graph=False)[0]
            
        else:
            # Mode 1: Exact HVP
            # Drift = s(x) - alpha * H(x) * theta
            # H * theta = grad( (s(x) * theta).sum() )
            
            hvp_term = (grad_x * theta).sum()
            hvp = torch.autograd.grad(hvp_term, x_in, retain_graph=False)[0]
            surrogate_drift = grad_x - self.alpha * hvp
            
        return surrogate_drift, grad_x, out.detach() if 'out' in locals() else None


    def step(self, x, model):
        if self.theta.size(0) != x.size(0):
             self.theta = torch.zeros_like(x).to(x.device)
        
        x_cur = x.float()
        a_s = []

        surrogate_drift, grad_x, log_p_cur = self._get_continuous_gradients(x_cur, self.theta, model)
        
        if log_p_cur is None:
             log_p_cur = self._unwrap_and_compute_score(model, x_cur).detach()

        for _ in range(self.n_steps):
            x_cur = x_cur.detach()
            grad_x = grad_x.detach()
            
            # 1. GWG Approximation
            # Logits ≈ -(2x - 1) * Drift * (2 / temp)
            # GWG paper Eq(3): difference ~ -(2x-1) * grad
            
            gwg_logits = -(2.0 * x_cur - 1.0) * surrogate_drift * (2.0 / self.temp)
            
            cd_forward = dists.Bernoulli(logits=gwg_logits)
            changes = cd_forward.sample()
            x_prop = (1. - x_cur) * changes + x_cur * (1. - changes)
            
            # 2. Metropolis-Hastings Correction 
            surrogate_drift_prop, grad_prop, log_p_prop_val = self._get_continuous_gradients(x_prop, self.theta, model)
            if log_p_prop_val is None:
                log_p_prop_val = self._unwrap_and_compute_score(model, x_prop).detach()
            
            # Reverse Logits
            gwg_logits_reverse = -(2.0 * x_prop - 1.0) * surrogate_drift_prop * (2.0 / self.temp)
            cd_reverse = dists.Bernoulli(logits=gwg_logits_reverse)
            
            # q(x'|x), q(x|x')
            lp_forward = cd_forward.log_prob(changes).sum(-1)
            lp_reverse = cd_reverse.log_prob(changes).sum(-1)
            
            # Target Probabilities (Score-Tilted)
            # log pi_theta(x) = log p(x) - alpha * theta^T s(x)
            log_target_cur = log_p_cur - self.alpha * (self.theta * grad_x).sum(-1)
            log_target_prop = log_p_prop_val - self.alpha * (self.theta * grad_prop).sum(-1)
            
            # MH Ratio
            log_alpha_ratio = log_target_prop + lp_reverse - log_target_cur - lp_forward
            
            rand = torch.rand_like(log_alpha_ratio).log()
            accept = (rand < log_alpha_ratio).float()
            
            # Update State
            x_cur = x_prop * accept[:, None] + x_cur * (1. - accept[:, None])
            grad_x = grad_prop * accept[:, None] + grad_x * (1. - accept[:, None])
            log_p_cur = log_p_prop_val * accept + log_p_cur * (1. - accept)
            
            # Update Drift (for next proposal)
            # Reuse the computed drift to avoid re-computation
            surrogate_drift = surrogate_drift_prop * accept[:, None] + surrogate_drift * (1. - accept[:, None])
            
            a_s.append(accept.mean().item())
            
            # 3. Update Theta (History)
            # theta_{n+1} = theta_n + a_n * (s(x_{n+1}) - theta_n)
            # Uses standard score s(x) (grad_x)
            
            self.n_steps_count += 1
            # Step size decay schedule: theta_step / (t^0.6)
            gamma_n = self.theta_step / (self.n_steps_count.float() ** 0.6)
            
            self.theta = self.theta + gamma_n * (grad_x - self.theta)
            
        self._ar = np.mean(a_s)
        return x_cur.detach()
        
# Gibbs-With-Gradients for binary data
class DiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, multi_hop=False, fixed_proposal=False, temp=2., step_size=1.0):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.fixed_proposal = fixed_proposal
        self.multi_hop = multi_hop
        self.temp = temp
        self.step_size = step_size
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        if self.multi_hop:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.Bernoulli(probs=delta.sigmoid() * self.step_size)
                for i in range(self.n_steps):
                    changes = cd.sample()
                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.Bernoulli(logits=(forward_delta * 2 / self.temp))
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes).sum(-1)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)


                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.Bernoulli(logits=(reverse_delta * 2 / self.temp))

                    lp_reverse = cd_reverse.log_prob(changes).sum(-1)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                    m_terms.append(m_term.mean().item())
                    prop_terms.append((lp_reverse - lp_forward).mean().item())
                self._ar = np.mean(a_s)
                self._mt = np.mean(m_terms)
                self._pt = np.mean(prop_terms)
        else:
            if self.fixed_proposal:
                delta = self.diff_fn(x, model)
                cd = dists.OneHotCategorical(logits=delta)
                for i in range(self.n_steps):
                    changes = cd.sample()

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
                    la = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
                    a_s.append(a.mean().item())
                self._ar = np.mean(a_s)
            else:
                for i in range(self.n_steps):
                    forward_delta = self.diff_fn(x_cur, model)
                    cd_forward = dists.OneHotCategorical(logits=forward_delta)
                    changes = cd_forward.sample()

                    lp_forward = cd_forward.log_prob(changes)

                    x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)

                    reverse_delta = self.diff_fn(x_delta, model)
                    cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

                    lp_reverse = cd_reverse.log_prob(changes)

                    m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
                    la = m_term + lp_reverse - lp_forward
                    a = (la.exp() > torch.rand_like(la)).float()
                    x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])

        return x_cur


# Gibbs-With-Gradients variant which proposes multiple flips per step
class MultiDiffSampler(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1., n_samples=1):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        self.n_samples = n_samples
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function(x, m) / self.temp


    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []

        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            cd_forward = dists.OneHotCategorical(logits=forward_delta)
            changes_all = cd_forward.sample((self.n_samples,))

            lp_forward = cd_forward.log_prob(changes_all).sum(0)

            changes = (changes_all.sum(0) > 0.).float()

            x_delta = (1. - x_cur) * changes + x_cur * (1. - changes)
            self._phops = (x_delta != x).float().sum(-1).mean().item()

            reverse_delta = self.diff_fn(x_delta, model)
            cd_reverse = dists.OneHotCategorical(logits=reverse_delta)

            lp_reverse = cd_reverse.log_prob(changes_all).sum(0)

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None] + x_cur * (1. - a[:, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).mean().item()
        return x_cur


class PerDimGibbsSampler(nn.Module):
    def __init__(self, dim, rand=False):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 1.
        self.rand = rand

    def step(self, x, model):
        sample = x.clone()
        lp_keep = model(sample).squeeze()
        if self.rand:
            changes = dists.OneHotCategorical(logits=torch.zeros((self.dim,))).sample((x.size(0),)).to(x.device)
        else:
            changes = torch.zeros((x.size(0), self.dim)).to(x.device)
            changes[:, self._i] = 1.

        sample_change = (1. - changes) * sample + changes * (1. - sample)

        lp_change = model(sample_change).squeeze()

        lp_update = lp_change - lp_keep
        update_dist = dists.Bernoulli(logits=lp_update)
        updates = update_dist.sample()
        sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
        self.changes[self._i] = updates.mean()
        self._i = (self._i + 1) % self.dim
        self._hops = (x != sample).float().sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


class PerDimMetropolisSampler(nn.Module):
    def __init__(self, dim, n_out, rand=False):
        super().__init__()
        self.dim = dim
        self.n_out = n_out
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))
        self._i = 0
        self._j = 0
        self._ar = 0.
        self._hops = 0.
        self._phops = 0.
        self.rand = rand

    def step(self, x, model):
        if self.rand:
            i = np.random.randint(0, self.dim)
        else:
            i = self._i

        logits = []
        ndim = x.size(-1)

        for k in range(ndim):
            sample = x.clone()
            sample_i = torch.zeros((ndim,))
            sample_i[k] = 1.
            sample[:, i, :] = sample_i
            lp_k = model(sample).squeeze()
            logits.append(lp_k[:, None])
        logits = torch.cat(logits, 1)
        dist = dists.OneHotCategorical(logits=logits)
        updates = dist.sample()
        sample = x.clone()
        sample[:, i, :] = updates
        self._i = (self._i + 1) % self.dim
        self._hops = ((x != sample).float().sum(-1) / 2.).sum(-1).mean().item()
        self._ar = self._hops
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.


# Gibbs-With-Gradients for categorical data
class DiffSamplerMultiDim(nn.Module):
    def __init__(self, dim, n_steps=10, approx=False, temp=1.):
        super().__init__()
        self.dim = dim
        self.n_steps = n_steps
        self._ar = 0.
        self._mt = 0.
        self._pt = 0.
        self._hops = 0.
        self._phops = 0.
        self.approx = approx
        self.temp = temp
        if approx:
            self.diff_fn = lambda x, m: utils.approx_difference_function_multi_dim(x, m) / self.temp
        else:
            self.diff_fn = lambda x, m: utils.difference_function_multi_dim(x, m) / self.temp

    def step(self, x, model):

        x_cur = x
        a_s = []
        m_terms = []
        prop_terms = []


        for i in range(self.n_steps):
            forward_delta = self.diff_fn(x_cur, model)
            # make sure we dont choose to stay where we are!
            forward_logits = forward_delta - 1e9 * x_cur
            #print(forward_logits)
            cd_forward = dists.OneHotCategorical(logits=forward_logits.view(x_cur.size(0), -1))
            changes = cd_forward.sample()

            # compute probability of sampling this change
            lp_forward = cd_forward.log_prob(changes)
            # reshape to (bs, dim, nout)
            changes_r = changes.view(x_cur.size())
            # get binary indicator (bs, dim) indicating which dim was changed
            changed_ind = changes_r.sum(-1)
            # mask out cuanged dim and add in the change
            x_delta = x_cur.clone() * (1. - changed_ind[:, :, None]) + changes_r

            reverse_delta = self.diff_fn(x_delta, model)
            reverse_logits = reverse_delta - 1e9 * x_delta
            cd_reverse = dists.OneHotCategorical(logits=reverse_logits.view(x_delta.size(0), -1))
            reverse_changes = x_cur * changed_ind[:, :, None]

            lp_reverse = cd_reverse.log_prob(reverse_changes.view(x_delta.size(0), -1))

            m_term = (model(x_delta).squeeze() - model(x_cur).squeeze())
            la = m_term + lp_reverse - lp_forward
            a = (la.exp() > torch.rand_like(la)).float()
            x_cur = x_delta * a[:, None, None] + x_cur * (1. - a[:, None, None])
            a_s.append(a.mean().item())
            m_terms.append(m_term.mean().item())
            prop_terms.append((lp_reverse - lp_forward).mean().item())
        self._ar = np.mean(a_s)
        self._mt = np.mean(m_terms)
        self._pt = np.mean(prop_terms)

        self._hops = (x != x_cur).float().sum(-1).sum(-1).mean().item()
        return x_cur


class GibbsSampler(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.changes = torch.zeros((dim,))
        self.change_rate = 0.
        self.p = nn.Parameter(torch.zeros((dim,)))

    def step(self, x, model):
        sample = x.clone()
        for i in range(self.dim):
            lp_keep = model(sample).squeeze()

            xi_keep = sample[:, i]
            xi_change = 1. - xi_keep
            sample_change = sample.clone()
            sample_change[:, i] = xi_change

            lp_change = model(sample_change).squeeze()

            lp_update = lp_change - lp_keep
            update_dist = dists.Bernoulli(logits=lp_update)
            updates = update_dist.sample()
            sample = sample_change * updates[:, None] + sample * (1. - updates[:, None])
            self.changes[i] = updates.mean()
        return sample

    def logp_accept(self, xhat, x, model):
        # only true if xhat was generated from self.step(x, model)
        return 0.
