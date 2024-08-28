"""Markov chain Monte Carlo methods"""

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pygem_input as pygem_prms

# z-normalization functions
def z_normalize(params, means, std_devs):
    return (params - means) / std_devs

# inverse z-normalization
def inverse_z_normalize(z_params, means,  std_devs):
    return z_params * std_devs + means

def log_normal_density(x, **kwargs):
    """
    Computes the log probability density of a normal distribution.

    Parameters:
    - x: Input tensor where you want to evaluate the log probability.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.

    Returns:
        Log probability density at the given input tensor x.
    """
    for key, value in kwargs.items():
        kwargs[key] = torch.tensor([value])
    mu, sigma = kwargs['mu'], kwargs['sigma']

    return (
        -0.5*np.log(2*np.pi) -                      # constant term
        torch.log(sigma) -                          # logarithm of the determinant of the covariance matrix
        0.5*(((x-mu)/sigma)**2)                     # exponential term
    )

def log_gamma_density(x, **kwargs):
    """
    Computes the log probability density of a Gamma distribution.

    Parameters:
    - x: Input tensor where you want to evaluate the log probability.
    - alpha: Shape parameter of the Gamma distribution.
    - beta: Rate parameter (1/scale) of the Gamma distribution.

    Returns:
        Log probability density at the given input tensor x.
    """
    for key, value in kwargs.items():
        kwargs[key] = torch.tensor([value])
    alpha, beta = kwargs['alpha'], kwargs['beta']   # shape, scale
    return alpha * torch.log(beta) + (alpha - 1) * torch.log(x) - beta * x - torch.lgamma(alpha)

def log_truncated_normal(x, **kwargs):
    """
    Computes the log probability density of a truncated normal distribution.

    Parameters:
    - x: Input tensor where you want to evaluate the log probability.
    - mu: Mean of the normal distribution.
    - sigma: Standard deviation of the normal distribution.
    - a: Lower truncation bound.
    - b: Upper truncation bound.

    Returns:
        Log probability density at the given input tensor x.
    """
    for key, value in kwargs.items():
        kwargs[key] = torch.tensor([value])
    mu, sigma, a, b = kwargs['mu'], kwargs['sigma'], kwargs['a'], kwargs['b']
    # Standardize
    standard_x = (x - mu) / sigma
    standard_a = (a - mu) / sigma
    standard_b = (b - mu) / sigma
    
    # PDF of the standard normal distribution
    pdf = torch.exp(-0.5 * standard_x**2) / np.sqrt(2 * torch.pi)
    
    # CDF of the standard normal distribution using the error function
    cdf_upper = 0.5 * (1 + torch.erf(standard_b / np.sqrt(2)))
    cdf_lower = 0.5 * (1 + torch.erf(standard_a / np.sqrt(2)))
    
    normalization = cdf_upper - cdf_lower
    
    return torch.log(pdf) - torch.log(normalization)

# mapper dictionary - maps to appropriate log probability density function for given distribution `type`
function_map = {
    'normal': log_normal_density,
    'gamma': log_gamma_density,
    'truncated_normal': log_truncated_normal
}

# mass balance posterior class
class mbPosterior:
    def __init__(self, mb_obs, sigma_obs, priors, mb_func, mb_args=None, potential_fxns=None, **kwargs):
        self.mb_obs = mb_obs
        self.sigma_obs = sigma_obs
        self.prior_params = priors
        self.mb_func = mb_func
        self.mb_args = mb_args
        self.potential_functions = potential_fxns if potential_fxns is not None else []
        self.mb_pred = None

        # get mean and std for each parameter type
        self.means = torch.tensor([params['mu'] if 'mu' in params else 0 for params in priors.values()])
        self.stds = torch.tensor([params['sigma'] if 'sigma' in params else 1 for params in priors.values()])

    # update modelprms for evaluation
    def update_modelprms(self, m):
        for i, k in enumerate(['tbias','kp','ddfsnow']):
            self.mb_args[1][k] = float(m[i])
        self.mb_args[1]['ddfice'] = self.mb_args[1]['ddfsnow'] / pygem_prms.ddfsnow_iceratio 

    # get mb_pred
    def get_mb_pred(self, m):
        if self.mb_args:
            self.update_modelprms(m)

            self.mb_pred = self.mb_func(*self.mb_args)
        else:
            self.mb_pred = self.mb_func([*m])

    # get total log prior density
    def log_prior(self, m):
        log_prior = []
        for i, (key, params) in enumerate(self.prior_params.items()):
            params_copy = params.copy()
            prior_type = params_copy.pop('type')
            function_to_call = function_map[prior_type]
            log_prior.append(function_to_call(m[i], **params_copy))
        log_prior = torch.stack(log_prior).sum()
        return log_prior

    # get log likelihood
    def log_likelihood(self):
        return log_normal_density(self.mb_obs, **{'mu': self.mb_pred, 'sigma': self.sigma_obs})
    
    # get log potential (sum up as any declared potential functions)
    def log_potential(self, m):
        log_potential = 0
        for potential_function in self.potential_functions:
            log_potential += potential_function(*m, **{'massbal':self.mb_pred})
        return log_potential

    # get log posterior (sum of log prior, log likelihood and log potential)
    def log_posterior(self, m):
        # anytime log_posterior is called for a new step, calculate the predicted mass balance
        self.get_mb_pred(m)
        return self.log_prior(m) + self.log_likelihood() + self.log_potential(m), self.mb_pred

# Metropolis-Hastings Markoc chain Monte Carlo class
class Metropolis:
    def __init__(self, means, stds):
        # Initialize chains
        self.steps = []
        self.P_chain = []
        self.m_chain = []
        self.m_primes = []
        self.mb_chain = []
        self.mb_primes = []
        self.means = means
        self.stds = stds

    def sample(self, m_0, log_posterior, h=0.1, n_samples=1000, burnin=0, thin_factor=1, progress_bar=False):
        # Compute initial unscaled log-posterior
        P_0, mb_0 = log_posterior(inverse_z_normalize(m_0, self.means, self.stds))

        n = len(m_0)

        # Draw samples
        iterable = range(n_samples)
        if progress_bar:
            iterable = tqdm(iterable)

        for i in iterable:
            # Propose new value according to
            # proposal distribution Q(m) = N(m_0,h)
            step = torch.randn(n)*h
            m_prime = m_0 + step

            # record step
            self.steps.append(step)

            # Compute new unscaled log-posterior
            P_1, mb_prime = log_posterior(inverse_z_normalize(m_prime, self.means, self.stds))

            # Compute logarithm of probability ratio
            log_ratio = P_1 - P_0

            # Convert to non-log space
            ratio = torch.exp(log_ratio)

            # If proposed value is more probable than current value, accept.
            # If not, then accept proportional to the probability ratios
            if ratio>torch.rand(1):
                m_0 = m_prime
                P_0 = P_1
                mb_0 = mb_prime

            # Only append to the chain if we're past burn-in.
            if i>burnin:
                # Only append every j-th sample to the chain
                if i%thin_factor==0:
                    self.P_chain.append(P_0)
                    self.m_chain.append(m_0)
                    self.m_primes.append(m_prime)
                    self.mb_chain.append(mb_0)
                    self.mb_primes.append(mb_prime)

        return torch.tensor(self.P_chain), torch.vstack(self.m_chain), torch.tensor(self.mb_chain), torch.vstack(self.m_primes), torch.tensor(self.mb_primes), torch.vstack(self.steps)
    
### some other useful functions ###

# acceptance rate, calculated as rolloing average of probability
def acceptance_rate(P_chain, window_size=100):
    return np.convolve(P_chain, np.ones(window_size)/window_size, mode='valid')

def effective_n(x):
    """
    Compute the effective sample size of a trace.

    Takes the trace and computes the effective sample size
    according to its detrended autocorrelation.

    Parameters
    ----------
    x : list or array of chain samples

    Returns
    -------
    effective_n : int
        effective sample size
    """
    try:
        # detrend trace using mean to be consistent with statistics
        # definition of autocorrelation
        x = np.asarray(x)
        x = (x - x.mean())
        # compute autocorrelation (note: only need second half since
        # they are symmetric)
        rho = np.correlate(x, x, mode='full')
        rho = rho[len(rho)//2:]
        # normalize the autocorrelation values
        #  note: rho[0] is the variance * n_samples, so this is consistent
        #  with the statistics definition of autocorrelation on wikipedia
        # (dividing by n_samples gives you the expected value).
        rho_norm = rho / rho[0]
        # Iterate until sum of consecutive estimates of autocorrelation is
        # negative to avoid issues with the sum being -0.5, which returns an
        # effective_n of infinity
        negative_autocorr = False
        t = 1
        n = len(x)
        while not negative_autocorr and (t < n):
            if not t % 2:
                negative_autocorr = sum(rho_norm[t-1:t+1]) < 0
            t += 1
        return int(n / (1 + 2*rho_norm[1:t].sum()))
    except:
        return None


def plot_chain(m_primes, m_chain, P_chain, title, ms=1, fontsize=8):
    # Plot the trace of the parameters
    fig, axes = plt.subplots(5, 1, figsize=(6, 8), sharex=True)
    m_chain = m_chain.detach().numpy()
    m_primes = m_primes.detach().numpy()
    P_chain = P_chain.detach().numpy()

    # get n_eff
    neff = [effective_n(arr) for arr in m_chain.T]

    axes[0].plot([],[],label=f'mean={np.mean(m_chain[:, 0]):.3f}\nstd={np.std(m_chain[:, 0]):.3f}')
    l0 = axes[0].legend(loc='upper right',handlelength=0, borderaxespad=0, fontsize=fontsize)

    axes[0].plot(m_primes[:, 0],'.',ms=ms, label='proposed', c='tab:blue')
    axes[0].plot(m_chain[:, 0],'.',ms=ms, label='accepted', c='tab:orange')
    hands, ls = axes[0].get_legend_handles_labels()

    # axes[0].add_artist(leg)
    axes[0].set_ylabel(r'$T_{bias}$', fontsize=fontsize)

    axes[1].plot(m_primes[:, 1],'.',ms=ms, c='tab:blue')
    axes[1].plot(m_chain[:, 1],'.',ms=ms, c='tab:orange')
    axes[1].plot([],[],label=f'mean={np.mean(m_chain[:, 1]):.3f}\nstd={np.std(m_chain[:, 1]):.3f}')
    l1 = axes[1].legend(loc='upper right',handlelength=0, borderaxespad=0, fontsize=fontsize)
    axes[1].set_ylabel(r'$K_p$', fontsize=fontsize)

    axes[2].plot(m_primes[:, 2],'.',ms=ms, c='tab:blue')
    axes[2].plot(m_chain[:, 2],'.',ms=ms, c='tab:orange')
    axes[2].plot([],[],label=f'mean={np.mean(m_chain[:, 2]):.3f}\nstd={np.std(m_chain[:, 2]):.3f}')
    l2 = axes[2].legend(loc='upper right',handlelength=0, borderaxespad=0, fontsize=fontsize)
    axes[2].set_ylabel(r'$fsnow$', fontsize=fontsize)

    axes[3].plot(m_primes[:, 3],'.',ms=ms, c='tab:blue')
    axes[3].plot(m_chain[:, 3],'.',ms=ms, c='tab:orange')
    axes[3].plot([],[],label=f'mean={np.mean(m_chain[:, 3]):.3f}\nstd={np.std(m_chain[:, 3]):.3f}')
    l3 = axes[3].legend(loc='upper right',handlelength=0, borderaxespad=0, fontsize=fontsize)
    axes[3].set_ylabel(r'$\dot{{b}}$', fontsize=fontsize)

    axes[4].plot(np.exp(P_chain),'tab:orange', lw=1)
    axes[4].plot(acceptance_rate(np.exp(P_chain)), 'k', label='moving avg.', lw=1)
    l4 = axes[4].legend(loc='upper right',handlelength=0, borderaxespad=0, fontsize=fontsize)
    axes[4].set_ylabel(r'$AR$', fontsize=fontsize)

    for i, ax in enumerate(axes):
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis="both",direction="inout")
        if i==4:
            continue
        ax.plot([],[],label=f'n_eff={neff[i]}')
        if i==0:
            hands, ls = ax.get_legend_handles_labels()
            ax.legend(handles=[hands[1],hands[2],hands[3]], labels=[ls[1],ls[2],ls[3]], loc='upper left', borderaxespad=0, handlelength=0, fontsize=fontsize)
        else:
            ax.legend(loc='upper left', borderaxespad=0, handlelength=0, fontsize=fontsize)

    axes[0].add_artist(l0)
    axes[1].add_artist(l1)
    axes[2].add_artist(l2)
    axes[3].add_artist(l3)
    axes[4].add_artist(l4)
    axes[0].set_xlim([0, m_chain.shape[0]])
    axes[0].set_title(title, fontsize=fontsize)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0)
    plt.show()
    return