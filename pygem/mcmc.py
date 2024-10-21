"""Markov chain Monte Carlo methods"""
import sys
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pygem_input as pygem_prms
torch.set_default_dtype(torch.float64)
plt.rcParams["font.family"] = "arial"
plt.rcParams['font.size'] = 8
plt.rcParams['legend.fontsize'] = 6

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
        if isinstance(value, torch.Tensor):
            pass
        elif isinstance(value, float):
            kwargs[key] = torch.tensor([value])
        else:
            kwargs[key] = torch.tensor(value)
    mu, sigma = kwargs['mu'], kwargs['sigma']

    # flatten arrays and get dimensionality
    x = x.flatten()
    mu = mu.flatten()
    sigma = sigma.flatten()
    k = mu.shape[-1]

    return torch.tensor([
                        -k/2.*torch.log(torch.tensor(2*np.pi)) - 
                        torch.log(sigma).nansum() -
                        0.5*(((x-mu)/sigma)**2).nansum()
                        ])

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
log_prob_fxn_map = {
    'normal': log_normal_density,
    'gamma': log_gamma_density,
    'truncated_normal': log_truncated_normal
}

# mass balance posterior class
class mbPosterior:
    def __init__(self, obs, priors, mb_func, mb_args=None, potential_fxns=None, **kwargs):
        # obs will be passed as a list, where each item is a tuple with the first element being the mean observation, and the second being the variance
        self.obs = obs
        self.prior_params = priors
        self.mb_func = mb_func
        self.mb_args = mb_args
        self.potential_functions = potential_fxns if potential_fxns is not None else []
        self.preds = None

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
            self.preds = self.mb_func(*self.mb_args)
        else:
            self.preds = self.mb_func([*m])
        if not isinstance(self.preds, tuple):
            self.preds = [self.preds]

    # get total log prior density
    def log_prior(self, m):
        log_prior = []
        for i, (key, params) in enumerate(self.prior_params.items()):
            params_copy = params.copy()
            prior_type = params_copy.pop('type')
            function_to_call = log_prob_fxn_map[prior_type]
            log_prior.append(function_to_call(m[i], **params_copy))
        log_prior = torch.stack(log_prior).sum()
        return log_prior

    # get log likelihood
    def log_likelihood(self):
        log_likehood = 0
        for i, pred in enumerate(self.preds):
            log_likehood+=log_normal_density(self.obs[i][0], **{'mu': pred, 'sigma': self.obs[i][1]})
        return log_likehood
    
    # get log potential (sum up as any declared potential functions)
    def log_potential(self, m):
        log_potential = 0
        for potential_function in self.potential_functions:
            log_potential += potential_function(*m, **{'massbal':self.preds[0]})
        return log_potential

    # get log posterior (sum of log prior, log likelihood and log potential)
    def log_posterior(self, m):
        # anytime log_posterior is called for a new step, calculate the predicted mass balance
        self.get_mb_pred(m)
        return self.log_prior(m) + self.log_likelihood() + self.log_potential(m), self.preds

# Metropolis-Hastings Markov chain Monte Carlo class
class Metropolis:
    def __init__(self, means, stds):
        # Initialize chains
        self.steps = []
        self.P_chain = []
        self.m_chain = []
        self.m_primes = []
        self.preds_chain = {}
        self.preds_primes = {}
        self.naccept = 0
        self.acceptance = []
        self.n_rm = 0
        self.means = means
        self.stds = stds

    def get_constant_prefix(self, threshold=100):
        n_rm = 0
        if threshold>0:
            # Convert to NumPy array if it's not already
            list_of_arrays = np.array(self.m_chain)
            # Start by checking the first `threshold` elements
            if np.all(list_of_arrays[:threshold] == list_of_arrays[0]):
                n_rm = threshold-1
                # Find the index where the constant value changes
                for i in range(threshold, len(list_of_arrays)):
                    if not np.array_equal(list_of_arrays[i], list_of_arrays[0]):
                        break
                    else:
                        n_rm += 1
        return n_rm
    
    def rm_constant_prefix(self, n_rm):
        self.P_chain = self.P_chain[n_rm:]
        self.m_chain = self.m_chain[n_rm:]
        self.m_primes = self.m_primes[n_rm:]
        self.steps = self.steps[n_rm:]
        self.acceptance = self.acceptance[n_rm:]
        for j in self.preds_primes.keys():
            self.preds_primes[j] = self.preds_primes[j][n_rm:]
            self.preds_chain[j] = self.preds_chain[j][n_rm:]
        return

    def sample(self, m_0, log_posterior, n_samples=1000, h=0.1, burnin=0, thin_factor=1, trim=True, progress_bar=False):
        # Compute initial unscaled log-posterior
        P_0, pred_0 = log_posterior(inverse_z_normalize(m_0, self.means, self.stds))

        n = len(m_0)

        # Create a tqdm progress bar if enabled
        pbar = tqdm(total=n_samples) if progress_bar else None

        i=0
        # Draw samples
        while i < n_samples:
            # Propose new value according to
            # proposal distribution Q(m) = N(m_0,h)
            step = torch.randn(n)*h
            m_prime = m_0 + step

            # Compute new unscaled log-posterior
            P_1, pred_1 = log_posterior(inverse_z_normalize(m_prime, self.means, self.stds))

            # Compute logarithm of probability ratio
            log_ratio = P_1 - P_0

            # Convert to non-log space
            ratio = torch.exp(log_ratio)

            # If proposed value is more probable than current value, accept.
            # If not, then accept proportional to the probability ratios
            if ratio>torch.rand(1):
                m_0 = m_prime
                P_0 = P_1
                pred_0 = pred_1
                # update naccept
                self.naccept += 1

            # Only append to the chain if we're past burn-in.
            if i>burnin:
                # Only append every j-th sample to the chain
                if i%thin_factor==0:
                    self.steps.append(step)
                    self.P_chain.append(P_0)
                    self.m_chain.append(m_0)
                    self.m_primes.append(m_prime)
                    self.acceptance.append(self.naccept / (i + (thin_factor*self.n_rm)))
                    for j in range(len(pred_1)):
                        if j not in self.preds_chain.keys():
                            self.preds_chain[j]=[]
                            self.preds_primes[j]=[]
                        self.preds_chain[j].append(pred_0[j])
                        self.preds_primes[j].append(pred_1[j])

            # trim off any initial steps that are stagnant
            if (i == (n_samples-1)) and (trim):
                self.n_rm = self.get_constant_prefix(threshold=int(n_samples/thin_factor*.1))
                if self.n_rm > 0:
                    if self.n_rm < (len(self.m_chain))*.8:
                        self.rm_constant_prefix(self.n_rm)  # remove the appropriate number of samples
                        i-=int((self.n_rm-1)*thin_factor)   # back track the iterator
                    trim = False                            # set trim to False as to only perform one time

            # increment iterator
            i+=1

            # update progress bar
            if pbar:
                pbar.update(1)

        # Close the progress bar if it was used
        if pbar:
            pbar.close()

        return torch.vstack(self.m_chain), \
                self.preds_chain, \
                torch.vstack(self.m_primes), \
                self.preds_primes, \
                torch.vstack(self.steps), \
                self.acceptance
    
### some other useful functions ###

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


def plot_chain(m_primes, m_chain, mb_obs, ar, title, ms=1, fontsize=8, show=False, fpath=None):
    # Plot the trace of the parameters
    fig, axes = plt.subplots(5, 1, figsize=(6, 8), sharex=True)
    m_chain = m_chain.detach().numpy()
    m_primes = m_primes.detach().numpy()

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

    axes[3].fill_between(np.arange(len(ar)),mb_obs[0]-(2*mb_obs[1]),mb_obs[0]+(2*mb_obs[1]),color='grey',alpha=.3)
    axes[3].fill_between(np.arange(len(ar)),mb_obs[0]-mb_obs[1],mb_obs[0]+mb_obs[1],color='grey',alpha=.3)
    axes[3].plot(m_primes[:, 3],'.',ms=ms, c='tab:blue')
    axes[3].plot(m_chain[:, 3],'.',ms=ms, c='tab:orange')
    axes[3].plot([],[],label=f'mean={np.mean(m_chain[:, 3]):.3f}\nstd={np.std(m_chain[:, 3]):.3f}')
    l3 = axes[3].legend(loc='upper right',handlelength=0, borderaxespad=0, fontsize=fontsize)
    axes[3].set_ylabel(r'$\dot{{b}}$', fontsize=fontsize)

    axes[4].plot(ar,'tab:orange', lw=1)
    axes[4].plot(np.convolve(ar, np.ones(100)/100, mode='valid'), 'k', label='moving avg.', lw=1)
    l4 = axes[4].legend(loc='upper left',handlelength=.5, borderaxespad=0, fontsize=fontsize)
    axes[4].set_ylabel(r'$AR$', fontsize=fontsize)

    for i, ax in enumerate(axes):
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis="both",direction="inout")
        if i==4:
            continue
        ax.plot([],[],label=f'n_eff={neff[i]}')
        hands, ls = ax.get_legend_handles_labels()
        if i==0:
            ax.legend(handles=[hands[1],hands[2],hands[3]], labels=[ls[1],ls[2],ls[3]], loc='upper left', borderaxespad=0, handlelength=0, fontsize=fontsize)
        else:
            ax.legend(handles=[hands[-1]], labels=[ls[-1]], loc='upper left', borderaxespad=0, handlelength=0, fontsize=fontsize)

    axes[0].add_artist(l0)
    axes[1].add_artist(l1)
    axes[2].add_artist(l2)
    axes[3].add_artist(l3)
    axes[4].add_artist(l4)
    axes[0].set_xlim([0, m_chain.shape[0]])
    axes[0].set_title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0)
    if fpath:
        fig.savefig(fpath, dpi=400)
    if show:
        plt.show(block=True)  # wait until the figure is closed
    plt.close(fig)
    return


def plot_resid_hist(obs, preds, title, fontsize=8, show=False, fpath=None):
    # Plot the trace of the parameters
    fig, axes = plt.subplots(1, 1, figsize=(3, 2))
    # subtract obs from preds to get residuals
    diffs = np.concatenate([pred.flatten() - obs[0].flatten().numpy() for pred in preds])
    # mask nans to avoid error in np.histogram()
    diffs = diffs[~np.isnan(diffs)]
    # Calculate histogram counts and bin edges
    counts, bin_edges = np.histogram(diffs, bins=20)
    pct = counts / counts.sum() * 100
    bin_width = bin_edges[1] - bin_edges[0]
    axes.bar(bin_edges[:-1], pct, width=bin_width, edgecolor='black', color='gray', align='edge')
    axes.set_xlabel('residuals (pred - obs)', fontsize=fontsize)
    axes.set_ylabel('count (%)', fontsize=fontsize)
    axes.set_title(title, fontsize=fontsize)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0)
    if fpath:
        fig.savefig(fpath, dpi=400)
    if show:
        plt.show(block=True)  # wait until the figure is closed
    plt.close(fig)
    return