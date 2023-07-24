import torch
from torch import nn
from torch.distributions import (
    Bernoulli,
    Independent,
    Laplace,
    Normal,
    StudentT,
    kl_divergence,
)
from torch.distributions.transforms import AffineTransform


class Null(nn.Module):
    def make_posterior(self, *args):
        return None

    def forward(self, *args):
        return 0


class MultivariateNormal(nn.Module):
    def __init__(
        self,
        mean: float,
        variance: float,
        shape: int,
    ):
        super().__init__()
        # Create distribution parameter tensors
        means = torch.ones(shape) * mean
        logvars = torch.log(torch.ones(shape) * variance)
        self.mean = nn.Parameter(means, requires_grad=True)
        self.logvar = nn.Parameter(logvars, requires_grad=False)

    def make_posterior(self, post_mean, post_std):
        return Independent(Normal(post_mean, post_std), 1)

    def forward(self, post_mean, post_std):
        # Create the posterior distribution
        posterior = self.make_posterior(post_mean, post_std)
        # Create the prior and posterior
        prior_std = torch.exp(0.5 * self.logvar)
        prior = Independent(Normal(self.mean, prior_std), 1)
        # Compute KL analytically
        kl_batch = kl_divergence(posterior, prior)
        return torch.mean(kl_batch)


class AutoregressiveMultivariateNormal(nn.Module):
    def __init__(
        self,
        tau: float,
        nvar: float,
        shape: int,
    ):
        super().__init__()
        # Create the distribution parameters
        logtaus = torch.log(torch.ones(shape) * tau)
        lognvars = torch.log(torch.ones(shape) * nvar)
        self.logtaus = nn.Parameter(logtaus, requires_grad=True)
        self.lognvars = nn.Parameter(lognvars, requires_grad=True)

    def make_posterior(self, post_mean, post_std):
        return Independent(Normal(post_mean, post_std), 2)

    def log_prob(self, sample):
        # Compute alpha and process variance
        alphas = torch.exp(-1.0 / torch.exp(self.logtaus))
        logpvars = self.lognvars - torch.log(1 - alphas**2)
        # Create autocorrelative transformation
        transform = AffineTransform(loc=0, scale=alphas)
        # Align previous samples and compute means and stddevs
        prev_samp = torch.roll(sample, shifts=1, dims=1)
        means = transform(prev_samp)
        stddevs = torch.ones_like(means) * torch.exp(0.5 * self.lognvars)
        # Correct the first time point
        means[:, 0] = 0.0
        stddevs[:, 0] = torch.exp(0.5 * logpvars)
        # Create the prior and compute the log-probability
        prior = Independent(Normal(means, stddevs), 2)
        return prior.log_prob(sample)

    def forward(self, post_mean, post_std):
        posterior = self.make_posterior(post_mean, post_std)
        sample = posterior.rsample()
        log_q = posterior.log_prob(sample)
        log_p = self.log_prob(sample)
        kl_batch = log_q - log_p
        return torch.mean(kl_batch)


class MultivariateStudentT(nn.Module):
    def __init__(
        self,
        loc: float,
        scale: float,
        df: int,
        shape: int,
    ):
        super().__init__()
        # Create the distribution parameters
        loc = torch.ones(shape) * scale
        self.loc = nn.Parameter(loc, requires_grad=True)
        logscale = torch.log(torch.ones(shape) * scale)
        self.logscale = nn.Parameter(logscale, requires_grad=True)
        self.df = df

    def make_posterior(self, post_loc, post_scale):
        # TODO: Should probably be inferring degrees of freedom along with loc and scale
        return Independent(StudentT(self.df, post_loc, post_scale), 1)

    def forward(self, post_loc, post_scale):
        # Create the posterior distribution
        posterior = self.make_posterior(post_loc, post_scale)
        # Create the prior distribution
        prior_scale = torch.exp(self.logscale)
        prior = Independent(StudentT(self.df, self.loc, prior_scale), 1)
        # Approximate KL divergence
        sample = posterior.rsample()
        log_q = posterior.log_prob(sample)
        log_p = prior.log_prob(sample)
        kl_batch = log_q - log_p
        return torch.mean(kl_batch)


class SparseMultivariateNormal(nn.Module):
    def __init__(
        self,
        mean: float,
        scale: float,  # scale (b) parameter for Laplace distribution
        shape: int,
    ):
        super().__init__()
        # Create distribution parameter tensors
        means = torch.ones(shape) * mean
        scales = torch.ones(shape) * scale
        self.mean = nn.Parameter(means, requires_grad=True)
        self.scale = nn.Parameter(scales, requires_grad=False)

    def make_posterior(self, post_mean, post_std):
        return Independent(Laplace(post_mean, post_std), 1)

    def forward(self, post_mean, post_std):
        # Create the posterior distribution
        posterior = self.make_posterior(post_mean, post_std)
        # Create the prior with Laplace distribution
        prior = Independent(Laplace(self.mean, self.scale), 1)
        # Compute KL analytically
        kl_batch = kl_divergence(posterior, prior)
        return torch.mean(kl_batch)


class SparseBernoulli(nn.Module):
    def __init__(
        self,
        prob: float,
        shape: int,
    ):
        super().__init__()
        # Create distribution parameter tensors
        probs = torch.ones(shape) * prob
        self.probs = nn.Parameter(probs, requires_grad=True)

    def make_posterior(self, post_probs, post_std):
        return Independent(Bernoulli(post_probs), 1)

    def forward(self, post_probs):
        # Create the posterior distribution
        posterior = self.make_posterior(post_probs, None)
        # Create the prior with Bernoulli distribution
        prior = Independent(Bernoulli(self.probs), 1)
        # Compute KL analytically
        kl_batch = kl_divergence(posterior, prior)
        return torch.mean(kl_batch)
