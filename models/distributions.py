import torch.distributions as dist
import torch
from torch import nn
from torch.nn import functional as F
from numbers import Number
import numpy as np
import utils


class Mixture(dist.Distribution):
    def __init__(self, base_ditributions, weights=None):
        super(Mixture, self).__init__(batch_shape=base_ditributions[0].batch_shape,
                                      event_shape=base_ditributions[0].event_shape)
        self.base_ditributions = base_ditributions
        self.weights = weights
        if self.weights is None:
            k = len(self.base_ditributions)
            self.weights = torch.ones((k,)) / float(k)

    def log_prob(self, x, detach=False):
        ens = None
        for prob, w in zip(self.base_ditributions, self.weights):
            logp = prob.log_prob(x) + torch.log(w)
            if detach:
                logp = logp.detach()
            if ens is None:
                ens = logp
            else:
                _t = torch.stack([ens, logp])
                ens = torch.logsumexp(_t, dim=0)
        return ens

    def one_sample(self, labels=False):
        k = np.random.choice(len(self.weights), p=utils.tonp(self.weights))
        if labels:
            return self.base_ditributions[k].sample(), k
        return self.base_ditributions[k].sample()

    def sample(self, sample_shape=torch.Size(), labels=False):
        if len(sample_shape) == 0:
            return self.one_sample()
        elif len(sample_shape) == 1:
            res, ys = [], []
            for i in range(sample_shape[0]):
                if labels:
                    samples, y = self.one_sample(labels=labels)
                    res.append(samples)
                    ys.append(y)
                else:
                    res.append(self.one_sample())
            if labels:
                return torch.stack(res), np.stack(ys)
            else:
                return torch.stack(res)
        elif len(sample_shape) == 2:
            res, y = [], []
            for _ in range(sample_shape[0]):
                res.append([])
                y.append([])
                for _ in range(sample_shape[1]):
                    if labels:
                        samples, y = self.one_sample(labels=labels)
                        res[-1].append(samples)
                        ys[-1].append(y)
                    else:
                        res[-1].append(self.one_sample())
                res[-1] = torch.stack(res[-1])
            if labels:
                return torch.stack(res), np.stack(ys)
            else:
                return torch.stack(res)
        else:
            raise NotImplementedError


class GMM(nn.Module):
    def __init__(self, k=None, dim=None, means=None, covariances=None, weights=None, normalize=False):
        super(GMM, self).__init__()
        if k is None and means is None:
            raise NotImplementedError

        if means is None:
            covars = torch.rand(k, dim, dim)
            covars = torch.matmul(covars, covars.transpose(1, 2))
            self.means = nn.ParameterList([nn.Parameter(m) for m in torch.randn(k, dim)])
            self.cov_factors = nn.ParameterList([nn.Parameter(torch.cholesky(cov)) for cov in covars])
            self.weights = nn.Parameter(torch.FloatTensor([1./k] * k))
            self.k = k
        else:
            self.means = nn.ParameterList([nn.Parameter(m) for m in means])
            self.cov_factors = nn.ParameterList([nn.Parameter(torch.cholesky(cov)) for cov in covariances])
            self.weights = nn.Parameter(weights)
            self.k = weights.shape[0]

        self.normalize = normalize

    def get_weights(self):
        if self.normalize:
            return F.softmax(self.weights, dim=0)
        return self.weights

    def get_dist(self):
        base_distributions = []
        for m, covf in zip(self.means, self.cov_factors):
            if covf.dim() == 1:
                covf = torch.diag(covf)
            cov = torch.mm(covf, covf.t())
            base_distributions.append(dist.MultivariateNormal(m, covariance_matrix=cov))
        return Mixture(base_distributions, weights=self.get_weights())

    def set_covariance(self, k, cov):
        self.cov_factors[k].data = torch.cholesky(cov)

    def set_params(self, means=None, covars=None, pi=None):
        if pi is not None:
            self.weights.data = torch.log(pi) if self.normalize else pi
        for k in range(self.k):
            if means is not None:
                self.means[k].data = means[k]
            if covars is not None:
                self.set_covariance(k, covars[k])

    @property
    def covariances(self):
        return torch.stack([torch.mm(covf, covf.t()) for covf in self.cov_factors])

    def log_prob(self, x, k=None):
        if k is None:
            p = self.get_dist()
            return p.log_prob(x)
        else:
            p = self.get_dist()
            return p.base_ditributions[k].log_prob(x) + torch.log(p.weights[k])

    def sample(self, sample_shape=torch.Size(), labels=False):
        p = self.get_dist()
        return p.sample(sample_shape, labels=labels)


class MultivariateNormalDiag(torch.distributions.Normal):
    def log_prob(self, x):
        logp = super().log_prob(x)
        return logp.sum(1)


class GaussianDiag(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.logsigma = nn.Parameter(torch.zeros((dim,)))
        self.mean = nn.Parameter(torch.zeros((dim,)), requires_grad=False)

    def _get_dist(self):
        scale = F.softplus(self.logsigma)
        return MultivariateNormalDiag(self.mean, scale)

    def log_prob(self, x):
        p = self._get_dist()
        return p.log_prob(x)

    def log_prob_full(self, x):
        p = self._get_dist()
        return p.log_prob(x)[:, None]


class GmmPrior(nn.Module):
    def __init__(self, k=None, dim=None, full_dim=None, means=None, covariances=None, weights=None, cov_type='diag'):
        super().__init__()
        if k is None and means is None:
            raise NotImplementedError

        if cov_type not in ['diag']:
            raise NotImplementedError
        self.cov_type = cov_type

        if full_dim is None:
            full_dim = dim

        if means is None:
            means = torch.randn(k, dim) * np.sqrt(2)
            if cov_type == 'diag':
                # covariances = torch.log(torch.rand(k, dim) * 0.5)
                covariances = torch.zeros((k, dim))
            weights = torch.FloatTensor([1./k] * k)

        self.means = nn.Parameter(means)
        self.cov_factors = nn.Parameter(covariances)
        self.weights = nn.Parameter(weights)
        self.k = self.weights.shape[0]
        self.dim = self.means.shape[1]
        self.full_dim = full_dim
        self.sn_dim = self.full_dim - self.dim

    def get_logpi(self):
        return F.log_softmax(self.weights, dim=0)

    def get_dist(self):
        base_distributions = []
        for m, covf in zip(self.means, self.cov_factors):
            m = torch.cat([torch.zeros((self.sn_dim,)).to(m.device), m])
            if self.cov_type == 'diag':
                covf = torch.cat([torch.zeros((self.sn_dim,)).to(covf.device), covf])
                # TODO: softplus seems to be more stable
                scale = torch.exp(covf * 0.5)
                base_distributions.append(MultivariateNormalDiag(m, scale))

        pi = torch.exp(self.get_logpi())
        return Mixture(base_distributions, weights=pi)

    def log_prob(self, x, k=None):
        logpi = self.get_logpi()
        if k is None:
            p = self.get_dist()
            return p.log_prob(x)
        else:
            p = self.get_dist()
            return p.base_ditributions[k].log_prob(x) + logpi[k]

    def log_prob_full(self, x):
        return torch.stack([self.log_prob(x, k=k) for k in range(self.k)]).transpose(0, 1)

    def log_prob_full_fast(self, x):
        if self.cov_type != 'diag':
            raise NotImplementedError
        var = torch.exp(self.cov_factors)
        logp = -(x[:, None] - self.means[None])**2 / (2. * var[None]) - 0.5 * self.cov_factors
        return logp.sum(2) + self.get_logpi()[None] - 0.5 * np.log(2 * np.pi) * self.dim

    def sample(self, sample_shape=torch.Size(), labels=False):
        p = self.get_dist()
        return p.sample(sample_shape, labels=labels)

    def set_params(self, means=None, covars=None, pi=None):
        if pi is not None:
            self.weights.data = torch.log(pi) if self.normalize else pi
        if means is not None:
            self.means.data = torch.tensor(means)
        if covars is not None:
            self.cov_factors.data = torch.log(covars)
