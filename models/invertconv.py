import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def num_pixels(tensor):
    assert tensor.dim() == 4
    return tensor.size(2) * tensor.size(3)


class _BaseInvertibleConv2d(nn.Module):
    def _get_w(self):
        raise NotImplementedError

    def _get_w_inv(self):
        raise NotImplementedError

    def _log_det(self, x=None):
        raise NotImplementedError

    def forward(self, x, log_det_jac, z):
        log_det_jac += self._log_det(x)
        W = self._get_w().to(x)
        W = W.unsqueeze(-1).unsqueeze(-1)
        return F.conv2d(x, W), log_det_jac, z

    def g(self, x, z):
        W = self._get_w_inv()
        W = W.unsqueeze(-1).unsqueeze(-1)
        return F.conv2d(x, W), z


class InvertibleConv2d(_BaseInvertibleConv2d):
    '''
    Diederik P. Kingma, Prafulla Dhariwal
    "Glow: Generative Flow with Invertible 1x1 Convolutions"
    https://arxiv.org/pdf/1807.03039.pdf
    '''
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.W = nn.Parameter(torch.Tensor(features, features))
        self.reset_parameters()

    def reset_parameters(self):
        # nn.init.orthogonal_(self.W)
        self.W.data = torch.eye(self.features).to(self.W.device)

    def _get_w(self):
        return self.W

    def _get_w_inv(self):
        return torch.inverse(self.W.double()).float()

    def _log_det(self, x):
        return torch.slogdet(self.W.double())[1].float() * num_pixels(x)

    def extra_repr(self):
        return 'InvertibleConv2d({:d})'.format(self.features)


def householder_matrix(v, size=None):
    """
    householder_matrix(Tensor, size=None) -> Tensor

    Arguments
        v: Tensor of size [Any,]
        size: `int` or `None`. The size of the resulting matrix.
            size >= v.size(0)
    Output
        I - 2 v^T * v / v*v^T: Tensor of size [size, size]
    """
    size = size or v.size(0)
    assert size >= v.size(0)
    v = torch.cat([torch.ones(size - v.size(0), device=v.device), v])
    I = torch.eye(size, device=v.device)
    outer = torch.ger(v, v)
    inner = torch.dot(v, v) + 1e-16
    return I - 2 * outer / inner


def naive_cascade(vectors, size=None):
    """
    naive_cascade([Tensor, Tensor, ...], size=None) -> Tensor
    naive implementation

    Arguments
        vectors: list of Tensors of size [Any,]
        size: `int` or `None`. The size of the resulting matrix.
            size >= max(v.size(0) for v in vectors)
    Output
        Q: `torch.Tensor` of size [size, size]
    """
    size = size or max(v.size(0) for v in vectors)
    assert size >= max(v.size(0) for v in vectors)
    device = vectors[0].device
    Q = torch.eye(size, device=device)
    for v in vectors:
        Q = torch.mm(Q, householder_matrix(v, size=size))
    return Q


class HHConv2d_1x1(_BaseInvertibleConv2d):
    def __init__(self, features, factors=None):
        super().__init__()
        self.features = features
        self.factors = factors or range(2, features + 1)

        # init vectors
        self.vectors = []
        for i, f in enumerate(self.factors):
            vec = nn.Parameter(torch.Tensor(f))
            self.register_parameter('vec_{}'.format(i), vec)
            self.vectors.append(vec)

        self.reset_parameters()
        self.cascade = naive_cascade

    def reset_parameters(self):
        for v in self.vectors:
            v.data.uniform_(-1, 1)
            with torch.no_grad():
                v /= (torch.norm(v) + 1e-16)

    def _get_w(self):
        return self.cascade(self.vectors, self.features)

    def _get_w_inv(self):
        return self._get_w().t()

    def _log_det(self, x):
        return 0.


class QRInvertibleConv2d(HHConv2d_1x1):
    """
    Hoogeboom, Emiel and Berg, Rianne van den and Welling, Max
    "Emerging Convolutions for Generative Normalizing Flows"
    https://arxiv.org/pdf/1901.11137.pdf
    """
    def __init__(self, features, factors=None, act='softplus'):
        super().__init__(features, factors=factors)
        self.act = act
        if act == 'softplus':
            self.s_factor = nn.Parameter(torch.zeros((features,)))
        elif act == 'no':
            self.s_factor = nn.Parameter(torch.ones((features,)))
        else:
            raise NotImplementedError

        self.r = nn.Parameter(torch.zeros((features, features)))

    def _get_w(self):
        Q = super()._get_w()
        if self.act == 'softplus':
            R = torch.diag(F.softplus(self.s_factor))
        elif self.act == 'no':
            R = torch.diag(self.s_factor)

        R += torch.triu(self.r, diagonal=1)
        return Q.to(R) @ R

    def _log_det(self, x=None):
        if self.act == 'softplus':
            return torch.log(F.softplus(self.s_factor)).sum() * num_pixels(x)
        elif self.act == 'no':
            return torch.log(torch.abs(self.s_factor)).sum() * num_pixels(x)

    def _get_w_inv(self):
        Q = super()._get_w().to(self.s_factor)
        if self.act == 'softplus':
            R = torch.diag(F.softplus(self.s_factor))
        elif self.act == 'no':
            R = torch.diag(self.s_factor)

        R += torch.triu(self.r, diagonal=1)
        return torch.inverse(R.double()).float() @ Q.t()


class DummyCondInvertibleConv2d(InvertibleConv2d):
    def forward(self, x, y, log_det_jac, z):
        return super().forward(x, log_det_jac, z)

    def g(self, x, y, z):
        return super().g(x, z)
