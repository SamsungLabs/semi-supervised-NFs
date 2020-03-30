import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


def coupling(x1, x2, net, log_det_jac=None, eps=1e-6):
    scale, shift = net(x2).split(x1.size(1), dim=1)
    # TODO: deal with scale
    # scale = torch.tanh(scale)
    # scale = torch.exp(scale)
    scale = torch.sigmoid(scale + 2.) + eps
    x1 = (x1 + shift) * scale
    if log_det_jac is not None:
        if scale.dim() == 4:
            log_det_jac += torch.log(scale).sum((1, 2, 3))
        else:
            log_det_jac += torch.log(scale).sum(1)
    return x1


def coupling_inv(x1, x2, net, eps=1e-6):
    scale, shift = net(x2).split(x1.size(1), dim=1)
    # TODO: deal with scale
    # scale = torch.tanh(scale)
    # scale = torch.exp(scale)
    scale = torch.sigmoid(scale + 2.) + eps
    x1 = x1 / scale - shift
    return x1


def get_mask(xs, mask_type):
    if 'checkerboard' in mask_type:
        unit0 = np.array([[0.0, 1.0], [1.0, 0.0]])
        unit1 = -unit0 + 1.0
        unit = unit0 if mask_type == 'checkerboard0' else unit1
        unit = np.reshape(unit, [1, 2, 2])
        b = np.tile(unit, [xs[0], xs[1]//2, xs[2]//2])
    elif 'channel' in mask_type:
        white = np.ones([xs[0]//2, xs[1], xs[2]])
        black = np.zeros([xs[0]//2, xs[1], xs[2]])
        if mask_type == 'channel0':
            b = np.concatenate([white, black], 0)
        else:
            b = np.concatenate([black, white], 0)

    if list(b.shape) != list(xs):
        b = np.tile(b, (1, 2, 2))[:, :xs[1], :xs[2]]

    return b


class MaskedCouplingLayer(nn.Module):
    def __init__(self, shape, mask_type, net):
        super().__init__()
        mask = torch.FloatTensor(get_mask(shape, mask_type))
        self.mask = nn.Parameter(mask[None], requires_grad=False)
        self.net = net
        self.eps = 1e-6

    def extra_repr(self):
        return 'MaskedCouplingLayer(mask=checkerboard)'

    def forward(self, x, log_det_jac, z):
        return self.f(x, log_det_jac, z)

    def f(self, x, log_det_jac, z):
        x1 = self.mask * x
        s, t = self.net(x1).split(x1.size(1), dim=1)
        logs = -F.softplus(-s-2.)
        logs *= (1 - self.mask)
        s = torch.sigmoid(s + 2.)
        s = (1 - self.mask) * s
        s += self.mask
        t = (1 - self.mask) * t
        x = x1 + (1 - self.mask) * (x * s + t)
        log_det_jac += torch.sum(logs, dim=(1, 2, 3))
        return x, log_det_jac, z

    def g(self, x, z):
        x1 = self.mask * x
        s, t = self.net(x1).split(x1.size(1), dim=1)
        s = torch.sigmoid(s + 2.) + self.eps
        x = x1 + (1 - self.mask) * (x - t) / s
        return x, z


class ConditionalMaskedCouplingLayer(nn.Module):
    def __init__(self, shape, mask_type, net):
        super().__init__()
        mask = torch.FloatTensor(get_mask(shape, mask_type))
        self.mask = nn.Parameter(mask[None], requires_grad=False)
        self.net = net
        self.eps = 1e-6

    def forward(self, x, y, log_det_jac, z):
        return self.f(x, y, log_det_jac, z)

    def f(self, x, y, log_det_jac, z):
        x1 = self.mask * x

        assert y.dim() == 2
        y = y[..., None, None].repeat((1, 1, x1.shape[2], x1.shape[3]))

        s, t = self.net(torch.cat([x1, y], dim=1)).split(x1.size(1), dim=1)
        logs = -F.softplus(-s-2.)
        logs *= (1 - self.mask)
        s = torch.sigmoid(s + 2.)
        s = (1 - self.mask) * s
        s += self.mask
        t = (1 - self.mask) * t
        x = x1 + (1 - self.mask) * (x * s + t)
        log_det_jac += torch.sum(logs, dim=(1, 2, 3))
        return x, log_det_jac, z

    def g(self, x, y, z):
        x1 = self.mask * x

        assert y.dim() == 2
        y = y[..., None, None].repeat((1, 1, x1.shape[2], x1.shape[3]))

        s, t = self.net(torch.cat([x1, y], dim=1)).split(x1.size(1), dim=1)
        s = torch.sigmoid(s + 2.)
        x = x1 + (1 - self.mask) * (x - t) / s
        return x, z


class CouplingLayer(nn.Module):
    """
    Coupling layer with channelwise mask applied twice.
    (e.g. see RealNVP https://arxiv.org/pdf/1605.08803.pdf for details)
    """
    def __init__(self, netfunc):
        super().__init__()
        self.net1 = netfunc()
        self.net2 = netfunc()

    def extra_repr(self):
        return 'CouplingLayer(mask=channel)'

    def forward(self, x, log_det_jac, z):
        return self.f(x, log_det_jac, z)

    def f(self, x, log_det_jac, z):
        C = x.size(1) // 2
        x1, x2 = x.split(C, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        x1 = coupling(x1, x2, self.net1, log_det_jac)
        x2 = coupling(x2, x1, self.net2, log_det_jac)

        return torch.cat([x1, x2], dim=1), log_det_jac, z

    def g(self, x, z):
        C = x.size(1) // 2
        x1, x2 = x.split(C, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        x2 = coupling_inv(x2, x1, self.net2)
        x1 = coupling_inv(x1, x2, self.net1)

        return torch.cat([x1, x2], dim=1), z


class ConditionalCouplingLayer(CouplingLayer):
    def forward(self, x, y, log_det_jac, z):
        return self.f(x, y, log_det_jac, z)

    def extra_repr(self):
        return 'ConditionalCouplingLayer(mask=channel)'

    def f(self, x, y, log_det_jac, z):
        C = x.size(1) // 2
        x1, x2 = x.split(C, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        assert y.dim() == 2
        y = y[..., None, None].repeat((1, 1, x1.shape[2], x1.shape[3]))

        x1 = coupling(x1, torch.cat([x2, y], dim=1), self.net1, log_det_jac)
        x2 = coupling(x2, torch.cat([x1, y], dim=1), self.net2, log_det_jac)

        return torch.cat([x1, x2], dim=1), log_det_jac, z

    def g(self, x, y, z):
        C = x.size(1) // 2
        x1, x2 = x.split(C, dim=1)
        x1, x2 = x1.contiguous(), x2.contiguous()

        assert y.dim() == 2
        y = y[..., None, None].repeat((1, 1, x1.shape[2], x1.shape[3]))

        x2 = coupling_inv(x2, torch.cat([x1, y], dim=1), self.net2)
        x1 = coupling_inv(x1, torch.cat([x2, y], dim=1), self.net1)

        return torch.cat([x1, x2], dim=1), z


class ConditionalShift(nn.Module):
    def __init__(self, channels, nfactors):
        super().__init__()
        self.factors = nn.Embedding(nfactors, channels)

    def forward(self, x, y, log_det_jac, z):
        return self.f(x, y, log_det_jac, z)

    def f(self, x, y, log_det_jac, z):
        shift = self.factors(y)
        return x - shift.view((x.size(0), -1, 1, 1)), log_det_jac, z

    def g(self, x, y, z):
        shift = self.factors(y)
        return x + shift.view((x.size(0), -1, 1, 1)), z
