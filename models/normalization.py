import torch
import torch.nn as nn
import torch.nn.functional as F


def num_pixels(tensor):
    assert tensor.dim() == 4
    return tensor.size(2) * tensor.size(3)


class ActNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, flow=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.logs = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.requires_init = nn.Parameter(torch.ByteTensor(1), requires_grad=False)
        self.reset_parameters()
        self.flow = flow

    def reset_parameters(self):
        self.logs.data.zero_()
        self.bias.data.zero_()
        self.requires_init.data.fill_(True)

    def init_data_dependent(self, x):
        with torch.no_grad():
            x_ = x.transpose(0, 1).contiguous().view(self.num_features, -1)
            mean = x_.mean(1)
            var = x_.var(1)
            logs = -torch.log(torch.sqrt(var) + 1e-6)
            self.logs.data.copy_(logs.data)
            self.bias.data.copy_(mean.data)

    def forward(self, x, log_det_jac=None, z=None):
        assert x.size(1) == self.num_features
        if self.requires_init:
            self.requires_init.data.fill_(False)
            self.init_data_dependent(x)

        size = [1] * x.ndimension()
        size[1] = self.num_features
        x = (x - self.bias.view(*size)) * torch.exp(self.logs.view(*size))
        if not self.flow:
            return x
        log_det_jac += self.logs.sum() * num_pixels(x)
        return x, log_det_jac, z

    def g(self, x, z):
        size = [1] * x.ndimension()
        size[1] = self.num_features
        x = x * torch.exp(-self.logs.view(*size)) + self.bias.view(*size)
        return x, z

    def inverse(self, x):
        size = [1] * x.ndimension()
        size[1] = self.num_features
        x = x * torch.exp(-self.logs.view(*size)) + self.bias.view(*size)
        return x

    def log_det(self):
        return self._log_det

    def extra_repr(self):
        return 'ActNorm({}, requires_init={})'.format(self.num_features, bool(self.requires_init.item()))


class DummyCondActNorm(ActNorm):
    def forward(self, x, y, log_det_jac=None, z=None):
        return super().forward(x, log_det_jac, z)

    def g(self, x, y, z):
        return super().g(x, z)
