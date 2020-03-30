import torch
from torch import nn
import numpy as np
import torch.distributions as dist
import torch.nn.functional as F


class Conv2dZeros(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__(in_channels, out_channels, kernel_size, padding=padding)
        self.weight.data.zero_()
        self.bias.data.zero_()


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input, *inputs):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return [output] + list(inputs)

    def g(self, input, *inputs):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size, input_height, output_width, output_depth) for t_t in spl]
        output = torch.stack(stacks, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size,
                                                                                       output_height,
                                                                                       output_width,
                                                                                       output_depth)
        output = output.permute(0, 3, 1, 2)
        return [output] + list(inputs)

    def extra_repr(self):
        return 'SpaceToDepth({0:d}x{0:d})'.format(self.block_size)


class CondSpaceToDepth(SpaceToDepth):
    def forward(self, x, y, log_det, z):
        return super().forward(x, log_det, z)

    def g(self, x, y, z):
        return super().g(x, z)


class FactorOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_shape = None

    def forward(self, x, log_det_jac, z):
        self.out_shape = list(x.shape)[1:]
        self.inp_shape = list(x.shape)[1:]
        self.out_shape[0] = self.out_shape[0] // 2
        self.out_shape[0] += self.out_shape[0] % 2

        k = self.out_shape[0]
        if z is None:
            return x[:, k:], log_det_jac, x[:, :k].reshape((x.shape[0], -1))
        z = torch.cat([z, x[:, :k].view((x.shape[0], -1))], dim=1)
        return x[:, k:], log_det_jac, z

    def g(self, x, z):
        k = np.prod(self.out_shape)
        x = torch.cat([z[:, -k:].view([x.shape[0]] + self.out_shape), x], dim=1)
        z = z[:, :-k]
        return x, z

    def extra_repr(self):
        return 'FactorOut({:s} -> {:s})'.format(str(self.inp_shape), str(self.out_shape))


class CondFactorOut(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_shape = None

    def extra_repr(self):
        return 'FactorOut({:s} -> {:s})'.format(str(self.inp_shape), str(self.out_shape))

    def forward(self, x, y, log_det_jac, z):
        self.out_shape = list(x.shape)[1:]
        self.inp_shape = list(x.shape)[1:]
        self.out_shape[0] = self.out_shape[0] // 2
        self.out_shape[0] += self.out_shape[0] % 2

        k = self.out_shape[0]
        if z is None:
            return x[:, k:], log_det_jac, x[:, :k].reshape((x.shape[0], -1))
        z = torch.cat([z, x[:, :k].view((x.shape[0], -1))], dim=1)
        return x[:, k:], log_det_jac, z

    def g(self, x, y, z):
        k = np.prod(self.out_shape)
        x = torch.cat([z[:, -k:].view([x.shape[0]] + self.out_shape), x], dim=1)
        z = z[:, :-k]
        return x, z


class ToLogits(nn.Module):
    '''
    Maps interval [0, 1] to (-inf, +inf) via inversion of sigmoid
    '''
    alpha = 0.05

    def forward(self, x, log_det_jac, z):
        # [0, 1] -> [alpha/2, 1 - alpha/2]
        x = x * (1 - self.alpha) + self.alpha * 0.5
        log_det_jac += torch.sum(-torch.log(x) - torch.log(1-x) + np.log(1 - self.alpha), dim=[1, 2, 3])
        x = torch.log(x) - torch.log(1-x)
        return x, log_det_jac, z

    def g(self, x, z):
        x = torch.sigmoid(x)
        x = (x - self.alpha * 0.5) / (1. - self.alpha)
        return x, z

    def extra_repr(self):
        return 'ToLogits()'


class InverseLogits(nn.Module):
    def forward(self, x, log_det_jac, z):
        log_det_jac += torch.sum(-F.softplus(-x) - F.softplus(x), dim=[1, 2, 3])
        x = torch.sigmoid(x)
        return x, log_det_jac, z

    def g(self, x, z):
        x = torch.log(x) - torch.log(1 - x)
        return x, z

    def extra_repr(self):
        return 'InverseLogits()'


class CondToLogits(nn.Module):
    '''
    Maps interval [0, 1] to (-inf, +inf) via inversion of sigmoid
    '''
    alpha = 0.05

    def forward(self, x, y, log_det_jac, z):
        # [0, 1] -> [alpha/2, 1 - alpha/2]
        x = x * (1 - self.alpha) + self.alpha * 0.5
        log_det_jac += torch.sum(-torch.log(x) - torch.log(1-x) + np.log(1 - self.alpha), dim=[1, 2, 3])
        x = torch.log(x) - torch.log(1-x)
        return x, log_det_jac, z

    def g(self, x, y, z):
        x = torch.sigmoid(x)
        x = (x - self.alpha * 0.5) / (1. - self.alpha)
        return x, z


class DummyCond(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, y, log_det_jac, z):
        return self.module.forward(x, log_det_jac, z)

    def g(self, x, y, z):
        return self.module.g(x, z)


class IdFunction(nn.Module):
    def forward(self, *inputs):
        return inputs

    def g(self, *inputs):
        return inputs


class UniformWithLogits(dist.Distribution):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def log_prob(self, x):
        return torch.sum(-F.softplus(-x) - F.softplus(x), dim=1)

    def sample(self, shape):
        x = torch.rand(list(shape) + [self.dim])
        return torch.log(x) - torch.log(1 - x)
