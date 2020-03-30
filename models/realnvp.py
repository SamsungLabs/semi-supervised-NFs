import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn import functional as F
from models.normalization import ActNorm


class RealNVPold(nn.Module):
    def __init__(self, nets, nett, masks, prior, device=None):
        super().__init__()

        self.prior = prior
        self.mask = nn.Parameter(masks, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

        self.to(device)
        self.device = device

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            if x.dim() == 2:
                log_det_J -= s.sum(dim=1)
            else:
                log_det_J -= s.sum(dim=(1, 2, 3))

        return z, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x = self.g(z)
        return x


def get_toy_nvp(prior=None, device=None):
    def nets():
        return nn.Sequential(nn.Linear(2, 32),
                             nn.LeakyReLU(),
                             nn.Linear(32, 2),
                             nn.Tanh()
                             )

    def nett():
        return nn.Sequential(nn.Linear(2, 32),
                             nn.LeakyReLU(),
                             nn.Linear(32, 2)
                             )

    if prior is None:
        prior = distributions.MultivariateNormal(torch.zeros(2).to(device),
                                                 torch.eye(2).to(device))

    masks = torch.from_numpy(np.array([[0, 1], [1, 0]] * 3).astype(np.float32))
    return RealNVPold(nets, nett, masks, prior, device=device)


class NFGMM(RealNVPold):
    def log_prob(self, x, k=None):
        if k is None:
            z, logp = self.f(x)
            return self.prior.log_prob(z) + logp
        else:
            z, logp = self.f(x)
            return self.prior.log_prob(z, k=k) + logp


def gmm_prior(k):
    covars = torch.rand(args.gmm_k, 2, 2)
    covars = torch.matmul(covars, covars.transpose(1, 2))
    prior = distributions.GMM(torch.randn(args.gmm_k, 2), covars, torch.FloatTensor([0.5] * args.gmm_k),
                              normalize=args.prior_train_algo == 'GD')


class WNConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)
        self.scale = nn.Parameter(torch.ones((1,)))
        self.scale.reg = True
        self.eps = 1e-6

    def forward(self, input):
        w = self.weight / (torch.norm(self.weight) + self.eps) * self.scale
        return F.conv2d(input, w, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class MABatchNorm2d(nn.BatchNorm2d):
    def forward(self, input):
        # Code from PyTroch repo:
        self._check_input_dim(input)

        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # --- My code ---

        if self.training:
            mean = torch.mean(input, (0, 2, 3), keepdim=True)
            var = torch.mean((input - mean)**2, (0, 2, 3), keepdim=True)
            self.running_mean.data = self.running_mean.data * (1 - self.momentum) + self.momentum * mean.data.squeeze()
            self.running_var.data = self.running_var.data * (1 - self.momentum) + self.momentum * var.data.squeeze()
            mean = mean * self.momentum + (1 - self.momentum) * self.running_mean[None, :, None, None]
            var = var * self.momentum + (1 - self.momentum) * self.running_var[None, :, None, None]
        else:
            mean = self.running_mean[None, :, None, None]
            var = self.running_var[None, :, None, None]

        input = (input - mean) / (var + self.eps)
        input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input


class ResNetBlock(nn.Module):
    def __init__(self, channels, use_bn=True):
        super().__init__()
        modules = []
        if use_bn:
            modules.append(nn.BatchNorm2d(channels))
        modules += [
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            WNConv2d(channels, channels, 3)]
        if use_bn:
            modules.append(nn.BatchNorm2d(channels))
        modules += [
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            WNConv2d(channels, channels, 3)]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x) + x


class SplitAndNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(data=torch.FloatTensor([1.]))
        self.scale.reg = True

    def forward(self, x):
        k = x.shape[1] // 2
        s, t = x[:, :k], x[:, k:]
        return torch.tanh(s) * self.scale, t


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

    assert list(b.shape) == list(xs)

    return b


class CouplingLayer(nn.Module):
    def __init__(self, mask_type, shape, net):
        super().__init__()
        mask = torch.FloatTensor(get_mask(shape, mask_type))
        self.mask = nn.Parameter(mask[None], requires_grad=False)
        self.net = net

    def forward(self, x, log_det_jac, z):
        return self.f(x, log_det_jac, z)

    def f(self, x, log_det_jac, z):
        x1 = self.mask * x
        s, t = self.net(x1)
        s = (1 - self.mask) * s
        t = (1 - self.mask) * t
        x = x1 + (1 - self.mask) * (x * torch.exp(s) + t)
        log_det_jac += torch.sum(s, dim=(1, 2, 3))
        return x, log_det_jac, z

    def g(self, x, z):
        x1 = self.mask * x
        s, t = self.net(x1)
        x = x1 + (1 - self.mask) * (x - t) * torch.exp(-s)
        return x, z


class Invertable1x1Conv(nn.Module):
    # reference https://github.com/openai/glow/blob/eaff2177693a5d84a1cf8ae19e8e0441715b82f8/model.py#L438
    def __init__(self, channels):
        super().__init__()
        # Sample a random orthogonal matrix
        w_init = np.linalg.qr(np.random.randn(channels, channels))[0]
        self.weight = nn.Parameter(torch.FloatTensor(w_init))

    def forward(self, x, log_det_jac, z):
        x = F.conv2d(x, self.weight[:, :, None, None])
        log_det_jac += torch.logdet(self.weight) * np.prod(x.shape[2:])
        return x, log_det_jac, z

    def g(self, x, z):
        x = F.conv2d(x, torch.inverse(self.weight)[:, :, None, None])
        return x, z


class Housholder1x1Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.v = nn.Parameter(torch.ones((channels,)))
        self.id = nn.Parameter(torch.eye(channels), requires_grad=False)
        self.channels = channels

    def forward(self, x, log_det_jac, z):
        v = self.v
        w = self.id - 2 * v[:, None] @ v[None] / (v @ v)
        x = F.conv2d(x, w[..., None, None])
        # w is unitary so log_det = 0
        return x, log_det_jac, z

    def g(self, x, z):
        v = self.v
        w = self.id - 2 * v[:, None] @ v[None] / (v @ v)
        x = F.conv2d(x, w[..., None, None])
        return x, z


class Prior(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros((dim,)), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(dim), requires_grad=False)

    def log_prob(self, x):
        p = torch.distributions.MultivariateNormal(self.mean, self.cov)
        return p.log_prob(x)


class RealNVP(nn.Module):
    def __init__(self, modules, dim):
        super().__init__()
        self.modules_ = nn.ModuleList(modules)
        self.latent_len = -1
        self.x_shape = -1
        self.prior = Prior(dim)
        self.alpha = 0.05

    def f(self, x):
        x = x * (1 - self.alpha) + self.alpha * 0.5
        log_det_jac = torch.sum(-torch.log(x) - torch.log(1-x) + np.log(1 - self.alpha), dim=[1, 2, 3])
        x = torch.log(x) - torch.log(1-x)

        z = None
        for m in self.modules_:
            x, log_det_jac, z = m(x, log_det_jac, z)
        if z is None:
            z = torch.zeros((x.shape[0], 1))[:, :0].to(x.device)
        self.x_shape = list(x.shape)[1:]
        self.latent_len = z.shape[1]
        z = torch.cat([z, x.reshape((x.shape[0], -1))], dim=1)
        return x, log_det_jac, z

    def forward(self, x):
        return self.log_prob(x)

    def g(self, z):
        x = z[:, self.latent_len:].view([z.shape[0]] + self.x_shape)
        z = z[:, :self.latent_len]
        for m in reversed(self.modules_):
            x, z = m.g(x, z)
        x = torch.sigmoid(x)
        x = (x - self.alpha * 0.5) / (1. - self.alpha)
        return x

    def log_prob(self, x):
        x, log_det_jac, z = self.f(x)
        logp = self.prior.log_prob(z) + log_det_jac
        return logp


def get_cifar_realnvp():
    dim = 32**2 * 3
    channels = 64

    def get_net(in_channels, channels):
        net = nn.Sequential(
            nn.ReflectionPad2d(1),
            WNConv2d(in_channels, channels, 3),
            ResNetBlock(channels),
            ResNetBlock(channels),
            ResNetBlock(channels),
            ResNetBlock(channels),
            ResNetBlock(channels),
            ResNetBlock(channels),
            ResNetBlock(channels),
            ResNetBlock(channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            WNConv2d(channels, in_channels * 2, 3),
            SplitAndNorm()
        )
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=1e-6)
                m.scale.data.fill_(1e-5)
        return net

    model = [
        CouplingLayer('checkerboard0', [3, 32, 32], get_net(3, channels)),
        CouplingLayer('checkerboard1', [3, 32, 32], get_net(3, channels)),
        CouplingLayer('checkerboard0', [3, 32, 32], get_net(3, channels)),
        SpaceToDepth(2),
        CouplingLayer('channel0', [12, 16, 16], get_net(12, channels)),
        CouplingLayer('channel1', [12, 16, 16], get_net(12, channels)),
        CouplingLayer('channel0', [12, 16, 16], get_net(12, channels)),
        FactorOut([12, 16, 16]),
        CouplingLayer('checkerboard0', [6, 16, 16], get_net(6, channels)),
        CouplingLayer('checkerboard1', [6, 16, 16], get_net(6, channels)),
        CouplingLayer('checkerboard0', [6, 16, 16], get_net(6, channels)),
        CouplingLayer('checkerboard1', [6, 16, 16], get_net(6, channels)),
    ]
    realnvp = RealNVP(model, dim)
    return realnvp


def get_mnist_realnvp():
    dim = 28**2
    channels = 32

    def get_net(in_channels, channels):
        net = nn.Sequential(
            nn.ReflectionPad2d(1),
            WNConv2d(in_channels, channels, 3),
            ResNetBlock(channels),
            ResNetBlock(channels),
            ResNetBlock(channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            WNConv2d(channels, in_channels * 2, 3),
            SplitAndNorm()
        )
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=1e-6)
                m.scale.data.fill_(1e-5)
        return net

    model = [
        CouplingLayer('checkerboard0', [1, 28, 28], get_net(1, channels)),
        CouplingLayer('checkerboard1', [1, 28, 28], get_net(1, channels)),
        CouplingLayer('checkerboard0', [1, 28, 28], get_net(1, channels)),
        SpaceToDepth(2),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        FactorOut([4, 14, 14]),
        CouplingLayer('checkerboard0', [2, 14, 14], get_net(2, channels)),
        CouplingLayer('checkerboard1', [2, 14, 14], get_net(2, channels)),
        CouplingLayer('checkerboard0', [2, 14, 14], get_net(2, channels)),
        CouplingLayer('checkerboard1', [2, 14, 14], get_net(2, channels)),
    ]
    realnvp = RealNVP(model, dim)
    return realnvp


class ConcatNet(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, in_channels, 3, padding=1),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(True),
            nn.Conv2d(channels, in_channels, 3, padding=1),
        )
        self.split = SplitAndNorm()

    def forward(self, x):
        x = torch.cat([self.net1(x), self.net2(x)], dim=1)
        return self.split(x)


def get_pie(channels=32):
    def get_net(in_channels, channels):
        net = ConcatNet(in_channels, channels)
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=1e-6)
                m.bias.data.fill_(0.)
        return net

    model = [
        SpaceToDepth(2),
        Housholder1x1Conv(4),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        ActNorm(4),
        Housholder1x1Conv(4),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        ActNorm(4),
        Housholder1x1Conv(4),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        ActNorm(4),
        Housholder1x1Conv(4),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        ActNorm(4),
        Housholder1x1Conv(4),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        ActNorm(4),
        Housholder1x1Conv(4),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        ActNorm(4),
        Housholder1x1Conv(4),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        ActNorm(4),
        Housholder1x1Conv(4),
        CouplingLayer('channel0', [4, 14, 14], get_net(4, channels)),
        CouplingLayer('channel1', [4, 14, 14], get_net(4, channels)),
        ActNorm(4),
        Housholder1x1Conv(4),
        SpaceToDepth(14),
        Housholder1x1Conv(784),
        CouplingLayer('channel0', [784, 1, 1], get_net(784, channels)),
        CouplingLayer('channel1', [784, 1, 1], get_net(784, channels)),
        ActNorm(784),
        Housholder1x1Conv(784),
        CouplingLayer('channel0', [784, 1, 1], get_net(784, channels)),
        CouplingLayer('channel1', [784, 1, 1], get_net(784, channels)),
        ActNorm(784),
        Housholder1x1Conv(784),
        CouplingLayer('channel0', [784, 1, 1], get_net(784, channels)),
        CouplingLayer('channel1', [784, 1, 1], get_net(784, channels)),
        ActNorm(784),
        Housholder1x1Conv(784),
        CouplingLayer('channel0', [784, 1, 1], get_net(784, channels)),
        CouplingLayer('channel1', [784, 1, 1], get_net(784, channels)),
        ActNorm(784),
        Housholder1x1Conv(784),
        CouplingLayer('channel0', [784, 1, 1], get_net(784, channels)),
        CouplingLayer('channel1', [784, 1, 1], get_net(784, channels)),
        ActNorm(784),
        Housholder1x1Conv(784),
        CouplingLayer('channel0', [784, 1, 1], get_net(784, channels)),
        CouplingLayer('channel1', [784, 1, 1], get_net(784, channels)),
        ActNorm(784),
        Housholder1x1Conv(784),
        CouplingLayer('channel0', [784, 1, 1], get_net(784, channels)),
        CouplingLayer('channel1', [784, 1, 1], get_net(784, channels)),
        ActNorm(784),
        Housholder1x1Conv(784),
        CouplingLayer('channel0', [784, 1, 1], get_net(784, channels)),
        CouplingLayer('channel1', [784, 1, 1], get_net(784, channels)),
        ActNorm(784),
        Housholder1x1Conv(784),
    ]

    dim = 784
    realnvp = RealNVP(model, dim)
    return realnvp


def get_realnvp(k, l, in_shape, channels, use_bn=False):
    dim = int(np.prod(in_shape))

    def get_net(in_channels, channels):
        net = nn.Sequential(
            nn.ReflectionPad2d(1),
            WNConv2d(in_channels, channels, 3),
            ResNetBlock(channels, use_bn),
            ResNetBlock(channels, use_bn),
            ResNetBlock(channels, use_bn),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            WNConv2d(channels, in_channels * 2, 3),
            SplitAndNorm()
        )
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=1e-6)
                m.scale.data.fill_(1e-5)
        return net

    shape = tuple(in_shape)
    model = []
    for _ in range(l):
        for i in range(k):
            model.append(Housholder1x1Conv(shape[0]))
            model.append(CouplingLayer('checkerboard{}'.format(i % 2), shape, get_net(shape[0], channels)))
        model += [SpaceToDepth(2)]
        shape = (shape[0] * 4, shape[1] // 2, shape[2] // 2)
        for i in range(k):
            model.append(Housholder1x1Conv(shape[0]))
            model.append(CouplingLayer('channel{}'.format(i % 2), shape, get_net(shape[0], channels)))
        model += [FactorOut(list(shape))]
        shape = (shape[0] // 2, shape[1], shape[2])

    model += [
        CouplingLayer('checkerboard0', shape, get_net(shape[0], channels)),
        CouplingLayer('checkerboard1', shape, get_net(shape[0], channels)),
        CouplingLayer('checkerboard0', shape, get_net(shape[0], channels)),
        CouplingLayer('checkerboard1', shape, get_net(shape[0], channels)),
    ]
    realnvp = RealNVP(model, dim)
    return realnvp


class MyModel(nn.Module):
    def __init__(self, flow, prior):
        super().__init__()
        self.flow = flow
        self.prior = prior

    def _flow_term(self, x):
        _, log_det, z = self.flow([x, None, None])

        # TODO: get rid of this
        if z.numel() != x.numel():
            log_det += self.flow.pie.residual()

        logp = log_det + self.prior.log_prob(z)
        return logp, z

    def log_prob(self, x):
        logp, z = self._flow_term(x)
        return logp + self.prior.log_prob(z)

    def log_prob_full(self, x):
        logp, z = self._flow_term(x)
        log_prior = torch.stack([self.prior.log_prob(z, k=k) for k in range(self.prior.k)])
        return logp[:, None] + log_prior.transpose(0, 1)


class MyPie(nn.Module):
    def __init__(self, pie):
        super().__init__()
        self.pie = pie

    def forward(self, x):
        x, _, _ = x
        z = self.pie(x)
        return None, self.pie.log_det(), z
