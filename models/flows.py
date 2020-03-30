import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from models.normalization import ActNorm, DummyCondActNorm
from models.invertconv import InvertibleConv2d, HHConv2d_1x1, QRInvertibleConv2d, DummyCondInvertibleConv2d
from models.coupling import CouplingLayer, MaskedCouplingLayer, ConditionalCouplingLayer, ConditionalMaskedCouplingLayer
from models.utils import Conv2dZeros, SpaceToDepth, FactorOut, ToLogits, CondToLogits, CondFactorOut, CondSpaceToDepth
from models.utils import DummyCond, IdFunction
import warnings


class Flow(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_ = nn.ModuleList(modules)
        self.latent_len = -1
        self.x_shape = -1

    def f(self, x):
        z = None
        log_det_jac = torch.zeros((x.shape[0],)).to(x.device)
        for m in self.modules_:
            x, log_det_jac, z = m(x, log_det_jac, z)
        if z is None:
            z = torch.zeros((x.shape[0], 1))[:, :0].to(x.device)
        self.x_shape = list(x.shape)[1:]
        self.latent_len = z.shape[1]
        z = torch.cat([z, x.reshape((x.shape[0], -1))], dim=1)
        return log_det_jac, z

    def forward(self, x):
        return self.f(x)

    def g(self, z):
        x = z[:, self.latent_len:].view([z.shape[0]] + self.x_shape)
        z = z[:, :self.latent_len]
        for m in reversed(self.modules_):
            x, z = m.g(x, z)
        return x


class ConditionalFlow(Flow):
    def f(self, x, y):
        z = None
        log_det_jac = torch.zeros((x.shape[0],)).to(x.device)
        for m in self.modules_:
            x, log_det_jac, z = m(x, y, log_det_jac, z)
        if z is None:
            z = torch.zeros((x.shape[0], 1))[:, :0].to(x.device)
        self.x_shape = list(x.shape)[1:]
        self.latent_len = z.shape[1]
        z = torch.cat([z, x.reshape((x.shape[0], -1))], dim=1)
        return log_det_jac, z

    def g(self, z, y):
        x = z[:, self.latent_len:].view([z.shape[0]] + self.x_shape)
        z = z[:, :self.latent_len]
        for m in reversed(self.modules_):
            x, z = m.g(x, y, z)
        return x

    def forward(self, x, y):
        return self.f(x, y)


class DiscreteConditionalFlow(ConditionalFlow):
    def __init__(self, modules, num_cat, emb_dim):
        super().__init__(modules)
        self.embeddings = nn.Embedding(num_cat, emb_dim)

        self.embeddings.weight.data.zero_()
        l = torch.arange(self.embeddings.weight.data.shape[0])
        self.embeddings.weight.data[l, l] = 1.

    def f(self, x, y):
        return super().f(x, self.embeddings(y))

    def g(self, z, y):
        return super().g(z, self.embeddings(y))


class FlowPDF(nn.Module):
    def __init__(self, flow, prior):
        super().__init__()
        self.flow = flow
        self.prior = prior

    def log_prob(self, x):
        log_det, z = self.flow(x)
        return log_det + self.prior.log_prob(z)


class DeepConditionalFlowPDF(nn.Module):
    def __init__(self, flow, deep_prior, yprior, deep_dim, shallow_prior=None):
        super().__init__()
        self.flow = flow
        self.shallow_prior = shallow_prior
        self.deep_prior = deep_prior
        self.yprior = yprior
        self.deep_dim = deep_dim

    def log_prob(self, x, y):
        if x.dim() == 2:
            x = x[..., None, None]
        if self.deep_dim == x.shape[1]:
            log_det, z = self.flow(x, y)
            return log_det + self.deep_prior.log_prob(z)
        else:
            log_det, z = self.flow(x[:, -self.deep_dim:], y)
            return log_det + self.deep_prior.log_prob(z) + self.shallow_prior.log_prob(x[:, :-self.deep_dim].squeeze())

    def log_prob_joint(self, x, y):
        return self.log_prob(x, y) + self.yprior.log_prob(y)


class ConditionalFlowPDF(nn.Module):
    def __init__(self, flow, prior, emb=True):
        super().__init__()
        self.flow = flow
        self.prior = prior

    def log_prob(self, x, y):
        log_det, z = self.flow(x, y)
        return log_det + self.prior.log_prob(z)


class DiscreteConditionalFlowPDF(DeepConditionalFlowPDF):
    def log_prob_full(self, x):
        sup = self.yprior.enumerate_support().to(x.device)
        logp = []

        n_uniq = sup.size(0)
        y = sup.repeat((x.size(0), 1)).t().reshape((1, -1)).t()[:, 0]
        logp = self.log_prob(x.repeat([n_uniq] + [1]*(len(x.shape)-1)), y)
        return logp.reshape((n_uniq, x.size(0))).t() + self.yprior.log_prob(sup[None])

    def log_prob(self, x, y=None):
        if y is not None:
            return super().log_prob(x, y)
        else:
            logp_joint = self.log_prob_full(x)
            return torch.logsumexp(logp_joint, dim=1)

    def log_prob_posterior(self, x):
        logp_joint = self.log_prob_full(x)
        return logp_joint - torch.logsumexp(logp_joint, dim=1)[:, None]


class ResNetBlock(nn.Module):
    def __init__(self, channels, use_bn=False):
        super().__init__()
        modules = []
        if use_bn:
            # modules.append(nn.BatchNorm2d(channels))
            ActNorm(channels, flow=False)
        modules += [
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3)]
        if use_bn:
            # modules.append(nn.BatchNorm2d(channels))
            ActNorm(channels, flow=False)
        modules += [
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3)]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x) + x


class ResNetBlock1x1(nn.Module):
    def __init__(self, channels, use_bn=False):
        super().__init__()
        modules = []
        if use_bn:
            ActNorm(channels, flow=False)
        modules += [
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)]
        if use_bn:
            ActNorm(channels, flow=False)
        modules += [
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x) + x


def get_resnet1x1(in_channels, channels, out_channels=None):
    if out_channels is None:
        out_channels = in_channels * 2
    net = nn.Sequential(
        nn.Conv2d(in_channels, channels, 1, padding=0),
        ResNetBlock1x1(channels, use_bn=True),
        ResNetBlock1x1(channels, use_bn=True),
        ResNetBlock1x1(channels, use_bn=True),
        ResNetBlock1x1(channels, use_bn=True),
        ActNorm(channels, flow=False),
        nn.ReLU(),
        Conv2dZeros(channels, out_channels, 1, padding=0),
    )
    return net


def get_resnet(in_channels, channels, out_channels=None):
    if out_channels is None:
        out_channels = in_channels * 2
    net = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, channels, 3),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ActNorm(channels, flow=False),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        Conv2dZeros(channels, out_channels, 3, 0),
    )
    return net


def get_resnet8(in_channels, channels, out_channels=None):
    if out_channels is None:
        out_channels = in_channels * 2
    net = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, channels, 3),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ActNorm(channels, flow=False),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        Conv2dZeros(channels, out_channels, 3, 0),
    )
    return net


def netfunc_for_coupling(in_channels, hidden_channels, out_channels, k=3):
    def foo():
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, k, padding=int(k == 3)),
            nn.ReLU(False),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(False),
            Conv2dZeros(hidden_channels, out_channels, k, padding=int(k == 3))
        )

    return foo


def get_flow(num_layers, k_factor, in_channels=1, hid_dim=[256], conv='full', hh_factors=2,
             cond=False, emb_dim=10, n_cat=10, net='shallow'):
    modules = [
        DummyCond(ToLogits()) if cond else ToLogits(),
    ]
    channels = in_channels

    if conv == 'full':
        convf = lambda x: InvertibleConv2d(x)
    elif conv == 'hh':
        convf = lambda x: HHConv2d_1x1(x, factors=[x]*hh_factors)
    elif conv == 'qr':
        convf = lambda x: QRInvertibleConv2d(x, factors=[x]*hh_factors)
    elif conv == 'qr-abs':
        convf = lambda x: QRInvertibleConv2d(x, factors=[x]*hh_factors, act='no')
    elif conv == 'no':
        convf = lambda x: IdFunction()
    else:
        raise NotImplementedError

    if net == 'shallow':
        couplingnetf = lambda x, y: netfunc_for_coupling(x, hid_dim[0], y)
    elif net == 'resnet':
        couplingnetf = lambda x, y: lambda: get_resnet(x, hid_dim[0], out_channels=y)
    else:
        raise NotImplementedError

    for l in range(num_layers):
        # TODO: FIX
        warnings.warn('==== "get_flow" reduce spatial dimensions only 4 times!!! ====')
        if l != 4:
            if cond:
                modules.append(DummyCond(SpaceToDepth(2)))
            else:
                modules.append(SpaceToDepth(2))
            channels *= 4
        for k in range(k_factor):
            if cond:
                modules.append(DummyCond(ActNorm(channels)))
                modules.append(DummyCond(convf(channels)))
                modules.append(ConditionalCouplingLayer(couplingnetf(channels // 2 + emb_dim, channels)))
            else:
                modules.append(ActNorm(channels))
                modules.append(convf(channels))
                modules.append(CouplingLayer(couplingnetf(channels // 2, channels)))

        if l != num_layers - 1:
            if cond:
                modules.append(DummyCond(FactorOut()))
            else:
                modules.append(FactorOut())

            channels //= 2
            channels -= channels % 2

    return DiscreteConditionalFlow(modules, n_cat, emb_dim) if cond else Flow(modules)


def get_flow_cond(num_layers, k_factor, in_channels=1, hid_dim=256, conv='full', hh_factors=2, num_cat=10, emb_dim=10):
    modules = []
    channels = in_channels
    for l in range(num_layers):
        for k in range(k_factor):
            modules.append(DummyCondActNorm(channels))
            if conv == 'full':
                modules.append(DummyCondInvertibleConv2d(channels))
            elif conv == 'hh':
                modules.append(DummyCond(HHConv2d_1x1(channels, factors=[channels]*hh_factors)))
            elif conv == 'qr':
                modules.append(DummyCond(QRInvertibleConv2d(channels, factors=[channels]*hh_factors)))
            elif conv == 'qr-abs':
                modules.append(DummyCond(QRInvertibleConv2d(channels, factors=[channels]*hh_factors, act='no')))
            else:
                raise NotImplementedError

            netf = lambda: get_resnet1x1(channels//2 + emb_dim, hid_dim, channels)
            modules.append(ConditionalCouplingLayer(netf))

        if l != num_layers - 1:
            modules.append(CondFactorOut())
            channels //= 2
            channels -= channels % 2

    return DiscreteConditionalFlow(modules, num_cat, emb_dim)


def mnist_flow(num_layers=5, k_factor=4, logits=True, conv='full', hh_factors=2, hid_dim=[32, 784]):
    modules = []
    if logits:
        modules.append(ToLogits())

    channels = 1
    hd = hid_dim[0]
    kernel = 3
    for l in range(num_layers):
        if l < 2:
            modules.append(SpaceToDepth(2))
            channels *= 4
        elif l == 2:
            modules.append(SpaceToDepth(7))
            channels *= 49
            hd = hid_dim[1]
            kernel = 1

        for k in range(k_factor):
            modules.append(ActNorm(channels))
            if conv == 'full':
                modules.append(InvertibleConv2d(channels))
            elif conv == 'hh':
                modules.append(HHConv2d_1x1(channels, factors=[channels]*hh_factors))
            elif conv == 'qr':
                modules.append(QRInvertibleConv2d(channels, factors=[channels]*hh_factors))
            elif conv == 'qr-abs':
                modules.append(QRInvertibleConv2d(channels, factors=[channels]*hh_factors, act='no'))
            else:
                raise NotImplementedError
            modules.append(CouplingLayer(netfunc_for_coupling(channels, hd, k=kernel)))

        if l != num_layers - 1:
            modules.append(FactorOut())
            channels //= 2
            channels -= channels % 2

    return Flow(modules)


def mnist_masked_glow(conv='full', hh_factors=2):
    def get_net(in_channels, channels):
        net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, channels, 3),
            ResNetBlock(channels, use_bn=True),
            ResNetBlock(channels, use_bn=True),
            ResNetBlock(channels, use_bn=True),
            ResNetBlock(channels, use_bn=True),
            ActNorm(channels, flow=False),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            Conv2dZeros(channels, in_channels * 2, 3, 0),
        )
        return net

    if conv == 'full':
        convf = lambda x: InvertibleConv2d(x)
    elif conv == 'qr':
        convf = lambda x: QRInvertibleConv2d(x, [x]*hh_factors)
    elif conv == 'hh':
        convf = lambda x: HHConv2d_1x1(x, [x]*hh_factors)
    else:
        raise NotImplementedError

    modules = [
        ToLogits(),
        convf(1),
        MaskedCouplingLayer([1, 28, 28], 'checkerboard0', get_net(1, 64)),
        ActNorm(1),
        convf(1),
        MaskedCouplingLayer([1, 28, 28], 'checkerboard1', get_net(1, 64)),
        ActNorm(1),
        convf(1),
        MaskedCouplingLayer([1, 28, 28], 'checkerboard0', get_net(1, 64)),
        ActNorm(1),
        SpaceToDepth(2),
        convf(4),
        CouplingLayer(lambda: get_net(2, 64)),
        ActNorm(4),
        convf(4),
        CouplingLayer(lambda: get_net(2, 64)),
        ActNorm(4),

        FactorOut(),

        convf(2),
        MaskedCouplingLayer([2, 14, 14], 'checkerboard0', get_net(2, 64)),
        ActNorm(2),
        convf(2),
        MaskedCouplingLayer([2, 14, 14], 'checkerboard1', get_net(2, 64)),
        ActNorm(2),
        convf(2),
        MaskedCouplingLayer([2, 14, 14], 'checkerboard0', get_net(2, 64)),
        ActNorm(2),
        SpaceToDepth(2),
        convf(8),
        CouplingLayer(lambda: get_net(4, 64)),
        ActNorm(8),
        convf(8),
        CouplingLayer(lambda: get_net(4, 64)),
        ActNorm(8),

        FactorOut(),

        convf(4),
        MaskedCouplingLayer([4, 7, 7], 'checkerboard0', get_net(4, 64)),
        ActNorm(4),
        convf(4),
        MaskedCouplingLayer([4, 7, 7], 'checkerboard1', get_net(4, 64)),
        ActNorm(4),
        convf(4),
        MaskedCouplingLayer([4, 7, 7], 'checkerboard0', get_net(4, 64)),
        ActNorm(4),
        convf(4),
        CouplingLayer(lambda: get_net(2, 64)),
        ActNorm(4),
        convf(4),
        CouplingLayer(lambda: get_net(2, 64)),
        ActNorm(4),
    ]

    return Flow(modules)


def toy2d_flow(conv='full', hh_factors=2, l=5):
    def netf():
        return nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 2, 1)
        )

    if conv == 'full':
        convf = lambda x: InvertibleConv2d(x)
    elif conv == 'qr':
        convf = lambda x: QRInvertibleConv2d(x, [x]*hh_factors)
    elif conv == 'hh':
        convf = lambda x: HHConv2d_1x1(x, [x]*hh_factors)
    else:
        raise NotImplementedError

    modules = []
    for _ in range(l):
        modules.append(convf(2))
        modules.append(CouplingLayer(netf))
        modules.append(ActNorm(2))
    return Flow(modules)
