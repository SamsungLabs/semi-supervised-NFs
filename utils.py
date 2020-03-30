import numpy as np
import torch
from sklearn import datasets
import os
import torchvision
from torchvision import transforms
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import warnings
from models import flows, coupling


def viz_array_grid(array, rows, cols, padding=0, channels_last=False, normalize=False, **kwargs):
    # normalization
    '''
    Args:
        array: (N_images, N_channels, H, W) or (N_images, H, W, N_channels)
        rows, cols: rows and columns of the plot. rows * cols == array.shape[0]
        padding: padding between cells of plot
        channels_last: for Tensorflow = True, for PyTorch = False
        normalize: `False`, `mean_std`, or `min_max`
    Kwargs:
        if normalize == 'mean_std':
            mean: mean of the distribution. Default 0.5
            std: std of the distribution. Default 0.5
        if normalize == 'min_max':
            min: min of the distribution. Default array.min()
            max: max if the distribution. Default array.max()
    '''
    if not channels_last:
        array = np.transpose(array, (0, 2, 3, 1))

    array = array.astype('float32')

    if normalize:
        if normalize == 'mean_std':
            mean = kwargs.get('mean', 0.5)
            mean = np.array(mean).reshape((1, 1, 1, -1))
            std = kwargs.get('std', 0.5)
            std = np.array(std).reshape((1, 1, 1, -1))
            array = array * std + mean
        elif normalize == 'min_max':
            min_ = kwargs.get('min', array.min())
            min_ = np.array(min_).reshape((1, 1, 1, -1))
            max_ = kwargs.get('max', array.max())
            max_ = np.array(max_).reshape((1, 1, 1, -1))
            array -= min_
            array /= max_ + 1e-9

    batch_size, H, W, channels = array.shape
    assert rows * cols == batch_size

    if channels == 1:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1)))
        array = array[:, :, :, 0]
    elif channels == 3:
        canvas = np.ones((H * rows + padding * (rows - 1),
                          W * cols + padding * (cols - 1),
                          3))
    else:
        raise TypeError('number of channels is either 1 of 3')

    for i in range(rows):
        for j in range(cols):
            img = array[i * cols + j]
            start_h = i * padding + i * H
            start_w = j * padding + j * W
            canvas[start_h: start_h + H, start_w: start_w + W] = img

    canvas = np.clip(canvas, 0, 1)
    canvas *= 255.0
    canvas = canvas.astype('uint8')
    return canvas


def params_norm(parameters):
    sq = 0.
    n = 0
    for p in parameters:
        sq += (p**2).sum()
        n += torch.numel(p)
    return np.sqrt(sq.item() / float(n))


def tonp(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def batch_eval(f, loader):
    res = []
    for x in loader:
        res.append(f(x))
    return res


def bits_dim(ll, dim, bits=256):
    return np.log2(bits) - ll / dim / np.log(2)


class LinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, num_epochs, last_epoch=-1):
        self.num_epochs = max(num_epochs, 1)
        super(LinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        res = []
        for lr in self.base_lrs:
            res.append(np.maximum(lr * np.minimum(-(self.last_epoch + 1) * 1. / self.num_epochs + 1., 1.), 0.))
        return res


class HatLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warm_up, num_epochs, last_epoch=-1):
        if warm_up == 0:
            warnings.warn('====> HatLR with warm_up=0 !!! <====')

        self.num_epochs = max(num_epochs, 1)
        self.warm_up = warm_up
        self.warm_schedule = LinearLR(optimizer, warm_up + 1)
        self.warm_schedule.step()
        self.anneal_schedule = LinearLR(optimizer, num_epochs - warm_up)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch + 1 < self.warm_up:
            return [lr - x for lr, x in zip(self.base_lrs, self.warm_schedule.get_lr())]
        return self.anneal_schedule.get_lr()

    def step(self, epoch=None):
        super().step(epoch=epoch)
        if self.last_epoch + 1 < self.warm_up:
            self.warm_schedule.step()
        else:
            self.anneal_schedule.step()

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class BaseLR(torch.optim.lr_scheduler._LRScheduler):
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def init_kmeans(k, dataloader, model=None, epochs=1, device=None):
    kmeans = MiniBatchKMeans(k, batch_size=dataloader.batch_size)
    for _ in range(epochs):
        for x, _ in dataloader:
            if model:
                x = model(x.to(device))
            x = tonp(x)
            kmeans.partial_fit(x)

    mu = kmeans.cluster_centers_
    dim = mu.shape[1]
    cov = np.zeros((k, dim, dim))
    n = np.zeros((k,))
    for x, _ in dataloader:
        if model:
            x = model(x.to(device))
        x = tonp(x)
        labels = kmeans.predict(x)
        for k in range(k):
            c = labels == k
            n[k] += np.sum(c)
            d = x[c] - mu[None, k]
            cov[k] += np.matmul(d[..., None], d[:, None]).sum(0)
    cov /= n[:, None, None]
    pi = n / n.sum()
    return mu, cov, pi


def create_flow(args, data_shape):
    if args.model == 'toy':
        flow = flows.toy2d_flow(args.conv, args.hh_factors, args.l)
    elif args.model == 'id':
        flow = flows.Flow([])
    elif args.model == 'mnist':
        flow = flows.mnist_flow(num_layers=args.l, k_factor=args.k, logits=args.logits,
                                conv=args.conv, hh_factors=args.hh_factors, hid_dim=args.hid_dim)
    elif args.model == 'mnist-masked':
        flow = flows.mnist_masked_glow(conv=args.conv, hh_factors=args.hh_factors)
    elif args.model == 'ffjord':
        # TODO: add FFJORD model
        raise NotImplementedError
    else:
        raise NotImplementedError

    return flow


def create_cond_flow(args):
    if args.ssl_model == 'cond-flow':
        flow = flows.get_flow_cond(args.ssl_l, args.ssl_k, in_channels=args.ssl_dim, hid_dim=args.ssl_hd,
                                   conv=args.ssl_conv, hh_factors=args.ssl_hh, num_cat=args.ssl_nclasses)
    elif args.ssl_model == 'cond-shift':
        flow = flows.ConditionalFlow([
            coupling.ConditionalShift(args.ssl_dim, args.ssl_nclasses)
        ])
    return flow


class MovingMetric(object):
    def __init__(self):
        self.n = 0
        self.sum = 0.

    def add(self, x):
        assert np.ndim(x) == 1
        self.n += len(x)
        self.sum += np.sum(x)

    def avg(self):
        return self.sum / self.n if self.n != 0 else np.nan
