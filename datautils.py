import numpy as np
import torch
from sklearn import datasets
import os
import torchvision
from torchvision import transforms
from models.distributions import GMM
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.utils.data
import PIL
from torchvision.datasets import ImageFolder
import warnings
import utils

DATA_ROOT = './'


mean = {
    'mnist': (0.1307,),
    'cifar10': (0.4914, 0.4822, 0.4465)
}

std = {
    'mnist': (0.3081,),
    'cifar10': (0.2470, 0.2435, 0.2616)
}


class UniformNoise(object):
    def __init__(self, bits=256):
        self.bits = bits

    def __call__(self, x):
        with torch.no_grad():
            noise = torch.rand_like(x)
            # TODO: generalize. x assumed to be normalized to [0, 1]
            return (x * (self.bits - 1) + noise) / self.bits

    def __repr__(self):
        return "UniformNoise"


def load_dataset(data, train_bs, test_bs, num_examples=None, data_root=DATA_ROOT, shuffle=True,
                 seed=42, supervised=-1, logs_root='', sup_sample_weight=-1, sup_only=False, device=None):
    bits = None
    sampler = None
    if data in ['moons', 'circles']:
        if data == 'moons':
            x, y = datasets.make_moons(n_samples=int(num_examples * 1.5), noise=0.1, random_state=seed)
            train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=num_examples, random_state=seed)
        elif data == 'circles':
            x, y = datasets.make_circles(n_samples=int(num_examples * 1.5), noise=0.1, factor=0.2, random_state=seed)
            train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=num_examples, random_state=seed)

        if supervised not in [-1, len(train_y), 0]:
            unsupervised_idxs, _ = train_test_split(np.arange(len(train_y)), test_size=supervised, stratify=train_y)
            train_y[unsupervised_idxs] = -1
        elif supervised == 0:
            train_y[:] = -1

        torch.save({
            'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y,
        }, os.path.join(logs_root, 'data.torch'))

        trainset = torch.utils.data.TensorDataset(torch.FloatTensor(train_x[..., None, None]),
                                                  torch.LongTensor(train_y))
        testset = torch.utils.data.TensorDataset(torch.FloatTensor(test_x[..., None, None]),
                                                 torch.LongTensor(test_y))
        data_shape = [2, 1, 1]
        bits = np.nan
    elif data == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            UniformNoise(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            UniformNoise(),
        ])
        trainset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=test_transform)

        if num_examples != -1 and num_examples != len(trainset) and num_examples is not None:
            idxs, _ = train_test_split(np.arange(len(trainset)), train_size=num_examples, random_state=seed,
                                       stratify=utils.tonp(trainset.targets))
            trainset.data = trainset.data[idxs]
            trainset.targets = trainset.targets[idxs]

        if supervised == 0:
            trainset.targets[:] = -1
        elif supervised != -1:
            unsupervised_idxs, _ = train_test_split(np.arange(len(trainset.targets)),
                                                    test_size=supervised, stratify=trainset.targets)
            trainset.targets[unsupervised_idxs] = -1

        if sup_only:
            mask = trainset.targets != -1
            trainset.targets = trainset.targets[mask]
            trainset.data = trainset.data[mask]

        data_shape = (1, 28, 28)
        bits = 256
    else:
        raise NotImplementedError

    nw = 2
    if sup_sample_weight == -1:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=shuffle,
                                                  num_workers=nw, pin_memory=True)
    else:
        sampler = ImbalancedDatasetSampler(trainset, sup_weight=sup_sample_weight)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, sampler=sampler,
                                                  num_workers=nw, pin_memory=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False,
                                             num_workers=nw, pin_memory=True)
    return trainloader, testloader, data_shape, bits


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/sampler.py
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None, num_samples=None, sup_weight=1.):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {
            -1: 0
        }
        sup = 0
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label == -1:
                label_to_count[-1] += 1
            else:
                sup += 1
                label_to_count[label] = sup
        for k in label_to_count:
            if k != -1:
                label_to_count[k] = sup

        # weight for each sample
        weights = []
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            w = 1 if label == -1 else sup_weight
            weights.append(w / label_to_count[label])

        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.targets[idx].item()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class FastMNIST(torchvision.datasets.MNIST):
    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).float().div(255)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target
