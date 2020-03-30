import myexman
import torch
from logger import Logger
import torchvision
import os
from torch import nn
import torch.nn.functional as F
import utils
import warnings
import numpy as np


parser = myexman.ExParser(file=os.path.basename(__file__))
parser.add_argument('--name', default='')
# Data
parser.add_argument('--data', default='')
parser.add_argument('--emb')
parser.add_argument('--dim', default=196, type=int)
# Optimization
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = Logger('logs', base=args.root)

if args.emb == "f":
    emb_train = np.load(os.path.join(args.data, 'zf_train.npy'))[:, -args.dim:]
    emb_test = np.load(os.path.join(args.data, 'zf_test.npy'))[:, -args.dim:]
elif args.emb == 'h':
    emb_train = np.load(os.path.join(args.data, 'zh_train.npy'))
    emb_test = np.load(os.path.join(args.data, 'zh_test.npy'))
else:
    raise NotImplementedError

y_train = np.load(os.path.join(args.data, 'y_train.npy'))
y_test = np.load(os.path.join(args.data, 'y_test.npy'))


trainset = torch.utils.data.TensorDataset(torch.FloatTensor(emb_train - emb_train.mean(0)[None]), torch.LongTensor(y_train))
testset = torch.utils.data.TensorDataset(torch.FloatTensor(emb_test - emb_test.mean(0)[None]), torch.LongTensor(y_test))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=256)
testloader = torch.utils.data.DataLoader(testset, batch_size=256)


net = nn.Sequential(
    nn.Linear(args.dim, 256),
    nn.LeakyReLU(),

    nn.Linear(256, 256),
    nn.Dropout(0.3),
    nn.LeakyReLU(),

    nn.Linear(256, 256),
    nn.LeakyReLU(),

    nn.Linear(256, 256),
    nn.LeakyReLU(),

    nn.Linear(256, len(np.unique(y_train))),
).to(device)

opt = torch.optim.Adam(net.parameters(), lr=args.lr)
lr_schedule = utils.LinearLR(opt, args.epochs)

for e in range(1, 1 + args.epochs):
    net.train()
    train_loss = 0.
    train_acc = 0.
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        p = net(x)
        loss = F.cross_entropy(p, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss += loss.item() * x.size(0)
        train_acc += (p.argmax(1) == y).sum().item()

    train_loss /= len(trainloader.dataset)
    train_acc /= len(trainloader.dataset)

    net.eval()
    x, y = map(lambda a: a.to(device), next(iter(testloader)))
    p = net(x)
    test_acc = (p.argmax(1) == y).float().mean().item()

    logger.add_scalar(e, 'train.loss', train_loss)
    logger.add_scalar(e, 'train.acc', train_acc)
    logger.add_scalar(e, 'test.acc', test_acc)
    logger.iter_info()
    logger.save()
