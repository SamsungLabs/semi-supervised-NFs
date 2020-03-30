import myexman
import torch
import utils
import datautils
import os
from logger import Logger
import time
import numpy as np
from models import flows, distributions
import matplotlib.pyplot as plt
from algo.em import init_kmeans2plus_mu
import warnings
from sklearn.mixture import GaussianMixture
import torch.nn.functional as F
from tqdm import tqdm
import sys


def get_metrics(model, loader):
    logp, acc = [], []
    for x, y in loader:
        x = x.to(device)
        log_det, z = model.flow(x)
        log_prior_full = model.prior.log_prob_full(z)
        pred = torch.softmax(log_prior_full, dim=1).argmax(1)
        logp.append(utils.tonp(log_det + model.prior.log_prob(z)))
        acc.append(utils.tonp(pred) == utils.tonp(y))
    return np.mean(np.concatenate(logp)), np.mean(np.concatenate(acc))


parser = myexman.ExParser(file=os.path.basename(__file__))
parser.add_argument('--name', default='')
parser.add_argument('--verbose', default=0, type=int)
parser.add_argument('--save_dir', default='')
parser.add_argument('--test_mode', default='')
# Data
parser.add_argument('--data', default='mnist')
parser.add_argument('--num_examples', default=-1, type=int)
parser.add_argument('--data_seed', default=0, type=int)
parser.add_argument('--sup_sample_weight', default=-1, type=float)
# parser.add_argument('--aug', dest='aug', action='store_true')
# parser.add_argument('--no_aug', dest='aug', action='store_false')
# parser.set_defaults(aug=True)
# Optimization
parser.add_argument('--opt', default='adam')
parser.add_argument('--ssl_alg', default='em')
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--lr_schedule', default='linear')
parser.add_argument('--lr_warmup', default=10, type=int)
parser.add_argument('--lr_gamma', default=0.5, type=float)
parser.add_argument('--lr_steps', type=int, nargs='*', default=[])
parser.add_argument('--log_each', default=1, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pretrained', default='')
parser.add_argument('--weight_decay', default=0., type=float)
parser.add_argument('--sup_ohe', dest='sup_ohe', action='store_true')
parser.add_argument('--no_sup_ohe', dest='sup_ohe', action='store_false')
parser.set_defaults(sup_ohe=True)
parser.add_argument('--clip_gn', default=100., type=float)
# Model
parser.add_argument('--model', default='flow')
parser.add_argument('--logits', dest='logits', action='store_true')
parser.add_argument('--no_logits', dest='logits', action='store_false')
parser.set_defaults(logits=True)
parser.add_argument('--conv', default='full')
parser.add_argument('--hh_factors', default=2, type=int)
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--l', default=2, type=int)
parser.add_argument('--hid_dim', type=int, nargs='*', default=[])
# Prior
parser.add_argument('--ssl_model', default='cond-flow')
parser.add_argument('--ssl_dim', default=-1, type=int)
parser.add_argument('--ssl_l', default=2, type=int)
parser.add_argument('--ssl_k', default=4, type=int)
parser.add_argument('--ssl_hd', default=256, type=int)
parser.add_argument('--ssl_conv', default='full')
parser.add_argument('--ssl_hh', default=2, type=int)
parser.add_argument('--ssl_nclasses', default=10, type=int)
# SSL
parser.add_argument('--supervised', default=0, type=int)
parser.add_argument('--sup_weight', default=1., type=float)
parser.add_argument('--cl_weight', default=0, type=float)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load data
np.random.seed(args.data_seed)
torch.manual_seed(args.data_seed)
torch.cuda.manual_seed_all(args.data_seed)
trainloader, testloader, data_shape, bits = datautils.load_dataset(args.data, args.train_bs, args.test_bs,
                                                                   seed=args.data_seed, shuffle=False)

# Seed for training process
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Create model
dim = int(np.prod(data_shape))
if args.ssl_dim == -1:
    args.ssl_dim = dim
deep_prior = distributions.GaussianDiag(args.ssl_dim)
shallow_prior = distributions.GaussianDiag(dim - args.ssl_dim)
yprior = torch.distributions.Categorical(logits=torch.zeros((args.ssl_nclasses,)).to(device))
ssl_flow = flows.get_flow_cond(args.ssl_l, args.ssl_k, in_channels=args.ssl_dim, hid_dim=args.ssl_hd,
                               conv=args.ssl_conv, hh_factors=args.ssl_hh, num_cat=args.ssl_nclasses)
ssl_flow = torch.nn.DataParallel(ssl_flow.to(device))
prior = flows.DiscreteConditionalFlowPDF(ssl_flow, deep_prior, yprior, deep_dim=args.ssl_dim,
                                         shallow_prior=shallow_prior)

flow = utils.create_flow(args, data_shape)
flow = torch.nn.DataParallel(flow.to(device))

model = flows.FlowPDF(flow, prior).to(device)

if args.pretrained != '':
    model.load_state_dict(torch.load(os.path.join(args.pretrained, 'model.torch'), map_location=device))


# def get_embeddings(loader, model):
#     zf, zh, labels = [], [], []
#     for x, y in loader:
#         x = x.to(device)
#         print(model.log_prob(x).mean())
#         z_ = flow(x)[1]
#         zf.append(utils.tonp(z_).mean())
#         # z_ = z_[:, -args.ssl_dim:, None, None]
#         print(model.prior.log_prob(z_))
#         print(torch.zeros((z_.shape[0],)).to(z_.device))
#         print(model.prior.flow(z_, y))
#         # print(x.device, z_.device)
#         # print(torch.zeros((z_.shape[0],)).to(z_.device))
#         # print(z_.shape)
#         sys.exit(0)
#         log_det_jac = torch.zeros((x.shape[0],)).to(x.device)
#         ssl_flow.module.f(z_, y.to(device))
#         zh.append(utils.tonp())
#         labels.append(utils.tonp(y))
#     return np.concatenate(zf), np.concatenate(zh), np.concatenate(labels)

def get_embeddings(loader, model):
    zf, zh, labels = [], [], []
    for x, y in tqdm(loader):
        z_ = model.flow(x)[1]
        zf.append(utils.tonp(z_))
        zh.append(utils.tonp(model.prior.flow(z_[:, -args.ssl_dim:, None, None], y)[1]))
        labels.append(utils.tonp(y))
    return np.concatenate(zf), np.concatenate(zh), np.concatenate(labels)


y_test = np.array(testloader.dataset.targets)
y_train = np.array(trainloader.dataset.targets)

if args.test_mode == 'perm':
    idxs = np.random.permutation(10000)[:5000]
    testloader.dataset.data[idxs] = 255 - testloader.dataset.data[idxs]
    testloader.dataset.targets[idxs] = 1 - testloader.dataset.targets[idxs]
elif args.test_mode == '':
    pass
elif args.test_mode == 'inv':
    testloader.dataset.data = 255 - testloader.dataset.data
    testloader.dataset.targets = 1 - testloader.dataset.targets
else:
    raise NotImplementedError

with torch.no_grad():
    zf_train, zh_train, _ = get_embeddings(trainloader, model)
    zf_test, zh_test, _ = get_embeddings(testloader, model)


np.save(os.path.join(args.save_dir, 'zf_train'), zf_train)
np.save(os.path.join(args.save_dir, 'zh_train'), zh_train)
np.save(os.path.join(args.save_dir, 'y_train'), y_train)

np.save(os.path.join(args.save_dir, 'zf_test'), zf_test)
np.save(os.path.join(args.save_dir, 'zh_test'), zh_test)
np.save(os.path.join(args.save_dir, 'y_test'), y_test)
