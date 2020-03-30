import myexman
import torch
import utils
import datautils
import os
from logger import Logger
import time
import numpy as np
from models import flows
import matplotlib.pyplot as plt
from models import distributions
import sys
from tqdm import tqdm


def get_logp(model, loader):
    logp = []
    for x, _ in loader:
        x = x.to(device)
        logp.append(utils.tonp(model.log_prob(x)))
    return np.concatenate(logp)


parser = myexman.ExParser(file=os.path.basename(__file__))
parser.add_argument('--name', default='')
parser.add_argument('--save_dir', default='')
# Data
parser.add_argument('--data', default='mnist')
parser.add_argument('--data_seed', default=0, type=int)
parser.add_argument('--aug', dest='aug', action='store_true')
parser.add_argument('--no_aug', dest='aug', action='store_false')
parser.set_defaults(aug=False)
# Optimization
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--lr_schedule', default='linear')
parser.add_argument('--lr_warmup', default=10, type=int)
parser.add_argument('--lr_gamma', default=0.5, type=float)
parser.add_argument('--lr_steps', type=int, nargs='*', default=[])
parser.add_argument('--log_each', default=1, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--pretrained', default='')
parser.add_argument('--weight_decay', default=0., type=float)
parser.add_argument('--clip_gv', default=1e9, type=float)
parser.add_argument('--clip_gn', default=100., type=float)
# Model
parser.add_argument('--model', default='flow')
parser.add_argument('--logits', dest='logits', action='store_true')
parser.add_argument('--no-logits', dest='logits', action='store_false')
parser.set_defaults(logits=True)
parser.add_argument('--conv', default='full')
parser.add_argument('--hh_factors', default=2, type=int)
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--l', default=2, type=int)
parser.add_argument('--hid_dim', type=int, nargs='*', default=[])
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

fmt = {
    'time': '.3f',
    'lr': '.1e',
}
logger = Logger('logs', base=args.root, fmt=fmt)

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
prior = distributions.GaussianDiag(dim).to(device)

flow = utils.create_flow(args, data_shape)
flow = torch.nn.DataParallel(flow.to(device))
model = flows.FlowPDF(flow, prior).to(device)

if args.pretrained is not None and args.pretrained != '':
    model.load_state_dict(torch.load(args.pretrained))


def get_embeddings(loader, model):
    zf, zh, labels = [], [], []
    for x, y in tqdm(loader):
        z_ = model.flow(x)[1]
        zf.append(utils.tonp(z_))
        labels.append(utils.tonp(y))
    return np.concatenate(zf), np.concatenate(labels)


with torch.no_grad():
    zf_train, y_train = get_embeddings(trainloader, model)
    zf_test, y_test = get_embeddings(testloader, model)


np.save(os.path.join(args.save_dir, 'zf_train'), zf_train)
np.save(os.path.join(args.save_dir, 'y_train'), y_train)

np.save(os.path.join(args.save_dir, 'zf_test'), zf_test)
np.save(os.path.join(args.save_dir, 'y_test'), y_test)
