import datetime
import os
from pathlib import Path

import matplotlib
import pyro
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.BO.bayesianoptimisation import BayesianOptimizer
from src.blackboxpipe import BlackBoxPipe
from src.resnet import ResNet
from src.utils import load_npz_kmnist, get_git_revision_short_hash

# Seeding & githash for reproducibility.
pyro.set_rng_seed(0)
torch.manual_seed(0)
git_hash = get_git_revision_short_hash()

# (0) Setup your computation device / plotting method. ------------------------
TEST = False
ROOT_DATA = 'Data/Raw/'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

INIT_LAMB = -3  # 0.01
EPS = 0.
NOISE = 0.
# SEARCH_SPACE = (10e-5, 10e-1)
SEARCH_SPACE = (-5, -1)
GPCONFIG = dict(initial_var=0.5, initial_length=0.5, noise=0.)

if TEST:
    BUDGET = 4
    EPOCHS = 1
    BATCH_SIZE = 1

    resnet_config = dict(img_size=(28, 28),
                         architecture=((1, 2), (2, 2, 2)),
                         no_classes=10)

    matplotlib.use('TkAgg')

else:
    # FULLRUN CONFIG
    BATCH_SIZE = 5
    EPOCHS = 5

    BUDGET = 10

    resnet_config = dict(img_size=(28, 28),
                         architecture=(
                             (1, 8), (8, 16, 16), (16, 16, 16), (16, 16, 16),
                             (16, 32, 32), (32, 32, 32)),
                         no_classes=10)

    matplotlib.use('Agg')

# Define the Name of the RUN.
s = '{:%Y%m%d_%H%M%S}'
timestamp = s.format(datetime.datetime.now())
RUNIDX = 'run_{}_{}'.format(git_hash, timestamp)  # Run name

print(RUNIDX)

# (1) loading data & preprocessing according to
# https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py
# Load the data ---------------------------------------------------------------
x_train, x_test, y_train, y_test = load_npz_kmnist(
    folder=ROOT_DATA,
    files=['kmnist-train-imgs.npz', 'kmnist-test-imgs.npz',
           'kmnist-train-labels.npz', 'kmnist-test-labels.npz'])

if TEST:
    n = 100  # len(x_train)
    x_train = x_train[:n]
    y_train = y_train[:n]
    x_test = x_test[:int(n / 10)]
    y_test = y_test[:int(n / 10)]

# Plot an example image.
# plot_kmnist(x_train, y_train,
#             labelpath=root_data + 'kmnist_classmap.csv',
#             idx=2)

# (2) Adjust the Data & create datapipeline. ----------------------------------
# Adjust X to 0 - 1 range.
x_train /= 255.
x_test /= 255.

# Convert y float to int.
y_train = y_train.type(torch.LongTensor)
y_test = y_test.type(torch.LongTensor)

# Add channel information/dim (greyscale image).
x_train = torch.unsqueeze(x_train, dim=1)
x_test = torch.unsqueeze(x_test, dim=1)

# Descriptive info of the dataset.
print("y's shape: {}\nx's shape: {}".format(y_train.shape, x_train.shape))

# Create Dataset & dataloader for train & test.
trainset = TensorDataset(x_train, y_train)
testset = TensorDataset(x_test, y_test)

trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0)
testloader = DataLoader(
    testset, batch_size=1,
    shuffle=True, num_workers=0)

# (3) Model setup.
resnet = ResNet(**resnet_config)
resnet.to(DEVICE)

# (4) Create, track & run-config for a model with sgd under a specific
# learning rate.
root = os.getcwd()
modeldir = root + '/models/{}'.format(RUNIDX)
Path(modeldir).mkdir(parents=True, exist_ok=True)
pipe = BlackBoxPipe(
    resnet, trainloader, testloader, epochs=EPOCHS,
    path=modeldir, device=DEVICE)

# (5) Pass closure object to BO, which is the bridge from the model to bo.
bo_config = dict(
    search_space=SEARCH_SPACE,
    budget=BUDGET,
    noise=NOISE)
bo = BayesianOptimizer(
    **bo_config,
    # To optimize on log10 scale: lambda function
    closure=lambda x: pipe.evaluate_model_with_SGD(10 ** x))

bo.optimize(eps=EPS, initial_guess=INIT_LAMB, gp_config=GPCONFIG)
bo.plot_bo()

# Write out the final image.
bo.tracker.fig.savefig('{}/bo_{}.pdf'.format(modeldir, git_hash),
                       bbox_inches='tight')

# Write out the configs & interesting run-data.
bo.tracker.save(modeldir)
pipe.flush(modeldir)

if TEST:
    # RUN ONLY when on local machine
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.BO.botracker import BoTracker

    botracker = BoTracker.load(modeldir)
    botracker.plot_bo()

    root = os.getcwd()
    # filename = root + '/models/server_return/pickle_fullrun3/fullrun3.pkl'
    file = 'blackboxpipe.pkl'
    filename = '{}/models/{}/{}'.format(root, RUNIDX, file)
    with open(filename, 'rb') as handle:
        d = pickle.load(handle)

    plt.plot(np.arange(len(d['trainlosses'][0])),
             d['trainlosses'][0].detach().numpy())
    plt.show()

    plt.close()
    confused = d['confusion_matrices'][0].numpy()
    sns.heatmap(confused, annot=True)
    plt.show()
