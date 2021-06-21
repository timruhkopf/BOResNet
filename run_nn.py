"""
Author: Tim Ruhkopf
Email: timruhkopf@gmail.com
Purpose: This script is intended to check iff the ResNet Model really learns
    sth given sufficient capability.
"""

import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
import matplotlib
import os
import pickle
import datetime
from pathlib import Path

from src.resnet import ResNet
from src.utils import load_npz_kmnist, get_git_revision_short_hash
from src.blackboxpipe import BlackBoxPipe

matplotlib.use('Agg')

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

if TEST:
    BUDGET = 3
    EPOCHS = 1
    BATCH_SIZE = 1

    resnet_config = dict(img_size=(28, 28),
                         architecture=((1, 2), (2, 2, 2)),
                         no_classes=10)

else:
    # FULLRUN CONFIG
    EPOCHS = 10
    BATCH_SIZE = 8
    BUDGET = 10

    resnet_config = dict(img_size=(28, 28),
                         architecture=(
                             (1, 8), (8, 16, 16), (16, 16, 16), (16, 16, 16),
                             (16, 32, 32), (32, 32, 32)),
                         no_classes=10)

# Define the Name of the RUN.
s = '{:%Y%m%d_%H%M%S}'
timestamp = s.format(datetime.datetime.now())
RUNIDX = 'run_{}_{}'.format(git_hash, timestamp)  # Run name

# (1) loading data & preprocessing according to
# https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py
# Load the data ---------------------------------------------------------------
x_train, x_test, y_train, y_test = load_npz_kmnist(
    folder=ROOT_DATA,
    files=['kmnist-train-imgs.npz', 'kmnist-test-imgs.npz',
           'kmnist-train-labels.npz', 'kmnist-test-labels.npz'])

if TEST:
    n = 1001  # len(x_train)
    x_train = x_train[:n]
    y_train = y_train[:n]
    x_test = x_test[:int(n / 10)]
    y_test = y_test[:int(n / 10)]

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

# (3) Model setup. ------------------------------------------------------------
resnet = ResNet(**resnet_config)
resnet.to(DEVICE)

# (4) Create, track & run-config for a model with sgd under a specific
# learning rate.
# optionally create the directory
root = os.getcwd()
modeldir = root + '/models/{}'.format(RUNIDX)
Path(modeldir).mkdir(parents=True, exist_ok=True)

# create the model's training & testing protocol
pipe = BlackBoxPipe(
    resnet, trainloader, testloader, epochs=EPOCHS,
    path=modeldir, device=DEVICE)

# pipe.evaluate_model_with_SGD(0.003)
pipe.evaluate_model_with_SGD(0.001)

# remove the already written out model
del pipe.model
del pipe.trainloader
del pipe.testloader

filename = '{}/{}.pkl'.format(modeldir, RUNIDX)
with open(filename, 'wb') as handle:
    pickle.dump(pipe, handle, protocol=pickle.HIGHEST_PROTOCOL)

if TEST:
    # Pickle the tracker object
    # to be capable of instantating the class once more and load the info
    from src.blackboxpipe import BlackBoxPipe
    from src.resnet import ResNet
    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use('TkAgg')

    modelfolder = ''
    file = ''
    filename = '{}/{}'.format(modelfolder, file )
    with open(filename, 'rb') as handle:
        pipepickled = pickle.load(handle)

    # plot the confusion matrix of the models
    model_idx = 0
    c_mat = pipepickled.confusion_matrices[model_idx]
    confused = pd.DataFrame(c_mat.numpy())
    sns.heatmap(confused, annot=True)
    plt.show()
