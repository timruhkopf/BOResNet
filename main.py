import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
import matplotlib
import os
import pickle
from pathlib import Path

from src.resnet import ResNet
from src.utils import load_npz_kmnist, plot_kmnist
from src.blackboxpipe import BlackBoxPipe
from src.bo import BayesianOptimizer

matplotlib.use('Agg')

# Seeding for reproducibility.
pyro.set_rng_seed(0)
torch.manual_seed(0)

# (0) Setup your computation device / plotting method. ------------------------
TEST = False
RUNIDX = 2

BATCH_SIZE = 4
EPOCHS = 5
INIT_LAMB = 0.01
EPS = 0.
NOISE = 0.01
SEARCH_SPACE = (0.001, 0.02)
BUDGET = 10

ROOT_DATA = 'Data/Raw/'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Fullrun setup.
resnet_config = dict(img_size=(28, 28),
                     architecture=(
                         (1, 16), (16, 16, 16), (16, 16, 16), (16, 16, 16),
                         (16, 32, 32), (32, 32, 32), (32, 64, 64)),
                     no_classes=10)

if TEST:
    BUDGET = 3
    EPOCHS = 1
    BATCH_SIZE = 1
    # test_setup
    resnet_config = dict(img_size=(28, 28),
                         architecture=((1, 2), (2, 2, 2)),
                         no_classes=10)

# (1) loading data & preprocessing according to
# https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py
# Load the data
x_train, x_test, y_train, y_test = load_npz_kmnist(
    folder=ROOT_DATA,
    files=['kmnist-train-imgs.npz', 'kmnist-test-imgs.npz',
           'kmnist-train-labels.npz', 'kmnist-test-labels.npz'])

# perfectly balanced training /test datasets
# --> use accuracy as quality measure
# import pandas as pd
# pd.Series(y_train.numpy()).value_counts()
# pd.Series(y_test.numpy()).value_counts()


# Test training.
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
    testset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0)

# (3) Model setup.
resnet = ResNet(**resnet_config)
resnet.to(DEVICE)

# (4) Create, track & run-config for a model with sgd under a specific
# learning rate.
root = os.getcwd()
modeldir = root + '/models/fullrun{}'.format(RUNIDX)
Path(modeldir).mkdir(parents=True, exist_ok=True)
pipe = BlackBoxPipe(
    resnet, trainloader, testloader, epochs=EPOCHS,
    path=modeldir + '/model_', device=DEVICE)

# Check consecutive runs are tracked.
# pipe.evaluate_model_with_SGD(lr=0.001)
# pipe.evaluate_model_with_SGD(lr=0.005)

# (5) Pass closure object to BO, which is the bridge from the model to bo.
bo_config = dict(
    search_space=SEARCH_SPACE,
    budget=BUDGET)
bo = BayesianOptimizer(
    **bo_config,
    closure=pipe.evaluate_model_with_SGD)

bo.optimize(eps=EPS, initial_lamb=INIT_LAMB, noise=NOISE)

# Write out the final image.
root = os.getcwd()
bo.fig.savefig(root + '/Plots/bo_fullrun{}.pdf'.format(RUNIDX),
               bbox_inches='tight')

# Write out the configs & interesting run-data.
pickledict = dict(
    resnet_confit=resnet_config,
    bo_config=bo_config,
    losses=pipe.trainlosses,
    incumbent=bo.incumbent,
    costs=bo.cost,
    inquired=bo.inquired,
    accuracy=pipe.acc,
    bo_fig=bo.fig,
    bo_axes=bo.axes,
    bo_fig_handle=bo.fig_handle)

modeldir = root + '/models/pickle/'.format(RUNIDX)
Path(modeldir).mkdir(parents=True, exist_ok=True)
filename = modeldir + '/fullrun{}.pkl'.format(RUNIDX)
with open(filename, 'wb') as handle:
    pickle.dump(pickledict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if TEST:
    # Load pickle & analyse it.
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    with open(filename, 'rb') as handle:
        b = pickle.load(handle)

    print(b['incumbent'], '\n', b['inquired'])

    matplotlib.use('TkAgg')
    plt.plot(np.arange(len(b['losses'][0])),
             b['losses'][0].detach().numpy())
    plt.show()
