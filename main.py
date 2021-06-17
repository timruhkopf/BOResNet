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

# setup your computation device / plotting method
matplotlib.use('Agg')
BATCH_SIZE = 4
ROOT_DATA = 'Data/Raw/'
RUNIDX = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

bo_config = dict(
    search_space=(0.001, 0.02),
    budget=10)

resnet_config = dict(img_size=(28, 28),
                     architecture=(
                         (1, 16), (16, 16, 16), (16, 16, 16), (16, 32, 32),
                         (32, 32, 32), (32, 64, 64)),
                     no_classes=10)

# seeding for reproducibility
pyro.set_rng_seed(0)
torch.manual_seed(0)

# (0) loading data & preporcessing according to
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

# testing if training starts at all

# for testing purposes
# n = 100  # len(x_train)
# x_train = x_train[:n]
# y_train = y_train[:n]
# x_test = x_test[:int(n / 10)]
# y_test = y_test[:int(n / 10)]

# plot an example image
# plot_kmnist(x_train, y_train,
#             labelpath=root_data + 'kmnist_classmap.csv',
#             idx=2)

# adjust X to 0 - 1 range
x_train /= 255.
x_test /= 255.

# make y's 1d vector single examples by adding a dimension
# y_train = torch.unsqueeze(y_train, dim=1)
# y_test = torch.unsqueeze(y_test, dim=1)

# convert y float to int
y_train = y_train.type(torch.LongTensor)
y_test = y_test.type(torch.LongTensor)

# add channel information (greyscale image)
x_train = torch.unsqueeze(x_train, dim=1)
x_test = torch.unsqueeze(x_test, dim=1)

# descriptive info of the dataset
print("y's shape: {}\nx's shape: {}".format(y_train.shape, x_train.shape))

# create Dataset & dataloader for Train & Test.
trainset = TensorDataset(x_train, y_train)
testset = TensorDataset(x_test, y_test)

trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=1)
testloader = DataLoader(
    testset, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=1)

# setup the model
# resnet9 = ResNet(
#     img_size=(28, 28),
#     architecture=((1, 8), (8, 8, 8), (8, 8, 8),
#                   (8, 8, 8), (8, 16, 16)),
#     no_classes=10)
#
# resnet3 = ResNet(
#     img_size=(28, 28),
#     architecture=((1, 8), (8, 8, 8)),
#     no_classes=10)

resnet = ResNet(**resnet_config)
resnet.to(DEVICE)

# create, track & run a model with sgd under a specific learning rate
root = os.getcwd()
modeldir = root + '/models/fullrun{}'.format(RUNIDX)
Path(modeldir).mkdir(parents=True, exist_ok=True)
pipe = BlackBoxPipe(
    resnet, trainloader, testloader, epochs=5,
    path=modeldir + '/model_')

# check consecutive runs are tracked
# pipe.evaluate_model_with_SGD(lr=0.001)
# pipe.evaluate_model_with_SGD(lr=0.005)

# pass closure object to BO, which is the bridge from the model to bo
bo = BayesianOptimizer(
    **bo_config,
    closure=pipe.evaluate_model_with_SGD)

bo.optimize(eps=0., initial_lamb=0.01)

# write out the final image
root = os.getcwd()
bo.fig.savefig(root + '/Plots/bo_fullrun{}.pdf'.format(RUNIDX),
               bbox_inches='tight')

# write out the configs & interesting rundata
pickledict = dict(
    resnet_confit=resnet_config,
    bo_config=bo_config,
    losses=pipe.trainlosses,
    incumbent=bo.incumbent,
    costs=bo.costs,
    inquired=bo.inquired,
    accuracy=bo.acc)

modeldir = root + '/models/pickle/'.format(RUNIDX)
Path(modeldir).mkdir(parents=True, exist_ok=True)
filename = modeldir + '/fullrun{}.pkl'.format(RUNIDX)
with open(filename, 'wb') as handle:
    pickle.dump(pickledict, handle, protocol=pickle.HIGHEST_PROTOCOL)

# with open('filename.pickle', 'rb') as handle:
#     b = pickle.load(handle)


# # deprec
# import matplotlib.pyplot as plt
#
# plt.plot(runs.trainlosses[0].detach().numpy())
#
# bo.inquired
# runs.lrs
