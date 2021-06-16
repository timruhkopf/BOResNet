import torch
from torch.utils.data import TensorDataset, DataLoader
import pyro
import matplotlib
import os

from src.resnet import ResNet
from src.utils import load_npz_kmnist, plot_kmnist
from src.runs import RUNS
from src.bo import BayesianOptimizer

# setup your computation device / plotting method
matplotlib.use('Agg')
BATCH_SIZE = 4
ROOT_DATA = 'Data/Raw/'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

# FIXME: change this back to the full dataset!
n = 10000  # len(x_train)
x_train = x_train[:n]
y_train = y_train[:n]
x_test = x_test[:int(n / 10)]
y_test = y_test[:int(n / 10)]

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

# descriptive info
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

# iterable = testloader.__iter__()
# x, y = next(iterable)

# setup the model
# resnet9 = ResNet(
#     img_size=(28, 28),
#     architecture=((1, 8), (8, 8, 8), (8, 8, 8),
#                   (8, 8, 8), (8, 16, 16)),
#     no_classes=10)

# resnet3 = ResNet(
#     img_size=(28, 28),
#     architecture=((1, 8), (8, 8, 8), (8, 16, 16)),
#     no_classes=10)

resnet9 = ResNet(
    img_size=(28, 28),
    architecture=((1, 16), (16, 16, 16), (16, 16, 16), (16, 32, 32),
                  (32, 32, 32), (32, 64, 64)),
    no_classes=10)

resnet9.to(DEVICE)

# create, track & run a model with sgd under a specific learning rate
root = os.getcwd()
runs = RUNS(
    resnet9, trainloader, testloader, epochs=2,
    path=root + '/models/fullrun/model_')

# runs.evaluate_model_with_SGD(lr=0.001)
# runs.evaluate_model_with_SGD(lr=0.005)
# runs.evaluate_model_with_SGD(lr=0.003)


# pass closure object to BO
bo = BayesianOptimizer(
    search_space=(0.001, 0.02),
    budget=10,  # FIXME: change this
    closure=runs.evaluate_model_with_SGD)

bo.optimize(eps=0., initial_lamb=0.01)

root = os.getcwd()
# bo.fig.legend()  # FIXME: add legend to the plot
bo.fig.savefig(root + '/Plots/bo_fullrun.pdf', bbox_inches='tight')
# print()

#
# # TODO write out the BO images & check for single run on entire dataset,
# #  multiple epochs
# # deprec
# import matplotlib.pyplot as plt
#
# plt.plot(runs.trainlosses[0].detach().numpy())
#
# bo.inquired
# runs.lrs
#
# # TODO: check that proposed !!!!!
