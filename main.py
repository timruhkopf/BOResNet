import torch
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
import os

from src.ResNet import ResNet
from src.utils import load_npz_kmnist, plot_kmnist
from src.RUNS import RUNS
from src.BO import BayesianOptimizer

matplotlib.use('TkAgg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 10

# (0) loading data & preporcessing according to
# https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py
# Load the data
root_data = 'Data/Raw/'
x_train, x_test, y_train, y_test = load_npz_kmnist(
    folder=root_data,
    files=['kmnist-train-imgs.npz', 'kmnist-test-imgs.npz',
           'kmnist-train-labels.npz', 'kmnist-test-labels.npz'])

# testing if training starts at all
# FIXME: change this back to the full dataset!
n = 100  # len(x_train)
x_train = x_train[:n]
y_train = y_train[:n]
x_test = x_test[:int(n/10)]
y_test = y_test[:int(n/10)]

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
print('y\'s shape: {}\nx\'s shape: {}'.format(y_train.shape, x_train.shape))

# create Dataset & dataloader for Train & Test.
trainset = TensorDataset(x_train, y_train)
testset = TensorDataset(x_test, y_test)

trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=1)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=True, num_workers=1)

# iterable = testloader.__iter__()
# x, y = next(iterable)

# setup the model
# skip = 2
# resnet9 = ResNet(
#     img_size=(28, 28),
#     architecture=((1, 8), (8, 8, 8), (8, 8, 8),
#                   (8, 8, 8), (8, 16, 16)),
#     no_classes=10)

resnet3 = ResNet(
    img_size=(28, 28),
    architecture=((1, 8), (8, 8, 8), (8, 16, 16)),
    no_classes=10)

# resnet3.to(device)

# create, track & run a model with sgd under a specific learning rate
runs = RUNS(resnet3, trainloader, testloader, epochs=2)
# runs.evaluate_model_with_SGD(lr=0.001)
# runs.evaluate_model_with_SGD(lr=0.005)
# runs.evaluate_model_with_SGD(lr=0.003)


# pass closure object to BO
bo = BayesianOptimizer(search_space=(0.001, 0.01),
                       budget=5,  # FIXME: change this
                       closure=runs.evaluate_model_with_SGD)
bo.optimize(eps=0., initial_lamb=0.003)

root = os.getcwd()
bo.fig.legend() # FIXME: add legend to the plot
bo.fig.savefig(root +"/Plots/bo_run1.pdf", bbox_inches='tight')
print()


# TODO write out the BO images & check for single run on entire dataset,
#  multiple epochs
# deprec
import matplotlib.pyplot as plt

plt.plot(runs.trainlosses[0].detach().numpy())

bo.inquired
runs.lrs

# TODO: check that proposed !!!!!