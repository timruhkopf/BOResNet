import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD

from src.utils import load_npz_kmnist, plot_kmnist
import matplotlib

matplotlib.use('TkAgg')

batch_size = 10

# (0) loading data & preporcessing according to
# https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py

# Load the data
root_data = 'Data/Raw/'
x_train, x_test, y_train, y_test = load_npz_kmnist(
    folder=root_data,
    files=['kmnist-train-imgs.npz', 'kmnist-test-imgs.npz',
           'kmnist-train-labels.npz', 'kmnist-test-labels.npz'])

# plot an example image
plot_kmnist(x_train, y_train,
            labelpath=root_data + 'kmnist_classmap.csv',
            idx=2)

# make y's 1d vector single examples by adding a dimension
y_train = torch.unsqueeze(y_train, dim=1)
y_test = torch.unsqueeze(y_test, dim=1)

# descriptive info
print('y\'s shape: {}\nx\'s shape: {}'.format(y_train.shape, x_train.shape))

# preprocess the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Train simple Resnet-9 Architecture once.
nn.CrossEntropyLoss()
