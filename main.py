import torch
import torch.nn as nn
from torch.optim import SGD
from src.utils import load_npz_kmnist, plot_kmnist
import matplotlib

matplotlib.use('TkAgg')

# (0) loading data & preporcessing according to
# https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py

# Load the data
root_data = 'Data/Raw/'
x_train = load_npz_kmnist(root_data + 'kmnist-train-imgs.npz')
x_test = load_npz_kmnist(root_data + 'kmnist-test-imgs.npz')
y_train = load_npz_kmnist(root_data + 'kmnist-train-labels.npz')
y_test = load_npz_kmnist(root_data + 'kmnist-test-labels.npz')

plot_kmnist(x_train, y_train,
            labelpath=root_data + 'kmnist_classmap.csv',
            idx=2)

# descriptive info
print('y\'s shape: {}\nx\'s shape: {}'.format(y_train.shape, x_train.shape))

# preprocess the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Train simple Resnet-9 Architecture once.
nn.CrossEntropyLoss()
