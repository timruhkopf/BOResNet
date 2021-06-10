import torch
from torch.utils.data import TensorDataset, DataLoader

from src.ResNet import ResNet
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

# adjust X to 0 - 1 range
x_train /= 255.
x_test /= 255.

# make y's 1d vector single examples by adding a dimension
y_train = torch.unsqueeze(y_train, dim=1)
y_test = torch.unsqueeze(y_test, dim=1)

# add channel information (greyscale image)
x_train = torch.unsqueeze(x_train, dim=1)
x_test = torch.unsqueeze(x_test, dim=1)

# descriptive info
print('y\'s shape: {}\nx\'s shape: {}'.format(y_train.shape, x_train.shape))

# create Dataset & dataloader for Train & Test.
trainset = TensorDataset(x_train, y_train)
testset = TensorDataset(x_test, y_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=1)
testloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=1)

# setup the model
# skip = 2
resnet9 = ResNet(
    architecture=((1, 64), (64, 64, 64), (64, 64, 64),
                  (64, 64, 64),  (64, 64, 64),  (128, 128, 128)))

# set up the black box function as closure

def evaluate_model_with_SGD(model, dataloader, epochs, lr):
    """
     Train model once
    training the model on the data from the dataloader for n epochs using sgd,
    with a specified learning rate.
    :param model: instance to a nn.Module
    :param dataloader:
    :param epochs: int.
    :param lr: float.
    :return:
    """
    # TODO look how dataloader was supposed to be sampeled from init
    #  iterator, then next()?
    optimizer = SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = list()  # todo change to tensor
    for e in range(epochs):
        # TODO change syntax to the following:
        # generate an example
        # dataiter = iter(trainloader)
        # images, labels = next(dataiter)
        for X, y in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(model(X), y).backward()
            optimizer.step()
            losses.append(loss)

    return losses
