import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import SGD
import matplotlib

from src.ResNet import ResNet
from src.utils import load_npz_kmnist, plot_kmnist

matplotlib.use('TkAgg')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 2

# (0) loading data & preporcessing according to
# https://github.com/rois-codh/kmnist/blob/master/benchmarks/kuzushiji_mnist_cnn.py
# Load the data
root_data = 'Data/Raw/'
x_train, x_test, y_train, y_test = load_npz_kmnist(
    folder=root_data,
    files=['kmnist-train-imgs.npz', 'kmnist-test-imgs.npz',
           'kmnist-train-labels.npz', 'kmnist-test-labels.npz'])

# testing if training starts at all
x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

# plot an example image
# plot_kmnist(x_train, y_train,
#             labelpath=root_data + 'kmnist_classmap.csv',
#             idx=2)

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

trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=1)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=True, num_workers=1)

# setup the model
# skip = 2
# resnet9 = ResNet(
#     img_size=(28, 28),
#     architecture=((1, 8), (8, 8, 8), (8, 8, 8),
#                   (8, 8, 8), (8, 16, 16)),
#     no_classes=10)

resnet3 = ResNet(
    img_size=(28, 28),
    architecture=((1, 8), (8, 8, 8)),
    no_classes=10)

resnet3.to(device)

iterloader = iter(testloader)
images, labels = next(iterloader)


class RUNS:
    def __init__(self, model, trainloader, testloader, epochs):
        """
        RUN is class to gather all the information across Individual
        Calls to evaluate_model_with_SGD.

        This is a naive tracer.
        :param model: instance to a nn.Module
        :param trainloader:
        :param testloader:
        :param epochs: int. No. of cycles through trainloader
        """
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs

        # track the information
        self.trainlosses = []

    # set up the black box function as closure
    def evaluate_model_with_SGD(self, lr):
        """
         Train model once
        training the model on the data from the dataloader for n epochs using sgd,
        with a specified learning rate.

        :param lr: float.
        :return:
        """
        optimizer = SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.trainlosses.append([torch.zeros(len(self.trainloader))])

        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.trainloader):
                # images.to(device)
                # labels.to(device)

                optimizer.zero_grad()
                y_pred = self.model.forward(images)
                loss = loss_fn(y_pred, labels)

                loss.backward()
                optimizer.step()
                self.trainlosses[-1][i] = loss.sum()



        print('Finished Training')

        # evaluate the cost function on D^test
        cost = torch.tensor([0.])
        # test_acc = torch.tensor([0.])
        with torch.no_grad():
            for images, labels in self.testloader:
                y_pred = self.model.forward(images)
                cost += nn.CrossEntropyLoss(y_pred, labels)

                # _, prediction = torch.max(y_pred.data, 1)

                # test_acc += torch.sum(prediction == labels.data)

        # test_acc = test_acc / len(dataset)
        print('Finished Testing')
        return cost


runs = RUNS(resnet3, testloader, epochs=1)

# TODO write a closure to evaluate optimized resnet on D_validation --> in
#  order to calculate the cost function in BO.
runs.evaluate_model_with_SGD(lr=0.001)

# pass runs.evaluate_model_with_SGD to BO's closure argument
