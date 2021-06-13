import torch
import torch.nn as nn
from torch.optim import SGD


class RUNS:
    def __init__(self, model, trainloader, testloader, epochs):
        """
        RUN is class to gather all the information across Individual
        Calls to evaluate_model_with_SGD.

        This is can act as a naive tracer.
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

    def evaluate_model_with_SGD(self, lr):
        """
        Train model once & evaluate it on the test dataset.
        Training the model on the data from the dataloader for n epochs
        using sgd, with a specified learning rate.

        :param lr: float. learning rate of SGD.
        :return: torch.Tensor. loss of the model trained with lr evaluated
        on the test model.
        """
        optimizer = SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.trainlosses.append([torch.zeros(len(self.trainloader))])

        self.train(optimizer, loss_fn)
        return self.test()  # == cost

    def train(self, optimizer, loss_fn):

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

    def test(self):
        # evaluate the cost function on D^test
        cost = torch.tensor([0.])
        # test_acc = torch.tensor([0.])
        with torch.no_grad():
            for images, labels in self.testloader:
                y_pred = self.model.forward(images)
                cost += nn.CrossEntropyLoss(y_pred, labels)

                # TODO add metrics such as accuracy on test data
                # _, prediction = torch.max(y_pred.data, 1)
                # test_acc += torch.sum(prediction == labels.data)

        # test_acc = test_acc / len(dataset)
        print('Finished Testing')
        return cost
