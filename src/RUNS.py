import torch
import torch.nn as nn
from torch.optim import SGD
import datetime


class RUNS:
    def __init__(self, model, trainloader, testloader, epochs, path=None):
        """
        RUN is class to gather all the information across Individual
        Calls to evaluate_model_with_SGD.

        This is can act as a naive tracer.
        :param model: instance to a nn.Module
        :param trainloader:
        :param testloader:
        :param epochs: int. No. of cycles through trainloader
        :param path: str. path to a folder/modelbase name, i.e. all models are
        saved to this folder using the modelbase name and a time stamp.
        """
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs

        # track the information
        self.trainlosses = []
        self.lrs = []
        self.costs = []
        self.acc = []

        self.path = path

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

        # pre-allocate a loss tensor for the current run
        # both for plotting purposes
        no_losses = len(self.trainloader.dataset) * \
                    self.epochs / self.trainloader.batch_size
        self.trainlosses.append(torch.zeros(int(no_losses)))
        self.lrs.append(lr)

        self.train(optimizer, loss_fn)
        cost = self.test()

        # save the state of the model & reset the parameters
        # ensuring independent initialisation & model "realisations"
        if self.path is not None:
            s = '{:%Y%m%d_%H%M%S}'
            timestamp = s.format(datetime.datetime.now())
            torch.save(self.model.state_dict(), self.path + timestamp)

        self.model.reset_parameters()
        return cost

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
                train_idx = int(i + len(self.trainloader) * epoch)
                self.trainlosses[-1][train_idx] = loss

        # deprec
        # plt.plot(self.trainlosses[-1].detach().numpy())

        print('Finished training')

    def test(self):
        # evaluate the cost function on D^test
        cost = torch.tensor([0.])
        # test_acc = torch.tensor([0.])
        loss_fn = nn.CrossEntropyLoss()
        test_acc = torch.tensor(0.)
        with torch.no_grad():
            for images, labels in self.testloader:
                y_pred = self.model.forward(images)

                # TODO metric only for testing
                cost += loss_fn(y_pred, labels)

                # accuracy on test data
                _, prediction = torch.max(y_pred.data, 1)
                test_acc += torch.sum(prediction == labels.data)

            test_acc = test_acc / len(self.testloader)
        self.acc.append(test_acc)

        print('Finished testing')

        self.costs.append(cost)
        return cost
