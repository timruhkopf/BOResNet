import datetime
import pickle

import torch
import torch.nn as nn
from torch.optim import SGD

from src.utils import get_git_revision_short_hash


class BlackBoxPipe:
    def __init__(self, model, trainloader, testloader, epochs,
                 device, path=None):
        """
        BlackBoxPipe is a naive tracer class to gather all the information
        across individual calls to the evaluate_model_with_SGD function.

        :param model: Instance to a nn.Module subclass.
        :param trainloader: Instance to torch.utils.data.Dataloader.
        :param testloader: Instance to torch.utils.data.Dataloader.
        :param epochs: int. Number of cycles through trainloader.
        :param device: the device that is used for computation during training
        :param path: str. path to a folder/modelbase name, i.e. all models are
        saved to this folder using the modelbase name and a time stamp.
        """
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs

        # Tracked information pre-allocation
        self.trainlosses = []
        self.lrs = []
        self.costs = []
        self.acc = []
        self.confusion_matrices = []
        self.trackstep = 1000

        self.git_commit = get_git_revision_short_hash()

        self.path = path
        self.device = device

    def evaluate_model_with_SGD(self, lr):
        """
        'Black-Box' model.

        This method specifies the training & testing procedure on self.model
        using SGD & the specified learning rate using n epochs on the
        training data. Subsequently, the final model is evaluated on the entire
        testloader once. Uses nn.CrossEntropyLoss in the process.

        :param lr: float. learning rate of SGD.
        :return: torch.Tensor. loss of the model trained with lr evaluated
        on the test model.
        """
        # Reset the model to ensure independent observations
        self.model.reset_parameters()

        optimizer = SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # Pre-allocate a loss tensor for the current run
        # both for plotting purposes.
        no_losses = ((len(self.trainloader.dataset) * self.epochs
                      / self.trainloader.batch_size)
                     // self.trackstep) + 1  # at step 0 loss is evaluated!
        self.trainlosses.append(torch.zeros(int(no_losses)))
        self.lrs.append(lr)

        # Train and evaluate the model
        self.train(optimizer, loss_fn)
        cost = self.test(loss_fn)

        # Save the state of the model & reset the parameters
        # ensuring independent initialisation & model "realisations".
        if self.path is not None:
            git_hash = get_git_revision_short_hash()
            s = '{:%Y%m%d_%H%M%S}'
            timestamp = s.format(datetime.datetime.now())
            torch.save(self.model.state_dict(),
                       '{}/model_{}'.format(self.path, timestamp))
            print('Saved model_{}'.format(timestamp))
        return cost

    def train(self, optimizer, loss_fn):
        """
        Train self.model for self.epochs on the training set.

        :param optimizer: subclass to torch.optim.Optimizer
        :param loss_fn: callable, taking a true vector y & the models
        corresponding prediction.
        :return: None. changes self.model's parameters inplace.
        """
        self.model.train()
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.trainloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                y_pred = self.model.forward(images)
                loss = loss_fn(y_pred, labels)

                loss.backward()
                optimizer.step()

                # write out every 1000's step
                if i % 1000 == 0:
                    train_idx = (int(i + len(self.trainloader) * epoch)
                                 // self.trackstep)
                    self.trainlosses[-1][train_idx] = loss

        print('Finished training')

    def test(self, loss_fn):
        """
        Evaluate the model on the testloader using loss function.

        :param loss_fn: callable. Takes the model's prediction and the
        testloaders' second element as input to compute the loss.
        :return: torch.Tensor.: the accumulated loss.
        """
        # Evaluate accuracy & cost function on D^test.
        num_correct = 0
        num_samples = 0
        cost = 0.
        self.confusion_matrices.append(
            torch.zeros(self.model.no_classes, self.model.no_classes))

        self.model.eval()
        with torch.no_grad():
            for x, y in self.testloader:
                x = x.to(self.device)
                y = y.to(self.device)
                scores = self.model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)

                cost += loss_fn(scores, y)

                # Sort each prediction to the appropriate part of the
                # confusion matrix. Works with batched input.
                for t, p in zip(y.view(-1), predictions.view(-1)):
                    self.confusion_matrices[-1][t.long(), p.long()] += 1

        avg_cost = cost / num_samples
        acc = float(num_correct) / float(num_samples) * 100
        print(f'Got {num_correct} / {num_samples} with accuracy {acc:.2f}')

        # Alternative way to calculate accuracy (from confusion).
        print('Confusion matrix:\n', self.confusion_matrices[-1])

        self.acc.append(acc)
        self.costs.append(avg_cost)
        print('Finished testing')

        return avg_cost

    def flush(self, path):
        """
        Write out all tracked information to disk.
        Information included are the trainlosses (every 1000th step),
        the learningrates, the associated costs (final average
        CrossEntropyLoss on the entire testdataset, the achieved accuracies and
        the associated confusion matrices.

        :param path: str. Specifies the folder in which the output is written.
        """
        pickledict = dict(
            trainlosses=self.trainlosses, lrs=self.lrs, costs=self.costs,
            acc=self.acc, confusion_matrices=self.confusion_matrices)

        filename = path + '/blackboxpipe.pkl'
        with open(filename, 'wb') as handle:
            pickle.dump(pickledict, handle, protocol=pickle.HIGHEST_PROTOCOL)
