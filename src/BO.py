import torch

import torch.nn as nn
from torch.optim import SGD


class BayesianOptimizer(torch.optim):
    def __init__(self, budget):
        # TODO check how to instantiate the model anew for each evaluation
        #  of SGD with a specific learning rate!
        self.budget = budget
        pass

    def gaussian_process(self):
        """library for gp: https://pyro.ai/examples/gp.html"""
        pass

    def expected_improvement(self):
        """function definition based on the lecture slides"""
        pass

    def bo_loop(self):
        """pseudo code:

        Require: Search space Λ , cost function c, acquisition function u, pre-
            dictive model ĉ, maximal number of function evaluations T
        Result : Best configuration λ̂ (according to D or ĉ)

        (1) Initialize data D (0) with initial observations
        for t = 1 to T do
            Fit predictive model ĉ^(t) on D^(t−1)
            Select next query point:
                λ^(t) ∈ arg max_{λ ∈Λ} u( λ | D^(t−1) , ĉ^(t))

            Query c(λ^(t))
            Update data: D^(t) ← D^(t−1) ∪ {<λ^(t) , c(λ^(t))>}

        return arg_min_λ c(λ^(t)) from {λ_t}_t=1 ^T
        the return value is called the INCUMBANT
        """
        pass

    def evaluate_model_with_SGD(self, model, dataloader, epochs, lr):
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
