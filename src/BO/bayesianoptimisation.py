import torch
import matplotlib.pyplot as plt

from src.BO.botracker import BoTracker
from src.BO.expectedimprovment import ExpectedImprovement
from src.BO.gaussianprocess import GaussianProcess


class BayesianOptimizer:
    def __init__(self, search_space, budget, closure, noise):
        self.search_space = search_space
        self.budget = budget
        self.closure = closure

        self.tracker = BoTracker(search_space, budget, noise)

        # Make BoTracker's arguments available in this instance.
        # Be aware of the shared object structure (and "right of ownership")
        names = ['costs', 'inquired', 'gprs', 'incumbent', 'inc_idx', 'ei']
        for n in names:
            self.__setattr__(n, self.tracker.__getattribute__(n))

    def get_candidate(self, eps, precision):
        # Fit gpr under constraints

        GaussianProcess(X, y, initial_var, initial_length, noise)
        self.gprs.append()

        # Find max EI
        self.gpr_t = self.gprs[-1]  # make avail to EI
        candidate = ExpectedImprovement.max_ei(self, eps, precision)

        pass

    def optimize(self, initial_guess, eps, precision):

        # Update tracker with user input:
        self.tracker.eps = eps
        self.tracker.precision = precision

        # check if initial_guess is in search space
        if initial_guess is None:
            initial_guess = torch.distributions.Uniform(
                *self.search_space).sample([1])
        elif (initial_guess < self.search_space[0] or
              initial_guess > self.search_space[1]):
            raise ValueError('initial_guess must respect the search space.')

        # Inquire costs of initial_guess.
        self.inquired[0] = initial_guess
        self.costs[0] = self.closure(initial_guess)

        for t in range(1, self.budget):
            # Max. expected improvement based on gpr's cost estimates
            self.tracker.inquired[t] = self.get_candidate(eps, precision)

            # Inquire costs of next candidate.
            self.costs[t] = self.closure(self.tracker.inquired[t])

            # Replace the incumbent if necessary.
            self.incumbent, self.inc_idx, _ = min(
                [(self.incumbent, self.inc_idx, self.costs[self.inc_idx]),
                 (self.inquired[t], t, self.costs[t])],
                key=lambda x: x[2])


if __name__ == '__main__':

    class BlackBoxPipe:
        def run_config(self, model, trainloader, testloader, config):
            # Setup.
            self.model = model
            self.trainloader = trainloader
            self.testloader = testloader
            self.optimizer = torch.optim.SGD(config['lr'])
            self.loss_fn = torch.nn.CrossEntropyLoss()

            # Single run.
            self.train(config['epochs'])
            cost = self.test()

            return cost

        def train(self, epochs):
            for e in epochs:
                self.model
                self.loss_fn
                self.optimizer
                pass

        def test(self):
            cost = None
            return cost


    import pickle

    # On REMOTE -------------------------------------------------------
    pipe = BlackBoxPipe()
    bo = BayesianOptimizer(search_space=(-5, -1),
                           budget=10, closure=pipe.run_config)

    bo.optimize()

    path = None
    gpr_config = None

    # Dump remote tracker:
    bo.tracker.save(path)

    # Obtain tracker
    new_tracker = BoTracker.load(path, gpr_config)
