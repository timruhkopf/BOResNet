import torch

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

    def plot_bo(self, n_test=500):
        self.tracker.plot_bo(n_test)
        # TODO add savefig write out. This way, BO can be called at the end
        #  of optimisation to get a first glance of the process.
        #  Fine tuning is still available on local machine by restoring
        #  BoTracker object!

    def optimize(self, initial_guess, eps, gp_config, precision=400):
        """

        :param initial_guess: initial guess on the
        :param eps:
        :param precision:
        :param gp_config:
        :return:
        """

        # Update tracker with user input:
        self.tracker.eps = eps
        self.tracker.precision = precision

        # Check if initial_guess is in search space.
        if initial_guess is None:
            initial_guess = torch.distributions.Uniform(
                *self.search_space).sample([1])
        elif (initial_guess < self.search_space[0] or
              initial_guess > self.search_space[1]):
            raise ValueError('initial_guess must respect the search space.')

        # Inquire costs of initial_guess.
        self.inquired[0] = initial_guess
        self.incumbent[0] = initial_guess
        self.inc_idx = 0
        self.costs[0] = self.closure(initial_guess)

        for t in range(1, self.budget):
            # (0) Fit the Gaussian Process to the observed data.
            self.gpr_t = GaussianProcess(
                x=self.inquired[:t], y=self.costs[:t], **gp_config)
            self.gprs.append(self.gpr_t)
            self.gpr_t.fit_hyperparam(400)

            # (1) Find max EI.

            # Naive method using linspace and ei evals.
            self.inquired[t] = ExpectedImprovement.max_ei(self, precision, eps)

            # Fixme: Gradient based ei does not work due to GP failing to pass
            #  the gradient through
            # self.inquired[t] = ExpextedImprov_grad.max_ei(
            #     self, iteration=t, eps=eps)

            # (2) Inquire costs of next candidate.
            self.costs[t] = self.closure(self.inquired[t])

            # (3) Replace the incumbent if necessary.
            incumbent, self.inc_idx, _ = min(
                [(self.incumbent[t - 1], self.inc_idx,
                  self.costs[self.inc_idx]),
                 (self.inquired[t], t, self.costs[t])],
                key=lambda x: x[2])

            self.incumbent[t] = incumbent


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


    # On REMOTE -------------------------------------------------------
    pipe = BlackBoxPipe()
    bo = BayesianOptimizer(search_space=(-5, -1),
                           budget=10, closure=pipe.run_config)

    bo.optimize()

    path = None
    gpr_config = None

    # Dump remote tracker:
    bo.tracker.save(path)

    # Remove everything
    del pipe, bo

    # Obtain tracker
    new_tracker = BoTracker.load(path, gpr_config)
