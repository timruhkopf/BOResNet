import torch

from src.bo.botracker import BoTracker
from src.bo.expectedimprovment import ExpectedImprovement
from src.bo.gaussianprocess import GaussianProcess


class BayesianOptimizer:
    def __init__(self, search_space, budget, closure):
        """
        Bayesian Optimization working on a 1d search space.

        Using a Gaussian Process, the cost function is evaluated,
        Subsequently, based on this approximation expected improvement is
        calculated and its max value (=new hyperparameter) is queried from the
        black box model.

        :param search_space: tuple of floats, giving the interval bounds of
        the one dimensional continuous search space.
        :param budget: int. number of function evaluations.
        :param closure: parametrized & callable function that is to be
        optimized
        """
        self.search_space = search_space
        self.budget = budget
        self.closure = closure

        self.tracker = BoTracker(search_space, budget)

        # Make BoTracker's arguments available in this instance.
        # Be aware of the shared object structure (and "right of ownership").
        names = ['costs', 'inquired', 'gprs', 'incumbent', 'inc_idx', 'ei',
                 'max_ei']
        for n in names:
            self.__setattr__(n, self.tracker.__getattribute__(n))

    def plot_bo(self, n_test=500):
        """
        Is a direct call to self.tracker.plot_bo. See its documentation
        for details.
        """
        self.tracker.plot_bo(n_test)

    def optimize(self, initial_guess, eps, gp_config, precision=500):
        """
        Execute bayesian optimization on the provided closure.

        # THE ALGORITHM TO DO THIS:
        Require: Search space Λ , cost function c, acquisition function u, pre-
            dictive model ĉ, maximal number of function evaluations T
        Result : Best configuration λ̂ (according to D or ĉ)

        (1) Initialize data D (0) with initial observations. (done at init
        of BO)
        for t = 1 to T do
            Fit predictive model ĉ^(t) on D^(t−1)
            Select next query point:
                λ^(t) ∈ arg max_{λ ∈Λ} u( λ | D^(t−1) , ĉ^(t))

            Query c(λ^(t))
            Update data: D^(t) ← D^(t−1) ∪ {<λ^(t) , c(λ^(t))>}

        return arg_min_λ c(λ^(t)) from {λ_t}_t=1 ^T
        the return value is called the INCUMBENT

        :param initial_guess: torch.Tensor. Optional initial guess. Default
        is sampling a value uniformly from the search space.
        :param eps: float. This parameter allows the user to tip the balance of
        EI towards exploration or exploitation.
        :param precision: int. Number of values placed on the search space
        and evaluated in the expected improvement to determine the max ei.
        :param gp_config: dict. Configurational details of GP. See
        GaussianProcess documentation for details

        :return: torch.tensor. Incumbent hyperparameter.
        Writes all is data including the GaussianProcess
        objects directly to a BoTracker instance. This allows to store and
        recover all the results of an optimisation run.
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

        return self.incumbent[t]
