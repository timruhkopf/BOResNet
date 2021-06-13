import torch
import torch.nn as nn
import torch.distributions as td
from torch.optim import SGD

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from math import pi
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

pyro.set_rng_seed(0)


# TODO Seeding
class BayesianOptimizer:
    def __init__(self, search_space, budget, closure):
        """
        BO working on a 1d search space
        :param search_space: tuple of floats, giving the interval bounds of
        the one dimensional continuous search space.
        :param budget: int. number of function evaluations.
        :param closure: parametrized & callable function that is to be
        optimized
        """
        # TODO check how to instantiate the model anew for each evaluation
        #  of SGD with a specific learning rate!
        self.budget = budget - 1
        self.closure = closure

        if search_space[0] > search_space[1]:
            raise ValueError('Searchspace Argument order is not correct: ('
                             'lower, upper)')
        self.search_space = search_space

        # sample the first data point at random.
        self.inquired = torch.zeros(budget)
        self.cost = torch.zeros(budget)

    def gaussian_process(self, X, y, num_steps=2000, noise=0.):
        """
        fit the Gaussian process to the observed data <X, y>.
        library for gp: https://pyro.ai/examples/gp.html
        :param X: tensor.
        :param y: tensor.
        :param num_steps: int. Number of ADAM steps to optimize the
        :returns None, but adds self.gpr_t to self, which is the MAP
        predictive Model, ready for inquiry.
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y tensors do not match')

        # now do MAP estimation:
        pyro.clear_param_store()
        # kernel = gp.kernels.Exponential(input_dim=1,
        #                                 variance=torch.tensor(5.),
        #                                 lengthscale=torch.tensor(10.))
        # TODO Prevent variance & lengthscale optimization
        # kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.),
        #                         lengthscale=torch.tensor(10.))
        kernel = gp.kernels.Matern32(input_dim=1, variance=torch.tensor(5.),
                                     lengthscale=torch.tensor(10.))
        self.gpr_t = gp.models.GPRegression(X, y, kernel,
                                            noise=torch.tensor(noise),
                                            jitter=1e-5)

        # note that our priors have support on the positive reals
        self.gpr_t.kernel.lengthscale = pyro.nn.PyroSample(
            dist.LogNormal(0.0, 1.0))
        self.gpr_t.kernel.variance = pyro.nn.PyroSample(
            dist.LogNormal(0.0, 1.0))

        optimizer = torch.optim.Adam(self.gpr_t.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = []

        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(self.gpr_t.model, self.gpr_t.guide)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # plt.plot(losses)

        print('GP Parameter\n\t'
              'Variance: {}\n\t'
              'Lengthscale: {}\n\t'
              'Noise: {}'.format(self.gpr_t.kernel.variance.item(),
                                 self.gpr_t.kernel.lengthscale.item(),
                                 self.gpr_t.noise.item()))

    def bo_plot(self, X, y, acquisition=False, n_test=500,
                closure=None):
        """

        :param X: torch.Tensor.
        :param y: torch.Tensor.
        :param n_test: int. Number of points for plotting the function.
        :param closure: callable. This allows to plot the true cost
        function, if known.
        :return:
        """

        # TODO beautify plot: legend, title, axis
        # TODO save plots
        plt.figure(figsize=(12, 6))
        # plot the datapoints
        plt.plot(X.numpy(), y.numpy(), 'kx')

        # generate points at which gp is evaluated at for plotting
        Xtest = torch.linspace(*self.search_space, n_test)
        if closure is not None:
            plt.plot(Xtest.numpy(), closure(Xtest.numpy()))

        # compute predictive mean and variance for each of these points
        with torch.no_grad():
            mean, cov = self.gpr_t(Xtest, full_cov=True, noiseless=False)
            # standard deviation at each input point x (testpoints)
            sd = cov.diag().sqrt()

        # plot the mean prediction
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)

        # plot the two-sigma uncertainty about the mean
        plt.fill_between(Xtest.numpy(),
                         # "confidence-bands"
                         (mean - 2.0 * sd).numpy(),
                         (mean + 2.0 * sd).numpy(),
                         color='C0', alpha=0.3)

        if acquisition:
            plt.plot(Xtest.numpy(), self.expected_improvement(Xtest, eps=0))

            # TODO cache that value rather than recompute!
            next_lamb = self.max_ei(precision=50)
            plt.plot(next_lamb.numpy(),
                     self.expected_improvement(next_lamb, eps=0).numpy(),
                     'v')

        plt.xlim(*self.search_space)
        plt.show()

    def expected_improvement(self, lamb, eps=0.):
        """
        function definition based on the lecture slides
        https://learn.ki-campus.org/courses/automl-luh2021/items/7rd8zSXREMWYBfbVTXLcci
        """
        with torch.no_grad():
            # inquire the MAP estimate for mu^t(lamb), sd^t(lamb)
            mu_t, cov_t = self.gpr_t(lamb, full_cov=True, noiseless=False)
            var = cov_t.diag()
            # FIXME pyros' GP may produce negative & zero variance predictions
            #  (~= -9e-7) ! to avoid producing nans in the following
            #  calculations, they are set to 1e-10 instead.
            var[var <= 0] = 1e-10
            sd_t = var.sqrt()

        # calculate EI
        Z = (self.cost[self.inc_idx] - mu_t - eps) / sd_t
        u = sd_t * (Z * td.Normal(0., 1.).cdf(Z) +
                    torch.exp(td.Normal(0., 1.).log_prob(Z)) +
                    # Normalizing constant of std. Normal required only for
                    # plotting the exact EI:
                    torch.tensor([2. * pi]) ** -1 / 2)

        return u

    def max_ei(self, precision=200, eps=0.):
        """
        Naive optimization of the Expected improvement
        This function uses a trivial grid evaluation across the search space
        and evaluates self.expected_improvement for each of these grid points.
        lastly, it returns lamb* = argmax_{lamb} u(lamb).

        :param precision: number of evenly spaced EI evaluations on the
        search space.
        :returns the maximum value of the current EI function
        """
        lamb = torch.linspace(*self.search_space,
                              steps=precision)

        u = self.expected_improvement(lamb, eps=eps)

        # find lamb* = argmax_{lamb} u(lamb)
        argmax = u.max(0)[1]
        return lamb[argmax].reshape((1,))  # = lamb*

    def max_ei_sgd(self, ei_budget=1000):
        """
        # DEPREC
        The original idea was to maximize ei using autograd: d u / d lamb,
        but since the function may be multimodal that may be parted by flat
        regions (especially if the model does not assume noisy observations
        c(lamb) + e). In this case SGD fails and most likely will stay get
        stuck in local minima.
        :param ei_budget: int. number of SGD steps
        """
        # Select next query point:
        # optimize over self.expected_improvement() to find max value lamb.
        lamb = torch.nn.Parameter(torch.tensor([self.incumbent]))
        optimizer_EI = torch.optim.SGD([lamb], lr=0.05)
        lamb.grad = None
        losses = []

        for i in range(ei_budget):
            optimizer_EI.zero_grad()
            loss = self.expected_improvement(lamb, eps=0.)
            loss.backward()
            optimizer_EI.step()
            losses.append(loss.item())

        lamb.data = lamb
        return lamb

    def optimize(self, eps=0., initial_lamb=None):
        """
        Execute the bayesian optimization on the closure.

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

        :param eps:
        :param initial_lamb: torch.Tensor. optional initial guess. Default
        is sampling a value uniformly from the search space.
        """

        # initialize D^(0)
        if initial_lamb is None:
            self.inquired[0] = td.Uniform(*self.search_space).sample([1])
        else:
            self.inquired[0] = initial_lamb

        # fixme: this saves array object to the list!
        self.cost[0] = self.closure(float(self.inquired[0].numpy()))
        self.incumbent = self.inquired[0]
        self.inc_idx = 0

        # optimize using the budget
        for t in range(1, self.budget + 1):
            print('Current incumbent: {} '.format(self.incumbent))
            # Fit predictive model
            # TODO find a third party implementation, that allows online
            #  computation (adding new values rather than creating an
            #  entirely new GP
            self.gaussian_process(X=self.inquired[:t], y=self.cost[:t])

            # select next point to query
            # TODO move precision to arguments
            lamb = self.max_ei(precision=200, eps=eps)
            self.inquired[t] = lamb

            # plot the gp + acquisition function
            # TODO write out plot for t > 1
            self.bo_plot(X=self.inquired[:t],
                         y=self.cost[:t],
                         acquisition=True)

            # Query cost function
            self.cost[t] = self.closure(lamb.data.numpy()[0])

            # replace the incumbent if necessary
            self.incumbent, self.inc_idx, _ = min([
                (self.incumbent, self.inc_idx, self.cost[self.inc_idx]),
                (lamb.data, t, self.cost[t])],
                key=lambda x: x[2])
        print()
        return self.incumbent
