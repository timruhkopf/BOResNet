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

# TODO consider inherrit from torch.optim
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
        self.budget = budget
        self.closure = closure
        self.search_space = search_space

        # sample the first data point at random.
        self.inquired = torch.zeros(budget)
        self.cost = torch.zeros(budget)

        # TODO consider prior knowledge: different function
        self.inquired[0] = td.Uniform(*search_space).sample([1])
        self.cost[0] = closure(self.inquired[0])
        self.incumbent = self.inquired[0]
        self.inc_idx = 0

    def gaussian_process(self, X, y):
        """library for gp: https://pyro.ai/examples/gp.html"""

        # kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.),
        #                         lengthscale=torch.tensor(10.))
        #
        # # TODO consider Exponential kernel instead
        # # kernel = gp.kernels.Exponential()
        # # TODO consider noise < 1 to smaller uncertainty around observed
        # #  points.
        # gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(1.))
        #
        # optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
        # loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        # losses = []
        # num_steps = 2000
        # for i in range(num_steps):
        #     optimizer.zero_grad()
        #     loss = loss_fn(gpr.model, gpr.guide)
        #     loss.backward()
        #     optimizer.step()
        #     losses.append(loss.item())

        # now do MAP estimation:
        # Define the same model as before.
        pyro.clear_param_store()
        kernel = gp.kernels.Exponential(input_dim=1,
                                        variance=torch.tensor(5.),
                                        lengthscale=torch.tensor(10.))

        # RBF(input_dim=1, variance=torch.tensor(5.),
        #                     lengthscale=torch.tensor(10.))
        gpr = gp.models.GPRegression(X, y, kernel, noise=torch.tensor(0.))

        # note that our priors have support on the positive reals
        gpr.kernel.lengthscale = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))
        gpr.kernel.variance = pyro.nn.PyroSample(dist.LogNormal(0.0, 1.0))

        optimizer = torch.optim.Adam(gpr.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = []
        num_steps = 1000
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # plt.plot(losses)


        self.gpr_t = gpr
        gpr.kernel.variance.item()
        gpr.kernel.lengthscale.item()
        gpr.noise.item()

    def gp_plot(self, X, y, model, n_test=500):
        """

        :param X:
        :param y:
        :param model:
        :param n_test:
        :return:
        """

        plt.figure(figsize=(12, 6))
        # plot the datapoints
        plt.plot(X.numpy(), y.numpy(), 'kx')

        # generate points at which gp is evaluated at for plotting
        Xtest = torch.linspace(*self.search_space, n_test)

        # compute predictive mean and variance for each of these points
        with torch.no_grad():
            mean, cov = model(Xtest, full_cov=True, noiseless=False)

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

        plt.xlim(*self.search_space)

        # TODO add acquisition function & display incumbent

    def expected_improvement(self, lamb, eps):
        """
        function definition based on the lecture slides
        https://learn.ki-campus.org/courses/automl-luh2021/items/7rd8zSXREMWYBfbVTXLcci
        """

        # PSEUDO CODE:
        # mu = None
        # sigma = None
        # Z = (self.cost[self.inc_idx] - mu(lamb) + eps) / sigma(lamb)
        #
        # # if sigma(lamb) >0
        # u_t = sigma(lamb) * (Z * td.Normal.cdf(Z) +
        #                      torch.exp(td.Normal.log_prob(lamb)) +
        #                      # Normalizing constant std. Normal:
        #                      torch.tensor([2. * pi]) ** -1 / 2)
        #
        # else sigma(lamb) =0
        # u_t = torch.tensor([0.])
        pass

    def bo_loop(self):
        """
        pseudo code:

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
        the return value is called the INCUMBENT
        """

        for t in range(1, self.budget):
            # Fit predictive model
            self.gaussian_process(X=self.inquired[:t], y=self.cost[:t])

            # Select next query point:
            # optimize over self.expected_improvement() to find max value lamb.
            lamb = None
            self.inquired[t] = lamb

            # Query cost function
            self.cost[t] = self.closure(lamb)

            # replace the incumbent if necessary
            # TODO check if we deal with loss, that min on neg is correct.
            #  this is a call for a convention on max / min the function!
            self.incumbent, self.inc_idx, _ = min([
                (self.incumbent, self.inc_idx, self.cost[self.inc_idx]),
                (lamb, t, self.cost[t])],
                key=lambda x: x[2])

        return self.incumbent

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
