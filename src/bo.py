import torch
# import torch.nn as nn
import torch.distributions as td
# from torch.optim import SGD
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
from math import pi
import matplotlib.pyplot as plt

from tqdm import tqdm


class BayesianOptimizer:
    def __init__(self, search_space, budget, closure):
        """
        Bayesian Optimization working on a 1d search space.

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
            msg = 'Searchspace Argument order is not correct: (lower, upper)'
            raise ValueError(msg)
        self.search_space = search_space

        # Sample the first data point at random.
        self.inquired = torch.zeros(budget)
        self.cost = torch.zeros(budget)

        # Plot preallocation to gather info along the way rather than
        # recomputing it.
        nrows = self.budget // 2 + self.budget % 2
        self.fig, self.axes = plt.subplots(nrows, 2, sharex=True)
        self.axes = self.axes.flatten()
        title = 'Bayesian Optimization for steps 2-{}'
        # self.fig.set_figheight( self.budget)
        self.fig.suptitle(title.format(budget))
        self.fig_handle = {}

        for ax in self.axes:
            ax.set_xlim(*search_space)

    def gaussian_process(self, X, y, num_steps=2000, noise=0.):
        """
        Fit the Gaussian process to the observed data <X, y>.
        library for gp: https://pyro.ai/examples/gp.html
        :param X: tensor.
        :param y: tensor.
        :param num_steps: int. Number of ADAM steps to optimize the
        :returns None, but adds self.gpr_t to self, which is the MAP
        predictive Model, ready for inquiry.
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y tensors do not match')

        # Do MAP estimation of the GP.
        pyro.clear_param_store()

        # Select the Kernel & set up the GP
        # kernel = gp.kernels.Exponential(input_dim=1,
        #                                 variance=torch.tensor(5.),
        #                                 lengthscale=torch.tensor(10.))
        # kernel = gp.kernels.RBF(input_dim=1, variance=torch.tensor(5.),
        #                         lengthscale=torch.tensor(10.))
        kernel = gp.kernels.Matern32(
            input_dim=1, variance=torch.tensor(1.),
            lengthscale=torch.tensor(1.))
        self.gpr_t = gp.models.GPRegression(
            X, y, kernel,
            noise=torch.tensor(noise),
            jitter=1e-5)

        # TODO consider storing & saving the gprs (for debug purposes)

        # Fitting GP using ELBO -----------------------------------------------
        optimizer = torch.optim.Adam(self.gpr_t.parameters(), lr=0.005)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        losses = []
        for i in range(num_steps):
            optimizer.zero_grad()
            loss = loss_fn(self.gpr_t.model, self.gpr_t.guide)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        msg = 'GP Parameter\n\tVariance: {}\n\tLengthscale: {}\n\tNoise: {}'
        print(msg.format(
            self.gpr_t.kernel.variance.item(),
            self.gpr_t.kernel.lengthscale.item(),
            self.gpr_t.noise.item()))

    def bo_plot(self, X, y, ax, acquisition=True, n_test=500,
                closure=None):
        """
        Plotting the BO process' current iteration.

        incl. the costs' approximation using a Gaussian Process, the
        corresponding expected improvement acquisition function, the next
        selected point of inquiry (max. expected improvement) and the
        currently observed points.

        Notice, that in order to create a single legend,
        this approach (https://stackoverflow.com/a/43940144) is utilized.

        :param X: torch.Tensor.
        :param y: torch.Tensor.
        :param ax: matplotlib axes to plot on.
        :param n_test: int. Number of points for plotting the function.
        :param acquisition: boolean, indicating as to whether the
        acquisition function and its max value shall be plotted.
        :param closure: callable. This allows to plot the true cost
        function, if known. Mainly for testing purposes.
        :return: None, changes self.axes inplace.
        """
        # Plot the observed datapoints.
        obs = ax.plot(X.numpy(), y.numpy(), 'kx', label='Observed')

        # Generate points at which GP is evaluated at for plotting.
        Xtest = torch.linspace(*self.search_space, n_test)
        if closure is not None:
            ax.plot(Xtest.numpy(), closure(Xtest.numpy()))

        # Compute predictive mean and variance for each of test points.
        with torch.no_grad():
            mean, cov = self.gpr_t(Xtest, full_cov=True, noiseless=False)

            # Standard deviation at each input point x (testpoints).
            sd = cov.diag().sqrt()

        # Plot the GP mean prediction.
        gp_mean = ax.plot(
            Xtest.numpy(), mean.numpy(), 'r',
            label='GP mean', lw=2)

        # Plot the two-sigma uncertainty about the GP mean.
        gp_sigma = ax.fill_between(
            Xtest.numpy(),
            # "confidence-bands"
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            label='GP +/-2 * sd', color='C0', alpha=0.3)

        if acquisition:
            ei = ax.plot(
                Xtest.numpy(),
                self.expected_improvement(Xtest, eps=0),
                label='EI')

            # TODO cache that value rather than recompute!
            next_lamb = self.max_ei(precision=50)
            ei_max = ax.plot(
                next_lamb.numpy(),
                self.expected_improvement(next_lamb, eps=0).numpy(),
                'v', label='EI max')

        # Create (unique) handles for the legend.
        # TODO move this to this functions decorator to execute only
        #  at first execution
        # TODO: can't i simply use the first ax and attach only to it the
        #  legend?
        labels = ['Observed', 'gp mean', 'gp 2*sigma', 'EI', 'EI max']
        ax_obj = [obs, gp_mean, gp_sigma, ei, ei_max]
        if not bool(self.fig_handle):
            # only fill fig_handle in the first bo_plot call
            for name, axs in zip(labels, ax_obj):
                if isinstance(axs, list):
                    self.fig_handle[name] = axs[0]
                else:
                    self.fig_handle[name] = axs

    def expected_improvement(self, lamb, eps=0.):
        """
        Calculate the Expected Improvement.

        function definition based on the lecture slides
        https://learn.ki-campus.org/courses/automl-luh2021/items/7rd8zSXREMWYBfbVTXLcci

        :param lamb: torch.Tensor. The values at which the EI is supposed to be
         evaluated at.
        :param eps: float. This parameter allows the user to tip the balance of
        EI towards exploration or exploitation.
        :return: torch.Tensor. The EI evaluated at lamb.
        """
        with torch.no_grad():
            # Inquire the MAP estimate for mu^t(lamb), sd^t(lamb) from GP
            mu_t, cov_t = self.gpr_t(lamb, full_cov=True, noiseless=False)
            var = cov_t.diag()

            # CAREFULL: pyros' GP may produce negative & zero variance
            # predictions (~= -9e-7) ! to avoid producing nans in the following
            # calculations, they are set to 1e-10 instead.
            var[var <= 0] = 1e-10
            sd_t = var.sqrt()

        # Calculate EI.
        Z = (self.cost[self.inc_idx] - mu_t - eps) / sd_t
        u = (sd_t
             * (Z * td.Normal(0., 1.).cdf(Z)
                + torch.exp(td.Normal(0., 1.).log_prob(Z))
                # Normalizing constant of std. Normal required only for
                # plotting the exact EI:
                + torch.tensor([2. * pi]) ** -1 / 2))

        return u

    def max_ei(self, precision=200, eps=0.):
        """
        Naive optimization of the Expected improvement
        This function uses a trivial grid evaluation across the search space
        and evaluates self.expected_improvement for each of these grid points.
        lastly, it returns lamb* = argmax_{lamb} u(lamb).

        :param precision: number of evenly spaced EI evaluations on the
        search space.
        :param eps: float. This parameter allows the user to tip the balance of
        EI towards exploration or exploitation.
        :returns the maximum value of the current EI function
        """
        # Evaluate EI on the entire searchspace.
        lamb = torch.linspace(*self.search_space, steps=precision)
        u = self.expected_improvement(lamb, eps=eps)

        # find lamb = argmax_{lamb} u(lamb)
        argmax = u.max(0)[1]
        return lamb[argmax].reshape((1,))  # = lamb*

    def optimize(self, eps=0., initial_lamb=None):
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

        :param eps: float. This parameter allows the user to tip the balance of
        EI towards exploration or exploitation.
        :param initial_lamb: torch.Tensor. optional initial guess. Default
        is sampling a value uniformly from the search space.
        :return: The incumbent; i.e. the hyperparameter, that minimizes the
        cost function.
        """

        if (initial_lamb < self.search_space[0] or
                initial_lamb > self.search_space[1]):
            raise ValueError('initial_lamb must respect the search space.')

        # Initialize D^(0)
        if initial_lamb is None:
            self.inquired[0] = td.Uniform(*self.search_space).sample([1])
        else:
            self.inquired[0] = initial_lamb

        self.cost[0] = self.closure(float(self.inquired[0].numpy()))
        self.incumbent = self.inquired[0]
        self.inc_idx = 0

        # Optimize lamb using the budget of function evaluations
        for t in tqdm(range(1, self.budget + 1)):
            print('Current incumbent: {} '.format(self.incumbent))
            # Fit predictive model
            # TODO find a third party implementation, that allows online
            #  computation (adding new values rather than creating an
            #  entirely new GP
            self.gaussian_process(X=self.inquired[:t], y=self.cost[:t])

            # Select next point to query.
            # TODO move precision to self.optimize arguments
            lamb = self.max_ei(precision=200, eps=eps)
            self.inquired[t] = lamb

            # Save current iter's bo_plot: the gp + acquisition function.
            # TODO write out plot for t > 1
            self.bo_plot(X=self.inquired[:t],
                         y=self.cost[:t],
                         ax=self.axes[t - 1],
                         acquisition=True)

            # Query cost function.
            self.cost[t] = self.closure(lamb.data.numpy()[0])

            # Replace the incumbent if necessary.
            self.incumbent, self.inc_idx, _ = min(
                [(self.incumbent, self.inc_idx, self.cost[self.inc_idx]),
                 (lamb.data, t, self.cost[t])],
                key=lambda x: x[2])

        # Place the legend with only one unique marker across all axes.
        self.fig.subplots_adjust(right=0.85)
        self.fig.legend(handles=self.fig_handle.values(),
                        borderaxespad=0.1,
                        loc="center right",
                        prop={'size': 6})  # scale the legend

        return self.incumbent
