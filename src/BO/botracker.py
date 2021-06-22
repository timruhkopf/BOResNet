import pickle

import matplotlib.pyplot as plt
import torch

from src.BO.expectedimprovment import ExpectedImprovement


class BoTracker:
    """
    Purpose of this object is to be easily storable. This way, all the
    information of a Bo run can be saved to disk, restored and the plot is
    computed locally rather than remotely. Alterations to the plot are far
    more easily done. Simply overwrite this models plot method before
    restoring an instance from disk.
    """

    def __init__(self, search_space, budget, noise):
        self.search_space = search_space
        self.budget = budget

        # x & y to the observed cost function
        self.costs = torch.zeros(self.budget)
        self.inquired = torch.zeros(self.budget + 1)

        # Gaussian Process objects
        self.gprs = []
        self.noise = noise
        self.inc_idx = 0
        self.incumbent = torch.zeros(self.budget)

        # List of Expected Improvements at each step
        self.ei = []

    def save(self, path):
        # write out gpr models
        gprs = {'gpr_{}'.format(t): gpr for t, gpr in enumerate(self.gprs)}
        torch.save(gprs, '{}/gpr_models'.format(path))

        filename = '{}/BoTracker.pkl'.format(path)
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path, noise=0.):
        """
        Load a BoTracker instance from disk.

        :param path: folder of the residing file /BoTracker.pkl
        # TODO make path the filepath
        :param noise: originally assumed noise of GP.
        :return: Instance to BoTracker.

        :example:
        import os

        path = os.getcwd()
        original = BoTracker((0., 1.), 10)
        original.serialize(path + '/botracker.pkl')
        unpickled = BoTracker.deserialize(path + '/botracker.pkl', **config)
        """

        # Load from disk.
        filename = '{}/BoTracker.pkl'.format(path)
        checkpoint = torch.load('{}/gpr_models'.format(path))
        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        # Instantiate GP's and load their state.
        obj.gprs = []
        for t, k in enumerate(checkpoint):
            obj.gprs.append(checkpoint[k])

        return obj

    def plot_bo(self, n_test):
        """
        :param n_test: int. Number of points at which the plot is evaluated.
        """
        # DO NOT share y! early bad uncertainty estimates may yield
        # non-interpretable visual.
        nrows = self.budget // 2 + self.budget % 2

        self.fig, self.axes = plt.subplots(
            nrows, 2, sharex=True)
        self.axes = self.axes.flatten()

        # Remove excess plot (if there is one)
        if self.budget % 2 > 0:
            self.fig.delaxes(self.axes[-1])

        title = 'Bayesian Optimization for steps 2-{}'
        self.fig.suptitle(title.format(self.budget))

        X_test = torch.linspace(*self.search_space, n_test)
        for t in range(1, self.budget + 1):
            ax = self.axes[t]
            ax.set_xlim(*self.search_space)

            # TODO for common labels: remove labels to share a single label
            #  per side
            # ax.set_xlabel('.', color=(0, 0, 0, 0))
            # ax.set_ylabel('.', color=(0, 0, 0, 0))

            # (a) Plot the observed data points.
            obs = ax.scatter(self.inquired[:t].numpy(),
                          self.costs[:t].numpy(),
                          'kx', label='Observed')

            # (b) Plot the current incumbent.
            # Annotate the plot with exact value.
            inc = ax.scatter(self.incumbent[t].numpy(), self.costs[t].numpy(),
                          '^', label='Incumbent')

            # (c1) Plot the cost approximation & uncertainty.
            with torch.no_grad():
                mean, sd = self.gprs[t](X_test)

            # (c2) Plot lower-bound-constrained uncertainty:
            lower = torch.minimum(torch.zeros_like(sd),
                                  mean - 2 * sd).numpy()
            upper = (mean + 2 * sd).numpy()

            # (e) Plot next candidate
            max_ei = ax.plot()

            # (d) Plot expected improvement on other axis.
            ei = ExpectedImprovement.eval(self, X_test, self.eps)
            self.plot(X_test.numpy(), ei.numpy(), label='EI')

        # TODO add common labels for x & y (once only)
        # self.fig.add_subplot(111, frame_on=False)
        # plt.tick_params(labelcolor="none", bottom=False, left=False)
        # plt.xlabel("Common X-Axis")
        # plt.ylabel("Common Y-Axis")

        plt.show()
