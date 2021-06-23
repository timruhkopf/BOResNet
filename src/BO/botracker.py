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
        self.noise = noise  # TODO remove noise argument!
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
        # TODO remove noise argument!
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

    def plot_bo(self, n_test=500):
        """
        :param n_test: int. Number of points at which the plot is evaluated.
        """

        plt.rcParams["figure.figsize"] = (20, 20)
        # DO NOT share y! early bad uncertainty estimates may yield
        # non-interpretable visual.
        nrows = self.budget // 2 + self.budget % 2
        self.fig, self.axes = plt.subplots(nrows, 2, sharex=True)
        self.axes = self.axes.flatten().tolist()

        # Remove excess plot (if there is one)
        # Notice, that the first obs. is inquired without a gp, so gp is
        # shorter by one.
        FLAG_REMOVED = False
        if (self.budget - 1) % 2 > 0:
            self.fig.delaxes(self.axes[-1])
            self.axes.pop()
            FLAG_REMOVED = True

        title = 'Bayesian Optimization for steps 2-{}'
        self.fig.suptitle(title.format(self.budget))

        X_test = torch.linspace(*self.search_space, n_test)

        for t, ax in enumerate(self.axes):
            ax.set_xlim(*self.search_space)

            # TODO for common labels: remove labels to share a single label
            #  per side
            # ax.set_xlabel('.', color=(0, 0, 0, 0))
            # ax.set_ylabel('.', color=(0, 0, 0, 0))

            # (a) Plot the observed data points.
            obs = ax.plot(self.inquired[:t + 1].numpy(),
                          self.costs[:t + 1].numpy(),
                          'kx', label='Observed')

            # (b) Plot the current incumbent.
            # Annotate the plot with exact value.
            incumb = self.incumbent[t].numpy()
            incumb_cost = self.costs[self.inc_idx].numpy()
            inc = ax.plot(incumb, incumb_cost, 'o', label='Incumbent')

            # (c1) Plot the cost approximation & uncertainty.
            self.gpr_t = self.gprs[t]
            with torch.no_grad():
                mean, _, sd = self.gpr_t.predict(X_test)

            gp_mean = ax.plot(
                X_test.numpy(), mean.numpy(), 'r',
                label='GP mean', lw=2)

            # (c2) Plot lower-bound-constrained uncertainty:
            # "confidence-bands"
            lower = (mean - 2 * sd).numpy()
            upper = (mean + 2 * sd).numpy()

            gp_sigma = ax.fill_between(
                X_test.numpy(), lower, upper,
                label='GP +/-2 * sd', color='C0', alpha=0.3)

            # EI: plot on the right axis:
            ax_ei_scale = ax.twinx()

            # (d) Plot expected improvement on other axis.
            ei = ExpectedImprovement.eval(self, X_test, self.eps)
            ax_ei_scale.plot(X_test.numpy(), ei.numpy(), label='EI')

            # (e) Plot next candidate
            # max_val = ExpectedImprovement.max_ei(self) # actual recompute
            max_val = self.inquired[t + 1].reshape([1])  # read from runhistory
            ei_val = ExpectedImprovement.eval(self, max_val).numpy()
            max_ei = ax_ei_scale.plot(max_val.numpy(), ei_val, 'v',
                                      label='Max EI')

            handles, labels = ax.get_legend_handles_labels()
            eihandles, eilabels = ax_ei_scale.get_legend_handles_labels()

            handles.extend(eihandles)
            labels.extend(eilabels)
            self.fig.legend(handles, labels, loc='lower right')

            # TODO add common labels for x & y (once only)
            # ax = self.fig.add_subplot(111, frame_on=False)
            #
            # ax.tick_params(labelcolor="none", bottom=False, left=False,
            # top=False, right=False)
            # ax.set_xlabel("X-axis")
            #
            # ax.set_ylabel("Common Y-Axis")
            # ei_axis = ax.twinx()
            # ei_axis.set_ylabel('Common Y2-Axis')
            # ax.axis('off')
            # ei_axis..set_visible(False)

            # FIXME: Add x-ticks to the lower right
            # if FLAG_REMOVED:
            #     self.axes[-2].set_xticks(
            #         torch.linspace(*self.search_space,
            #                        abs(int(self.search_space[0]
            #                        - self.search_space[1]))))
