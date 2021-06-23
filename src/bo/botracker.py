import pickle

import matplotlib.pyplot as plt
import torch

from src.bo.expectedimprovment import ExpectedImprovement


class BoTracker:
    def __init__(self, search_space, budget):
        """
        Tracker.

        BoTracker is a tracking device, upon which BayesianOptimizer writes
        all its data. Purpose of this object is to be easily storable &
        recoverable. This way, all the information of a Bo run can be saved to
        disk, restored and the plot is computed locally rather than remotely.
        Alterations to the plot are far more easily done. Simply overwrite
        this models plot method before restoring an instance from disk.

        :param search_space: tuple of floats, giving the interval bounds of
        the one dimensional continuous search space.
        :param budget: int. number of function evaluations.
        """
        self.search_space = search_space
        self.budget = budget

        # x & y to the observed cost function
        self.costs = torch.zeros(self.budget)
        self.inquired = torch.zeros(self.budget + 1)

        # Gaussian Process objects
        self.gprs = []
        self.inc_idx = 0
        self.incumbent = torch.zeros(self.budget)

        # List of Expected Improvements at each step
        self.ei = []
        self.max_ei = []

    def save(self, path):
        """
        Save the BoTracker to disk; including gpr_models (in a seperate file).
        :param path: str. Path to a folder on disk
        """
        # Write out gpr models.
        gprs = {'gpr_{}'.format(t): gpr for t, gpr in enumerate(self.gprs)}
        torch.save(gprs, '{}/gpr_models'.format(path))

        # Write out BoTracker data.
        filename = '{}/BoTracker.pkl'.format(path)
        with open(filename, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path):
        """
        Load a BoTracker instance from disk.

        :param path: str. Folder of the residing file /BoTracker.pkl &
        gpr_models files
        :return: Instance to BoTracker.  With all the data recovered from the
        BoTracker.pkl & gpr_models files

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
        1d plotting method of a Bayesian optimisation run

        Plotting multiple iterations including
        (1) the observed hyperparameters and their costs,
        (2) the GPs surrogate estimate of the cost function including mean and
        two times the standard deviation as crude uncertainty estimate,
        (3) expected improvement of the current iteration,
        (4) maximal expected improvement and
        (5) the current incumbent (i.e. best hyperparameter so far).

        :param n_test: int. Number of points at which the plot is evaluated.
        """
        # Set figure size to avoid overplotting of twinx.
        plt.rcParams["figure.figsize"] = (20, 20)

        # DO NOT share y! early bad uncertainty estimates may yield
        # non-interpretable visual.
        nrows = self.budget // 2 + self.budget % 2
        self.fig, self.axes = plt.subplots(nrows, 2)  # sharex=True
        self.axes = self.axes.flatten().tolist()

        # Remove excess plot (if there is one)
        # Notice, that the first obs. is inquired without a gp, so gp is
        # shorter by one.
        if (self.budget - 1) % 2 > 0:
            self.fig.delaxes(self.axes[-1])
            self.axes.pop()

        # Remove ticks & labels from plots (simulate sharex=True - this is
        # the simplest way to work around it, when removing an excess plot.
        for ax in self.axes[:-2]:
            ax.xaxis.set_visible(False)

        title = 'Bayesian Optimization for steps 2-{}'
        self.fig.suptitle(title.format(self.budget))

        X_test = torch.linspace(*self.search_space, n_test)

        for t, ax in enumerate(self.axes):
            ax.set_xlim(*self.search_space)

            # (a) Plot the observed data points.
            obs = ax.plot(self.inquired[:t + 1].numpy(),
                          self.costs[:t + 1].numpy(),
                          'kx', label='Observed')

            # (b) Plot the current incumbent.
            # Annotate the plot with exact value.
            incumb = self.incumbent[t].numpy()
            incidx = torch.argmin(self.costs[:t + 1])
            incumb_cost = self.costs[incidx].numpy()
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
            if len(self.ei) > 0:
                # Read ei from disk.
                ei = self.ei[t]
            else:
                # Evaluate ei again.
                ei = ExpectedImprovement.eval(self, X_test, self.eps)

            ax_ei_scale.plot(X_test.numpy(), ei.numpy(), label='EI')

            # (e) Plot next candidate.
            if len(self.max_ei):
                max_val = self.inquired[t + 1].reshape(
                    [1])  # read from runhistory
                ei_val = self.max_ei[t]
            else:
                max_val = ExpectedImprovement.max_ei(self)  # actual recompute
                ei_val = ExpectedImprovement.eval(self, max_val).numpy()

            max_ei = ax_ei_scale.plot(max_val.numpy(), ei_val, 'v',
                                      label='Max EI')

        handles, labels = ax.get_legend_handles_labels()
        eihandles, eilabels = ax_ei_scale.get_legend_handles_labels()

        handles.extend(eihandles)
        labels.extend(eilabels)
        self.fig.legend(handles, labels, loc='lower right')

        # Set common labels.
        self.fig.text(0.5, 0.04, 'Learning rate (10^lr)', ha='center')
        self.fig.text(0.07, 0.5, 'Avg. CrossEntropyLoss', va='center',
                      rotation='vertical')
        self.fig.text(0.95, 0.5, 'Expected Improvement', va='center',
                      rotation='vertical')
