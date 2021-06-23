import unittest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pyro
import torch
import torch.distributions as td

from src.bo.bayesianoptimisation import BayesianOptimizer
from src.utils import get_git_revision_short_hash

matplotlib.use('TkAgg')


class Test_BayesianOptimisation(unittest.TestCase):
    @staticmethod
    def g(x):
        return (x - 10 ** -2) ** 2

    @staticmethod
    def f(x):
        return torch.sin(3 * x) + 4.

    @staticmethod
    def h(x):
        """
        h is constructed to be defined on [0,1].
        this function is used to check if lambda x: h(10**x) is
        sufficient to change to log scale during optimisation.
        """
        return torch.sin(3 * x) + 4.

    def setUp(self) -> None:
        self.COMMIT = get_git_revision_short_hash()

        # SEARCH_SPACE = (10e-5, 10e-1)
        self.SEARCH_SPACE = (-5., -1.)
        self.BUDGET = 10
        self.NOISE = 0.
        self.INIT_lAMB = td.Uniform(*self.SEARCH_SPACE).sample([1])
        self.EPS = 0.

        self.path = '/home/tim/PycharmProjects/BOResNet/test/tmp/'
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def test_save_and_load_tracker(self):
        """Ensure, that bo is plottable after loading from tracker."""
        # Plotting the function.
        x = td.Uniform(*self.SEARCH_SPACE).sample([1000])
        y = self.f(x)
        plt.scatter(x.numpy(), y.numpy())
        plt.show()

        self.bo = BayesianOptimizer(
            search_space=self.SEARCH_SPACE,
            budget=self.BUDGET,
            closure=self.f)

        gp_config = dict(initial_var=0.5, initial_length=0.5, noise=0.)
        self.bo.optimize(initial_guess=self.INIT_lAMB, eps=self.EPS,
                         gp_config=gp_config)

        # Dump the model.
        self.bo.tracker.save(self.path)

        # Recover the bo.
        tracker = self.bo.tracker.load(self.path)

        # Plot from the recovered.
        tracker.plot_bo()

        # Save figure to disk
        tracker.fig.savefig('{}/botest_{}.pdf'.format(self.path, self.COMMIT),
                            bbox_inches='tight')

        # TODO write a viable unittest statement

    def test_log_scale_plotting(self):
        """Check if transformation such as log_scale works with plotting."""
        SEARCH_SPACE = (10e-5, 10e-1)

        pyro.set_rng_seed(2)
        torch.manual_seed(2)

        # Plotting the function on original scale.
        # x = td.Uniform(*SEARCH_SPACE).sample([1000])
        # y = self.h(x)
        # plt.scatter(x.numpy(), y.numpy())
        # plt.show()
        #
        # # Plotting func on log10 scale.
        # x = td.Uniform(*self.SEARCH_SPACE).sample([1000])
        # y = self.h(10 ** x)
        # plt.scatter(x.numpy(), y.numpy())
        # plt.show()
        #
        # plt.close()

        # transformation to logscale
        transformed_closure = lambda x: self.h(10 ** x)
        bo = BayesianOptimizer(
            search_space=self.SEARCH_SPACE,  # -5, -1
            budget=self.BUDGET,
            closure=transformed_closure)

        gp_config = dict(initial_var=0.5, initial_length=0.5, noise=0.)
        bo.optimize(initial_guess=self.INIT_lAMB, eps=self.EPS,
                    gp_config=gp_config)

        bo.plot_bo()

        # Save figure to disk
        bo.tracker.fig.savefig('{}/botest_{}_log10.pdf'.format(
            self.path, self.COMMIT), bbox_inches='tight')
        plt.close()

        self.assertTrue(torch.allclose(bo.incumbent[-1], torch.tensor(-5.)))


if __name__ == '__main__':
    unittest.main(exit=False)
