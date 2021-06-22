import unittest
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.distributions as td

from src.BO.bayesianoptimisation import BayesianOptimizer
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
        return

    def setUp(self) -> None:
        self.COMMIT = get_git_revision_short_hash()

        # SEARCH_SPACE = (10e-5, 10e-1)
        self.SEARCH_SPACE = (-5., -1.)
        self.BUDGET = 10
        self.NOISE = 0.
        self.INIT_lAMB = td.Uniform(*self.SEARCH_SPACE).sample([1])
        self.EPS = 0.

        # Plotting the function.
        x = td.Uniform(*self.SEARCH_SPACE).sample([1000])
        y = self.f(x)
        plt.scatter(x.numpy(), y.numpy())
        plt.show()

        self.bo = BayesianOptimizer(
            search_space=self.SEARCH_SPACE,
            budget=self.BUDGET,
            closure=self.f,
            noise=self.NOISE)

        gp_config = dict(initial_var=0.5, initial_length=0.5, noise=0.)
        self.bo.optimize(initial_guess=self.INIT_lAMB, eps=self.EPS,
                         gp_config=gp_config)

        # 10 ** self.bo.incumbent

        self.path = '/home/tim/PycharmProjects/BOResNet/test/tmp/'
        Path(self.path).mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        pass

    def test_save_and_load_tracker(self):
        # root = os.getcwd()
        # bo.fig.savefig(root + '/testplot/bo_test_powersof10.pdf',
        #                bbox_inches='tight')
        # plt.show()

        # Dump the model.
        self.bo.tracker.save(self.path)

        # Recover the bo.
        tracker = self.bo.tracker.load(self.path)

        # Plot from the recovered.
        tracker.plot_bo()
        plt.show()

        # Save figure to disk
        tracker.fig.savefig('{}/botest_{}.pdf'.format(self.path, self.COMMIT),
                            bbox_inches='tight')


if __name__ == '__main__':
    unittest.main(exit=False)
