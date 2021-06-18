import unittest
import torch
import torch.distributions as td
from src.bo import BayesianOptimizer
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


class Test_BO(unittest.TestCase):

    @staticmethod
    def c(x):
        """
        black box function that is easy to query for testing
        :param x: float32,
        :return torch.Tensor of dtype torch.float32
        """
        return torch.tensor(0.01 * (x ** 4 - 8 * x ** 3 - 12 * x +
                                    24) + 4)

    @staticmethod
    def f(x):
        """search space: (-2, 10)"""
        return torch.exp(-(x - 2) ** 2) + \
               torch.exp(-(x - 6) ** 2 / 10) + \
               1 / (x ** 2 + 1)

    def test_bo_on_explicit_function(self):
        """
        Use explicit black box function Test_bo.c, that can be inquired
        easily. This allows to look at the bo part (incl. GP configuration)
        & ei more easily.
        :return:
        """
        SEARCH_SPACE = (-5, 8)
        BUDGET = 4
        NOISE = 0.
        INIT_lAMB = 0.01
        EPS = 0.
        # # Plotting the exemplary function.
        # xs = torch.linspace(-2, 10, 10000)
        # plt.plot(xs, Test_BO.f(xs))
        # plt.show()

        lamb = td.Uniform(*SEARCH_SPACE).sample([100])
        y = torch.tensor(
            0.01 * (lamb ** 4 - 8 * lamb ** 3 - 12 * lamb + 24) + 5,
            dtype=torch.float32)

        # plt.plot(lamb.numpy(), y.numpy(), 'kx')

        bo = BayesianOptimizer(
            search_space=SEARCH_SPACE,
            budget=BUDGET,
            closure=Test_BO.c,
            scale='ident')

        # X = td.Uniform(*(-5, 8)).sample([3])
        # y = Test_BO.c(X)
        # gpr = bo.gaussian_process(X=X, y=y)
        # bo.bo_plot(X, y, gpr, n_test=500, closure=Test_BO.c)

        bo.optimize(initial_lamb=INIT_lAMB, eps=EPS, noise=NOISE)

        plt.show()

        root = os.getcwd()
        bo.fig.savefig(root + '/testplot/bo_testrun.pdf', bbox_inches='tight')

        # TODO write a viable test from this, that is not stochastic!
        # self.assertEqual(True, False, msg='')

    def test_log_scale(self):
        """
        Look at a data example more similar to the learning rate: log scale
        example.
        """
        # SEARCH_SPACE = (10e-5, 10e-1)
        SEARCH_SPACE = (-5, -1)
        BUDGET = 10
        NOISE = 0.
        INIT_lAMB = torch.distributions.Uniform(*SEARCH_SPACE).sample([1])
        EPS = 0.

        def g(x):
            return 2000 * (x - 10 ** -2) ** 2

            # Plotting the function.

        x = td.Uniform(*(10 ** -5, 10 ** -1)).sample([1000])
        y = g(x)
        plt.scatter(x.numpy(), y.numpy())
        plt.show()

        bo = BayesianOptimizer(
            search_space=SEARCH_SPACE,
            budget=BUDGET,
            closure=h,
            scale=None)

        bo.optimize(initial_lamb=INIT_lAMB, eps=EPS, noise=NOISE)
        bo.estimated_gpr_param

        10 ** bo.incumbent
        root = os.getcwd()
        bo.fig.savefig(root + '/testplot/bo_test_powersof10.pdf',
                       bbox_inches='tight')
        plt.show()

        # TODO write a viable test from this, that is not stochastic!


if __name__ == '__main__':
    unittest.main(exit=False)
