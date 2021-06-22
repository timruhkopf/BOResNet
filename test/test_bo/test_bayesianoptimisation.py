import os
import unittest

import matplotlib.pyplot as plt
import torch.distributions as td

from src.BO.bayesianoptimisation import BayesianOptimizer


class Test_BayesianOptimisation(unittest.TestCase):
    @staticmethod
    def g(x):
        return 2000 * (x - 10 ** -2) ** 2

    def setUp(self) -> None:
        # SEARCH_SPACE = (10e-5, 10e-1)
        SEARCH_SPACE = (-5, -1)
        BUDGET = 10
        NOISE = 0.
        INIT_lAMB = td.Uniform(*SEARCH_SPACE).sample([1])
        EPS = 0.

        # Plotting the function.
        x = td.Uniform(*(10 ** -5, 10 ** -1)).sample([1000])
        y = self.g(x)
        # plt.scatter(x.numpy(), y.numpy())
        # plt.show()

        bo = BayesianOptimizer(
            search_space=SEARCH_SPACE,
            budget=BUDGET,
            closure=self.g,
            noise=NOISE)

        gp_config = dict(initial_var=0.5, initial_length=0.1, noise=0.)
        bo.optimize(initial_guess=INIT_lAMB, eps=EPS, gp_config=gp_config)

        10 ** bo.incumbent

        root = os.getcwd()
        bo.fig.savefig(root + '/testplot/bo_test_powersof10.pdf',
                       bbox_inches='tight')
        plt.show()

    def tearDown(self) -> None:
        pass

    def test_save_and_load_tracker(self):
        self.assertEqual(True, False, msg='')


if __name__ == '__main__':
    unittest.main(exit=False)
