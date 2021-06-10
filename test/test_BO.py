import unittest
import torch
import torch.distributions as td
from src.BO import BayesianOptimizer


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

    def test_gp(self):
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use('TkAgg')
        xs = torch.linspace(-2, 10, 10000)
        # plt.plot(xs, Test_BO.f(xs))
        # plt.show()



        lamb = td.Uniform(*(-5, 8)).sample([100])
        y = torch.tensor(
            0.01 * (lamb ** 4 - 8 * lamb ** 3 - 12 * lamb + 24) + 5,
            dtype=torch.float32)

        # plt.plot(lamb.numpy(), y.numpy(), 'kx')

        bo = BayesianOptimizer(search_space=(-5, 8), budget=10,
                               closure=Test_BO.c)

        # X = td.Uniform(*(-5, 8)).sample([3])
        # y = Test_BO.c(X)
        # gpr = bo.gaussian_process(X=X, y=y)
        # bo.bo_plot(X, y, gpr, n_test=500, closure=Test_BO.c)

        bo.optimize()

        # self.assertEqual(True, False, msg='')


if __name__ == '__main__':
    unittest.main(exit=False)
