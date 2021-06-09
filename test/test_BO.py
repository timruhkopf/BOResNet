import unittest
import torch
import torch.distributions as td
from src.BO import BayesianOptimizer


class Test_BO(unittest.TestCase):

    @staticmethod
    def c(lamb):
        """
        black box function that is easy to query for testing
        :param lamb: float32,
        :return torch.Tensor of dtype torch.float32
        """
        return torch.tensor(
            0.01 * (lamb ** 4 - 8 * lamb ** 3 - 12 * lamb + 24),
            dtype=torch.float32)

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_gp(self):
        lamb = td.Uniform(*(-5, 8)).sample([100])
        y = torch.tensor(0.01 * (lamb ** 4 - 8 * lamb ** 3 - 12 * lamb + 24),
                         dtype=torch.float32)
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.use('TkAgg')
        plt.plot(lamb.numpy(), y.numpy(), 'kx')

        bo = BayesianOptimizer(search_space=(-5, 8), budget=10,
                               closure=Test_BO.c)

        X = lamb = td.Uniform(*(-5, 8)).sample([3])
        y = Test_BO.c(X)
        bo.gaussian_process(X=X, y=y)
        bo.gp_plot(X, y, bo.gpr_t, n_test=500)
        # self.assertEqual(True, False, msg='')


if __name__ == '__main__':
    unittest.main(exit=False)
