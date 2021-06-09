import unittest
import torch


class Test_BO(unittest.TestCase):

    @staticmethod
    def f(lamb):
        """black box function that is easy to query for testing
        :param lamb: float32,
        :return torch.Tensor of dtype torch.float32 """
        return torch.tensor([lamb ** 4 - 8 * lamb ** 3 - 12 * lamb + 24],
                            dtype=torch.float32)

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_(self):
        Test_BO.f(2.)

        self.assertEqual(True, False, msg='')


if __name__ == '__main__':
    unittest.main(exit=False)
