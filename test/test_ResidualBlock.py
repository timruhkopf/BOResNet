import unittest
import torch
import torch.nn as nn
from src.utils import load_npz_kmnist
import os

from src.ResidualBlock import ResidualBlock


class Test_ResidualBlock(unittest.TestCase):

    def no_convs(self, model):
        return len([l for l in model.layers if isinstance(l, nn.Conv2d)])

    def setUp(self) -> None:
        os.chdir("..")
        os.getcwd()
        root_data = 'Data/Raw/'
        self.x_train, self.y_train = load_npz_kmnist(
            folder=root_data,
            files=['kmnist-train-imgs.npz', 'kmnist-train-labels.npz'])

        self.y_train = torch.unsqueeze(self.y_train, dim=1)
        self.x_train = torch.reshape(self.x_train, (-1, 1, 28, 28))
        print('train_shape', self.x_train.shape)

    def test_forward_dimensions(self):
        residblock = ResidualBlock(
            cunits=(1, 64, 64, 128),
            kernel=3)

        self.assertEqual(residblock.forward(self.x_train[:1]).shape, \
                         torch.Size([1, 128, 28, 28]))

    def test_no_of_convolutions(self):
        """check that the scalable residblock actually has the appropriate
        number of convolutions."""

        # unequal sized skip implies 1x1 convoution on skipped image
        # case 1: 3 convolutions + 1x1 convolution: 4 total
        residblock = ResidualBlock(
            cunits=(1, 64, 64, 64),
            kernel=3)

        self.assertEqual(self.no_convs(residblock), 4,
                         'number of convolutions are incorrect')

        residblock = ResidualBlock(
            cunits=(64, 64, 64),
            kernel=3)

        self.assertEqual(self.no_convs(residblock), 2,
                         'number of convolutions are incorrect')

    def test_cunits_placed_to_convs(self):
        residblock = ResidualBlock(
            cunits=(1, 2, 3),
            kernel=3)

        channels = [(l.in_channels, l.out_channels)
                    for l in residblock.layers \
                    if isinstance(l, nn.Conv2d)]

        # notice that the 1x1 convolution on the skip applies with (1,3)
        self.assertEqual(channels, [(1, 2), (2, 3), (1, 3)])


if __name__ == '__main__':
    unittest.main(exit=False)
