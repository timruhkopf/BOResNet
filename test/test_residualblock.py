import unittest
import torch
import torch.nn as nn

import os
from copy import deepcopy

from src.residualblock import ResidualBlock
from src.utils import load_npz_kmnist


class Test_ResidualBlock(unittest.TestCase):

    def no_convs(self, model):
        return len([l for l in model.layers if isinstance(l, nn.Conv2d)])

    def setUp(self) -> None:

        # TODO instead of loading these arrays, consider creating random
        #  tensors.
        # os.chdir("..")
        # os.getcwd()
        # root_data = 'Data/Raw/'
        # self.x_train, self.y_train = load_npz_kmnist(
        #     folder=root_data,
        #     files=['kmnist-train-imgs.npz', 'kmnist-train-labels.npz'])
        #
        # self.y_train = torch.unsqueeze(self.y_train, dim=1)
        # self.x_train = torch.reshape(self.x_train, (-1, 1, 28, 28))
        # print('train_shape', self.x_train.shape)

        batch_size = 1
        self.x = torch.rand([batch_size, 1, 28, 28])
        self.y = torch.rand([batch_size, 8, 28, 28])

    def test_forward_dimensions(self):
        """Test, that the forward path produces the expected shape"""
        residblock = ResidualBlock(
            cunits=(1, 64, 64, 128),
            kernel_size=3)

        self.assertEqual(residblock.forward(self.x[:1]).shape, \
                         torch.Size([1, 128, 28, 28]))

    def test_naive_training(self):
        """run residblock model on an image and check, that the weights
        change & gradients are non zero + the same image has different
        predictions before and after training for a single step."""
        from torch.utils.data import TensorDataset, DataLoader
        from torch.optim import Adam
        from copy import deepcopy


        residblock = ResidualBlock(
            cunits=(1, 8, 8),
            kernel_size=3)

        # # hard coded forward bath
        # train_iter = iter(trainloader)
        # x, y = next(train_iter)
        #
        # y_hat = residblock.layers[0].forward(x) # conv
        # y_hat = residblock.layers[1].forward(y_hat) # bn
        # y_hat = residblock.layers[2].forward(y_hat) # relu
        # y_hat = residblock.layers[3].forward(y_hat) # conv
        # y_hat = residblock.layers[4].forward(y_hat) # bn
        # y_hat += residblock.layers[5].forward(x) # identity or 1x1
        # y_hat = residblock.layers[6].forward(y_hat)
        #
        # self.assertTrue(torch.equal(y_hat, residblock.forward(x)))

        state0 = deepcopy(residblock.state_dict())
        oldstate_prediction = residblock.forward(self.x)

        optimizer = Adam(residblock.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()

        # training step
        optimizer.zero_grad()
        loss = loss_fn(residblock.forward(self.x), self.y)
        loss.backward()
        optimizer.step()

        msg = '{}\'s grad is None still.'
        msg_weights = '{}\'s weights have not changed'
        for name, p in residblock.named_parameters():
            self.assertIsNotNone(p.grad, msg.format(name))
            self.assertFalse(torch.allclose(p.data, state0[name]), msg_weights)

        newstate_prediction = residblock.forward(self.x)
        lossdiff = loss_fn(oldstate_prediction, newstate_prediction)

        if torch.allclose(lossdiff, torch.tensor(0.)):
            raise ValueError('The weights did not change during trainingstep')

    def test_no_of_convolutions(self):
        """check that the scalable residblock actually has the appropriate
        number of convolutions."""

        # unequal sized skip implies 1x1 convoution on skipped image
        # case 1: 3 convolutions + 1x1 convolution: 4 total
        residblock = ResidualBlock(
            cunits=(1, 64, 64, 64),
            kernel_size=3)

        self.assertEqual(self.no_convs(residblock), 4,
                         'number of convolutions are incorrect')

        residblock = ResidualBlock(
            cunits=(64, 64, 64),
            kernel_size=3)

        self.assertEqual(self.no_convs(residblock), 2,
                         'number of convolutions are incorrect')

    def test_cunits_placed_to_convs(self):
        """check that the cunits argument is translated to the appropriate
        amount of channels in the respective convolutions"""
        residblock = ResidualBlock(
            cunits=(1, 2, 3),
            kernel_size=3)

        channels = [(l.in_channels, l.out_channels)
                    for l in residblock.layers \
                    if isinstance(l, nn.Conv2d)]

        # notice that the 1x1 convolution on the skip applies with (1,3)
        self.assertEqual(channels, [(1, 2), (2, 3), (1, 3)])

    # def test_reset_parameters(self):
    #     """check that all weights and biases are resampled"""
    #     residblock = ResidualBlock(
    #         cunits=(1, 2, 3),
    #         kernel_size=3)
    #
    #     state0 = deepcopy(residblock.state_dict())
    #     residblock.reset_parameters()
    #     state1 = deepcopy(residblock.state_dict())
    #
    #     # TODO BE CAREFULL BN has weights and biasses that do not change,
    #     #  if the model was not trainied!
    #     #  ASK if the weights & biases of nn.conv & nn.linear have changed!!
    #     change = [(name, torch.allclose(p0, p)) for (name, p0), p in
    #               zip(state0.items(), state1.values()) \
    #               if 'weight' in name or 'bias' in name]


if __name__ == '__main__':
    unittest.main(exit=False)
