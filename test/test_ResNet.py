import unittest
import torch
# import torch.nn as nn
# from src.utils import load_npz_kmnist
# import os

from src.ResNet import ResNet


class Test_ResidualBlock(unittest.TestCase):

    # def setUp(self) -> None:
    #     os.chdir("..")
    #     os.getcwd()
    #     root_data = 'Data/Raw/'
    #     self.x_train, self.y_train = load_npz_kmnist(
    #         folder=root_data,
    #         files=['kmnist-train-imgs.npz', 'kmnist-train-labels.npz'])
    #
    #     self.y_train = torch.unsqueeze(self.y_train, dim=1)
    #     self.x_train = torch.reshape(self.x_train, (-1, 1, 28, 28))
    #     print('train_shape', self.x_train.shape)

    def test_architechture_scaling_data(self):
        """
        check that the scaling of the image does not break the net's
        definition (linear == fully connected layer is critical in thin
        regard)
        """
        resnet = ResNet(
            img_size=(28, 28),
            architecture=((1, 64), (64, 64, 64), (64, 128, 128)),
            no_classes=10)

        self.assertEqual(resnet(torch.ones((1, 1, 28, 28))).shape,
                         torch.Size([1, 10]))

        resnet = ResNet(
            img_size=(40, 40),
            architecture=((1, 64), (64, 64, 64), (64, 128, 128)),
            no_classes=10)

        self.assertEqual(resnet(torch.ones((1, 1, 40, 40))).shape,
                         torch.Size([1, 10]))

    def test_scaling_architechture(self):
        """check different architechtures incl. irregular sized resnets
        and differently sized channels work as well."""
        # using a different net to the ones used in
        # test_architechture_scaling_data (more resnets skip=2 & halved
        # no. of channels.
        resnet = ResNet(
            img_size=(28, 28),
            architecture=((1, 32), (32, 32, 32), (32, 32, 32), (32, 64, 64)),
            no_classes=10)

        self.assertEqual(resnet(torch.ones((1, 1, 28, 28))).shape,
                         torch.Size([1, 10]))

        # dotted connection (32, 64, 64) implies 1x1 conv!
        # here with a skip of 3
        resnet = ResNet(
            img_size=(28, 28),
            architecture=((1, 32), (32, 32, 32), (32, 32, 32, 32),
                          (32, 64, 64)),
            no_classes=10)

        self.assertEqual(resnet(torch.ones((1, 1, 28, 28))).shape,
                         torch.Size([1, 10]))

        # dotted connection (32, 64, 64, 64) implies 1x1 conv!
        # here with a skip of 3
        resnet = ResNet(
            img_size=(28, 28),
            architecture=((1, 32), (32, 32, 32), (32, 32, 32),
                          (32, 64, 64, 64)),
            no_classes=10)

        self.assertEqual(resnet(torch.ones((1, 1, 28, 28))).shape,
                         torch.Size([1, 10]))

    def test_naive_training(self):
        """run resnet model on an image and check, that the weights
        change & gradients are non zero + the same image has different
        predictions before and after training for a single step.
        Notice, that here MSE loss and not CrossEntropy loss is used.
        This is out of mere convenience."""
        import torch.nn as nn
        from torch.optim import Adam
        from copy import deepcopy

        x = torch.rand([1, 1, 28, 28])
        y = torch.rand([1, 10])

        resnet = ResNet(
            img_size=(28, 28),
            architecture=((1, 32), (32, 32, 32), (32, 32, 32, 32),
                          (32, 64, 64)),
            no_classes=10)
        state0 = deepcopy(resnet.state_dict())
        oldstate_prediction = resnet.forward(x)

        optimizer = Adam(resnet.parameters(), lr=0.005)
        loss_fn = nn.MSELoss()

        # training step
        optimizer.zero_grad()
        loss = loss_fn(resnet.forward(x), y)
        loss.backward()
        optimizer.step()

        msg = '{}\'s grad is None still.'
        msg_weights = '{}\'s weights have not changed'
        for name, p in resnet.named_parameters():
            self.assertIsNotNone(p.grad, msg.format(name))
            self.assertFalse(torch.allclose(p.data, state0[name]), msg_weights)

        newstate_prediction = resnet.forward(x)
        lossdiff = loss_fn(oldstate_prediction, newstate_prediction)

        if torch.allclose(lossdiff, torch.tensor(0.)):
            raise ValueError('The weights did not change during trainingstep')

        print('finished training')


if __name__ == '__main__':
    unittest.main(exit=False)
