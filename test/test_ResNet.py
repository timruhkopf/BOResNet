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
        """check different architechtures incl. irregular sized residblocks
        and differently sized channels work as well."""
        # using a different net to the ones used in
        # test_architechture_scaling_data (more residblocks skip=2 & halved
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
        """check the model trains; the gradients & weights change after one
        training step"""
        from torch.utils.data import TensorDataset, DataLoader
        from torch.optim import Adam
        from copy import deepcopy

        batch_size = 1
        x_train = torch.rand([2, 1, 28, 28])
        y_train = torch.randint(0, 10, size=(2, 1))

        trainset = TensorDataset(x_train, y_train)
        trainloader = DataLoader(trainset, batch_size=batch_size,
                                 shuffle=True, num_workers=1)




        # for img, label in trainloader:
        #     optimizer.zero_grad()
if __name__ == '__main__':
    unittest.main(exit=False)
