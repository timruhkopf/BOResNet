"""
Author: Tim Ruhkopf
License:
"""

import torch
import torch.nn as nn

from src.ResidualBlock import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, img_size, architecture, no_classes):
        """
        :param img_size: tuple: (row, col) dimensions of the input image.
        :param architecture: tuple of tuples describing the architecture by
        the channels produces by consecutive convolutions.
         e.g. ((1,64), (64, 64, 64) , (64, 128, 128))
         Be well aware that for two consecutive tuples, the last value of
         the former must be the first of the latter. e.g. (..., 64), (64, ...)
         for the architecture to make sense.
        :param no_classes:
        """
        super().__init__()
        self.architecture = architecture
        self.no_classes = no_classes

        # In-part of the NN without skip connections
        self.layers = [nn.Conv2d(*architecture[0], kernel_size=7, padding=1),
                       nn.MaxPool2d(2, 2)]

        # residual block expansion incl. solid & dashed lines
        self.layers.extend([ResidualBlock(channels, kernel_size=3)
                            for channels in architecture[1:]])

        # final layers for linear combination & class prediction
        # assuming square image, linear dim is caused by 7x7 conv & max pooling
        linear_dim = architecture[-1][-1] * int(((img_size[0]-4)/2/2)**2)
        self.layers.extend([nn.MaxPool2d(2, 2),
                            nn.Flatten(),
                            nn.Linear(in_features=linear_dim,
                                      out_features=no_classes),
                            nn.Softmax(dim=-1)])

        # TODO: simplify model to work with sequential!
        self.layers = nn.Sequential(*self.layers)

    def __repr__(self):
        base = 'ResNet(architecture={}, no_classes={})'.format(
            self.architecture, self.no_classes)

        sublayers = '\n\t'.join([str(l) for l in self.layers])
        return '{}\n\t{}'.format(base, sublayers)

    def forward(self, x):
        return self.layers(x)



    #     for layer in self.layers:
    #
    #         x = layer(x)
    #
    #
    #
    #     return x

    # class ResNet9:
    #     def __init__(self, cconfig, skip=2):
    #         """
    #
    #         :param cconfig: tuple of tuples: describes the entire architecture.
    #         e.g. ((28, 28),
    #         :param skip: number of conv layers between a skip connection
    #         """
    #         # TODO: check the cconfig input
    #         if cconfig == False:
    #             raise ValueError()
    #
    #         self.layers = [nn.Conv2d(1, 64, 7),
    #                        nn.MaxPool2d(2, 2),
    #
    #                        # (1) Residblock with skip = 2 & ident. map (same sized)
    #                        # cunits = (64, 64, 64)
    #                        # keep value around
    #                        nn.Conv2d(64, 64, 3, 1),
    #                        nn.BatchNorm2d(),
    #                        nn.ReLU(),
    #
    #                        nn.Conv2d(64, 64, 3, 1),
    #                        nn.BatchNorm2d(),
    #                        # add skip
    #                        nn.ReLU(),
    #
    #                        # (2) Residblock with skip = 2 & ident. map (same sized)
    #                        # cunits = (64, 64, 64)
    #                        # keep value around
    #                        nn.Conv2d(64, 64, 3, 1),
    #                        nn.BatchNorm2d(),
    #                        nn.ReLU(),
    #
    #                        nn.Conv2d(64, 64, 3, 1),
    #                        nn.BatchNorm2d(),
    #                        # add skip
    #                        nn.ReLU(),
    #
    #                        # (3) Residblock skip=2, 1x1 conv for W_s sizing
    #                        # cunits = (64, 128, 128)
    #                        # keep value X_0 around
    #                        nn.Conv2d(64, 128, 3, 1),
    #                        nn.BatchNorm2d(),
    #                        nn.ReLU(),
    #
    #                        nn.Conv2d(128, 128, 3, 1),
    #                        nn.BatchNorm2d(),
    #                        # add skip, but using torch.nn.Conv2d(64, 128, 1)(X_0)
    #                        nn.ReLU(),
    #
    #                        # (4) Residblock with skip = 2 & ident. map (same sized)
    #                        # cunits = (128, 128, 128)
    #                        # keep value around
    #                        nn.Conv2d(128, 128, 3, 1),
    #                        nn.BatchNorm2d(),
    #                        nn.ReLU(),
    #
    #                        nn.Conv2d(128, 128, 3, 1),
    #                        nn.BatchNorm2d(),
    #                        # add skip
    #                        nn.ReLU(),
    #
    #                        nn.MaxPool2d(),
    #                        nn.Flatten(),  # TODO add dim
    #                        nn.Linear(),
    #                        nn.Softmax()
    #                        ]
    #
    #         # input convolution without skip connection & subsequent layers
    #         # self.layers = [nn.Conv2d(cconfig[0]),
    #         #                nn.MaxPool2d(),
    #         #                *[ResidualBlock(param, skip) for param in cconfig[1:]],
    #         #                nn.MaxPool2d(2, 2)]
    #         #
    #         # add the last fully connected linear layer; based on the former
    #         # output size.
    #         # TODO adjust size after max pooling
    #         # self.layers.append(nn.Linear(torch.prod(self.layers[-1].shape[1:])))
    #
    #     def forward(self, X):
    #         # path through ResNet-Conv blocks
    #         for l in self.layers[:-1]:
    #             X = l(X)
    #
    #         # flatten from conv to fc and make prediction after linear layer
    #         X = torch.flatten(X, 1)
    #         return nn.Softmax(self.layers(X))
    #
    #
    # if __name__ == '__main__':
    #     # check forward path works
    #     X = None
    #     ResNet9()
    #     ResNet9.forward(X)
    #
    # m = nn.MaxPool2d(3, stride=2)
    # input = torch.randn(20, 16, 50)
    # output = m(input)
