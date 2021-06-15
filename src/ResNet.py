import torch.nn as nn

from src.ResidualBlock import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, img_size, architecture, no_classes):
        """
        Residual Neural Net
        Implements a scalable version of the Residual Neural Net from.

        @inproceedings{he2016deep,
          title={Deep residual learning for image recognition},
          author={He, Kaiming and Zhang, Xiangyu and Ren,
                  Shaoqing and Sun, Jian},
          booktitle={Proceedings of the IEEE conference on computer vision
                     and pattern recognition},
          pages={770--778},
          year={2016}
        }

        it is scalable in particular with regard to the number of
        ResdiualBlocks and the number of skiped layers.

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

        # Check the user input of cunits is correct:
        consecutive_units = [t0[-1] == t1[0] for t0, t1 in
                             zip(architecture, architecture[1:])]
        if not all(consecutive_units):
            msg = 'Architecture argument: The channels of consecutive ' \
                  'convolutions are not specified correctly.'
            raise ValueError(msg)

        # "In"-part of the NN without skip connections.
        self.layers = [nn.Conv2d(*architecture[0], kernel_size=7, padding=1),
                       nn.MaxPool2d(2, 2),
                       nn.ReLU()]

        # Residual block expansion incl. solid & dashed lines.
        self.layers.extend([ResidualBlock(channels, kernel_size=3)
                            for channels in architecture[1:]])

        # Final layers for linear combination & class prediction.
        # Assuming square image, its dim is changed by 7x7 conv & max
        # poolings. Original img e.g. 28 padded by 2 (left & right),
        # conv reduces image size by (kernel_size-1). Each max pool of kernel
        # 2 halves the remaining image size).
        # In the end linear layer's input is the flattened conv result
        # e.g. [1, 128, 6, 6] which is a 1 * 128 * 6**2 vector.
        last_conv_dim = ((img_size[0] + 2 - (7 - 1)) / 2 / 2) ** 2
        linear_dim = architecture[-1][-1] * int(last_conv_dim)
        self.layers.extend(
            [nn.MaxPool2d(2, 2), nn.Flatten(),
             nn.Linear(in_features=linear_dim, out_features=no_classes),
             nn.Softmax(dim=-1)])

        # Simplify model by working with sequential.
        self.layers = nn.Sequential(*self.layers)

    def __repr__(self):
        s = 'ResNet(architecture={}, no_classes={})'
        base = s.format(self.architecture, self.no_classes)

        sublayers = '\n\t'.join([str(l) for l in self.layers])
        return '{}\n\t{}'.format(base, sublayers)

    def forward(self, x):
        """
        Forward path of the nn.Module. See nn.Module for details
        :param x: torch.Tensor.
        :return: torch.Tensor
        """
        return self.layers(x)

    def reset_parameters(self):
        """
        Resampling all parameters of the model
        :return: None.
        """
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
