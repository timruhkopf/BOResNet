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
                       nn.MaxPool2d(2, 2),
                       nn.ReLU()]

        # residual block expansion incl. solid & dashed lines
        self.layers.extend([ResidualBlock(channels, kernel_size=3)
                            for channels in architecture[1:]])

        # final layers for linear combination & class prediction
        # assuming square image, linear dim is caused by 7x7 conv & max pooling
        last_conv_dim = ((img_size[0] + 2 - (7 - 1)) / 2 / 2) ** 2
        linear_dim = architecture[-1][-1] * int(last_conv_dim)
        # original img e.g. 28 padded by 2 (left & right), conv reduces
        # image size by (kernel_size-1). each max pool of kernel 2 halves
        # the remaining image size)
        # in the end linear layer's input is the flattened conv result
        # e.g. [1, 128, 6, 6] which is a 1 * 128 * 6**2 vector.
        self.layers.extend(
            [nn.MaxPool2d(2, 2), nn.Flatten(),
             nn.Linear(in_features=linear_dim, out_features=no_classes),
             nn.Softmax(dim=-1)])

        # simplify model by working with sequential
        self.layers = nn.Sequential(*self.layers)

    def __repr__(self):
        s = 'ResNet(architecture={}, no_classes={})'
        base = s.format(self.architecture, self.no_classes)

        sublayers = '\n\t'.join([str(l) for l in self.layers])
        return '{}\n\t{}'.format(base, sublayers)

    def forward(self, x):
        return self.layers(x)

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
