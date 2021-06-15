import torch
import torch.nn as nn
import itertools


class ResidualBlock(nn.Module):
    def __init__(self, cunits, kernel_size=3):
        """
        Same sized Convolution
        implements a scalable version of  the dashed and solid line residual
        skip connection blocks from

        @inproceedings{he2016deep,
          title={Deep residual learning for image recognition},
          author={He, Kaiming and Zhang, Xiangyu and Ren,
                  Shaoqing and Sun, Jian},
          booktitle={Proceedings of the IEEE conference on computer vision
                     and pattern recognition},
          pages={770--778},
          year={2016}
        }

        In case of cunits = (64, 64, 64), this amounts to a skip over two (
        no_skips = len(cunits)-1) convolutions:

        ----------------------------------
        |                                |
        x --> conv, bn, relu, conv, bn + x , relu ---> y

        In case of cunits = (64, 64, 64, 64), a skip over three convolutions
        -------------------------------------------------
        |                                                |
        x --> conv, bn, relu, conv, bn, relu, conv, bn + x , relu ---> y


        :param cunits: tuple of consecutive convolutional channels, describing
        the entire residblock's convolutions.
        Starting with no. of channels in the residblock's input tensor,
        and continuing with the consecutive channels introduced by convolution.

        e.g. (1, 64, 128) implies input tensor has 1 channel,
        so the first conv layer has 1 "in" channel and 64 "out" channels.
        the second conv layer then automatically has 64 "in" channels.
        The specified 128 implies this conv layer produces 128 "out" channels.

        In this scenario of unequal no. of channels between beginning and
        last conv., the skip connection cannot be identity, as the same
        sized images (same convolution) have a different number of channels.
        To adjust the size, a 1x1 convolution of appropriate no. of channels
        is used as skip connection before adding the scaled X to the
        residblock's output.
        :param kernel_size: the squared filter size used in all convolutions in
        the residblock.
        """

        # setup nn.Module's bookkeeping
        super().__init__()

        # save arguments for later usage
        self.cunits = cunits
        self.kernel_size = kernel_size

        # scalable version of repetitive parts (conv-bn-relu) to skip over
        # notice, that relu are applied in forward path!
        self.layers = torch.nn.ModuleList(itertools.chain.from_iterable(
            (nn.Conv2d(no_in, no_out, kernel_size=kernel_size, padding=1),
             nn.BatchNorm2d(no_out),
             nn.ReLU())
            for no_in, no_out in zip(cunits[:-2], cunits[1:-1])))

        # final (conv-bn-skip-relu)
        # add identity or 1x1 conv depending on the shape-change in residblock
        self.layers.extend(
            [nn.Conv2d(cunits[-2], cunits[-1], kernel_size, padding=1),
             nn.BatchNorm2d(cunits[-1]),
             # skip connection; either identity or 1x1 conv to adjust shape
             nn.Identity() if cunits[0] == cunits[-1] else \
                 nn.Conv2d(cunits[0], cunits[-1], 1),
             nn.ReLU()])

    def __repr__(self):
        base = 'Residualblock(cunits={}, kernel_size={}):'.format(
            str(self.cunits), self.kernel_size)

        sublayers = '\n\t'.join([str(l) for l in self.layers[:-2]])
        skip = '\n\t'.join([str(l) for l in self.layers[-2:]])

        s = '{}\n\t---scaled block---\n\t{}\n\t---skip connection---\n\t{}'
        return s.format(base, sublayers, skip)

    def forward(self, x):
        xskip = x

        # skipped block until skip connection
        for layer in self.layers[:-2]:
            x = layer(x)

        # relu(X+ ident(X_skip) or  relu(X+ conv1x1(X_skip).
        return self.layers[-1](x + self.layers[-2](xskip))

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
