import torch
import torch.nn as nn
import itertools


class ResidualBlock(nn.Module):
    def __init__(self, hunits, skip=2):
        """
        Same sized Convolution
        :param skip: int. >1
        :param hunits:
        """
        super().__init__()

        if skip < 2:
            msg = 'According to Deep Residual Learning for Image Recognition by' \
                  'He et. al 2015, skips < 2 are not reasonable.'
            raise ValueError(msg)

        self.skip = skip

        # comprehension to efficiently expand the (conv2d + BN + relu) to the
        # desired number of these units - dependent on the hunits parameter.
        self.layers = list(itertools.chain.from_iterable(
            (nn.Conv2d(*config), nn.BatchNorm2d(), nn.ReLU()) for config in
            hunits))

    def forward(self, X):
        for i, l in enumerate(self.layers):

            if i // 3 * self.skip == 0:
                # layers come in ordering (conv2d, BN, ReLU) and are repeated
                # skip number of times before a skip connection is established
                out = l(X) + X
                X = out

            else:
                out = l(X)

        return


if __name__ == '__main__':
    ResidualBlock(skip=3, )
