import torch
import torch.nn as nn


class HardcodedResNet(nn.Module):
    def __init__(self, img_size, no_classes, cunits, kernel_size=3):
        super().__init__()

        # IN ---------------
        self.conv0 = nn.Conv2d(cunits[0], cunits[1],
                               kernel_size=7, padding=1)
        self.relu0 = nn.ReLU()
        self.mp0 = nn.MaxPool2d(2, 2)

        # resiblock 1 ---------------
        ref = 0
        self.conv1 = nn.Conv2d(cunits[1 + ref], cunits[2 + ref],
                               kernel_size=kernel_size,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(cunits[2 + ref])
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(cunits[2 + ref], cunits[3 + ref],
                               kernel_size=kernel_size,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(cunits[3 + ref])

        # skip
        if cunits[1 + ref] == cunits[3 + ref]:
            self.skip2 = nn.Identity()
        else:
            self.skip2 = nn.Conv2d(cunits[0 + ref], cunits[+3], 1)
        self.relu2 = nn.ReLU()

        # residblock 2 -------------------
        ref = 2

        self.conv11 = nn.Conv2d(cunits[1 + ref], cunits[2 + ref],
                                kernel_size=kernel_size,
                                padding=1)
        self.bn11 = nn.BatchNorm2d(cunits[2 + ref])
        self.relu11 = nn.ReLU()

        self.conv21 = nn.Conv2d(cunits[2 + ref], cunits[3 + ref],
                                kernel_size=kernel_size,
                                padding=1)
        self.bn21 = nn.BatchNorm2d(cunits[3 + ref])

        # skip
        if cunits[1 + ref] == cunits[3 + ref]:
            self.skip21 = nn.Identity()
        else:
            self.skip21 = nn.Conv2d(cunits[0 + ref], cunits[ref+ 3], 1)
        self.relu21 = nn.ReLU()

        # OUT ---------------
        last_conv_dim = ((img_size + 2 - (7 - 1)) / 2 / 2) ** 2
        linear_dim = cunits[-1] * int(last_conv_dim)

        self.mplast = nn.MaxPool2d(2, 2)
        self.flat = nn.Flatten()
        self.lin = nn.Linear(in_features=linear_dim, out_features=no_classes)
        self.soft = nn.Softmax(dim=-1)

    def forward(self, x):
        y = self.conv0(x)
        y = self.relu0(y)
        y1 = self.mp0(y)

        # residblock1
        y = self.conv1(y1)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y2 = self.bn2(y) + self.skip2(y1)

        # residblock2
        y = self.conv11(y2)
        y = self.bn11(y)
        y = self.relu11(y)
        y = self.conv21(y)
        y = self.bn21(y) + self.skip21(y2)

        y = self.relu2(y)
        y = self.mplast(y)
        y = self.flat(y)
        y = self.lin(y)
        y = self.soft(y)

        return y

    def reset_parameters(self):
        for l in [self.conv0, self.conv1, self.conv11, self.conv21,
                  self.conv2, self.skip2, self.skip21]:
            l.reset_parameters()


if __name__ == '__main__':
    resnet = HardcodedResNet(28, 10, cunits=(1, 8, 8, 8, 16, 16, 16))
    x = torch.rand([1, 1, 28, 28])
    resnet.forward(x)

    from torch.optim import Adam

    optimizer = Adam(resnet.parameters(), lr=0.005)
    loss_fn = nn.CrossEntropyLoss()
    batch = 100
    x = torch.rand([batch, 1, 1, 28, 28])  # added dimension to allow direct
    # loop instead of torch.util.data.Dataloader
    y = torch.randint(10, (batch, 1))

    y = y.type(torch.LongTensor)
    resnet.train()
    losses = []
    for x, y in zip(x, y):
        optimizer.zero_grad()
        y_pred = resnet.forward(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()
        losses.append(loss)

print()
