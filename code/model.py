import torch
import torch.nn as nn
# import torch.nn.functional as F


def conv3x3(inplane, outplane, stride=1, groups=1, dilation=1):
    return nn.Conv2d(inplane, outplane, kernel_size=3, stride=stride, bias=False,
                     groups=groups, padding=dilation, dilation=dilation)


class DCENet(nn.Module):
    '''https://li-chongyi.github.io/Proj_Zero-DCE.html'''

    def __init__(self, n_LE=8, std=0.02):
        super().__init__()
        self.n_LE = n_LE  # number of iterations of enhancement
        self.std = std

        self.conv1 = conv3x3(3, 32)
        self.conv2 = conv3x3(32, 32)
        self.conv3 = conv3x3(32, 32)
        self.conv4 = conv3x3(32, 32)
        self.conv5 = conv3x3(64, 32)
        self.conv6 = conv3x3(64, 32)
        self.conv7 = conv3x3(64, 3 * n_LE)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=std)
                # nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        out1 = self.relu(self.conv1(x))

        out2 = self.relu(self.conv2(out1))
        out3 = self.relu(self.conv3(out2))
        out4 = self.relu(self.conv4(out3))

        out5 = self.relu(self.conv5(torch.cat((out4, out3), 1)))
        out6 = self.relu(self.conv6(torch.cat((out5, out2), 1)))

        out = self.tanh(self.conv7(torch.cat((out6, out1), 1)))

        return out
