''' BasicBlock module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385 '''
from lib.activations import Hermite
import torch.nn as nn
import torch.nn.functional as F

NUM_POL = 4


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        self.actib1 = Hermite()
        self.actib1_wts = self.actib1.get_vars(num_pol=NUM_POL)
        self.actib2 = Hermite()
        self.actib2_wts = self.actib2.get_vars(num_pol=NUM_POL)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False))

    def forward(self, x):
        out = F.softsign(
            self.actib1.hermite(self.bn1(x), self.actib1_wts, num_pol=NUM_POL))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        # V2L Architecture: Pull Softsign out here
        out = F.softsign(
            self.conv2(
                self.actib2.hermite(
                    self.bn2(out), self.actib2_wts, num_pol=NUM_POL)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.actir = Hermite()
        self.actir_wts = self.actir.get_vars(num_pol=NUM_POL)

        self.conv1 = conv3x3(3, 64)  # change to 3 if 3 channel
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_middle=False, from_middle=False):
        if not from_middle:
            out = F.softsign(
                self.actir.hermite(
                    self.bn1(self.conv1(x)), self.actir_wts, num_pol=NUM_POL))
            first_layer = self.layer1(out)

            out = self.layer2(first_layer)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            feat = out.view(out.size(0), -1)
        else:
            feat = x

        out = self.linear(feat)
        if out_middle:
            return out, feat
        else:
            return out


def ResNet18(n=10):
    return ResNet(PreActBlock, [2, 2, 2, 2], n)
