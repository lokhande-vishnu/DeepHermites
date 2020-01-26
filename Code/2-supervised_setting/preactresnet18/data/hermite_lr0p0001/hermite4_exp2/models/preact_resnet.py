'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.activations import Hermite

num_pol = 4

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()

        self.actib1 = Hermite()
        self.actib1_wts = self.actib1.get_vars(num_pol = num_pol)
        self.actib2 = Hermite()
        self.actib2_wts = self.actib2.get_vars(num_pol = num_pol)

        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.softsign(self.actib1.hermite(self.bn1(x), self.actib1_wts, num_pol = num_pol))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = F.softsign(self.conv2(self.actib2.hermite(self.bn2(out), self.actib2_wts, num_pol = num_pol)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()

        self.actib1 = Hermite()
        self.actib1_wts = self.actib1.get_vars(num_pol = num_pol)
        self.actib2 = Hermite()
        self.actib2_wts = self.actib2.get_vars(num_pol = num_pol)
        self.actib3 = Hermite()
        self.actib3_wts = self.actib3.get_vars(num_pol = num_pol)
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.softsign(self.actib1.hermite(self.bn1(x), self.actib1_wts, num_pol=num_pol))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = F.softsign(self.conv2(self.actib2.hermite(self.bn2(out), self.actib2_wts, num_pol = num_pol)))
        out = F.softsign(self.conv3(self.actib3_wts(self.bn3(out), self.actib3_wts, num_pol = num_pol)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.actir = Hermite()
        self.actir_wts = self.actir.get_vars(num_pol = num_pol)

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.softsign(self.actir.hermite(self.bn1(self.conv1(x)), self.actir_wts, num_pol=num_pol))
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    print('USING HERMITES !!!')
    print('NUM of POLS = ', num_pol)
    return PreActResNet(PreActBlock, [2,2,2,2])

def PreActResNet34():
    print('USING HERMITES !!!')
    print('NUM of POLS = ', num_pol)
    return PreActResNet(PreActBlock, [3,4,6,3])

def PreActResNet50():
    print('USING HERMITES !!!')
    print('NUM of POLS = ', num_pol)
    return PreActResNet(PreActBottleneck, [3,4,6,3])

def PreActResNet101():
    print('USING HERMITES !!!')
    print('NUM of POLS = ', num_pol)
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    print('USING HERMITES !!!')
    print('NUM of POLS = ', num_pol)
    return PreActResNet(PreActBottleneck, [3,8,36,3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()


'''
Note: 

There is a bn + relu missing in the preactresnet after conv1
lr = 0.1 is used for relu
weight decay is present

'''
