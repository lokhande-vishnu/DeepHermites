'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv

from models.preact_resnet import *
from utils import progress_bar

import numpy as np

EPOCH = '175'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--name', default='preactresnet_18_hermite', type=str, help='name of exp')
parser.add_argument('--gpu', default='2', type=str, help='gpuid in str')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# learing rate
current_lr = args.lr

# Model
print('==> Building model..')
net = PreActResNet18()
net = net.cuda()
#if device != 'cpu':
#    net = torch.nn.DataParallel(net)
cudnn.benchmark = True

# Load checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load('./checkpoint/ckpt_' + EPOCH + '.t7')
net.load_state_dict(checkpoint['net'])
print(checkpoint.keys())
best_acc = checkpoint['test_acc']
start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


net.eval()
test_loss = 0
correct = 0
total = 0

activations = torch.zeros(0, dtype = torch.long, device=torch.device('cuda'))
header = []
total_units = 0.
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, layeracts = net(inputs)

        nonzeros = torch.zeros(0, dtype=torch.long, device=torch.device('cuda'))
        for layeract in layeracts:
            if batch_idx == 0:
                header.append('layer_size:' + str(layeract.size(1)))
                total_units = total_units + float(layeract.size(1))
                
            nonzero = torch.sum(abs(layeract) >= 1e-5, 1, keepdim=True)
            #print('nonzero', nonzero.shape)
            nonzeros = torch.cat((nonzeros, nonzero),
                                     dim=1)

        activations = torch.cat((activations, nonzeros),
                                dim=0)
        #print('activations shape', activations.shape)
        
        loss = criterion(outputs, targets)
        
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    #print(activations.float().shape)
    #print(torch.mean(activations.float(), 0, keepdim = True).shape)
    mean_activations = torch.mean(activations.float(), 0, keepdim = True)
    active_units = mean_activations.sum()
    activations = torch.cat((mean_activations, activations.float()),
                            dim = 0)
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
            
    print('TEST ACCURACY Epoch {} , TestLoss: {:.4f}, TestAcc: {:.2f}, BEST_ACC:{:.2f}'\
          .format(start_epoch, test_loss, 100.*correct/total, best_acc))

    print('EPOCH NUMBER:{},  ActiveUnits/TotalUnits {:.1f}/{:.1f}={:.2f} '
          .format(EPOCH, active_units, total_units, 100*active_units/total_units))

    np.savetxt('relu_activations_' + EPOCH + '.csv',
               activations.cpu().numpy(),
               delimiter=",",
               header = ','.join(header))
