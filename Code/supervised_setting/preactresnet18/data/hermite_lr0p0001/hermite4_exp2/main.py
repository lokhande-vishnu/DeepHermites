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


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--name', default='preactresnet_18_hermite', type=str, help='name of exp')
parser.add_argument('--gpu', default='1', type=str, help='gpuid in str')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
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

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

def set_lr(optimizer, current_lr):
    for g in optimizer.param_groups:
        g['lr'] = current_lr
                    
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        if (batch_idx+1) %100 == 0:
            print('parameter update')
            print(list(net.parameters())[0])

            print('Epoch {} :[{}/{}], Acc: {:.2f}, Loss: {:.4f}'\
                  .format(epoch, batch_idx * len(inputs), len(trainloader.dataset), 100.*correct/total, train_loss/(batch_idx+1)))

    return train_loss, 100.*correct/total



def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
            
    print('TEST ACCURACY Epoch {} , TestLoss: {:.4f}, TestAcc: {:.2f}, BEST_ACC:{:.2f}'\
                .format(epoch, test_loss, 100.*correct/total, best_acc))
    return test_loss, 100.*correct/total, best_acc

def save_everything(epoch, test_acc):
    # Save checkpoint.
    state = {
        'epoch': epoch,
        'test_acc': test_acc,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt_' + str(epoch) + '.t7')

if __name__ == '__main__':
    with open('stat/{}.csv'.format(args.name), 'w+') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'learning_rate', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'best_acc'])
        writer.writeheader()
        
    for epoch in range(start_epoch, start_epoch+200):
        # Set learning rate
        if epoch == 81 or epoch == 122:
            current_lr *= 0.1
            set_lr(optimizer, current_lr)
        
        train_loss, train_acc = train(epoch)
        test_loss, test_acc, best_acc = test(epoch)
        
        with open('stat/{}.csv'.format(args.name), 'a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['epoch', 'learning_rate','train_loss', 'train_acc', 'test_loss', 'test_acc', 'best_acc'])
            writer.writerow({'epoch': epoch, 'learning_rate': current_lr, 'train_loss': train_loss, 'train_acc': train_acc, 'test_loss': test_loss, 'test_acc': test_acc, 'best_acc': best_acc})

        if epoch % 25 == 0:
            save_everything(epoch, test_acc)
            
        
                                    
        
        

    
