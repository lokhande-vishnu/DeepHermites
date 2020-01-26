from torch.utils.data import DataLoader
from lib.datasets.small_norb import SmallNORB
import copy
import numpy as np
import torch
import torchvision.transforms as transforms


def augment():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            10, translate=(0.1, 0.2), scale=(1.1, 1.5), shear=10),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            10, translate=(0.1, 0.2), scale=(1.1, 1.5), shear=10),
        transforms.ToTensor(),
    ])
    return transform_train, transform_test


def get_loaders(nb_labelled, batch_size, unlab_rat, lab_inds=[]):
    transform_train, transform_test = augment()

    trainset_l = SmallNORB(
        root='./data', train=True, download=True, transform=transform_train)
    test_set = SmallNORB(
        root='./data', train=False, download=True, transform=transform_test)
    nb_class = 5

    print('total training dataset - ')
    print(trainset_l.data.shape, len(trainset_l.labels))
    if len(lab_inds) == 0:
        lab_inds = []
        for i in range(nb_class):
            labels = np.array(trainset_l.labels)
            inds_i = np.where(labels == i)[0]
            inds_i = np.random.permutation(inds_i)
            lab_inds.extend(inds_i[0:int(nb_labelled / nb_class)].tolist())
        lab_inds = np.array(lab_inds)

    all_inds = np.arange(len(trainset_l.labels))
    unlab_inds = np.setdiff1d(all_inds, lab_inds)

    trainset_u = copy.deepcopy(trainset_l)
    trainset_u.data = torch.tensor(np.array(trainset_u.data)[unlab_inds])
    trainset_u.labels = torch.tensor(np.array(trainset_u.labels)[unlab_inds])
    trainloader_u = DataLoader(
        trainset_u, batch_size=batch_size, shuffle=False)
    print('unlabelled part of training dataset - ')
    print(trainset_u.data.shape, len(trainset_u.labels))

    trainset_l.data = torch.tensor(np.array(trainset_l.data)[lab_inds])
    trainset_l.labels = torch.tensor(np.array(trainset_l.labels)[lab_inds])
    print('labelled part of training dataset - ')
    print(trainset_l.data.shape, len(trainset_l.labels))
    trainloader_l = DataLoader(trainset_l, batch_size=batch_size, shuffle=True)

    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    loaders = {
        "trainloader_l": trainloader_l,
        "testloader": testloader,
        "trainloader_u": trainloader_u,
        "trainset_l": trainset_l,
        "test_set": test_set,
        "trainset_u": trainset_u,
        "lab_inds": lab_inds
    }
    return loaders


def shuffle_loader(trainset_u_org, batch_size=100):
    nb_unlabelled = len(trainset_u_org.labels)
    inds_all = np.arange(nb_unlabelled)
    inds_shuff = np.random.permutation(inds_all)
    trainset_u = copy.deepcopy(trainset_u_org)
    trainset_u.data = trainset_u_org.data[inds_shuff]
    trainset_u.labels = np.array(trainset_u_org.labels)[inds_shuff]
    trainloader_u = DataLoader(
        trainset_u, batch_size=batch_size, shuffle=False)
    return trainloader_u, torch.from_numpy(inds_shuff).cuda()